"""Load a curriculum from Markdown files with YAML frontmatter.

Layout
------

::

    <content_root>/<any-directory>/module.md   # the module's own frontmatter
    <content_root>/<any-directory>/<lesson>.md # one lesson each

A directory without a ``module.md`` is not a module and is not an error --
assets and scratch directories can sit next to content. Versioning is per
file (``schema_version``) rather than per content root, so one stale file
is reported by name instead of failing everything opaquely.

Two properties matter more than anything else here:

**Determinism.** Order comes from the declared ``order``/``slug`` values,
never from the order the filesystem hands back directory entries. Two loads
of the same content root produce equal :class:`~netstack_academy.curriculum.models.Curriculum`
objects, which is what lets everything downstream be compared and cached.

**Fail closed, but report everything.** Any validation problem leaves
``result.curriculum`` as ``None``: a half-valid course is never served,
because a lesson silently missing its quiz is indistinguishable to a
learner from a lesson that has none. At the same time, every problem in
every file is collected into ``result.errors`` before returning, so an
author fixes one round of content instead of rediscovering the next error
on each reload. ``strict=True`` turns that same aggregated list into a
raised :class:`CurriculumValidationError`.

Validation is two-tier, matching the two things a lesson can be. A
``draft`` needs only identity (``schema_version``, ``id``, ``slug``,
``title``, ``order``, ``status``) so an author can commit a title and a
paragraph and see it rendered. A ``published`` lesson is what a learner is
graded against, so it additionally needs the full teaching contract. Quiz
*coherence* -- an answer that names a real option, unique option and
question ids -- is enforced at both tiers, because an ungradeable question
is a bug regardless of whether anyone is meant to see it yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import frontmatter
import yaml

from netstack_academy.indexing.paths import is_safe_relative_path

from .models import (
    LESSON_STATUSES,
    MAX_REVIEW_LEVEL,
    SUPPORTED_SCHEMA_VERSION,
    Curriculum,
    Lab,
    Lesson,
    MasteryGate,
    Module,
    QuizOption,
    QuizQuestion,
    StructureReference,
    SymbolReference,
)
from .rendering import render_markdown

#: The file that makes a directory a module.
MODULE_FILENAME = "module.md"

_MARKDOWN_GLOB = "*.md"

_MODULE_KEYS = frozenset(
    {"schema_version", "id", "slug", "title", "order", "summary"}
)

_LESSON_KEYS = frozenset(
    {
        "schema_version",
        "id",
        "slug",
        "title",
        "order",
        "status",
        "summary",
        "objectives",
        "prerequisites",
        "packet_stage",
        "execution_context",
        "ownership",
        "locking",
        "rcu",
        "structures",
        "config_caveats",
        "version_caveats",
        "tracepoints",
        "source_symbols",
        "lab",
        "quiz",
        "mastery_gate",
    }
)

_SYMBOL_KEYS = frozenset({"name", "path"})
_STRUCTURE_KEYS = frozenset({"name", "fields"})
_LAB_KEYS = frozenset({"commands", "expected_observations", "cleanup"})
_QUESTION_KEYS = frozenset({"id", "prompt", "options", "answer", "explanation"})
_OPTION_KEYS = frozenset({"id", "text"})
_MASTERY_GATE_KEYS = frozenset({"min_quiz_score", "required_review_level"})

#: Free-text lesson fields: optional for a draft, required when published.
_TEXT_KEYS = (
    "summary",
    "packet_stage",
    "execution_context",
    "ownership",
    "locking",
    "rcu",
)

#: Plain lists of strings, all optional at both tiers.
_STRING_LIST_KEYS = (
    "objectives",
    "prerequisites",
    "config_caveats",
    "version_caveats",
    "tracepoints",
)

#: Published lessons need each of these to be present and non-empty.
_PUBLISHED_TEXT_KEYS = ("summary", "packet_stage", "execution_context")
_PUBLISHED_COLLECTION_KEYS = ("objectives", "source_symbols", "quiz")


@dataclass(frozen=True, slots=True)
class CurriculumError:
    """One validation problem, located precisely enough to fix.

    ``path`` is the offending file's path relative to the content root (or
    ``""`` for a problem with the root itself); ``field`` names the
    frontmatter key, using dotted/indexed notation for nested values
    (``lab.expected_observations``, ``quiz[0].answer``), and is ``None``
    when the whole file is the problem.
    """

    path: str
    field: str | None
    message: str


@dataclass(frozen=True, slots=True)
class CurriculumLoadResult:
    """The outcome of a load: a curriculum, or the reasons there isn't one."""

    curriculum: Curriculum | None
    errors: tuple[CurriculumError, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


class CurriculumValidationError(ValueError):
    """Raised by ``load_curriculum(..., strict=True)`` when content is invalid.

    Carries the same aggregated ``errors`` a non-strict load would have
    returned, so a caller that prefers exceptions still gets the whole list
    rather than only the first problem.
    """

    def __init__(self, errors: Sequence[CurriculumError]) -> None:
        self.errors: tuple[CurriculumError, ...] = tuple(errors)
        summary = "; ".join(
            f"{error.path or '<content root>'}: {error.message}" for error in self.errors
        )
        super().__init__(f"Invalid curriculum content ({len(self.errors)} error(s)): {summary}")


class _ErrorSink:
    """Collects the errors found in one file."""

    def __init__(self, path: str) -> None:
        self._path = path
        self.errors: list[CurriculumError] = []

    def add(self, message: str, *, field: str | None = None) -> None:
        self.errors.append(CurriculumError(path=self._path, field=field, message=message))

    def __bool__(self) -> bool:
        return bool(self.errors)


@dataclass(frozen=True, slots=True)
class _LoadedModule:
    """A parsed module plus the file it came from (for error reporting)."""

    module: Module
    source_path: str


def load_curriculum(
    content_root: Path | str, *, strict: bool = False
) -> CurriculumLoadResult:
    """Load, validate and order the curriculum rooted at ``content_root``."""
    root = Path(content_root)
    errors: list[CurriculumError] = []

    if not root.is_dir():
        errors.append(
            CurriculumError(
                path="",
                field=None,
                message=f"Curriculum content root does not exist: {root}",
            )
        )
        return _finish(None, errors, strict=strict)

    loaded: list[_LoadedModule] = []
    for directory in sorted(entry for entry in root.iterdir() if entry.is_dir()):
        if not (directory / MODULE_FILENAME).is_file():
            # Not a module: assets and scratch directories may live here.
            continue
        module = _load_module(directory, root, errors)
        if module is not None:
            loaded.append(module)

    _check_uniqueness(loaded, errors)
    _check_prerequisites(loaded, errors)

    ordered = tuple(
        entry.module
        for entry in sorted(
            loaded, key=lambda entry: (entry.module.order, entry.module.slug)
        )
    )
    curriculum = Curriculum(schema_version=SUPPORTED_SCHEMA_VERSION, modules=ordered)
    return _finish(curriculum, errors, strict=strict)


def _finish(
    curriculum: Curriculum | None,
    errors: Sequence[CurriculumError],
    *,
    strict: bool,
) -> CurriculumLoadResult:
    ordered_errors = tuple(
        sorted(errors, key=lambda error: (error.path, error.field or "", error.message))
    )
    if ordered_errors:
        if strict:
            raise CurriculumValidationError(ordered_errors)
        return CurriculumLoadResult(curriculum=None, errors=ordered_errors)
    return CurriculumLoadResult(curriculum=curriculum, errors=())


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _read_document(
    path: Path, sink: _ErrorSink
) -> tuple[dict[str, Any], str] | None:
    """Read one Markdown file into ``(frontmatter, body)``."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        sink.add(f"Could not read file: {exc}")
        return None

    try:
        metadata, body = frontmatter.parse(text)
    except yaml.YAMLError as exc:
        sink.add(f"Invalid YAML frontmatter: {exc}")
        return None

    if not isinstance(metadata, dict) or not metadata:
        sink.add("File has no YAML frontmatter")
        return None

    return metadata, body


def _load_module(
    directory: Path, root: Path, errors: list[CurriculumError]
) -> _LoadedModule | None:
    module_path = directory / MODULE_FILENAME
    relative_module_path = _relative_path(module_path, root)
    sink = _ErrorSink(relative_module_path)

    module_id: str | None = None
    slug: str | None = None
    title: str | None = None
    order: int | None = None
    summary: str | None = None

    document = _read_document(module_path, sink)
    if document is not None:
        metadata = document[0]
        _check_unknown_keys(metadata, _MODULE_KEYS, sink)
        _check_schema_version(metadata, sink)
        module_id = _required_text(metadata, "id", sink)
        slug = _required_text(metadata, "slug", sink)
        title = _required_text(metadata, "title", sink)
        order = _required_int(metadata, "order", sink)
        summary = _optional_text(metadata, "summary", sink)

    # Lessons are read even when the module itself is invalid, so one
    # broken module.md does not hide every problem underneath it.
    lessons = _load_lessons(directory, root, module_id or "", errors)

    errors.extend(sink.errors)
    if sink or module_id is None or slug is None or title is None or order is None:
        return None

    module = Module(
        id=module_id,
        slug=slug,
        title=title,
        order=order,
        summary=summary or "",
        lessons=lessons,
    )
    return _LoadedModule(module=module, source_path=relative_module_path)


def _load_lessons(
    directory: Path, root: Path, module_id: str, errors: list[CurriculumError]
) -> tuple[Lesson, ...]:
    lessons: list[Lesson] = []
    for lesson_path in sorted(directory.glob(_MARKDOWN_GLOB)):
        if lesson_path.name == MODULE_FILENAME or not lesson_path.is_file():
            continue
        lesson = _load_lesson(lesson_path, root, module_id, errors)
        if lesson is not None:
            lessons.append(lesson)

    return tuple(sorted(lessons, key=lambda lesson: (lesson.order, lesson.slug)))


def _load_lesson(
    path: Path, root: Path, module_id: str, errors: list[CurriculumError]
) -> Lesson | None:
    relative_path = _relative_path(path, root)
    sink = _ErrorSink(relative_path)

    document = _read_document(path, sink)
    if document is None:
        errors.extend(sink.errors)
        return None

    metadata, body = document
    _check_unknown_keys(metadata, _LESSON_KEYS, sink)
    _check_schema_version(metadata, sink)

    lesson_id = _required_text(metadata, "id", sink)
    slug = _required_text(metadata, "slug", sink)
    title = _required_text(metadata, "title", sink)
    order = _required_int(metadata, "order", sink)
    status = _lesson_status(metadata, sink)

    texts = {key: _optional_text(metadata, key, sink) for key in _TEXT_KEYS}
    lists = {key: _string_tuple(metadata, key, sink) for key in _STRING_LIST_KEYS}
    structures = _parse_structures(metadata.get("structures"), sink)
    source_symbols = _parse_symbols(metadata.get("source_symbols"), sink)
    lab = _parse_lab(metadata.get("lab"), sink)
    quiz = _parse_quiz(metadata.get("quiz"), sink, published=status == "published")
    mastery_gate = _parse_mastery_gate(metadata.get("mastery_gate"), sink)

    if status == "published":
        _check_published_contract(metadata, texts, lab, sink)

    errors.extend(sink.errors)
    if sink or lesson_id is None or slug is None or title is None or order is None:
        return None

    return Lesson(
        id=lesson_id,
        slug=slug,
        title=title,
        order=order,
        module_id=module_id,
        status=status or "draft",
        source_path=relative_path,
        summary=texts["summary"] or "",
        objectives=lists["objectives"],
        prerequisites=lists["prerequisites"],
        packet_stage=texts["packet_stage"],
        execution_context=texts["execution_context"],
        ownership=texts["ownership"],
        locking=texts["locking"],
        rcu=texts["rcu"],
        structures=structures,
        config_caveats=lists["config_caveats"],
        version_caveats=lists["version_caveats"],
        tracepoints=lists["tracepoints"],
        source_symbols=source_symbols,
        lab=lab,
        quiz=quiz,
        mastery_gate=mastery_gate,
        body_markdown=body,
        body_html=render_markdown(body),
    )


def _check_unknown_keys(
    metadata: Mapping[str, Any], allowed: frozenset[str], sink: _ErrorSink
) -> None:
    """Reject keys outside the schema.

    Ignoring them would turn an author's typo (``objectivez:`` for
    ``objectives:``) into content that renders but teaches nothing.
    """
    for key in sorted(str(key) for key in metadata if key not in allowed):
        sink.add(f"Unknown frontmatter key {key!r}", field=key)


def _check_schema_version(metadata: Mapping[str, Any], sink: _ErrorSink) -> None:
    version = metadata.get("schema_version")
    if version is None:
        sink.add("Missing required field 'schema_version'", field="schema_version")
    elif version != SUPPORTED_SCHEMA_VERSION:
        sink.add(
            f"Unsupported schema_version {version!r} "
            f"(this build understands {SUPPORTED_SCHEMA_VERSION})",
            field="schema_version",
        )


def _required_text(
    metadata: Mapping[str, Any], key: str, sink: _ErrorSink
) -> str | None:
    value = metadata.get(key)
    if value is None:
        sink.add(f"Missing required field {key!r}", field=key)
        return None
    if not isinstance(value, str) or not value.strip():
        sink.add(f"Field {key!r} must be a non-empty string, got {value!r}", field=key)
        return None
    return value


def _required_int(
    metadata: Mapping[str, Any], key: str, sink: _ErrorSink
) -> int | None:
    value = metadata.get(key)
    if value is None:
        sink.add(f"Missing required field {key!r}", field=key)
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        sink.add(f"Field {key!r} must be an integer, got {value!r}", field=key)
        return None
    return value


def _optional_text(
    metadata: Mapping[str, Any], key: str, sink: _ErrorSink
) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        sink.add(f"Field {key!r} must be a string, got {value!r}", field=key)
        return None
    return value


def _string_tuple(
    metadata: Mapping[str, Any], key: str, sink: _ErrorSink
) -> tuple[str, ...]:
    return _string_tuple_from(metadata, key, key, sink)


def _string_tuple_from(
    container: Mapping[str, Any], key: str, field: str, sink: _ErrorSink
) -> tuple[str, ...]:
    value = container.get(key)
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        sink.add(f"Field {field!r} must be a list of strings, got {value!r}", field=field)
        return ()

    entries: list[str] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, str) or not entry.strip():
            sink.add(
                f"{field}[{index}] must be a non-empty string, got {entry!r}",
                field=f"{field}[{index}]",
            )
            continue
        entries.append(entry)
    return tuple(entries)


def _lesson_status(metadata: Mapping[str, Any], sink: _ErrorSink) -> str | None:
    value = metadata.get("status")
    if value is None:
        sink.add("Missing required field 'status'", field="status")
        return None
    if value not in LESSON_STATUSES:
        sink.add(
            f"Unknown publication status {value!r} "
            f"(expected one of {', '.join(LESSON_STATUSES)})",
            field="status",
        )
        return None
    return str(value)


def _parse_structures(
    value: Any, sink: _ErrorSink
) -> tuple[StructureReference, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        sink.add(
            f"Field 'structures' must be a list of {{name, fields}} mappings, got {value!r}",
            field="structures",
        )
        return ()

    structures: list[StructureReference] = []
    for index, entry in enumerate(value):
        field = f"structures[{index}]"
        if not isinstance(entry, Mapping):
            sink.add(f"{field} must be a mapping with a 'name'", field=field)
            continue
        _check_nested_keys(entry, _STRUCTURE_KEYS, field, sink)
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            sink.add(f"{field} is missing a structure 'name'", field=f"{field}.name")
            continue
        structures.append(
            StructureReference(
                name=name, fields=_string_tuple_from(entry, "fields", f"{field}.fields", sink)
            )
        )
    return tuple(structures)


def _parse_symbols(value: Any, sink: _ErrorSink) -> tuple[SymbolReference, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        sink.add(
            f"Field 'source_symbols' must be a list of {{name, path}} mappings, got {value!r}",
            field="source_symbols",
        )
        return ()

    symbols: list[SymbolReference] = []
    for index, entry in enumerate(value):
        field = f"source_symbols[{index}]"
        if not isinstance(entry, Mapping):
            sink.add(f"{field} must be a mapping with a symbol 'name'", field=field)
            continue
        _check_nested_keys(entry, _SYMBOL_KEYS, field, sink)

        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            sink.add(f"{field} is missing a symbol 'name'", field=f"{field}.name")
            continue

        path = entry.get("path")
        if path is None:
            # A name alone is enough: the index resolves it, and an author
            # should not have to know a symbol's file to reference it.
            symbols.append(SymbolReference(name=name))
            continue

        if not isinstance(path, str) or not is_safe_relative_path(path):
            sink.add(
                f"{field} path {path!r} is not a safe repository-relative path",
                field=f"{field}.path",
            )
            continue

        symbols.append(SymbolReference(name=name, path=path))
    return tuple(symbols)


def _parse_lab(value: Any, sink: _ErrorSink) -> Lab:
    if value is None:
        return Lab()
    if not isinstance(value, Mapping):
        sink.add(
            f"Field 'lab' must be a mapping of commands/expected_observations/cleanup, "
            f"got {value!r}",
            field="lab",
        )
        return Lab()

    _check_nested_keys(value, _LAB_KEYS, "lab", sink)
    return Lab(
        commands=_string_tuple_from(value, "commands", "lab.commands", sink),
        expected_observations=_string_tuple_from(
            value, "expected_observations", "lab.expected_observations", sink
        ),
        cleanup=_string_tuple_from(value, "cleanup", "lab.cleanup", sink),
    )


def _parse_quiz(
    value: Any, sink: _ErrorSink, *, published: bool
) -> tuple[QuizQuestion, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        sink.add(f"Field 'quiz' must be a list of questions, got {value!r}", field="quiz")
        return ()

    questions: list[QuizQuestion] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(value):
        field = f"quiz[{index}]"
        if not isinstance(entry, Mapping):
            sink.add(f"{field} must be a mapping describing a question", field=field)
            continue
        _check_nested_keys(entry, _QUESTION_KEYS, field, sink)

        question_id = entry.get("id")
        if not isinstance(question_id, str) or not question_id.strip():
            sink.add(f"{field} is missing a question 'id'", field=f"{field}.id")
            continue
        if question_id in seen_ids:
            sink.add(
                f"Duplicate quiz question id {question_id!r}", field=f"{field}.id"
            )
            continue
        seen_ids.add(question_id)

        prompt = entry.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            sink.add(f"{field} is missing a question 'prompt'", field=f"{field}.prompt")
            continue

        options = _parse_options(entry.get("options"), field, sink)
        if options is None:
            continue

        answer = entry.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            # Enforced for drafts too: a question with no trusted answer
            # cannot be graded, so it is never merely unfinished.
            sink.add(
                f"{field} is missing an 'answer' naming one of its options",
                field=f"{field}.answer",
            )
            continue

        option_ids = {option.id for option in options}
        if answer not in option_ids:
            sink.add(
                f"{field} answer {answer!r} is not one of its option ids "
                f"({', '.join(sorted(option_ids))})",
                field=f"{field}.answer",
            )
            continue

        explanation = entry.get("explanation")
        if explanation is None:
            explanation = ""
        elif not isinstance(explanation, str):
            sink.add(
                f"{field} explanation must be a string, got {explanation!r}",
                field=f"{field}.explanation",
            )
            continue
        if published and not explanation.strip():
            sink.add(
                f"{field} requires an 'explanation' before the lesson is published",
                field=f"{field}.explanation",
            )
            continue

        questions.append(
            QuizQuestion(
                id=question_id,
                prompt=prompt,
                options=options,
                answer=answer,
                explanation=explanation,
            )
        )
    return tuple(questions)


def _parse_options(
    value: Any, question_field: str, sink: _ErrorSink
) -> tuple[QuizOption, ...] | None:
    field = f"{question_field}.options"
    if not isinstance(value, (list, tuple)) or not value:
        sink.add(f"{field} must be a non-empty list of {{id, text}} mappings", field=field)
        return None

    options: list[QuizOption] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(value):
        option_field = f"{field}[{index}]"
        if not isinstance(entry, Mapping):
            sink.add(f"{option_field} must be a mapping with 'id' and 'text'", field=option_field)
            return None
        _check_nested_keys(entry, _OPTION_KEYS, option_field, sink)

        option_id = entry.get("id")
        if not isinstance(option_id, str) or not option_id.strip():
            sink.add(f"{option_field} is missing an option 'id'", field=f"{option_field}.id")
            return None
        if option_id in seen_ids:
            sink.add(
                f"Duplicate quiz option id {option_id!r} in {question_field}",
                field=f"{option_field}.id",
            )
            return None
        seen_ids.add(option_id)

        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            sink.add(f"{option_field} is missing option 'text'", field=f"{option_field}.text")
            return None

        options.append(QuizOption(id=option_id, text=text))
    return tuple(options)


def _parse_mastery_gate(value: Any, sink: _ErrorSink) -> MasteryGate | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        sink.add(
            f"Field 'mastery_gate' must be a mapping with 'min_quiz_score', got {value!r}",
            field="mastery_gate",
        )
        return None

    _check_nested_keys(value, _MASTERY_GATE_KEYS, "mastery_gate", sink)

    score = value.get("min_quiz_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not 0 < float(score) <= 1
    ):
        sink.add(
            f"mastery_gate.min_quiz_score must be a fraction greater than 0 and at "
            f"most 1, got {score!r}",
            field="mastery_gate.min_quiz_score",
        )
        return None

    level = value.get("required_review_level", 0)
    if (
        isinstance(level, bool)
        or not isinstance(level, int)
        or not 0 <= level <= MAX_REVIEW_LEVEL
    ):
        sink.add(
            f"mastery_gate.required_review_level must be an integer between 0 and "
            f"{MAX_REVIEW_LEVEL}, got {level!r}",
            field="mastery_gate.required_review_level",
        )
        return None

    return MasteryGate(min_quiz_score=float(score), required_review_level=level)


def _check_nested_keys(
    container: Mapping[str, Any], allowed: frozenset[str], field: str, sink: _ErrorSink
) -> None:
    for key in sorted(str(key) for key in container if key not in allowed):
        sink.add(f"Unknown key {key!r} in {field}", field=f"{field}.{key}")


def _check_published_contract(
    metadata: Mapping[str, Any],
    texts: Mapping[str, str | None],
    lab: Lab,
    sink: _ErrorSink,
) -> None:
    """Require everything a learner is graded against.

    Emptiness is judged from the *frontmatter*, not from the parsed
    values, so a collection that was written but failed its own validation
    is reported once (by the specific error) rather than twice.
    """
    for key in _PUBLISHED_TEXT_KEYS:
        if not (texts.get(key) or "").strip():
            sink.add(f"Published lessons require a non-empty {key!r}", field=key)

    for key in _PUBLISHED_COLLECTION_KEYS:
        raw = metadata.get(key)
        if raw is None or (isinstance(raw, (list, tuple)) and not raw):
            sink.add(f"Published lessons require a non-empty {key!r}", field=key)

    if metadata.get("lab") is None:
        sink.add("Published lessons require a 'lab'", field="lab")
    else:
        if not lab.commands:
            sink.add(
                "Published lessons require at least one 'lab.commands' entry",
                field="lab.commands",
            )
        if not lab.expected_observations:
            # Commands without observations teach nothing: the learner has
            # no way to tell whether what they saw was the point.
            sink.add(
                "Published lessons require at least one 'lab.expected_observations' entry",
                field="lab.expected_observations",
            )

    if metadata.get("mastery_gate") is None:
        sink.add("Published lessons require a 'mastery_gate'", field="mastery_gate")


def _check_uniqueness(
    loaded: Sequence[_LoadedModule], errors: list[CurriculumError]
) -> None:
    """Ids and slugs are unique curriculum-wide.

    Slugs are the URL and deep-link identity, so two lessons sharing one
    would make a link ambiguous no matter which modules they live in.
    """
    module_ids: dict[str, str] = {}
    module_slugs: dict[str, str] = {}
    lesson_ids: dict[str, str] = {}
    lesson_slugs: dict[str, str] = {}

    for entry in loaded:
        module = entry.module
        for value, seen, label, field in (
            (module.id, module_ids, "module id", "id"),
            (module.slug, module_slugs, "module slug", "slug"),
        ):
            first = seen.get(value)
            if first is None:
                seen[value] = entry.source_path
            else:
                errors.append(
                    CurriculumError(
                        path=entry.source_path,
                        field=field,
                        message=f"Duplicate {label} {value!r} (already defined in {first})",
                    )
                )

        for lesson in module.lessons:
            for value, seen, label, field in (
                (lesson.id, lesson_ids, "lesson id", "id"),
                (lesson.slug, lesson_slugs, "lesson slug", "slug"),
            ):
                first = seen.get(value)
                if first is None:
                    seen[value] = lesson.source_path
                else:
                    errors.append(
                        CurriculumError(
                            path=lesson.source_path,
                            field=field,
                            message=f"Duplicate {label} {value!r} (already defined in {first})",
                        )
                    )


def _check_prerequisites(
    loaded: Sequence[_LoadedModule], errors: list[CurriculumError]
) -> None:
    known = {
        lesson.id for entry in loaded for lesson in entry.module.lessons
    }
    for entry in loaded:
        for lesson in entry.module.lessons:
            for prerequisite in lesson.prerequisites:
                if prerequisite not in known:
                    errors.append(
                        CurriculumError(
                            path=lesson.source_path,
                            field="prerequisites",
                            message=f"Unknown prerequisite lesson id {prerequisite!r}",
                        )
                    )
