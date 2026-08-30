"""Immutable models for authored course content.

A :class:`Curriculum` is an ordered tuple of :class:`Module`\\ s, each an
ordered tuple of :class:`Lesson`\\ s. Everything here is a frozen dataclass
holding tuples rather than lists, for two reasons: loaded content is shared
freely between services, views and templates, so no caller can corrupt
another's copy; and two loads of the same content root compare equal, which
is what makes the loader's determinism testable at all.

Lessons come in two tiers. A ``draft`` carries only its identity fields --
that is the state an author is in while writing, and the UI still has to
render it. A ``published`` lesson is what a learner is graded against, so
the loader additionally requires the whole teaching contract (see
:mod:`netstack_academy.curriculum.loader`). The models themselves stay
permissive: they describe the *shape* of a lesson, and
:func:`~netstack_academy.curriculum.loader.load_curriculum` owns which
shapes are acceptable at which tier.

:func:`public_quiz` is the one deliberately lossy view in this module. The
answer key and the explanations live on :class:`QuizQuestion`; anything
rendered into a page or serialized to a client gets
:class:`PublicQuizQuestion` instead, which has no field that could carry
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

#: The only content ``schema_version`` this code understands. Every module
#: and lesson file declares it, so one stale file is reported by name
#: rather than failing the whole content root opaquely.
SUPPORTED_SCHEMA_VERSION = 1

#: Publication tiers. ``draft`` is author-visible work in progress;
#: ``published`` is learner-facing and fully validated.
LessonStatus = Literal["draft", "published"]

LESSON_STATUSES: tuple[str, ...] = ("draft", "published")

#: Highest Leitner box a mastery gate may require. The learning slice's
#: spaced-review ladder is bounded by the same value; a gate demanding a
#: level no card can ever reach would be unsatisfiable by construction.
MAX_REVIEW_LEVEL = 5


@dataclass(frozen=True, slots=True)
class SymbolReference:
    """A kernel symbol a lesson is about.

    ``path`` is optional: an author should not need to know which file a
    symbol lives in to reference it, and for a non-``static`` symbol the
    index can resolve the name on its own. When present it is a repository
    -relative path, validated by the loader before it ever reaches here.
    """

    name: str
    path: str | None = None


@dataclass(frozen=True, slots=True)
class StructureReference:
    """A kernel structure, optionally narrowed to the fields that matter."""

    name: str
    fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Lab:
    """The hands-on part of a lesson.

    ``cleanup`` stays optional at every tier because not every lab mutates
    system state; ``commands`` and ``expected_observations`` are what make
    a lab teachable ("run this, and here is what you should see"), so a
    published lesson needs both.
    """

    commands: tuple[str, ...] = ()
    expected_observations: tuple[str, ...] = ()
    cleanup: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QuizOption:
    """One selectable answer, identified by a lesson-local ``id``."""

    id: str
    text: str


@dataclass(frozen=True, slots=True)
class QuizQuestion:
    """A single-choice question, including its trusted answer.

    ``answer`` names one of ``options``' ids. This is the *only* copy of
    the answer key in the system: grading reads it from here, never from a
    submission. See :func:`public_quiz` for the view handed outwards.
    """

    id: str
    prompt: str
    options: tuple[QuizOption, ...] = ()
    answer: str = ""
    explanation: str = ""


@dataclass(frozen=True, slots=True)
class MasteryGate:
    """What a learner must reach before a lesson counts as mastered."""

    min_quiz_score: float
    required_review_level: int = 0


@dataclass(frozen=True, slots=True)
class Lesson:
    """One lesson: identity, kernel context, a lab, a quiz, and its body.

    ``source_path`` is the lesson file's path relative to the content root
    (POSIX form), which is what makes an error message or an "edit this
    page" link point at a real file.
    """

    id: str
    slug: str
    title: str
    order: int
    module_id: str
    status: LessonStatus = "draft"
    source_path: str = ""
    summary: str = ""
    objectives: tuple[str, ...] = ()
    prerequisites: tuple[str, ...] = ()
    packet_stage: str | None = None
    execution_context: str | None = None
    ownership: str | None = None
    locking: str | None = None
    rcu: str | None = None
    structures: tuple[StructureReference, ...] = ()
    config_caveats: tuple[str, ...] = ()
    version_caveats: tuple[str, ...] = ()
    tracepoints: tuple[str, ...] = ()
    source_symbols: tuple[SymbolReference, ...] = ()
    lab: Lab = field(default_factory=Lab)
    quiz: tuple[QuizQuestion, ...] = ()
    mastery_gate: MasteryGate | None = None
    body_markdown: str = ""
    body_html: str = ""


@dataclass(frozen=True, slots=True)
class Module:
    """An ordered group of lessons."""

    id: str
    slug: str
    title: str
    order: int
    summary: str = ""
    lessons: tuple[Lesson, ...] = ()


@dataclass(frozen=True, slots=True)
class Curriculum:
    """The whole loaded course, in display order.

    Lookups are linear scans over the same :class:`Lesson` objects held by
    the modules, so ``lesson_by_slug(...) is lesson_by_id(...)`` for the
    same lesson: callers can compare identities and stash lessons without
    worrying about which accessor produced them.
    """

    schema_version: int
    modules: tuple[Module, ...] = ()

    @property
    def lessons(self) -> tuple[Lesson, ...]:
        """Every lesson, in curriculum order (module order, then lesson order)."""
        return tuple(lesson for module in self.modules for lesson in module.lessons)

    def module_by_id(self, module_id: str) -> Module | None:
        for module in self.modules:
            if module.id == module_id:
                return module
        return None

    def module_by_slug(self, slug: str) -> Module | None:
        for module in self.modules:
            if module.slug == slug:
                return module
        return None

    def lesson_by_id(self, lesson_id: str) -> Lesson | None:
        for lesson in self.lessons:
            if lesson.id == lesson_id:
                return lesson
        return None

    def lesson_by_slug(self, slug: str) -> Lesson | None:
        for lesson in self.lessons:
            if lesson.slug == slug:
                return lesson
        return None


@dataclass(frozen=True, slots=True)
class PublicQuizOption:
    """An option as shown to a learner."""

    id: str
    text: str


@dataclass(frozen=True, slots=True)
class PublicQuizQuestion:
    """A question as shown to a learner: prompt and options, nothing else.

    There is intentionally no ``answer`` or ``explanation`` field. Omitting
    the values would still leak them through anything that serializes the
    dataclass generically (``dataclasses.asdict``, a template that dumps
    the object, a JSON response), so the answer key is absent from the
    *type*, not merely blanked out on the instance.
    """

    id: str
    prompt: str
    options: tuple[PublicQuizOption, ...] = ()


def public_quiz(lesson: Lesson) -> tuple[PublicQuizQuestion, ...]:
    """Project ``lesson``'s quiz into its answer-free public form."""
    return tuple(
        PublicQuizQuestion(
            id=question.id,
            prompt=question.prompt,
            options=tuple(
                PublicQuizOption(id=option.id, text=option.text)
                for option in question.options
            ),
        )
        for question in lesson.quiz
    )
