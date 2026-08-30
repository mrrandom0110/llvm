"""Contract for :mod:`netstack_academy.curriculum.loader`.

The loader turns a directory of versioned Markdown-with-YAML-frontmatter
files into an ordered, validated :class:`Curriculum`. Two properties are
load-bearing for everything built on top of it:

- **Determinism.** Module and lesson order is derived from declared
  ``order``/``slug`` values, never from the order the filesystem happens to
  hand back, so the same content root always renders the same course.
- **Fail closed, report everything.** Any validation problem leaves
  ``result.curriculum`` as ``None`` (a half-valid course is never served),
  and *all* problems across *all* files are aggregated into
  ``result.errors`` so an author fixes one round of content instead of
  rediscovering the next error on every reload.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from netstack_academy.curriculum.loader import (
    CurriculumValidationError,
    load_curriculum,
)

from content_builder import (
    OMIT,
    lesson_frontmatter,
    write_lesson,
    write_module,
    write_raw_lesson,
)


def _messages(result: object) -> str:
    return "\n".join(error.message for error in result.errors)


def _error_paths(result: object) -> set[str]:
    return {error.path for error in result.errors}


def test_load_curriculum_reads_modules_and_lessons(content_root: Path) -> None:
    result = load_curriculum(content_root)

    assert result.errors == ()
    assert result.ok is True
    assert [module.slug for module in result.curriculum.modules] == [
        "rx-path",
        "tx-path",
    ]
    assert [lesson.id for lesson in result.curriculum.modules[0].lessons] == [
        "lesson-napi-poll"
    ]


def test_load_curriculum_exposes_schema_version(content_root: Path) -> None:
    result = load_curriculum(content_root)

    assert result.curriculum.schema_version == 1


def test_lesson_records_its_content_root_relative_source_path(
    content_root: Path,
) -> None:
    result = load_curriculum(content_root)

    lesson = result.curriculum.lesson_by_id("lesson-napi-poll")
    assert lesson.source_path == "20-rx/napi-poll.md"


def test_lesson_records_its_owning_module_id(content_root: Path) -> None:
    result = load_curriculum(content_root)

    lesson = result.curriculum.lesson_by_id("lesson-napi-poll")
    assert lesson.module_id == "module-rx"


def test_modules_are_sorted_by_declared_order_not_directory_name(
    content_root: Path,
) -> None:
    """The ``rx-path`` module lives in directory ``20-rx`` and ``tx-path``
    in ``10-tx``, so a loader that trusted directory names (or raw
    ``os.listdir`` order) would emit them the other way round.
    """
    result = load_curriculum(content_root)

    assert [module.order for module in result.curriculum.modules] == [1, 2]
    assert [module.slug for module in result.curriculum.modules] == [
        "rx-path",
        "tx-path",
    ]


def test_modules_with_equal_order_are_sorted_by_slug(tmp_path: Path) -> None:
    for directory, slug in (("b", "zeta"), ("a", "alpha")):
        module = write_module(
            tmp_path, directory=directory, module_id=f"module-{slug}", slug=slug, order=1
        )
        write_lesson(
            module,
            frontmatter=lesson_frontmatter(
                lesson_id=f"lesson-{slug}", slug=f"lesson-{slug}"
            ),
        )

    result = load_curriculum(tmp_path)

    assert [module.slug for module in result.curriculum.modules] == ["alpha", "zeta"]


def test_lessons_within_a_module_are_sorted_by_order_then_slug(
    tmp_path: Path,
) -> None:
    module = write_module(tmp_path, directory="rx", module_id="module-rx", slug="rx")
    for lesson_id, slug, order in (
        ("lesson-third", "gro", 30),
        ("lesson-second-b", "budget", 20),
        ("lesson-second-a", "alloc", 20),
        ("lesson-first", "irq", 10),
    ):
        write_lesson(
            module,
            frontmatter=lesson_frontmatter(lesson_id=lesson_id, slug=slug, order=order),
        )

    result = load_curriculum(tmp_path)

    assert [lesson.slug for lesson in result.curriculum.modules[0].lessons] == [
        "irq",
        "alloc",
        "budget",
        "gro",
    ]


def test_load_curriculum_is_deterministic_across_repeated_loads(
    content_root: Path,
) -> None:
    first = load_curriculum(content_root)
    second = load_curriculum(content_root)

    assert first.curriculum == second.curriculum


def test_duplicate_lesson_id_across_modules_is_rejected(content_root: Path) -> None:
    module = write_module(
        content_root, directory="30-extra", module_id="module-extra", slug="extra", order=3
    )
    write_lesson(
        module,
        frontmatter=lesson_frontmatter(lesson_id="lesson-napi-poll", slug="napi-poll-copy"),
    )

    result = load_curriculum(content_root)

    assert result.curriculum is None
    assert "lesson-napi-poll" in _messages(result)


def test_duplicate_lesson_slug_across_modules_is_rejected(content_root: Path) -> None:
    """Slugs are the stable, human-typeable identity used in URLs and deep
    links, so they must be unique across the whole curriculum, not merely
    within one module.
    """
    module = write_module(
        content_root, directory="30-extra", module_id="module-extra", slug="extra", order=3
    )
    write_lesson(
        module,
        frontmatter=lesson_frontmatter(lesson_id="lesson-other", slug="napi-poll"),
    )

    result = load_curriculum(content_root)

    assert result.curriculum is None
    assert "napi-poll" in _messages(result)


def test_duplicate_module_slug_is_rejected(content_root: Path) -> None:
    module = write_module(
        content_root, directory="30-extra", module_id="module-extra", slug="rx-path", order=3
    )
    write_lesson(
        module, frontmatter=lesson_frontmatter(lesson_id="lesson-extra", slug="extra")
    )

    result = load_curriculum(content_root)

    assert result.curriculum is None
    assert "rx-path" in _messages(result)


def test_unsupported_schema_version_is_rejected_with_the_offending_file(
    content_root: Path,
) -> None:
    module = write_module(
        content_root, directory="30-extra", module_id="module-extra", slug="extra", order=3
    )
    write_lesson(
        module,
        frontmatter=lesson_frontmatter(lesson_id="lesson-future", slug="future"),
        overrides={"schema_version": 99},
    )

    result = load_curriculum(content_root)

    assert result.curriculum is None
    assert "30-extra/future.md" in _error_paths(result)
    assert "99" in _messages(result)


def test_missing_required_lesson_field_names_the_file_and_the_field(
    tmp_path: Path,
) -> None:
    module = write_module(tmp_path, directory="rx", module_id="module-rx", slug="rx")
    write_lesson(module, filename="broken.md", overrides={"title": OMIT})

    result = load_curriculum(tmp_path)

    assert result.curriculum is None
    assert "rx/broken.md" in _error_paths(result)
    assert {error.field for error in result.errors} == {"title"}


def test_validation_errors_from_several_files_are_all_reported(
    tmp_path: Path,
) -> None:
    """An author fixing content needs the whole list, not just whichever
    file the loader happened to read first.
    """
    module = write_module(tmp_path, directory="rx", module_id="module-rx", slug="rx")
    write_lesson(
        module,
        filename="no-title.md",
        frontmatter=lesson_frontmatter(lesson_id="lesson-a", slug="a"),
        overrides={"title": OMIT},
    )
    write_lesson(
        module,
        filename="no-id.md",
        frontmatter=lesson_frontmatter(lesson_id="lesson-b", slug="b"),
        overrides={"id": OMIT},
    )
    write_lesson(
        module,
        filename="bad-version.md",
        frontmatter=lesson_frontmatter(lesson_id="lesson-c", slug="c"),
        overrides={"schema_version": 99},
    )

    result = load_curriculum(tmp_path)

    assert _error_paths(result) >= {
        "rx/no-title.md",
        "rx/no-id.md",
        "rx/bad-version.md",
    }


def test_unknown_frontmatter_key_is_rejected(tmp_path: Path) -> None:
    """Silently ignoring an unrecognized key turns an author's typo (e.g.
    ``objective:`` instead of ``objectives:``) into content that renders
    but teaches nothing.
    """
    module = write_module(tmp_path, directory="rx", module_id="module-rx", slug="rx")
    write_lesson(module, filename="typo.md", overrides={"objectivez": ["oops"]})

    result = load_curriculum(tmp_path)

    assert result.curriculum is None
    assert "objectivez" in _messages(result)


def test_lesson_file_without_frontmatter_is_reported_not_raised(
    tmp_path: Path,
) -> None:
    module = write_module(tmp_path, directory="rx", module_id="module-rx", slug="rx")
    write_raw_lesson(module, filename="bare.md", text="# Just a heading\n")

    result = load_curriculum(tmp_path)

    assert result.curriculum is None
    assert "rx/bare.md" in _error_paths(result)


def test_malformed_yaml_frontmatter_is_reported_not_raised(tmp_path: Path) -> None:
    module = write_module(tmp_path, directory="rx", module_id="module-rx", slug="rx")
    write_raw_lesson(
        module,
        filename="malformed.md",
        text="---\ntitle: [unclosed\n---\n\nbody\n",
    )

    result = load_curriculum(tmp_path)

    assert result.curriculum is None
    assert "rx/malformed.md" in _error_paths(result)


def test_symbol_reference_path_escaping_the_repository_is_rejected(
    tmp_path: Path,
) -> None:
    module = write_module(tmp_path, directory="rx", module_id="module-rx", slug="rx")
    write_lesson(
        module,
        filename="escape.md",
        overrides={
            "source_symbols": [{"name": "napi_poll", "path": "../../etc/passwd"}]
        },
    )

    result = load_curriculum(tmp_path)

    assert result.curriculum is None
    assert "../../etc/passwd" in _messages(result)


def test_absolute_symbol_reference_path_is_rejected(tmp_path: Path) -> None:
    module = write_module(tmp_path, directory="rx", module_id="module-rx", slug="rx")
    write_lesson(
        module,
        filename="absolute.md",
        overrides={"source_symbols": [{"name": "napi_poll", "path": "/etc/passwd"}]},
    )

    result = load_curriculum(tmp_path)

    assert result.curriculum is None
    assert "/etc/passwd" in _messages(result)


def test_symbol_reference_without_a_name_is_rejected(tmp_path: Path) -> None:
    module = write_module(tmp_path, directory="rx", module_id="module-rx", slug="rx")
    write_lesson(
        module, filename="nameless.md", overrides={"source_symbols": [{"path": "net/core/dev.c"}]}
    )

    result = load_curriculum(tmp_path)

    assert result.curriculum is None
    assert "name" in _messages(result)


def test_symbol_reference_may_omit_the_path(content_root: Path) -> None:
    """``netif_receive_skb`` is declared with a name only; a lesson author
    should not have to know a symbol's file to reference it.
    """
    result = load_curriculum(content_root)

    lesson = result.curriculum.lesson_by_id("lesson-napi-poll")
    by_name = {symbol.name: symbol for symbol in lesson.source_symbols}
    assert by_name["netif_receive_skb"].path is None
    assert by_name["napi_poll"].path == "net/core/dev.c"


def test_prerequisite_referencing_an_unknown_lesson_is_rejected(
    content_root: Path,
) -> None:
    module = write_module(
        content_root, directory="30-extra", module_id="module-extra", slug="extra", order=3
    )
    write_lesson(
        module,
        filename="dangling.md",
        frontmatter=lesson_frontmatter(lesson_id="lesson-dangling", slug="dangling"),
        overrides={"prerequisites": ["lesson-does-not-exist"]},
    )

    result = load_curriculum(content_root)

    assert result.curriculum is None
    assert "lesson-does-not-exist" in _messages(result)


def test_prerequisite_referencing_a_known_lesson_is_accepted(
    content_root: Path,
) -> None:
    module = write_module(
        content_root, directory="30-extra", module_id="module-extra", slug="extra", order=3
    )
    write_lesson(
        module,
        filename="downstream.md",
        frontmatter=lesson_frontmatter(lesson_id="lesson-downstream", slug="downstream"),
        overrides={"prerequisites": ["lesson-napi-poll"]},
    )

    result = load_curriculum(content_root)

    assert result.errors == ()
    lesson = result.curriculum.lesson_by_id("lesson-downstream")
    assert lesson.prerequisites == ("lesson-napi-poll",)


def test_directory_without_a_module_file_is_ignored(content_root: Path) -> None:
    """Asset/scratch directories sitting next to modules must not become
    modules and must not become errors.
    """
    assets = content_root / "assets"
    assets.mkdir()
    (assets / "diagram.md").write_text("not a lesson\n", encoding="utf-8")

    result = load_curriculum(content_root)

    assert result.errors == ()
    assert [module.slug for module in result.curriculum.modules] == [
        "rx-path",
        "tx-path",
    ]


def test_missing_content_root_is_reported_not_raised(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-content"

    result = load_curriculum(missing)

    assert result.curriculum is None
    assert result.errors != ()
    assert "no-such-content" in _messages(result)


def test_strict_mode_raises_with_every_aggregated_error(tmp_path: Path) -> None:
    module = write_module(tmp_path, directory="rx", module_id="module-rx", slug="rx")
    write_lesson(
        module,
        filename="no-title.md",
        frontmatter=lesson_frontmatter(lesson_id="lesson-a", slug="a"),
        overrides={"title": OMIT},
    )
    write_lesson(
        module,
        filename="no-id.md",
        frontmatter=lesson_frontmatter(lesson_id="lesson-b", slug="b"),
        overrides={"id": OMIT},
    )

    with pytest.raises(CurriculumValidationError) as excinfo:
        load_curriculum(tmp_path, strict=True)

    assert {error.path for error in excinfo.value.errors} >= {
        "rx/no-title.md",
        "rx/no-id.md",
    }


def test_strict_mode_returns_the_same_result_when_content_is_valid(
    content_root: Path,
) -> None:
    strict = load_curriculum(content_root, strict=True)
    lenient = load_curriculum(content_root)

    assert strict.curriculum == lenient.curriculum


def test_lesson_body_is_rendered_to_sanitized_html(tmp_path: Path) -> None:
    module = write_module(tmp_path, directory="rx", module_id="module-rx", slug="rx")
    write_lesson(
        module,
        filename="scripted.md",
        body="Intro.\n\n<script>steal()</script>\n\n```c\nint x = 1;\n```\n",
    )

    result = load_curriculum(tmp_path)

    lesson = result.curriculum.lesson_by_id("lesson-napi-poll")
    assert "<script>" not in lesson.body_html
    assert "steal()" not in lesson.body_html
    assert "int x = 1;" in lesson.body_html


def test_lesson_keeps_the_raw_markdown_body(tmp_path: Path) -> None:
    module = write_module(tmp_path, directory="rx", module_id="module-rx", slug="rx")
    write_lesson(module, filename="raw.md", body="Intro paragraph.\n")

    result = load_curriculum(tmp_path)

    lesson = result.curriculum.lesson_by_id("lesson-napi-poll")
    assert lesson.body_markdown.strip() == "Intro paragraph."


def test_draft_and_published_lessons_load_side_by_side(content_root: Path) -> None:
    result = load_curriculum(content_root)

    tx = result.curriculum.module_by_slug("tx-path")
    assert [lesson.status for lesson in tx.lessons] == ["published", "draft"]


def test_curriculum_lookup_by_slug_and_id_agree(content_root: Path) -> None:
    result = load_curriculum(content_root)

    assert result.curriculum.lesson_by_slug("napi-poll") is result.curriculum.lesson_by_id(
        "lesson-napi-poll"
    )


def test_curriculum_lookup_of_unknown_lesson_returns_none(content_root: Path) -> None:
    result = load_curriculum(content_root)

    assert result.curriculum.lesson_by_id("lesson-nope") is None
    assert result.curriculum.module_by_slug("nope") is None
