"""Contract for the lesson schema itself (:mod:`netstack_academy.curriculum.models`).

The schema is deliberately two-tier. A ``draft`` lesson exists so an author
can commit a title and a paragraph and see it in the UI, so nearly every
field may be missing or empty. A ``published`` lesson is what a learner is
graded against, so the full teaching contract -- objectives, kernel
context, source symbols, a runnable lab, a quiz, and a mastery gate -- is
mandatory.

Invariants that would make a quiz un-gradeable (an answer that is not one
of the options, duplicate option ids) are enforced for *both* tiers: a
half-written quiz is worse than no quiz.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from netstack_academy.curriculum.loader import load_curriculum

from content_builder import (
    OMIT,
    draft_lesson_frontmatter,
    lesson_frontmatter,
    write_lesson,
    write_module,
)


@pytest.fixture
def module_dir(tmp_path: Path) -> Path:
    return write_module(tmp_path, directory="rx", module_id="module-rx", slug="rx")


def _load_one(module_dir: Path, **write_kwargs: object):
    write_lesson(module_dir, **write_kwargs)  # type: ignore[arg-type]
    return load_curriculum(module_dir.parent)


def _messages(result: object) -> str:
    return "\n".join(error.message for error in result.errors)


def test_published_lesson_exposes_its_teaching_metadata(module_dir: Path) -> None:
    result = _load_one(module_dir)

    lesson = result.curriculum.lesson_by_id("lesson-napi-poll")
    assert lesson.title == "The NAPI poll loop"
    assert lesson.summary.startswith("How the NAPI poll loop")
    assert lesson.objectives == (
        "Explain when napi_poll runs",
        "Name the budget that bounds one poll",
    )
    assert lesson.packet_stage == "rx-softirq"
    assert lesson.execution_context == "softirq"
    assert lesson.ownership.startswith("The NAPI instance")


def test_published_lesson_exposes_locking_and_rcu_notes(module_dir: Path) -> None:
    result = _load_one(module_dir)

    lesson = result.curriculum.lesson_by_id("lesson-napi-poll")
    assert "NAPI_STATE_SCHED" in lesson.locking
    assert "rcu_read_lock()" in lesson.rcu


def test_published_lesson_exposes_structures_with_their_fields(
    module_dir: Path,
) -> None:
    result = _load_one(module_dir)

    lesson = result.curriculum.lesson_by_id("lesson-napi-poll")
    assert [structure.name for structure in lesson.structures] == ["struct napi_struct"]
    assert lesson.structures[0].fields == ("poll", "weight", "state")


def test_published_lesson_exposes_config_and_version_caveats(
    module_dir: Path,
) -> None:
    result = _load_one(module_dir)

    lesson = result.curriculum.lesson_by_id("lesson-napi-poll")
    assert lesson.config_caveats == ("CONFIG_RPS moves work to a remote CPU.",)
    assert lesson.version_caveats == ("Budget accounting changed in v5.15.",)


def test_published_lesson_exposes_tracepoints_and_source_symbols(
    module_dir: Path,
) -> None:
    result = _load_one(module_dir)

    lesson = result.curriculum.lesson_by_id("lesson-napi-poll")
    assert lesson.tracepoints == ("napi:napi_poll",)
    assert [symbol.name for symbol in lesson.source_symbols] == [
        "napi_poll",
        "netif_receive_skb",
    ]


def test_published_lesson_exposes_lab_commands_observations_and_cleanup(
    module_dir: Path,
) -> None:
    result = _load_one(module_dir)

    lab = result.curriculum.lesson_by_id("lesson-napi-poll").lab
    assert lab.commands == ("cat /proc/net/softnet_stat",)
    assert lab.expected_observations == ("The second column stays at zero.",)
    assert lab.cleanup == ("true",)


def test_published_lesson_exposes_quiz_questions_options_answers_explanations(
    module_dir: Path,
) -> None:
    result = _load_one(module_dir)

    question = result.curriculum.lesson_by_id("lesson-napi-poll").quiz[0]
    assert question.id == "q-context"
    assert question.prompt.startswith("In which context")
    assert [option.id for option in question.options] == ["a", "b"]
    assert question.answer == "b"
    assert question.explanation.startswith("NAPI polling is deferred")


def test_published_lesson_exposes_its_mastery_gate(module_dir: Path) -> None:
    result = _load_one(module_dir)

    gate = result.curriculum.lesson_by_id("lesson-napi-poll").mastery_gate
    assert gate.min_quiz_score == pytest.approx(0.8)
    assert gate.required_review_level == 2


def test_draft_lesson_loads_with_only_identity_fields(module_dir: Path) -> None:
    """A draft is what an author sees in the UI while writing; it must not
    require the fields they have not written yet.
    """
    result = _load_one(module_dir, frontmatter=draft_lesson_frontmatter())

    assert result.errors == ()
    lesson = result.curriculum.lesson_by_id("lesson-draft")
    assert lesson.status == "draft"
    assert lesson.summary == ""
    assert lesson.objectives == ()
    assert lesson.source_symbols == ()
    assert lesson.quiz == ()
    assert lesson.lab.commands == ()
    assert lesson.mastery_gate is None


@pytest.mark.parametrize(
    "missing_field",
    [
        "summary",
        "objectives",
        "packet_stage",
        "execution_context",
        "source_symbols",
        "lab",
        "quiz",
        "mastery_gate",
    ],
)
def test_published_lesson_requires_the_full_contract(
    module_dir: Path, missing_field: str
) -> None:
    result = _load_one(module_dir, overrides={missing_field: OMIT})

    assert result.curriculum is None
    assert {error.field for error in result.errors} == {missing_field}


@pytest.mark.parametrize("empty_field", ["objectives", "source_symbols", "quiz"])
def test_published_lesson_rejects_empty_required_collections(
    module_dir: Path, empty_field: str
) -> None:
    """Present-but-empty is the same failure as absent for a published
    lesson: there is nothing to teach or grade.
    """
    result = _load_one(module_dir, overrides={empty_field: []})

    assert result.curriculum is None
    assert {error.field for error in result.errors} == {empty_field}


def test_published_lesson_requires_lab_commands_and_expected_observations(
    module_dir: Path,
) -> None:
    result = _load_one(
        module_dir,
        overrides={"lab": {"commands": ["ip -s link"], "cleanup": ["true"]}},
    )

    assert result.curriculum is None
    assert "expected_observations" in _messages(result)


def test_published_lesson_allows_an_empty_lab_cleanup(module_dir: Path) -> None:
    """Not every lab mutates state, so cleanup stays optional even when the
    rest of the lab is required.
    """
    result = _load_one(
        module_dir,
        overrides={
            "lab": {
                "commands": ["ip -s link"],
                "expected_observations": ["Counters increase."],
            }
        },
    )

    assert result.errors == ()
    assert result.curriculum.lesson_by_id("lesson-napi-poll").lab.cleanup == ()


def test_quiz_answer_must_name_one_of_the_options(module_dir: Path) -> None:
    result = _load_one(
        module_dir,
        overrides={
            "quiz": [
                {
                    "id": "q1",
                    "prompt": "Which context?",
                    "options": [{"id": "a", "text": "Hard IRQ"}],
                    "answer": "z",
                    "explanation": "…",
                }
            ]
        },
    )

    assert result.curriculum is None
    assert "z" in _messages(result)


def test_quiz_options_must_have_unique_ids(module_dir: Path) -> None:
    result = _load_one(
        module_dir,
        overrides={
            "quiz": [
                {
                    "id": "q1",
                    "prompt": "Which context?",
                    "options": [
                        {"id": "a", "text": "Hard IRQ"},
                        {"id": "a", "text": "Softirq"},
                    ],
                    "answer": "a",
                    "explanation": "…",
                }
            ]
        },
    )

    assert result.curriculum is None
    assert "a" in _messages(result)


def test_quiz_question_ids_must_be_unique_within_a_lesson(module_dir: Path) -> None:
    question = {
        "id": "q1",
        "prompt": "Which context?",
        "options": [{"id": "a", "text": "Hard IRQ"}, {"id": "b", "text": "Softirq"}],
        "answer": "b",
        "explanation": "…",
    }
    result = _load_one(module_dir, overrides={"quiz": [question, dict(question)]})

    assert result.curriculum is None
    assert "q1" in _messages(result)


def test_quiz_question_without_an_answer_is_rejected_even_in_a_draft(
    module_dir: Path,
) -> None:
    """An ungradeable question is a bug at any publication status: the
    grader has no trusted answer to score against.
    """
    frontmatter = draft_lesson_frontmatter()
    frontmatter["quiz"] = [
        {
            "id": "q1",
            "prompt": "Which context?",
            "options": [{"id": "a", "text": "Hard IRQ"}],
        }
    ]
    result = _load_one(module_dir, frontmatter=frontmatter)

    assert result.curriculum is None
    assert "answer" in _messages(result)


def test_published_quiz_question_requires_an_explanation(module_dir: Path) -> None:
    result = _load_one(
        module_dir,
        overrides={
            "quiz": [
                {
                    "id": "q1",
                    "prompt": "Which context?",
                    "options": [
                        {"id": "a", "text": "Hard IRQ"},
                        {"id": "b", "text": "Softirq"},
                    ],
                    "answer": "b",
                }
            ]
        },
    )

    assert result.curriculum is None
    assert "explanation" in _messages(result)


@pytest.mark.parametrize("score", [-0.1, 0.0, 1.5])
def test_mastery_gate_score_must_be_a_fraction_above_zero(
    module_dir: Path, score: float
) -> None:
    result = _load_one(
        module_dir,
        overrides={"mastery_gate": {"min_quiz_score": score, "required_review_level": 1}},
    )

    assert result.curriculum is None
    assert "min_quiz_score" in _messages(result)


@pytest.mark.parametrize("level", [-1, 6])
def test_mastery_gate_review_level_must_be_within_the_leitner_range(
    module_dir: Path, level: int
) -> None:
    result = _load_one(
        module_dir,
        overrides={"mastery_gate": {"min_quiz_score": 0.8, "required_review_level": level}},
    )

    assert result.curriculum is None
    assert "required_review_level" in _messages(result)


def test_unknown_publication_status_is_rejected(module_dir: Path) -> None:
    result = _load_one(module_dir, overrides={"status": "sort-of-published"})

    assert result.curriculum is None
    assert "sort-of-published" in _messages(result)


def test_lesson_order_must_be_an_integer(module_dir: Path) -> None:
    result = _load_one(module_dir, overrides={"order": "first"})

    assert result.curriculum is None
    assert {error.field for error in result.errors} == {"order"}


def test_lesson_collections_are_immutable_tuples(module_dir: Path) -> None:
    """Views and services share loaded lessons freely; a mutable list would
    let one caller edit another caller's curriculum in place.
    """
    result = _load_one(module_dir)

    lesson = result.curriculum.lesson_by_id("lesson-napi-poll")
    assert isinstance(lesson.objectives, tuple)
    assert isinstance(lesson.tracepoints, tuple)
    assert isinstance(lesson.source_symbols, tuple)
    assert isinstance(lesson.quiz, tuple)
    assert isinstance(lesson.quiz[0].options, tuple)


def test_two_loads_of_the_same_lesson_compare_equal(module_dir: Path) -> None:
    write_lesson(module_dir, frontmatter=lesson_frontmatter())

    first = load_curriculum(module_dir.parent)
    second = load_curriculum(module_dir.parent)

    assert first.curriculum.lesson_by_id("lesson-napi-poll") == second.curriculum.lesson_by_id(
        "lesson-napi-poll"
    )
