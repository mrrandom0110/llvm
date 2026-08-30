"""Contract for :mod:`netstack_academy.learning.quiz` and quiz attempts.

Two rules define this module. Grading happens server-side against the
answers stored in the loaded lesson, so a response payload can never
influence its own score. And the public view of a quiz -- the shape handed
to a template or an API response -- has no field that could carry the
answer or its explanation, so a learner cannot read the key out of the page
before answering.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from netstack_academy.learning.quiz import (
    UnknownQuizOptionError,
    UnknownQuizQuestionError,
    grade_quiz,
    public_quiz,
)
from netstack_academy.learning.store import LearningStore

from learning_fakes import FakeClock
from lesson_factory import make_lesson, make_question

TWO_QUESTION_QUIZ = (
    make_question(question_id="q1", answer="b"),
    make_question(
        question_id="q2",
        prompt="What bounds one poll?",
        options=(("a", "The budget"), ("b", "The MTU")),
        answer="a",
        explanation="The poll budget bounds work per call.",
    ),
)


def test_public_quiz_exposes_prompt_and_options() -> None:
    lesson = make_lesson()

    questions = public_quiz(lesson)

    assert [question.id for question in questions] == ["q-context"]
    assert questions[0].prompt == "In which context does napi_poll run?"
    assert [option.id for option in questions[0].options] == ["a", "b"]
    assert [option.text for option in questions[0].options] == ["Hard IRQ", "Softirq"]


def test_public_quiz_question_has_no_answer_or_explanation_field() -> None:
    question = public_quiz(make_lesson())[0]

    field_names = {field.name for field in dataclasses.fields(question)}
    assert "answer" not in field_names
    assert "explanation" not in field_names
    assert not hasattr(question, "answer")
    assert not hasattr(question, "explanation")


def test_public_quiz_serialization_never_contains_the_answer_key() -> None:
    """Templates and JSON responses both serialize whatever they are given,
    so the leak test has to cover the serialized form, not just attribute
    access.
    """
    lesson = make_lesson(
        quiz=(
            make_question(
                question_id="q1",
                options=(("a", "Hard IRQ"), ("b", "Softirq")),
                answer="b",
                explanation="Deferred to NET_RX_SOFTIRQ.",
            ),
        )
    )

    serialized = json.dumps(
        [dataclasses.asdict(question) for question in public_quiz(lesson)]
    )

    assert "Deferred to NET_RX_SOFTIRQ." not in serialized
    assert '"answer"' not in serialized


def test_grade_quiz_scores_a_fully_correct_submission() -> None:
    lesson = make_lesson(quiz=TWO_QUESTION_QUIZ)

    grade = grade_quiz(lesson, {"q1": "b", "q2": "a"})

    assert grade.score == pytest.approx(1.0)
    assert grade.correct_count == 2
    assert grade.question_count == 2


def test_grade_quiz_scores_a_partially_correct_submission() -> None:
    lesson = make_lesson(quiz=TWO_QUESTION_QUIZ)

    grade = grade_quiz(lesson, {"q1": "b", "q2": "b"})

    assert grade.score == pytest.approx(0.5)
    assert grade.correct_count == 1


def test_grade_quiz_treats_an_unanswered_question_as_incorrect() -> None:
    lesson = make_lesson(quiz=TWO_QUESTION_QUIZ)

    grade = grade_quiz(lesson, {"q1": "b"})

    assert grade.correct_count == 1
    assert grade.question_count == 2
    results = {result.question_id: result for result in grade.results}
    assert results["q2"].response is None
    assert results["q2"].correct is False


def test_grade_quiz_uses_the_lessons_answers_not_the_submission() -> None:
    """The same responses graded against two lessons that differ *only* in
    their stored answers must score differently; nothing in the submission
    may act as an answer key.
    """
    responses = {"q1": "b"}
    matching = make_lesson(quiz=(make_question(question_id="q1", answer="b"),))
    opposing = make_lesson(quiz=(make_question(question_id="q1", answer="a"),))

    assert grade_quiz(matching, responses).score == pytest.approx(1.0)
    assert grade_quiz(opposing, responses).score == pytest.approx(0.0)


def test_grade_quiz_results_carry_the_correct_option_and_explanation() -> None:
    lesson = make_lesson()

    result = grade_quiz(lesson, {"q-context": "a"}).results[0]

    assert result.correct is False
    assert result.response == "a"
    assert result.correct_option_id == "b"
    assert result.explanation == "NAPI polling is deferred to NET_RX_SOFTIRQ."


def test_grade_quiz_rejects_a_response_for_an_unknown_question() -> None:
    lesson = make_lesson()

    with pytest.raises(UnknownQuizQuestionError):
        grade_quiz(lesson, {"q-context": "b", "q-injected": "b"})


def test_grade_quiz_rejects_a_response_naming_an_unknown_option() -> None:
    lesson = make_lesson()

    with pytest.raises(UnknownQuizOptionError):
        grade_quiz(lesson, {"q-context": "z"})


def test_grade_quiz_of_a_lesson_without_questions_scores_zero() -> None:
    lesson = make_lesson(status="draft", quiz=())

    grade = grade_quiz(lesson, {})

    assert grade.question_count == 0
    assert grade.correct_count == 0
    assert grade.score == pytest.approx(0.0)


def test_record_quiz_attempt_persists_the_server_side_score(
    store: LearningStore, clock: FakeClock
) -> None:
    lesson = make_lesson(quiz=TWO_QUESTION_QUIZ)

    attempt = store.record_quiz_attempt(lesson, responses={"q1": "b", "q2": "b"})

    assert attempt.lesson_id == lesson.id
    assert attempt.score == pytest.approx(0.5)
    assert attempt.correct_count == 1
    assert attempt.question_count == 2
    assert attempt.created_at == clock.now


def test_record_quiz_attempt_stores_the_submitted_responses(
    store: LearningStore,
) -> None:
    lesson = make_lesson(quiz=TWO_QUESTION_QUIZ)

    attempt = store.record_quiz_attempt(lesson, responses={"q1": "b", "q2": "a"})

    assert dict(attempt.responses) == {"q1": "b", "q2": "a"}


def test_record_quiz_attempt_rejects_an_unknown_question(
    store: LearningStore,
) -> None:
    lesson = make_lesson()

    with pytest.raises(UnknownQuizQuestionError):
        store.record_quiz_attempt(lesson, responses={"q-injected": "b"})

    assert store.list_quiz_attempts(lesson.id) == []


def test_list_quiz_attempts_is_ordered_oldest_first(
    store: LearningStore, clock: FakeClock
) -> None:
    lesson = make_lesson(quiz=TWO_QUESTION_QUIZ)
    store.record_quiz_attempt(lesson, responses={"q1": "a", "q2": "b"})
    clock.advance(days=1)
    store.record_quiz_attempt(lesson, responses={"q1": "b", "q2": "a"})

    attempts = store.list_quiz_attempts(lesson.id)

    assert [attempt.score for attempt in attempts] == pytest.approx([0.0, 1.0])


def test_quiz_attempts_are_scoped_to_their_lesson(store: LearningStore) -> None:
    first = make_lesson(lesson_id="lesson-a", slug="a")
    second = make_lesson(lesson_id="lesson-b", slug="b")
    store.record_quiz_attempt(first, responses={"q-context": "b"})

    assert store.list_quiz_attempts(second.id) == []


def test_quiz_attempts_survive_reopening_the_database(tmp_path: Path) -> None:
    db_path = tmp_path / "learning.sqlite3"
    lesson = make_lesson()
    with LearningStore.open(db_path, clock=FakeClock()) as first:
        first.record_quiz_attempt(lesson, responses={"q-context": "b"})

    with LearningStore.open(db_path, clock=FakeClock()) as second:
        attempts = second.list_quiz_attempts(lesson.id)

    assert [attempt.score for attempt in attempts] == pytest.approx([1.0])
