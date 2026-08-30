"""Server-side quiz grading.

Grading reads the answer key from the loaded
:class:`~netstack_academy.curriculum.models.Lesson`, never from the
submission. A response payload therefore contributes exactly one thing --
which option was chosen -- and nothing in it can influence its own score.

:func:`public_quiz` is re-exported from
:mod:`netstack_academy.curriculum.models`, where it lives because the
answer-free projection is a property of the content model rather than of
the learner's store. Importing it from here keeps the two halves of the
quiz contract (what a learner is shown, and how their answers are scored)
visible in one place.

Unrecognized input is an error rather than a silent zero: a response naming
a question or an option this lesson does not have means the submission and
the content have diverged (a stale page, a renamed question, or tampering),
and scoring it as merely "wrong" would hide that.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from netstack_academy.curriculum.models import (
    Lesson,
    PublicQuizOption,
    PublicQuizQuestion,
    public_quiz,
)

__all__ = [
    "PublicQuizOption",
    "PublicQuizQuestion",
    "QuestionResult",
    "QuizGrade",
    "UnknownQuizOptionError",
    "UnknownQuizQuestionError",
    "grade_quiz",
    "public_quiz",
]


class UnknownQuizQuestionError(ValueError):
    """Raised when a submission answers a question the lesson does not have."""


class UnknownQuizOptionError(ValueError):
    """Raised when a submission names an option the question does not have."""


@dataclass(frozen=True, slots=True)
class QuestionResult:
    """How one question was answered, and what the right answer was.

    This is the *post-submission* view, so it does carry
    ``correct_option_id`` and ``explanation`` -- that is the teaching part
    of a quiz. It must never be handed out before an attempt is recorded;
    :func:`public_quiz` is the shape for that.
    """

    question_id: str
    response: str | None
    correct: bool
    correct_option_id: str
    explanation: str


@dataclass(frozen=True, slots=True)
class QuizGrade:
    """The outcome of grading one submission against a lesson's answers."""

    lesson_id: str
    score: float
    correct_count: int
    question_count: int
    results: tuple[QuestionResult, ...] = ()


def grade_quiz(lesson: Lesson, responses: Mapping[str, str]) -> QuizGrade:
    """Score ``responses`` against ``lesson``'s trusted answer key.

    An unanswered question counts as incorrect (a learner who skips a
    question has not demonstrated the objective), while an *unrecognized*
    question or option raises.
    """
    questions = {question.id: question for question in lesson.quiz}

    for question_id in responses:
        if question_id not in questions:
            raise UnknownQuizQuestionError(
                f"Lesson {lesson.id!r} has no quiz question {question_id!r}"
            )

    results: list[QuestionResult] = []
    correct_count = 0
    for question in lesson.quiz:
        response = responses.get(question.id)
        if response is not None and response not in {
            option.id for option in question.options
        }:
            raise UnknownQuizOptionError(
                f"Question {question.id!r} of lesson {lesson.id!r} has no option "
                f"{response!r}"
            )

        correct = response is not None and response == question.answer
        if correct:
            correct_count += 1
        results.append(
            QuestionResult(
                question_id=question.id,
                response=response,
                correct=correct,
                correct_option_id=question.answer,
                explanation=question.explanation,
            )
        )

    question_count = len(lesson.quiz)
    score = correct_count / question_count if question_count else 0.0
    return QuizGrade(
        lesson_id=lesson.id,
        score=score,
        correct_count=correct_count,
        question_count=question_count,
        results=tuple(results),
    )
