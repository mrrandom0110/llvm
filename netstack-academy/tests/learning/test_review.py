"""Contract for spaced review in :mod:`netstack_academy.learning.review`.

The scheduler is a plain bounded Leitner ladder, levels 0..5: a correct
review moves up one box, an incorrect review drops all the way back to box
0, and the next due date is ``now`` plus that box's documented interval.
Nothing here is adaptive or probabilistic -- given a level, an outcome and
a clock reading, the next state is fully determined, which is what makes
"what is due today" answerable without guessing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from netstack_academy.learning.review import (
    LEITNER_INTERVAL_DAYS,
    MAX_LEITNER_LEVEL,
    InvalidLeitnerLevelError,
    interval_for_level,
    next_due,
    next_level,
    schedule_review,
)
from netstack_academy.learning.store import LearningStore

from learning_fakes import FakeClock
from lesson_factory import make_lesson

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


def test_ladder_is_bounded_at_five_levels() -> None:
    assert MAX_LEITNER_LEVEL == 5
    assert len(LEITNER_INTERVAL_DAYS) == MAX_LEITNER_LEVEL + 1


def test_intervals_are_documented_and_strictly_increasing() -> None:
    assert LEITNER_INTERVAL_DAYS[0] == 0
    assert list(LEITNER_INTERVAL_DAYS) == sorted(LEITNER_INTERVAL_DAYS)
    assert len(set(LEITNER_INTERVAL_DAYS)) == len(LEITNER_INTERVAL_DAYS)


@pytest.mark.parametrize("level", [0, 1, 2, 3, 4])
def test_a_correct_review_advances_exactly_one_level(level: int) -> None:
    assert next_level(level, correct=True) == level + 1


def test_a_correct_review_at_the_top_level_stays_there() -> None:
    assert next_level(MAX_LEITNER_LEVEL, correct=True) == MAX_LEITNER_LEVEL


@pytest.mark.parametrize("level", [0, 1, 2, 3, 4, 5])
def test_an_incorrect_review_resets_to_level_zero(level: int) -> None:
    assert next_level(level, correct=False) == 0


@pytest.mark.parametrize("level", [-1, 6, 100])
def test_levels_outside_the_ladder_are_rejected(level: int) -> None:
    with pytest.raises(InvalidLeitnerLevelError):
        next_level(level, correct=True)

    with pytest.raises(InvalidLeitnerLevelError):
        interval_for_level(level)


@pytest.mark.parametrize("level", [0, 1, 2, 3, 4, 5])
def test_next_due_is_now_plus_the_levels_documented_interval(level: int) -> None:
    assert next_due(level, now=NOW) == NOW + timedelta(days=LEITNER_INTERVAL_DAYS[level])


def test_level_zero_is_due_immediately() -> None:
    """A card the learner just got wrong has to come back in the same
    session, not tomorrow.
    """
    assert next_due(0, now=NOW) == NOW


def test_scheduling_requires_a_timezone_aware_clock_reading() -> None:
    """Due dates are compared against stored UTC timestamps; a naive
    ``datetime`` would compare wrongly (or raise) depending on the host's
    local zone.
    """
    with pytest.raises(ValueError):
        next_due(1, now=datetime(2026, 3, 1, 12, 0))


def test_schedule_review_combines_the_level_move_and_the_due_date() -> None:
    schedule = schedule_review(2, correct=True, now=NOW)

    assert schedule.level == 3
    assert schedule.next_due == NOW + timedelta(days=LEITNER_INTERVAL_DAYS[3])


def test_schedule_review_after_a_lapse_returns_to_the_bottom_of_the_ladder() -> None:
    schedule = schedule_review(5, correct=False, now=NOW)

    assert schedule.level == 0
    assert schedule.next_due == NOW


def test_scheduling_is_deterministic() -> None:
    assert schedule_review(3, correct=True, now=NOW) == schedule_review(
        3, correct=True, now=NOW
    )


def test_first_correct_review_creates_a_card_at_level_one(
    store: LearningStore, clock: FakeClock
) -> None:
    lesson = make_lesson()

    card = store.record_review(lesson.id, correct=True)

    assert card.lesson_id == lesson.id
    assert card.level == 1
    assert card.last_reviewed_at == clock.now
    assert card.next_due == clock.now + timedelta(days=LEITNER_INTERVAL_DAYS[1])


def test_review_card_is_absent_until_the_first_review(store: LearningStore) -> None:
    assert store.review_card("lesson-napi-poll") is None


def test_consecutive_correct_reviews_climb_the_ladder(
    store: LearningStore, clock: FakeClock
) -> None:
    store.record_review("lesson-napi-poll", correct=True)
    clock.advance(days=1)
    card = store.record_review("lesson-napi-poll", correct=True)

    assert card.level == 2
    assert card.next_due == clock.now + timedelta(days=LEITNER_INTERVAL_DAYS[2])


def test_an_incorrect_review_resets_the_stored_card(
    store: LearningStore, clock: FakeClock
) -> None:
    store.record_review("lesson-napi-poll", correct=True)
    clock.advance(days=1)
    store.record_review("lesson-napi-poll", correct=True)
    clock.advance(days=3)

    card = store.record_review("lesson-napi-poll", correct=False)

    assert card.level == 0
    assert card.next_due == clock.now


def test_due_reviews_exclude_cards_scheduled_for_the_future(
    store: LearningStore, clock: FakeClock
) -> None:
    store.record_review("lesson-napi-poll", correct=True)

    assert store.due_reviews() == []

    clock.advance(days=1)
    assert [card.lesson_id for card in store.due_reviews()] == ["lesson-napi-poll"]


def test_due_reviews_include_cards_due_exactly_now(
    store: LearningStore, clock: FakeClock
) -> None:
    store.record_review("lesson-napi-poll", correct=False)

    assert [card.lesson_id for card in store.due_reviews()] == ["lesson-napi-poll"]


def test_due_reviews_accept_an_explicit_clock_reading(
    store: LearningStore, clock: FakeClock
) -> None:
    store.record_review("lesson-napi-poll", correct=True)

    later = clock.now + timedelta(days=2)
    assert [card.lesson_id for card in store.due_reviews(now=later)] == [
        "lesson-napi-poll"
    ]


def test_due_reviews_are_ordered_by_due_date_then_lesson_id(
    store: LearningStore, clock: FakeClock
) -> None:
    store.record_review("lesson-b", correct=False)
    store.record_review("lesson-a", correct=False)
    store.record_review("lesson-c", correct=True)
    clock.advance(days=5)

    assert [card.lesson_id for card in store.due_reviews()] == [
        "lesson-a",
        "lesson-b",
        "lesson-c",
    ]


def test_review_cards_survive_reopening_the_database(tmp_path: Path) -> None:
    db_path = tmp_path / "learning.sqlite3"
    clock = FakeClock()
    with LearningStore.open(db_path, clock=clock) as first:
        first.record_review("lesson-napi-poll", correct=True)

    with LearningStore.open(db_path, clock=clock) as second:
        card = second.review_card("lesson-napi-poll")

    assert card.level == 1
    assert card.next_due == clock.now + timedelta(days=LEITNER_INTERVAL_DAYS[1])
