"""Spaced review scheduling: a bounded Leitner ladder.

Levels run 0..5. A correct review moves a card up exactly one box; an
incorrect one drops it all the way back to box 0, because a lapse means the
previous spacing was already too long to be trusted. The next due date is
``now`` plus the box's interval, and nothing else feeds into it: no
difficulty estimate, no per-learner decay, no randomness. Given a level, an
outcome and a clock reading, the next state is fully determined, which is
what makes "what is due today" a question with one answer rather than a
guess that drifts between runs.

The intervals below are the classic doubling-ish Leitner spacing. Box 0 is
deliberately ``0`` days: a card the learner just got wrong comes back in the
same session rather than tomorrow, when the mistake is still fresh.

Every entry point takes an explicit, timezone-aware ``now``. Due dates are
compared against stored UTC timestamps, so a naive ``datetime`` would
compare against the wrong instant (or raise) depending on the host's local
zone -- it is rejected rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

#: Highest box in the ladder. A card that reaches it stays there.
MAX_LEITNER_LEVEL = 5

#: Days until the next review, indexed by level.
LEITNER_INTERVAL_DAYS: tuple[int, ...] = (0, 1, 3, 7, 16, 35)


class InvalidLeitnerLevelError(ValueError):
    """Raised when a level falls outside the bounded 0..5 ladder."""


@dataclass(frozen=True, slots=True)
class ReviewSchedule:
    """Where a card lands after one review."""

    level: int
    next_due: datetime


def _check_level(level: int) -> None:
    if isinstance(level, bool) or not isinstance(level, int):
        raise InvalidLeitnerLevelError(f"Leitner level must be an integer, got {level!r}")
    if not 0 <= level <= MAX_LEITNER_LEVEL:
        raise InvalidLeitnerLevelError(
            f"Leitner level {level!r} is outside the ladder (0..{MAX_LEITNER_LEVEL})"
        )


def _check_now(now: datetime) -> datetime:
    if not isinstance(now, datetime):
        raise ValueError(f"'now' must be a datetime, got {now!r}")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError(
            "'now' must be timezone-aware; due dates are stored and compared in UTC"
        )
    return now.astimezone(timezone.utc)


def next_level(level: int, *, correct: bool) -> int:
    """The box a card moves to after a correct/incorrect review."""
    _check_level(level)
    if not correct:
        return 0
    return min(level + 1, MAX_LEITNER_LEVEL)


def interval_for_level(level: int) -> timedelta:
    """How long a card rests in ``level`` before it is due again."""
    _check_level(level)
    return timedelta(days=LEITNER_INTERVAL_DAYS[level])


def next_due(level: int, *, now: datetime) -> datetime:
    """When a card sitting in ``level`` becomes due, measured from ``now``."""
    return _check_now(now) + interval_for_level(level)


def schedule_review(level: int, *, correct: bool, now: datetime) -> ReviewSchedule:
    """Apply one review outcome: the new level and when it comes back."""
    moved = next_level(level, correct=correct)
    return ReviewSchedule(level=moved, next_due=next_due(moved, now=now))
