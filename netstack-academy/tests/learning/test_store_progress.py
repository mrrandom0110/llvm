"""Contract for lesson progress in :mod:`netstack_academy.learning.store`.

The store is the only durable thing a learner owns, so its schema is set up
the same way the symbol index's is (migrations recorded in the database,
foreign keys enforced, WAL journaling) and its status machine is explicit:
``not_started -> in_progress -> completed`` and nothing else. "Complete a
lesson I never opened" is a bug in the caller, not a shortcut, so it raises
rather than silently inventing a start time.
"""

from __future__ import annotations

import sqlite3
from datetime import timezone
from pathlib import Path

import pytest

from netstack_academy.learning.store import (
    InvalidStatusTransitionError,
    LearningStore,
)

from learning_fakes import FakeClock
from lesson_factory import make_lesson


def test_open_creates_the_database_file(tmp_path: Path) -> None:
    db_path = tmp_path / "learning.sqlite3"

    with LearningStore.open(db_path):
        pass

    assert db_path.exists()


def test_open_enables_wal_journal_mode(store: LearningStore) -> None:
    mode = store.connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert mode.lower() == "wal"


def test_open_enables_foreign_keys(store: LearningStore) -> None:
    assert store.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_open_records_a_schema_version(store: LearningStore) -> None:
    assert store.schema_version >= 1


def test_reopening_the_same_database_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "learning.sqlite3"

    with LearningStore.open(db_path) as first:
        first_version = first.schema_version
    with LearningStore.open(db_path) as second:
        second_version = second.schema_version

    assert first_version == second_version


def test_fresh_databases_have_identical_schema(tmp_path: Path) -> None:
    def schema_of(name: str) -> list[tuple[str]]:
        with LearningStore.open(tmp_path / name) as opened:
            return opened.connection.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY name"
            ).fetchall()

    assert schema_of("one.sqlite3") == schema_of("two.sqlite3")


def test_unknown_lesson_reports_not_started_progress(store: LearningStore) -> None:
    progress = store.get_progress("lesson-napi-poll")

    assert progress.status == "not_started"
    assert progress.started_at is None
    assert progress.completed_at is None


def test_start_lesson_moves_progress_to_in_progress(
    store: LearningStore, clock: FakeClock
) -> None:
    progress = store.start_lesson("lesson-napi-poll")

    assert progress.status == "in_progress"
    assert progress.started_at == clock.now
    assert progress.completed_at is None


def test_starting_an_already_started_lesson_keeps_the_original_start_time(
    store: LearningStore, clock: FakeClock
) -> None:
    """Reopening a lesson is the most common interaction there is; it must
    not rewrite history.
    """
    first_started_at = store.start_lesson("lesson-napi-poll").started_at
    clock.advance(hours=3)

    progress = store.start_lesson("lesson-napi-poll")

    assert progress.status == "in_progress"
    assert progress.started_at == first_started_at


def test_complete_lesson_records_a_completion_timestamp(
    store: LearningStore, clock: FakeClock
) -> None:
    store.start_lesson("lesson-napi-poll")
    clock.advance(minutes=45)

    progress = store.complete_lesson("lesson-napi-poll")

    assert progress.status == "completed"
    assert progress.completed_at == clock.now
    assert progress.started_at < progress.completed_at


def test_completing_a_lesson_that_was_never_started_is_rejected(
    store: LearningStore,
) -> None:
    with pytest.raises(InvalidStatusTransitionError):
        store.complete_lesson("lesson-napi-poll")

    assert store.get_progress("lesson-napi-poll").status == "not_started"


def test_completing_a_completed_lesson_twice_is_rejected(
    store: LearningStore, clock: FakeClock
) -> None:
    store.start_lesson("lesson-napi-poll")
    completed_at = store.complete_lesson("lesson-napi-poll").completed_at
    clock.advance(days=1)

    with pytest.raises(InvalidStatusTransitionError):
        store.complete_lesson("lesson-napi-poll")

    assert store.get_progress("lesson-napi-poll").completed_at == completed_at


def test_restarting_a_completed_lesson_is_rejected(store: LearningStore) -> None:
    store.start_lesson("lesson-napi-poll")
    store.complete_lesson("lesson-napi-poll")

    with pytest.raises(InvalidStatusTransitionError):
        store.start_lesson("lesson-napi-poll")

    assert store.get_progress("lesson-napi-poll").status == "completed"


def test_rejected_transition_names_both_states(store: LearningStore) -> None:
    with pytest.raises(InvalidStatusTransitionError) as excinfo:
        store.complete_lesson("lesson-napi-poll")

    message = str(excinfo.value)
    assert "not_started" in message
    assert "completed" in message


def test_list_progress_is_ordered_by_lesson_id(store: LearningStore) -> None:
    for lesson_id in ("lesson-c", "lesson-a", "lesson-b"):
        store.start_lesson(lesson_id)

    assert [progress.lesson_id for progress in store.list_progress()] == [
        "lesson-a",
        "lesson-b",
        "lesson-c",
    ]


def test_timestamps_are_timezone_aware_utc(store: LearningStore) -> None:
    progress = store.start_lesson("lesson-napi-poll")

    assert progress.started_at.tzinfo is not None
    assert progress.started_at.utcoffset() == timezone.utc.utcoffset(None)


def test_progress_survives_reopening_the_database(tmp_path: Path) -> None:
    db_path = tmp_path / "learning.sqlite3"
    with LearningStore.open(db_path, clock=FakeClock()) as first:
        first.start_lesson("lesson-napi-poll")
        first.complete_lesson("lesson-napi-poll")
        first.start_lesson("lesson-qdisc-dequeue")

    with LearningStore.open(db_path, clock=FakeClock()) as second:
        statuses = {
            progress.lesson_id: progress.status for progress in second.list_progress()
        }

    assert statuses == {
        "lesson-napi-poll": "completed",
        "lesson-qdisc-dequeue": "in_progress",
    }


def test_recording_a_quiz_attempt_creates_the_lesson_progress_row(
    store: LearningStore,
) -> None:
    """Attempts and review cards hang off a lesson's progress row, so
    recording either has to bring that row into existence rather than leave
    a dangling reference.
    """
    lesson = make_lesson()

    store.record_quiz_attempt(lesson, responses={"q-context": "b"})

    assert [progress.lesson_id for progress in store.list_progress()] == [lesson.id]


def test_foreign_keys_reject_a_quiz_attempt_for_an_unknown_lesson(
    store: LearningStore,
) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store.connection.execute(
            """
            INSERT INTO quiz_attempts (
                lesson_id, score, correct_count, question_count, created_at
            ) VALUES ('lesson-ghost', 1.0, 1, 1, '2026-03-01T12:00:00+00:00')
            """
        )


def test_close_closes_the_underlying_connection(tmp_path: Path) -> None:
    opened = LearningStore.open(tmp_path / "learning.sqlite3")
    opened.close()

    with pytest.raises(sqlite3.ProgrammingError):
        opened.connection.execute("SELECT 1")


def test_context_manager_closes_the_store(tmp_path: Path) -> None:
    with LearningStore.open(tmp_path / "learning.sqlite3") as opened:
        pass

    with pytest.raises(sqlite3.ProgrammingError):
        opened.connection.execute("SELECT 1")
