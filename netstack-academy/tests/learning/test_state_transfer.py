"""Contract for export/import in :mod:`netstack_academy.learning.store`.

Export is the learner's escape hatch: a portable JSON document holding
progress, notes, quiz attempts and review cards, with timestamps written as
ISO-8601 strings so the file survives being read by anything other than
this program.

Import is the matching, untrusted direction. The payload may come from
another machine or a text editor, so it is validated as a whole before a
single row changes: an unknown key, an unknown status, or a note path that
tries to escape the repository aborts the entire import and leaves the
store exactly as it was.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from netstack_academy.learning.store import LearningStore, StateImportError

from learning_fakes import FakeClock
from lesson_factory import make_lesson


def _populated(store: LearningStore) -> LearningStore:
    lesson = make_lesson()
    store.start_lesson(lesson.id)
    store.complete_lesson(lesson.id)
    store.start_lesson("lesson-qdisc-dequeue")
    store.upsert_lesson_note(lesson.id, "Budget is per poll.")
    store.upsert_symbol_note("napi_poll", "Called from net_rx_action.", relative_path="net/core/dev.c")
    store.record_quiz_attempt(lesson, responses={"q-context": "b"})
    store.record_review(lesson.id, correct=True)
    return store


def test_export_is_json_serializable(store: LearningStore) -> None:
    payload = _populated(store).export_state()

    assert json.loads(json.dumps(payload)) == payload


def test_export_carries_a_version_and_every_section(store: LearningStore) -> None:
    payload = _populated(store).export_state()

    assert payload["version"] >= 1
    assert {"progress", "notes", "quiz_attempts", "review_cards"} <= set(payload)


def test_export_writes_timestamps_as_iso_strings(store: LearningStore) -> None:
    payload = _populated(store).export_state()

    completed_at = payload["progress"][0]["completed_at"]
    assert isinstance(completed_at, str)
    assert completed_at.startswith("2026-03-01T12:00:00")


def test_export_import_round_trip_reproduces_the_learner_state(
    tmp_path: Path,
) -> None:
    with LearningStore.open(tmp_path / "source.sqlite3", clock=FakeClock()) as source:
        payload = _populated(source).export_state()
        expected = {
            "progress": {p.lesson_id: p.status for p in source.list_progress()},
            "notes": {(n.target_type, n.target_key): n.body for n in source.list_notes()},
            "attempts": [a.score for a in source.list_quiz_attempts("lesson-napi-poll")],
            "card": source.review_card("lesson-napi-poll").level,
        }

    with LearningStore.open(tmp_path / "target.sqlite3", clock=FakeClock()) as target:
        target.import_state(payload)

        assert {p.lesson_id: p.status for p in target.list_progress()} == expected["progress"]
        assert {
            (n.target_type, n.target_key): n.body for n in target.list_notes()
        } == expected["notes"]
        assert [
            a.score for a in target.list_quiz_attempts("lesson-napi-poll")
        ] == expected["attempts"]
        assert target.review_card("lesson-napi-poll").level == expected["card"]


def test_import_preserves_original_timestamps(tmp_path: Path) -> None:
    with LearningStore.open(tmp_path / "source.sqlite3", clock=FakeClock()) as source:
        payload = _populated(source).export_state()
        started_at = source.get_progress("lesson-napi-poll").started_at

    later_clock = FakeClock()
    later_clock.advance(days=30)
    with LearningStore.open(tmp_path / "target.sqlite3", clock=later_clock) as target:
        target.import_state(payload)

        assert target.get_progress("lesson-napi-poll").started_at == started_at


def test_import_replaces_existing_local_state(tmp_path: Path) -> None:
    """Import restores a snapshot rather than merging two divergent
    histories, so anything not in the payload is gone afterwards.
    """
    with LearningStore.open(tmp_path / "source.sqlite3", clock=FakeClock()) as source:
        payload = _populated(source).export_state()

    with LearningStore.open(tmp_path / "target.sqlite3", clock=FakeClock()) as target:
        target.start_lesson("lesson-only-local")
        target.import_state(payload)

        assert "lesson-only-local" not in {p.lesson_id for p in target.list_progress()}


def test_import_rejects_a_payload_without_a_version(store: LearningStore) -> None:
    payload = _populated(store).export_state()
    payload.pop("version")

    with pytest.raises(StateImportError):
        store.import_state(payload)


def test_import_rejects_an_unsupported_version(store: LearningStore) -> None:
    payload = _populated(store).export_state()
    payload["version"] = 999

    with pytest.raises(StateImportError):
        store.import_state(payload)


def test_import_rejects_unknown_top_level_keys(store: LearningStore) -> None:
    payload = _populated(store).export_state()
    payload["run_command"] = "rm -rf /"

    with pytest.raises(StateImportError) as excinfo:
        store.import_state(payload)

    assert "run_command" in str(excinfo.value)


def test_import_rejects_unknown_record_fields(store: LearningStore) -> None:
    payload = _populated(store).export_state()
    payload["progress"][0]["db_path"] = "/etc/passwd"

    with pytest.raises(StateImportError) as excinfo:
        store.import_state(payload)

    assert "db_path" in str(excinfo.value)


@pytest.mark.parametrize("unsafe_path", ["../../etc/passwd", "/etc/passwd"])
def test_import_rejects_a_note_path_that_escapes_the_repository(
    store: LearningStore, unsafe_path: str
) -> None:
    payload = _populated(store).export_state()
    for note in payload["notes"]:
        if note["target_type"] == "symbol":
            note["relative_path"] = unsafe_path

    with pytest.raises(StateImportError) as excinfo:
        store.import_state(payload)

    assert unsafe_path in str(excinfo.value)


def test_import_rejects_an_unknown_progress_status(store: LearningStore) -> None:
    payload = _populated(store).export_state()
    payload["progress"][0]["status"] = "mastered"

    with pytest.raises(StateImportError):
        store.import_state(payload)


def test_import_rejects_a_review_level_outside_the_ladder(
    store: LearningStore,
) -> None:
    payload = _populated(store).export_state()
    payload["review_cards"][0]["level"] = 42

    with pytest.raises(StateImportError):
        store.import_state(payload)


def test_import_rejects_a_quiz_score_outside_zero_to_one(
    store: LearningStore,
) -> None:
    payload = _populated(store).export_state()
    payload["quiz_attempts"][0]["score"] = 7.5

    with pytest.raises(StateImportError):
        store.import_state(payload)


def test_a_rejected_import_mutates_nothing(tmp_path: Path) -> None:
    """The invalid record here is the *second* progress entry, so a loader
    that validated row-by-row while writing would already have committed
    the first one by the time it failed.
    """
    with LearningStore.open(tmp_path / "source.sqlite3", clock=FakeClock()) as source:
        payload = _populated(source).export_state()

    payload["progress"].append(
        {
            "lesson_id": "lesson-imported",
            "status": "in_progress",
            "started_at": "2026-03-01T12:00:00+00:00",
            "completed_at": None,
        }
    )
    payload["progress"].append(
        {
            "lesson_id": "lesson-broken",
            "status": "teleported",
            "started_at": "2026-03-01T12:00:00+00:00",
            "completed_at": None,
        }
    )

    with LearningStore.open(tmp_path / "target.sqlite3", clock=FakeClock()) as target:
        target.start_lesson("lesson-only-local")
        before = [(p.lesson_id, p.status) for p in target.list_progress()]

        with pytest.raises(StateImportError):
            target.import_state(payload)

        assert [(p.lesson_id, p.status) for p in target.list_progress()] == before
        assert target.list_notes() == []


def test_import_does_not_mutate_the_payload_it_was_given(
    store: LearningStore,
) -> None:
    payload = _populated(store).export_state()
    original = copy.deepcopy(payload)

    store.import_state(payload)

    assert payload == original


def test_imported_state_survives_reopening_the_database(tmp_path: Path) -> None:
    with LearningStore.open(tmp_path / "source.sqlite3", clock=FakeClock()) as source:
        payload = _populated(source).export_state()

    target_path = tmp_path / "target.sqlite3"
    with LearningStore.open(target_path, clock=FakeClock()) as target:
        target.import_state(payload)

    with LearningStore.open(target_path, clock=FakeClock()) as reopened:
        assert reopened.get_progress("lesson-napi-poll").status == "completed"
