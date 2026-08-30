"""Contract for lesson and symbol notes in :mod:`netstack_academy.learning.store`.

Notes are upserts, not appends: a learner editing their note on
``napi_poll`` for the fifth time still has exactly one note, whose creation
time is preserved and whose update time moves. Symbol notes may carry the
file the symbol was read in, and that path is treated as untrusted input --
it is stored only if it is a safe repository-relative path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from netstack_academy.learning.store import LearningStore, UnsafeNotePathError

from learning_fakes import FakeClock


def test_upsert_lesson_note_stores_the_body(store: LearningStore) -> None:
    note = store.upsert_lesson_note("lesson-napi-poll", "Budget is per poll, not per packet.")

    assert note.target_type == "lesson"
    assert note.target_key == "lesson-napi-poll"
    assert note.body == "Budget is per poll, not per packet."


def test_get_lesson_note_returns_none_when_absent(store: LearningStore) -> None:
    assert store.get_lesson_note("lesson-napi-poll") is None


def test_upserting_the_same_lesson_note_updates_it_in_place(
    store: LearningStore, clock: FakeClock
) -> None:
    created = store.upsert_lesson_note("lesson-napi-poll", "First draft.")
    clock.advance(days=2)

    updated = store.upsert_lesson_note("lesson-napi-poll", "Second draft.")

    assert len(store.list_notes()) == 1
    assert updated.body == "Second draft."
    assert updated.created_at == created.created_at
    assert updated.updated_at == clock.now
    assert updated.updated_at > created.updated_at


def test_delete_lesson_note_reports_whether_anything_was_removed(
    store: LearningStore,
) -> None:
    store.upsert_lesson_note("lesson-napi-poll", "Note.")

    assert store.delete_lesson_note("lesson-napi-poll") is True
    assert store.delete_lesson_note("lesson-napi-poll") is False
    assert store.get_lesson_note("lesson-napi-poll") is None


def test_symbol_note_records_the_symbol_and_its_file(store: LearningStore) -> None:
    note = store.upsert_symbol_note(
        "napi_poll", "Called from net_rx_action.", relative_path="net/core/dev.c"
    )

    assert note.target_type == "symbol"
    assert note.target_key == "napi_poll"
    assert note.relative_path == "net/core/dev.c"


def test_symbol_note_may_omit_the_file(store: LearningStore) -> None:
    note = store.upsert_symbol_note("netif_receive_skb", "Entry into the stack.")

    assert note.relative_path is None


def test_symbol_notes_for_the_same_name_in_different_files_are_distinct(
    store: LearningStore,
) -> None:
    """Two ``static`` functions can share a name; a note about one must not
    overwrite the note about the other.
    """
    store.upsert_symbol_note("helper", "ipv4 flavour", relative_path="net/ipv4/a.c")
    store.upsert_symbol_note("helper", "ipv6 flavour", relative_path="net/ipv6/b.c")

    bodies = {note.relative_path: note.body for note in store.list_notes()}
    assert bodies == {"net/ipv4/a.c": "ipv4 flavour", "net/ipv6/b.c": "ipv6 flavour"}


def test_delete_symbol_note_matches_on_name_and_file(store: LearningStore) -> None:
    store.upsert_symbol_note("helper", "ipv4 flavour", relative_path="net/ipv4/a.c")
    store.upsert_symbol_note("helper", "ipv6 flavour", relative_path="net/ipv6/b.c")

    assert store.delete_symbol_note("helper", relative_path="net/ipv4/a.c") is True

    remaining = store.list_notes()
    assert [note.relative_path for note in remaining] == ["net/ipv6/b.c"]


@pytest.mark.parametrize(
    "unsafe_path",
    ["../../etc/passwd", "/etc/passwd", "net/../../etc/passwd", ""],
)
def test_symbol_note_rejects_an_unsafe_file_path(
    store: LearningStore, unsafe_path: str
) -> None:
    with pytest.raises(UnsafeNotePathError):
        store.upsert_symbol_note("napi_poll", "note", relative_path=unsafe_path)

    assert store.list_notes() == []


def test_note_body_must_not_be_blank(store: LearningStore) -> None:
    """Deleting is an explicit operation; saving an empty note is an
    accident, and silently keeping an empty row hides the learner's real
    intent from the export.
    """
    with pytest.raises(ValueError):
        store.upsert_lesson_note("lesson-napi-poll", "   ")

    assert store.list_notes() == []


def test_list_notes_is_deterministically_ordered(store: LearningStore) -> None:
    store.upsert_symbol_note("zeta", "z")
    store.upsert_lesson_note("lesson-b", "b")
    store.upsert_symbol_note("alpha", "a")
    store.upsert_lesson_note("lesson-a", "a")

    keys = [(note.target_type, note.target_key) for note in store.list_notes()]
    assert keys == [
        ("lesson", "lesson-a"),
        ("lesson", "lesson-b"),
        ("symbol", "alpha"),
        ("symbol", "zeta"),
    ]


def test_notes_survive_reopening_the_database(tmp_path: Path) -> None:
    db_path = tmp_path / "learning.sqlite3"
    with LearningStore.open(db_path, clock=FakeClock()) as first:
        first.upsert_lesson_note("lesson-napi-poll", "Kept across restarts.")

    with LearningStore.open(db_path, clock=FakeClock()) as second:
        note = second.get_lesson_note("lesson-napi-poll")

    assert note.body == "Kept across restarts."
