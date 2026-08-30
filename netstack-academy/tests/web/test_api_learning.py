"""Contract for the learning JSON API: progress, notes, quizzes, reviews, state.

Every endpoint here writes, and every write goes through the store that
already knows the rules -- the status machine, the note path check, the
answer key, the Leitner ladder. The API's job is to translate HTTP into those
calls and their refusals into typed status codes; it is not a second place
where a lesson can be completed without being started.

The grading rule is the one worth stating twice: a submission contributes
which option was chosen and nothing else. The score comes from the lesson.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from netstack_academy.learning.store import LearningStore

from academy_content import (
    CORRECT_OPTION_ID,
    QUESTION_ID,
    QUIZ_EXPLANATION_MARKER,
    WRONG_OPTION_ID,
)
from index_fixtures import DEV_C
from web_fakes import FakeClock

QUIZ_URL = "/api/lessons/napi-poll/quiz"


# ----------------------------------------------------------------------
# Progress
# ----------------------------------------------------------------------


def test_progress_reports_the_whole_curriculum(client: TestClient) -> None:
    payload = client.get("/api/progress").json()

    assert payload["lesson_count"] == 4
    assert payload["completed_count"] == 0
    assert payload["percent_complete"] == 0
    assert [module["slug"] for module in payload["modules"]] == ["rx-path", "tx-path"]


def test_progress_names_the_next_lesson(client: TestClient) -> None:
    payload = client.get("/api/progress").json()

    assert payload["next_lesson"]["slug"] == "napi-poll"


def test_progress_reports_each_lessons_state(client: TestClient) -> None:
    lessons = client.get("/api/progress").json()["modules"][0]["lessons"]

    napi = next(lesson for lesson in lessons if lesson["slug"] == "napi-poll")
    assert napi["progress_status"] == "not_started"
    assert napi["is_unlocked"] is True
    gro = next(lesson for lesson in lessons if lesson["slug"] == "gro-coalescing")
    assert gro["is_unlocked"] is False


def test_starting_a_lesson_records_it(
    client: TestClient, store: LearningStore
) -> None:
    response = client.post(
        "/api/lessons/napi-poll/progress", json={"status": "in_progress"}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["lesson_id"] == "lesson-napi-poll"
    assert payload["status"] == "in_progress"
    assert payload["started_at"]
    assert store.get_progress("lesson-napi-poll").status == "in_progress"


def test_starting_a_started_lesson_is_idempotent(client: TestClient) -> None:
    client.post("/api/lessons/napi-poll/progress", json={"status": "in_progress"})
    first = client.get("/api/progress").json()["in_progress_count"]

    client.post("/api/lessons/napi-poll/progress", json={"status": "in_progress"})

    assert client.get("/api/progress").json()["in_progress_count"] == first


def test_completing_a_started_lesson_records_the_time(client: TestClient) -> None:
    client.post("/api/lessons/napi-poll/progress", json={"status": "in_progress"})

    response = client.post(
        "/api/lessons/napi-poll/progress", json={"status": "completed"}
    )

    assert response.status_code == 200
    assert response.json()["completed_at"]


def test_completing_an_unstarted_lesson_is_a_conflict(
    client: TestClient, store: LearningStore
) -> None:
    """"Completed a lesson I never opened" is a client bug, and inventing a
    start time to paper over it would corrupt the only timeline the learner
    has.
    """
    response = client.post(
        "/api/lessons/napi-poll/progress", json={"status": "completed"}
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "invalid_transition"
    assert store.get_progress("lesson-napi-poll").status == "not_started"


def test_progress_for_an_unknown_lesson_is_not_found(client: TestClient) -> None:
    response = client.post(
        "/api/lessons/no-such-lesson/progress", json={"status": "in_progress"}
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "lesson_not_found"


@pytest.mark.parametrize("status", ["not_started", "finished", "", None, 7])
def test_progress_rejects_a_status_outside_the_machine(
    client: TestClient, status: object
) -> None:
    response = client.post(
        "/api/lessons/napi-poll/progress", json={"status": status}
    )

    assert response.status_code == 422


def test_progress_rejects_a_body_with_unexpected_fields(
    client: TestClient, store: LearningStore
) -> None:
    """A body carrying ``completed_at`` is either a stale client or an attempt
    to write a timestamp the server owns.
    """
    response = client.post(
        "/api/lessons/napi-poll/progress",
        json={"status": "in_progress", "completed_at": "2020-01-01T00:00:00+00:00"},
    )

    assert response.status_code == 422
    assert store.get_progress("lesson-napi-poll").status == "not_started"


# ----------------------------------------------------------------------
# Notes
# ----------------------------------------------------------------------


def test_saving_a_lesson_note_persists_it(
    client: TestClient, store: LearningStore
) -> None:
    response = client.put(
        "/api/lessons/napi-poll/note", json={"body": "Budget is per poll."}
    )

    assert response.status_code == 200
    assert response.json()["body"] == "Budget is per poll."
    note = store.get_lesson_note("lesson-napi-poll")
    assert note is not None and note.body == "Budget is per poll."


def test_saving_a_note_twice_replaces_it(
    client: TestClient, store: LearningStore
) -> None:
    client.put("/api/lessons/napi-poll/note", json={"body": "First."})
    client.put("/api/lessons/napi-poll/note", json={"body": "Second."})

    assert len(store.list_notes()) == 1
    note = store.get_lesson_note("lesson-napi-poll")
    assert note is not None and note.body == "Second."


@pytest.mark.parametrize("body", ["", "   ", "\n\t "])
def test_a_blank_note_is_refused(
    client: TestClient, store: LearningStore, body: str
) -> None:
    """Emptying a note is deleting it, and there is a verb for that."""
    response = client.put("/api/lessons/napi-poll/note", json={"body": body})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_note_body"
    assert store.list_notes() == []


def test_deleting_a_note_reports_whether_one_existed(client: TestClient) -> None:
    client.put("/api/lessons/napi-poll/note", json={"body": "Temporary."})

    first = client.delete("/api/lessons/napi-poll/note")
    second = client.delete("/api/lessons/napi-poll/note")

    assert first.json()["deleted"] is True
    assert second.json()["deleted"] is False


def test_a_note_on_an_unknown_lesson_is_not_found(client: TestClient) -> None:
    response = client.put("/api/lessons/nope/note", json={"body": "Anything."})

    assert response.status_code == 404


def test_saving_a_symbol_note_keeps_its_file(
    client: TestClient, store: LearningStore
) -> None:
    """Two ``static`` functions can share a name, so a symbol note is
    identified by name *and* file.
    """
    response = client.put(
        "/api/symbols/napi_poll/note",
        params={"path": DEV_C},
        json={"body": "Called from the softirq."},
    )

    assert response.status_code == 200
    note = store.get_symbol_note("napi_poll", relative_path=DEV_C)
    assert note is not None and note.body == "Called from the softirq."


def test_a_symbol_note_with_an_unsafe_path_is_refused(
    client: TestClient, store: LearningStore
) -> None:
    response = client.put(
        "/api/symbols/napi_poll/note",
        params={"path": "../../etc/passwd"},
        json={"body": "Nope."},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsafe_path"
    assert store.list_notes() == []


def test_notes_are_returned_verbatim_not_rendered(client: TestClient) -> None:
    """A note is the learner's own text. The API stores and returns exactly
    what was written; escaping is the template's job, at render time.
    """
    body = "Check `napi_poll` <not a tag> & co."

    response = client.put("/api/lessons/napi-poll/note", json={"body": body})

    assert response.json()["body"] == body


# ----------------------------------------------------------------------
# Quizzes
# ----------------------------------------------------------------------


def test_a_correct_submission_scores_one(client: TestClient) -> None:
    response = client.post(QUIZ_URL, json={"responses": {QUESTION_ID: CORRECT_OPTION_ID}})

    assert response.status_code == 200
    payload = response.json()
    assert payload["score"] == 1.0
    assert payload["correct_count"] == 1
    assert payload["question_count"] == 1


def test_a_wrong_submission_scores_zero(client: TestClient) -> None:
    payload = client.post(
        QUIZ_URL, json={"responses": {QUESTION_ID: WRONG_OPTION_ID}}
    ).json()

    assert payload["score"] == 0.0
    assert payload["correct_count"] == 0


def test_the_graded_response_explains_each_question(client: TestClient) -> None:
    """Explanations are the teaching part of a quiz -- after an attempt, and
    only after one.
    """
    payload = client.post(
        QUIZ_URL, json={"responses": {QUESTION_ID: WRONG_OPTION_ID}}
    ).json()

    result = payload["results"][0]
    assert result["question_id"] == QUESTION_ID
    assert result["correct"] is False
    assert result["correct_option_id"] == CORRECT_OPTION_ID
    assert QUIZ_EXPLANATION_MARKER in result["explanation"]


def test_the_score_is_computed_from_the_lesson_not_the_submission(
    client: TestClient, store: LearningStore
) -> None:
    """A submission that claims its own score is refused rather than trusted.
    """
    response = client.post(
        QUIZ_URL,
        json={
            "responses": {QUESTION_ID: WRONG_OPTION_ID},
            "score": 1.0,
            "correct_count": 1,
        },
    )

    assert response.status_code == 422
    assert store.list_quiz_attempts("lesson-napi-poll") == []


def test_a_recorded_attempt_carries_the_server_side_score(
    client: TestClient, store: LearningStore
) -> None:
    client.post(QUIZ_URL, json={"responses": {QUESTION_ID: WRONG_OPTION_ID}})

    attempts = store.list_quiz_attempts("lesson-napi-poll")
    assert [attempt.score for attempt in attempts] == [0.0]
    assert attempts[0].responses == {QUESTION_ID: WRONG_OPTION_ID}


def test_a_submission_reports_the_mastery_gate(client: TestClient) -> None:
    payload = client.post(
        QUIZ_URL, json={"responses": {QUESTION_ID: CORRECT_OPTION_ID}}
    ).json()

    assert payload["meets_mastery_gate"] is False


def test_an_unknown_question_is_refused(
    client: TestClient, store: LearningStore
) -> None:
    """A submission naming a question this lesson does not have means the page
    and the content have diverged; scoring it as merely wrong hides that.
    """
    response = client.post(QUIZ_URL, json={"responses": {"q-invented": "a"}})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_quiz_response"
    assert store.list_quiz_attempts("lesson-napi-poll") == []


def test_an_unknown_option_is_refused(
    client: TestClient, store: LearningStore
) -> None:
    response = client.post(QUIZ_URL, json={"responses": {QUESTION_ID: "z"}})

    assert response.status_code == 400
    assert store.list_quiz_attempts("lesson-napi-poll") == []


def test_a_submission_without_responses_is_a_validation_error(
    client: TestClient,
) -> None:
    assert client.post(QUIZ_URL, json={}).status_code == 422


def test_a_submission_with_non_string_responses_is_refused(
    client: TestClient,
) -> None:
    assert client.post(QUIZ_URL, json={"responses": {QUESTION_ID: 7}}).status_code == 422


def test_a_quiz_for_an_unknown_lesson_is_not_found(client: TestClient) -> None:
    response = client.post(
        "/api/lessons/nope/quiz", json={"responses": {QUESTION_ID: "a"}}
    )

    assert response.status_code == 404


# ----------------------------------------------------------------------
# Reviews
# ----------------------------------------------------------------------


def test_a_correct_review_moves_the_card_up_one_box(
    client: TestClient, clock: FakeClock
) -> None:
    response = client.post("/api/lessons/napi-poll/review", json={"correct": True})

    assert response.status_code == 200
    payload = response.json()
    assert payload["level"] == 1
    assert payload["next_due"] > clock.now.isoformat()


def test_an_incorrect_review_drops_the_card_to_the_bottom(
    client: TestClient,
) -> None:
    client.post("/api/lessons/napi-poll/review", json={"correct": True})

    payload = client.post(
        "/api/lessons/napi-poll/review", json={"correct": False}
    ).json()

    assert payload["level"] == 0


def test_a_review_is_visible_in_the_due_count(
    client: TestClient, store: LearningStore
) -> None:
    client.post("/api/lessons/napi-poll/review", json={"correct": False})

    assert client.get("/api/progress").json()["due_review_count"] == 1
    assert store.review_card("lesson-napi-poll") is not None


def test_a_review_without_an_outcome_is_a_validation_error(
    client: TestClient,
) -> None:
    assert client.post("/api/lessons/napi-poll/review", json={}).status_code == 422


def test_a_review_of_an_unknown_lesson_is_not_found(client: TestClient) -> None:
    response = client.post("/api/lessons/nope/review", json={"correct": True})

    assert response.status_code == 404


# ----------------------------------------------------------------------
# Export and import
# ----------------------------------------------------------------------


def test_export_returns_the_whole_learner_state(
    client: TestClient, store: LearningStore
) -> None:
    store.start_lesson("lesson-napi-poll")
    store.upsert_lesson_note("lesson-napi-poll", "Exported note.")

    payload = client.get("/api/state/export").json()

    assert payload == store.export_state()
    assert payload["progress"][0]["lesson_id"] == "lesson-napi-poll"


def test_export_of_a_fresh_installation_is_empty_but_valid(
    client: TestClient,
) -> None:
    payload = client.get("/api/state/export").json()

    assert payload["version"] == 1
    assert payload["progress"] == []
    assert payload["notes"] == []


def test_import_replaces_the_local_state(
    client: TestClient, store: LearningStore
) -> None:
    store.start_lesson("lesson-qdisc")
    document = {
        "version": 1,
        "progress": [
            {
                "lesson_id": "lesson-napi-poll",
                "status": "in_progress",
                "started_at": "2026-02-01T09:00:00.000000+00:00",
                "completed_at": None,
            }
        ],
        "notes": [],
        "quiz_attempts": [],
        "review_cards": [],
    }

    response = client.post("/api/state/import", json=document)

    assert response.status_code == 200
    assert store.get_progress("lesson-napi-poll").status == "in_progress"
    assert store.get_progress("lesson-qdisc").status == "not_started"


def test_an_invalid_import_changes_nothing(
    client: TestClient, store: LearningStore
) -> None:
    """The document is validated in full before anything is written, so a
    problem in the last record cannot leave the first ones applied.
    """
    store.start_lesson("lesson-qdisc")
    document = {
        "version": 1,
        "progress": [
            {
                "lesson_id": "lesson-napi-poll",
                "status": "in_progress",
                "started_at": "2026-02-01T09:00:00.000000+00:00",
                "completed_at": None,
            },
            {
                "lesson_id": "lesson-gro",
                "status": "teleported",
                "started_at": None,
                "completed_at": None,
            },
        ],
        "notes": [],
        "quiz_attempts": [],
        "review_cards": [],
    }

    response = client.post("/api/state/import", json=document)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_state_document"
    assert store.get_progress("lesson-qdisc").status == "in_progress"
    assert store.get_progress("lesson-napi-poll").status == "not_started"


def test_an_import_of_an_unsupported_version_is_refused(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/state/import",
        json={
            "version": 99,
            "progress": [],
            "notes": [],
            "quiz_attempts": [],
            "review_cards": [],
        },
    )

    assert response.status_code == 422


def test_an_import_carrying_an_unsafe_note_path_is_refused(
    client: TestClient, store: LearningStore
) -> None:
    response = client.post(
        "/api/state/import",
        json={
            "version": 1,
            "progress": [],
            "notes": [
                {
                    "target_type": "symbol",
                    "target_key": "napi_poll",
                    "relative_path": "../../etc/passwd",
                    "body": "escaped",
                    "created_at": "2026-02-01T09:00:00.000000+00:00",
                    "updated_at": "2026-02-01T09:00:00.000000+00:00",
                }
            ],
            "quiz_attempts": [],
            "review_cards": [],
        },
    )

    assert response.status_code == 422
    assert store.list_notes() == []


def test_export_and_import_round_trip(
    client: TestClient, store: LearningStore
) -> None:
    store.start_lesson("lesson-napi-poll")
    store.upsert_lesson_note("lesson-napi-poll", "Round trip.")
    store.record_review("lesson-napi-poll", correct=True)
    exported = client.get("/api/state/export").json()

    client.post("/api/state/import", json=exported)

    assert client.get("/api/state/export").json() == exported


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


def test_mutations_reach_the_database_on_disk(
    client: TestClient, state_dir: Path, clock: FakeClock
) -> None:
    """The API writes through the store to SQLite, not to process memory: a
    second reader of the same file sees the same state.
    """
    client.post("/api/lessons/napi-poll/progress", json={"status": "in_progress"})
    client.put("/api/lessons/napi-poll/note", json={"body": "On disk."})

    with LearningStore.open(state_dir / "learning.sqlite3", clock=clock) as reopened:
        assert reopened.get_progress("lesson-napi-poll").status == "in_progress"
        note = reopened.get_lesson_note("lesson-napi-poll")
        assert note is not None and note.body == "On disk."


def test_learning_endpoints_never_build_the_index(
    client: TestClient, orchestrator
) -> None:
    client.get("/api/progress")
    client.post("/api/lessons/napi-poll/progress", json={"status": "in_progress"})
    client.put("/api/lessons/napi-poll/note", json={"body": "No indexing."})

    assert orchestrator.call_count == 0
