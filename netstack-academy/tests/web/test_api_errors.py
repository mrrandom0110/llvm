"""Contract for the API's error shape and status codes.

Four codes carry all the meaning: 400 for input that is unsafe or names
something that cannot exist, 404 for something that could exist but does not,
409 for a request that conflicts with the state it found (an ambiguous symbol,
a status move the machine forbids), and 422 for a body or query that failed
validation.

Every one of them uses the same envelope with a stable machine-readable
``code``, because the page's JavaScript has to distinguish "you need to pick a
file" from "that lesson is gone" without parsing prose.

None of them echo the input back. A message that quotes an unsafe path turns
an error page into a reflection surface, and one that quotes a filesystem path
tells a reader something about the machine they did not ask about.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from netstack_academy.web.errors import ERROR_CODES

#: ``(method, url, kwargs, expected status, expected code)``.
ERROR_CASES = [
    (
        "GET",
        "/api/symbols/helper",
        {"params": {"path": "../../etc/passwd"}},
        400,
        "unsafe_path",
    ),
    (
        "PUT",
        "/api/symbols/napi_poll/note",
        {"params": {"path": "/etc/passwd"}, "json": {"body": "no"}},
        400,
        "unsafe_path",
    ),
    (
        "POST",
        "/api/lessons/napi-poll/quiz",
        {"json": {"responses": {"q-invented": "a"}}},
        400,
        "invalid_quiz_response",
    ),
    ("GET", "/api/symbols/no_such_symbol", {}, 404, "symbol_not_found"),
    ("GET", "/api/symbols/no_such_symbol/graph", {}, 404, "symbol_not_found"),
    (
        "POST",
        "/api/lessons/no-such-lesson/progress",
        {"json": {"status": "in_progress"}},
        404,
        "lesson_not_found",
    ),
    ("GET", "/api/symbols/helper", {}, 409, "symbol_ambiguous"),
    (
        "POST",
        "/api/lessons/napi-poll/progress",
        {"json": {"status": "completed"}},
        409,
        "invalid_transition",
    ),
    (
        "POST",
        "/api/lessons/napi-poll/progress",
        {"json": {"status": "teleported"}},
        422,
        "validation_error",
    ),
    (
        "PUT",
        "/api/lessons/napi-poll/note",
        {"json": {"body": "   "}},
        422,
        "invalid_note_body",
    ),
    (
        "POST",
        "/api/state/import",
        {"json": {"version": 99, "progress": [], "notes": []}},
        422,
        "invalid_state_document",
    ),
]

CASE_IDS = [f"{code}-{status}" for _, _, _, status, code in ERROR_CASES]


def _send(client: TestClient, method: str, url: str, kwargs: dict):
    return client.request(method, url, **kwargs)


@pytest.mark.parametrize(
    ("method", "url", "kwargs", "status", "code"), ERROR_CASES, ids=CASE_IDS
)
def test_errors_use_their_documented_status_and_code(
    client: TestClient,
    method: str,
    url: str,
    kwargs: dict,
    status: int,
    code: str,
) -> None:
    response = _send(client, method, url, kwargs)

    assert response.status_code == status
    assert response.json()["error"]["code"] == code


@pytest.mark.parametrize(
    ("method", "url", "kwargs", "status", "code"), ERROR_CASES, ids=CASE_IDS
)
def test_every_error_uses_the_same_envelope(
    client: TestClient,
    method: str,
    url: str,
    kwargs: dict,
    status: int,
    code: str,
) -> None:
    response = _send(client, method, url, kwargs)

    payload = response.json()
    assert set(payload) == {"error"}
    assert set(payload["error"]) <= {"code", "message", "details"}
    assert isinstance(payload["error"]["message"], str)
    assert payload["error"]["message"].strip()
    assert response.headers["content-type"].startswith("application/json")


@pytest.mark.parametrize(
    ("method", "url", "kwargs", "status", "code"), ERROR_CASES, ids=CASE_IDS
)
def test_error_codes_are_declared_in_one_place(
    method: str, url: str, kwargs: dict, status: int, code: str
) -> None:
    """One registry of codes, so a client can enumerate what it may receive.
    """
    assert code in ERROR_CODES


@pytest.mark.parametrize(
    ("method", "url", "kwargs", "status", "code"), ERROR_CASES, ids=CASE_IDS
)
def test_errors_leak_neither_paths_nor_tracebacks(
    client: TestClient,
    kernel_repo: Path,
    method: str,
    url: str,
    kwargs: dict,
    status: int,
    code: str,
) -> None:
    response = _send(client, method, url, kwargs)

    body = response.text
    assert "Traceback" not in body
    assert str(kernel_repo) not in body
    assert "sqlite3" not in body


def test_the_same_bad_request_always_reports_the_same_code(
    client: TestClient,
) -> None:
    first = client.get("/api/symbols/helper")
    second = client.get("/api/symbols/helper")

    assert first.json()["error"]["code"] == second.json()["error"]["code"]


def test_validation_errors_name_the_offending_field(client: TestClient) -> None:
    response = client.post(
        "/api/lessons/napi-poll/progress", json={"status": "teleported"}
    )

    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert "status" in str(error.get("details", ""))


def test_validation_errors_do_not_use_the_framework_default_shape(
    client: TestClient,
) -> None:
    """FastAPI's own 422 body is ``{"detail": [...]}``. Letting that through
    would leave the API with two error shapes, and the one a client hits most
    would be the one it cannot parse with the other.
    """
    payload = client.post("/api/lessons/napi-poll/quiz", json={}).json()

    assert "detail" not in payload
    assert payload["error"]["code"] == "validation_error"


def test_a_malformed_json_body_is_a_validation_error(client: TestClient) -> None:
    response = client.post(
        "/api/lessons/napi-poll/progress",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_an_unknown_api_path_is_a_json_not_found(client: TestClient) -> None:
    response = client.get("/api/no-such-endpoint")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] in ERROR_CODES


def test_page_errors_are_rendered_as_pages_not_envelopes(
    client: TestClient,
) -> None:
    """A human who follows a stale link gets a page they can read and navigate
    away from; only ``/api`` speaks JSON.
    """
    response = client.get("/lessons/no-such-lesson")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")
    assert 'href="/"' in response.text
