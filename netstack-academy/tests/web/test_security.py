"""Security contract for the served application.

Nothing here authenticates anyone: the app is bound to loopback and whoever
can reach it is the learner. That makes two classes of problem worth
defending against anyway.

The first is *content* that executes. Lesson bodies are Markdown with raw HTML
enabled, notes are the learner's own free text, and the search query is
whatever was in the URL -- a link from a chat window included. Bodies are
sanitized once at load time; notes and queries are escaped at render time; and
a content security policy without inline script means an injection that got
past both still has nothing to run.

The second is *state* that changes without being asked to. Every mutation is a
JSON request, which a cross-site form cannot make, and no mutation is reachable
with GET.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from netstack_academy.learning.store import LearningStore
from netstack_academy.settings import Settings

from academy_content import (
    BODY_XSS_MARKER,
    CORRECT_OPTION_ID,
    QUESTION_ID,
    QUIZ_EXPLANATION_MARKER,
)
from index_fixtures import DEV_C
from html_helpers import assert_no_inline_scripts

PAGES = ["/", "/modules/rx-path", "/lessons/napi-poll", "/symbols/napi_poll", "/search"]

NOTE_XSS = '<script>noteXss()</script><img src=x onerror="noteXss()">'


# ----------------------------------------------------------------------
# Response headers
# ----------------------------------------------------------------------


@pytest.mark.parametrize("path", PAGES + ["/api/progress", "/static/css/academy.css"])
def test_responses_set_the_basic_security_headers(
    client: TestClient, path: str
) -> None:
    headers = client.get(path).headers

    assert headers["x-content-type-options"] == "nosniff"
    assert headers["x-frame-options"].upper() == "DENY"
    assert "referrer-policy" in headers


@pytest.mark.parametrize("path", PAGES)
def test_pages_carry_a_content_security_policy(
    client: TestClient, path: str
) -> None:
    policy = client.get(path).headers["content-security-policy"]

    assert "default-src 'self'" in policy
    assert "script-src 'self'" in policy
    assert "object-src 'none'" in policy
    assert "frame-ancestors 'none'" in policy


@pytest.mark.parametrize("path", PAGES)
def test_the_policy_permits_no_script_execution_from_markup(
    client: TestClient, path: str
) -> None:
    policy = client.get(path).headers["content-security-policy"]
    script_directive = next(
        directive
        for directive in policy.split(";")
        if directive.strip().startswith("script-src")
    )

    assert "unsafe-inline" not in script_directive
    assert "unsafe-eval" not in policy


@pytest.mark.parametrize("path", PAGES)
def test_the_policy_names_no_external_origin(client: TestClient, path: str) -> None:
    """A policy listing a CDN is a policy that allows a CDN to run code here.
    """
    policy = client.get(path).headers["content-security-policy"]

    assert "http://" not in policy
    assert "https://" not in policy


@pytest.mark.parametrize("path", PAGES)
def test_pages_contain_no_inline_script(client: TestClient, path: str) -> None:
    assert_no_inline_scripts(client.get(path).text)


# ----------------------------------------------------------------------
# Escaping what the learner wrote
# ----------------------------------------------------------------------


def test_a_lesson_note_is_escaped_when_rendered(
    client: TestClient, store: LearningStore
) -> None:
    store.upsert_lesson_note("lesson-napi-poll", NOTE_XSS)

    html = client.get("/lessons/napi-poll").text

    assert "<script>noteXss" not in html
    assert "&lt;script&gt;" in html
    assert "onerror=" not in html.replace("&quot;", "").replace("&#34;", "")


def test_a_symbol_note_is_escaped_when_rendered(
    client: TestClient, store: LearningStore
) -> None:
    store.upsert_symbol_note("napi_poll", NOTE_XSS, relative_path=DEV_C)

    html = client.get("/symbols/napi_poll").text

    assert "<script>noteXss" not in html
    assert "&lt;script&gt;" in html


def test_a_hostile_note_survives_a_round_trip_unchanged(
    client: TestClient, store: LearningStore
) -> None:
    """Escaping is presentation. The stored text is exactly what was typed, so
    a note about ``<linux/skbuff.h>`` still says that when it comes back.
    """
    client.put("/api/lessons/napi-poll/note", json={"body": NOTE_XSS})

    note = store.get_lesson_note("lesson-napi-poll")
    assert note is not None and note.body == NOTE_XSS


def test_a_hostile_search_query_is_escaped(client: TestClient) -> None:
    html = client.get("/search", params={"q": NOTE_XSS}).text

    assert "<script>noteXss" not in html
    assert "&lt;script&gt;" in html


def test_authored_markup_never_executes(client: TestClient) -> None:
    """The lesson body is sanitized at load time, so the payload is gone from
    the model before any template sees it.
    """
    html = client.get("/lessons/napi-poll").text

    assert BODY_XSS_MARKER not in html
    assert "javascript:" not in html.lower()


# ----------------------------------------------------------------------
# The answer key
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/modules/rx-path",
        "/lessons/napi-poll",
        "/api/progress",
        "/search?q=napi_poll",
        "/api/search?q=napi_poll",
    ],
)
def test_no_surface_leaks_the_answer_key_before_an_attempt(
    client: TestClient, path: str
) -> None:
    assert QUIZ_EXPLANATION_MARKER not in client.get(path).text


def test_only_the_graded_response_carries_explanations(
    client: TestClient,
) -> None:
    graded = client.post(
        "/api/lessons/napi-poll/quiz",
        json={"responses": {QUESTION_ID: CORRECT_OPTION_ID}},
    )

    assert QUIZ_EXPLANATION_MARKER in graded.text
    assert QUIZ_EXPLANATION_MARKER not in client.get("/lessons/napi-poll").text


# ----------------------------------------------------------------------
# Deep links
# ----------------------------------------------------------------------


def test_editor_links_only_ever_use_a_known_scheme(client: TestClient) -> None:
    """Every absolute link on a symbol card is an editor deep link built by the
    one function allowed to build them.
    """
    html = client.get("/symbols/napi_poll").text

    schemes = {
        match.group(1).lower()
        for match in re.finditer(r'href="([a-z][a-z0-9+.\-]*):', html, re.IGNORECASE)
    }
    assert schemes <= {"cursor"}
    assert "cursor://vscode-remote/wsl+" in html


@pytest.mark.parametrize("scheme", ["javascript", "data", "file", "vbscript"])
def test_the_editor_scheme_is_a_closed_set(scheme: str) -> None:
    """The scheme is interpolated straight into an ``href``. Making it a
    configurable free-text field would make configuration an XSS vector.
    """
    with pytest.raises(ValidationError):
        Settings(editor_scheme=scheme)


def test_no_page_links_outside_the_kernel_repository(
    client: TestClient, kernel_repo: Path
) -> None:
    """Every editor link this app emits comes from the deep-link builder,
    which refuses a path that escapes the repository -- so the corrupt index
    row produces no link at all rather than a link somewhere else.
    """
    for path in ("/symbols/escaped_symbol", "/symbols/gone_symbol"):
        html = client.get(path).text
        assert "cursor://" not in html
        assert "/etc/passwd" not in html


# ----------------------------------------------------------------------
# Mutations
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/lessons/napi-poll/progress"),
        ("GET", "/api/lessons/napi-poll/quiz"),
        ("GET", "/api/lessons/napi-poll/review"),
        ("GET", "/api/state/import"),
        ("GET", "/api/index/ensure"),
    ],
)
def test_no_mutation_is_reachable_with_a_get(
    client: TestClient, method: str, path: str
) -> None:
    """A GET that writes is a link that writes, and this app is full of links.
    """
    assert client.request(method, path).status_code == 405


def test_a_cross_site_form_post_cannot_change_state(
    client: TestClient, store: LearningStore
) -> None:
    """Mutations take JSON. A form submission from another origin cannot set
    ``application/json``, so it never reaches a handler.
    """
    response = client.post(
        "/api/lessons/napi-poll/progress",
        data={"status": "in_progress"},
    )

    assert response.status_code in (415, 422)
    assert store.get_progress("lesson-napi-poll").status == "not_started"


def test_state_import_requires_json(client: TestClient) -> None:
    response = client.post("/api/state/import", data={"version": "1"})

    assert response.status_code in (415, 422)
