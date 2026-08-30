"""Contract for the module page.

A module page is a small thing -- a title, a summary, its lessons in order,
and how far the learner is through them -- but it is the page that makes the
course navigable without a search box, so its ordering and its breadcrumbs
carry the same weight as the dashboard's.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from netstack_academy.learning.store import LearningStore

from html_helpers import assert_page_shell, position_of, region
from web_fakes import RecordingOrchestrator


def test_module_page_is_served_as_html(client: TestClient) -> None:
    response = client.get("/modules/rx-path")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_module_page_has_the_shared_page_shell(client: TestClient) -> None:
    assert_page_shell(client.get("/modules/rx-path").text)


def test_module_page_names_the_module(client: TestClient) -> None:
    html = client.get("/modules/rx-path").text

    assert "Receive path" in html
    assert "How a frame becomes an sk_buff." in html


def test_module_page_breadcrumbs_lead_back_to_the_dashboard(
    client: TestClient,
) -> None:
    html = client.get("/modules/rx-path").text

    breadcrumb = region(html, 'aria-label="Breadcrumb"')
    assert 'href="/"' in breadcrumb
    assert "Receive path" in breadcrumb


def test_module_page_lists_its_lessons_in_order(client: TestClient) -> None:
    html = client.get("/modules/rx-path").text

    assert position_of(html, "/lessons/napi-poll") < position_of(
        html, "/lessons/gro-coalescing"
    )


def test_module_page_does_not_list_another_modules_lessons(
    client: TestClient,
) -> None:
    html = client.get("/modules/rx-path").text

    assert "/lessons/qdisc-dequeue" not in html


def test_module_page_shows_each_lessons_progress(
    client: TestClient, store: LearningStore
) -> None:
    store.start_lesson("lesson-napi-poll")

    html = client.get("/modules/rx-path").text

    assert 'data-progress-status="in_progress"' in html
    assert 'data-progress-status="not_started"' in html


def test_module_page_reports_module_completion(
    client: TestClient, store: LearningStore
) -> None:
    store.start_lesson("lesson-napi-poll")
    store.complete_lesson("lesson-napi-poll")

    html = client.get("/modules/rx-path").text

    assert re.search(r'aria-valuenow="50(\.0)?"', html)


def test_module_page_marks_a_locked_lesson(client: TestClient) -> None:
    """GRO names the NAPI lesson as a prerequisite, so the module listing has
    to say so rather than let a learner walk into it unprepared.
    """
    html = client.get("/modules/rx-path").text

    assert 'data-locked="true"' in html


def test_module_page_unlocks_a_lesson_once_its_prerequisite_is_complete(
    client: TestClient, store: LearningStore
) -> None:
    store.start_lesson("lesson-napi-poll")
    store.complete_lesson("lesson-napi-poll")

    html = client.get("/modules/rx-path").text

    assert 'data-locked="true"' not in html


def test_unknown_module_returns_a_not_found_page(client: TestClient) -> None:
    response = client.get("/modules/no-such-module")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")
    assert "not found" in response.text.lower()


def test_module_page_never_triggers_indexing(
    client: TestClient, orchestrator: RecordingOrchestrator
) -> None:
    client.get("/modules/rx-path")

    assert orchestrator.call_count == 0
