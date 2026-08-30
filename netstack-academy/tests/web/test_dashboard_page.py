"""Contract for the dashboard page.

The dashboard answers four questions a learner has when they sit down: what
is in this course, how far am I, what should I do next, and is the machinery
underneath me actually working. The last one matters more here than it would
in an ordinary application, because every symbol link on every lesson page
depends on an index that may be missing, stale, or built without ``clangd``
-- and a broken deep link is far more confusing than a page that says the
index has not been built yet.

Rendering the dashboard must not *build* that index, though: it is the page
a learner lands on, and a kernel-sized reindex is not a landing page.
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from netstack_academy.learning.store import LearningStore

from html_helpers import assert_page_shell, position_of
from index_fixtures import index_kernel_repo
from web_fakes import ExplodingOrchestrator, RecordingOrchestrator

STALE_HEAD = "0" * 40


def test_dashboard_is_served_as_html(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_dashboard_has_the_shared_page_shell(client: TestClient) -> None:
    assert_page_shell(client.get("/").text)


def test_dashboard_lists_modules_in_declared_order(client: TestClient) -> None:
    """The content directories are named ``20-rx`` and ``10-tx`` precisely so
    that a page listing them in filesystem order gets this wrong.
    """
    html = client.get("/").text

    assert position_of(html, "Receive path") < position_of(html, "Transmit path")


def test_dashboard_links_to_every_module(client: TestClient) -> None:
    html = client.get("/").text

    assert 'href="/modules/rx-path"' in html
    assert 'href="/modules/tx-path"' in html


def test_dashboard_lists_lessons_in_order_within_a_module(
    client: TestClient,
) -> None:
    html = client.get("/").text

    assert position_of(html, "/lessons/napi-poll") < position_of(
        html, "/lessons/gro-coalescing"
    )


def test_dashboard_marks_draft_lessons(client: TestClient) -> None:
    """A draft is visible -- an author has to be able to see it rendered --
    but a learner must be able to tell it apart from finished material.
    """
    html = client.get("/").text

    assert "Draft: byte queue limits" in html
    assert 'data-status="draft"' in html


def test_dashboard_reports_overall_progress(
    client: TestClient, store: LearningStore
) -> None:
    store.start_lesson("lesson-napi-poll")
    store.complete_lesson("lesson-napi-poll")

    html = client.get("/").text

    assert re.search(r'aria-valuenow="25(\.0)?"', html), "no overall progress value"
    assert "4" in html


def test_dashboard_reports_per_module_progress(
    client: TestClient, store: LearningStore
) -> None:
    store.start_lesson("lesson-napi-poll")
    store.complete_lesson("lesson-napi-poll")

    html = client.get("/").text

    assert re.search(r'aria-valuenow="50(\.0)?"', html)


def test_dashboard_names_the_next_lesson(client: TestClient) -> None:
    html = client.get("/").text

    marker = position_of(html, "data-next-lesson")
    assert "The NAPI poll loop" in html[marker : marker + 600]


def test_dashboard_next_lesson_advances_with_progress(
    client: TestClient, store: LearningStore
) -> None:
    store.start_lesson("lesson-napi-poll")
    store.complete_lesson("lesson-napi-poll")

    html = client.get("/").text

    marker = position_of(html, "data-next-lesson")
    assert "GRO coalescing" in html[marker : marker + 600]


def test_dashboard_marks_in_progress_lessons(
    client: TestClient, store: LearningStore
) -> None:
    store.start_lesson("lesson-napi-poll")

    html = client.get("/").text

    assert 'data-progress-status="in_progress"' in html


def test_dashboard_counts_due_reviews(
    client: TestClient, store: LearningStore
) -> None:
    store.record_review("lesson-napi-poll", correct=False)

    html = client.get("/").text

    marker = position_of(html, "data-due-reviews")
    assert re.search(r"\b1\b", html[marker : marker + 400])


def test_dashboard_links_to_the_lessons_that_are_due(
    client: TestClient, store: LearningStore
) -> None:
    store.record_review("lesson-napi-poll", correct=False)

    html = client.get("/").text

    marker = position_of(html, "data-due-reviews")
    assert "/lessons/napi-poll" in html[marker : marker + 800]


def test_dashboard_shows_no_due_reviews_when_nothing_is_scheduled(
    client: TestClient,
) -> None:
    html = client.get("/").text

    marker = position_of(html, "data-due-reviews")
    assert re.search(r"\b0\b", html[marker : marker + 400])


def test_dashboard_reports_the_active_kernel_head(
    client: TestClient, kernel_head: str
) -> None:
    html = client.get("/").text

    assert f'data-repository-head="{kernel_head}"' in html
    assert kernel_head[:12] in html


def test_dashboard_reports_the_indexed_head(
    client: TestClient, kernel_head: str
) -> None:
    html = client.get("/").text

    assert f'data-indexed-head="{kernel_head}"' in html
    assert 'data-index-stale="false"' in html


def test_dashboard_flags_an_index_built_at_another_commit(
    client: TestClient, index_storage, indexed_generation
) -> None:
    """An index built before the last ``git pull`` still answers queries --
    with line numbers that no longer match the file on disk. That is exactly
    the state a learner has to be told about rather than debug.
    """
    index_kernel_repo(index_storage, head=STALE_HEAD)

    html = client.get("/").text

    assert 'data-index-stale="true"' in html


def test_dashboard_says_the_index_has_not_run_this_session(
    client: TestClient,
) -> None:
    assert 'data-index-ensured="false"' in client.get("/").text


def test_dashboard_reports_provider_status_after_a_run(
    client: TestClient,
) -> None:
    client.post("/api/index/ensure")

    html = client.get("/").text

    assert 'data-index-ensured="true"' in html
    assert "ctags" in html
    assert "clangd" in html


def test_rendering_the_dashboard_never_triggers_indexing(
    client: TestClient, orchestrator: RecordingOrchestrator
) -> None:
    client.get("/")

    assert orchestrator.call_count == 0


def test_the_dashboard_renders_without_a_usable_index(make_client) -> None:
    """The dashboard is the page a learner sees before anything is indexed.
    """
    exploding = make_client(orchestrator_override=ExplodingOrchestrator())

    assert exploding.get("/").status_code == 200


def test_dashboard_offers_the_search_entry_point(client: TestClient) -> None:
    html = client.get("/").text

    assert "data-search-form" in html
    assert 'action="/search"' in html
