"""Contract for the lesson page.

This is the page the whole program exists to render, and the one with the
most to get wrong. It has to show the teaching contract a published lesson
carries -- where in the packet's life this happens, in which execution
context, who owns what, what locking applies, which structures matter, which
kernel configuration changes the answer, which tracepoints let you watch it
-- and then the authored body, the symbols to open in an editor, the lab, and
a quiz.

Two rules are absolute.

**The answer key is not on this page.** Not in a hidden input, not in a data
attribute, not in a comment. Grading happens server-side, so the page has no
reason to know the answer, and anything that knows it can leak it.

**Reading works without JavaScript.** The body, the objectives, the lab and
the context are server-rendered. JavaScript adds progress buttons, notes,
quiz submission and the call graph; it is not what makes the lesson legible.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from netstack_academy.learning.store import LearningStore

from academy_content import (
    BODY_PROSE_MARKER,
    BODY_XSS_MARKER,
    CORRECT_OPTION_ID,
    LAB_SENTINEL_PATH,
    QUESTION_ID,
    QUIZ_EXPLANATION_MARKER,
    WRONG_OPTION_ID,
)
from html_helpers import assert_page_shell, region
from web_fakes import RecordingOrchestrator

LESSON_URL = "/lessons/napi-poll"


def test_lesson_page_is_served_as_html(client: TestClient) -> None:
    response = client.get(LESSON_URL)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_lesson_page_has_the_shared_page_shell(client: TestClient) -> None:
    assert_page_shell(client.get(LESSON_URL).text)


def test_lesson_is_addressable_by_slug_and_by_id(client: TestClient) -> None:
    """A prerequisite names a lesson by id; a URL reads better with a slug.
    Both arrive at the same page.
    """
    by_slug = client.get(LESSON_URL)
    by_id = client.get("/lessons/lesson-napi-poll")

    assert by_slug.status_code == 200
    assert by_id.status_code == 200
    assert "The NAPI poll loop" in by_id.text


def test_lesson_breadcrumbs_name_the_dashboard_module_and_lesson(
    client: TestClient,
) -> None:
    breadcrumb = region(client.get(LESSON_URL).text, 'aria-label="Breadcrumb"')

    assert 'href="/"' in breadcrumb
    assert 'href="/modules/rx-path"' in breadcrumb
    assert "Receive path" in breadcrumb
    assert "The NAPI poll loop" in breadcrumb


def test_lesson_page_lists_its_objectives(client: TestClient) -> None:
    objectives = region(client.get(LESSON_URL).text, "data-objectives")

    assert "Explain when napi_poll runs" in objectives
    assert "Name the budget that bounds one poll" in objectives
    assert "<li" in objectives


def test_lesson_page_shows_the_packet_stage_and_execution_context(
    client: TestClient,
) -> None:
    context = region(client.get(LESSON_URL).text, "data-kernel-context")

    assert "rx-softirq" in context
    assert "softirq, with hard IRQs enabled" in context


def test_lesson_page_shows_ownership_and_locking(client: TestClient) -> None:
    context = region(client.get(LESSON_URL).text, "data-kernel-context")

    assert "owned by the device driver" in context
    assert "NAPI_STATE_SCHED" in context
    assert "rcu_read_lock()" in context


def test_lesson_page_shows_the_structures_and_their_fields(
    client: TestClient,
) -> None:
    structures = region(client.get(LESSON_URL).text, "data-structures")

    assert "struct napi_struct" in structures
    assert "weight" in structures
    assert "struct softnet_data" in structures
    assert "poll_list" in structures


def test_lesson_page_shows_every_config_caveat(client: TestClient) -> None:
    """A caveat that is only true with ``CONFIG_RPS`` is the difference
    between a correct answer and a wrong one, so all of them are shown.
    """
    caveats = region(client.get(LESSON_URL).text, "data-caveats")

    assert "CONFIG_RPS" in caveats
    assert "CONFIG_NET_RX_BUSY_POLL" in caveats
    assert "v5.15" in caveats


def test_lesson_page_shows_the_tracepoints(client: TestClient) -> None:
    tracepoints = region(client.get(LESSON_URL).text, "data-tracepoints")

    assert "napi:napi_poll" in tracepoints
    assert "net:netif_receive_skb" in tracepoints


def test_lesson_body_is_rendered_markdown(client: TestClient) -> None:
    body = region(client.get(LESSON_URL).text, "data-lesson-body", size=4000)

    assert "<h2" in body
    assert "<pre" in body
    assert "<code" in body
    assert "<table" in body


def test_lesson_body_escapes_kernel_include_syntax(client: TestClient) -> None:
    """``<linux/skbuff.h>`` has to be shown, not swallowed as a tag."""
    body = region(client.get(LESSON_URL).text, "data-lesson-body", size=4000)

    assert "&lt;linux/skbuff.h&gt;" in body


def test_lesson_body_keeps_ordinary_external_documentation_links(
    client: TestClient,
) -> None:
    body = region(client.get(LESSON_URL).text, "data-lesson-body", size=4000)

    assert "docs.kernel.org/networking/napi.html" in body


def test_lesson_body_carries_no_executable_markup(client: TestClient) -> None:
    """The authored body deliberately contains a ``<script>``, a
    ``javascript:`` link and an ``onerror`` handler. None of them reach the
    page: the body is sanitized once, at load time.
    """
    html = client.get(LESSON_URL).text

    assert BODY_XSS_MARKER not in html
    assert "javascript:" not in html.lower()
    assert "onerror" not in html.lower()


def test_lesson_body_is_readable_without_javascript(client: TestClient) -> None:
    html = client.get(LESSON_URL).text

    assert BODY_PROSE_MARKER in html


def test_lesson_page_links_each_source_symbol_to_its_card(
    client: TestClient,
) -> None:
    symbols = region(client.get(LESSON_URL).text, "data-source-symbols")

    assert "napi_poll" in symbols
    assert "/symbols/napi_poll" in symbols
    assert "/symbols/netif_receive_skb" in symbols


def test_lesson_symbol_links_carry_the_authored_path(client: TestClient) -> None:
    """``helper`` is ``static`` in two files. An authored path is the only
    thing that makes such a link unambiguous, so it has to survive into the
    URL.
    """
    symbols = region(client.get("/lessons/qdisc-dequeue").text, "data-source-symbols")

    assert "path=net%2Fipv4%2Fa.c" in symbols or "path=net/ipv4/a.c" in symbols


def test_lesson_page_shows_the_lab_commands_and_observations(
    client: TestClient,
) -> None:
    lab = region(client.get(LESSON_URL).text, "data-lab", size=2000)

    assert "cat /proc/net/softnet_stat" in lab
    assert "The second column stays at zero" in lab
    assert "<code" in lab


def test_lesson_page_shows_the_lab_cleanup(client: TestClient) -> None:
    lab = region(client.get(LESSON_URL).text, "data-lab", size=2000)

    assert "rm -f" in lab


def test_rendering_a_lab_never_runs_it(client: TestClient) -> None:
    """The lab is instructions for a human. The server shows them; it is not
    a shell.
    """
    sentinel = Path(LAB_SENTINEL_PATH)
    sentinel.unlink(missing_ok=True)

    client.get(LESSON_URL)

    assert not sentinel.exists()


def test_lesson_page_renders_the_quiz(client: TestClient) -> None:
    quiz = region(client.get(LESSON_URL).text, "data-quiz-form", size=2000)

    assert "In which context does napi_poll run?" in quiz
    assert "Hard IRQ" in quiz
    assert "Softirq" in quiz
    assert QUESTION_ID in quiz


def test_quiz_options_are_selectable_controls(client: TestClient) -> None:
    quiz = region(client.get(LESSON_URL).text, "data-quiz-form", size=2000)

    assert 'type="radio"' in quiz
    assert f'value="{WRONG_OPTION_ID}"' in quiz
    assert f'value="{CORRECT_OPTION_ID}"' in quiz


def test_quiz_never_ships_the_answer_key(client: TestClient) -> None:
    html = client.get(LESSON_URL).text

    assert QUIZ_EXPLANATION_MARKER not in html
    assert "data-answer" not in html
    assert 'name="answer"' not in html


def test_quiz_still_ships_no_answer_key_after_an_attempt(
    client: TestClient,
) -> None:
    """A page that starts showing explanations once an attempt exists is a
    page that shows them to the next reader of the same lesson.
    """
    client.post(
        "/api/lessons/napi-poll/quiz",
        json={"responses": {QUESTION_ID: CORRECT_OPTION_ID}},
    )

    assert QUIZ_EXPLANATION_MARKER not in client.get(LESSON_URL).text


def test_lesson_page_shows_the_mastery_gate(client: TestClient) -> None:
    gate = region(client.get(LESSON_URL).text, "data-mastery-gate", size=800)

    assert "80" in gate
    assert "2" in gate


def test_lesson_page_reports_attempts_and_the_best_score(
    client: TestClient,
) -> None:
    client.post(
        "/api/lessons/napi-poll/quiz",
        json={"responses": {QUESTION_ID: WRONG_OPTION_ID}},
    )
    client.post(
        "/api/lessons/napi-poll/quiz",
        json={"responses": {QUESTION_ID: CORRECT_OPTION_ID}},
    )

    gate = region(client.get(LESSON_URL).text, "data-mastery-gate", size=800)

    assert "100" in gate


def test_locked_lesson_says_so_and_names_its_prerequisite(
    client: TestClient,
) -> None:
    html = client.get("/lessons/gro-coalescing").text

    assert 'data-locked="true"' in html
    prerequisites = region(html, "data-prerequisites")
    assert "The NAPI poll loop" in prerequisites
    assert "/lessons/napi-poll" in prerequisites


def test_a_locked_lesson_is_still_readable(client: TestClient) -> None:
    """The lock is advice, not a paywall: a learner skipping ahead should see
    that they are skipping ahead, not a blank page.
    """
    response = client.get("/lessons/gro-coalescing")

    assert response.status_code == 200
    assert "GRO merges segments" in response.text


def test_completing_the_prerequisite_unlocks_the_lesson(
    client: TestClient, store: LearningStore
) -> None:
    store.start_lesson("lesson-napi-poll")
    store.complete_lesson("lesson-napi-poll")

    html = client.get("/lessons/gro-coalescing").text

    assert 'data-locked="false"' in html


def test_lesson_page_points_at_the_next_lesson(client: TestClient) -> None:
    html = client.get(LESSON_URL).text

    following = region(html, "data-next-lesson")
    assert "/lessons/gro-coalescing" in following
    assert "GRO coalescing" in following


def test_lesson_page_offers_progress_controls(client: TestClient) -> None:
    html = client.get(LESSON_URL).text

    assert 'data-progress-action="start"' in html
    assert 'data-progress-action="complete"' in html
    assert 'data-lesson-id="lesson-napi-poll"' in html


def test_lesson_page_shows_the_current_progress_status(
    client: TestClient, store: LearningStore
) -> None:
    store.start_lesson("lesson-napi-poll")

    assert 'data-progress-status="in_progress"' in client.get(LESSON_URL).text


def test_lesson_page_offers_a_note_editor(client: TestClient) -> None:
    html = client.get(LESSON_URL).text

    assert "data-note-form" in html
    assert "<textarea" in html


def test_lesson_page_shows_an_existing_note(
    client: TestClient, store: LearningStore
) -> None:
    store.upsert_lesson_note("lesson-napi-poll", "Budget is per poll, not per packet.")

    note = region(client.get(LESSON_URL).text, "data-note-form", size=1200)

    assert "Budget is per poll, not per packet." in note


def test_lesson_page_shows_the_review_state(
    client: TestClient, store: LearningStore
) -> None:
    store.record_review("lesson-napi-poll", correct=True)

    review = region(client.get(LESSON_URL).text, "data-review", size=800)

    assert "1" in review


def test_draft_lesson_renders_without_a_quiz_or_lab(client: TestClient) -> None:
    response = client.get("/lessons/bql-draft")

    assert response.status_code == 200
    assert "Draft: byte queue limits" in response.text
    assert "data-quiz-form" not in response.text


def test_unknown_lesson_returns_a_not_found_page(client: TestClient) -> None:
    response = client.get("/lessons/no-such-lesson")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")
    assert "not found" in response.text.lower()


def test_lesson_page_never_triggers_indexing(
    client: TestClient, orchestrator: RecordingOrchestrator
) -> None:
    """Symbol links are ordinary hrefs. Following one is what asks for the
    index; rendering the lesson is not.
    """
    client.get(LESSON_URL)

    assert orchestrator.call_count == 0


@pytest.mark.parametrize(
    "url",
    ["/lessons/napi-poll", "/lessons/gro-coalescing", "/lessons/qdisc-dequeue"],
)
def test_every_published_lesson_page_renders(client: TestClient, url: str) -> None:
    assert client.get(url).status_code == 200
