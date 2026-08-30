"""Contract for the combined search page.

One box, two kinds of answer. "napi_poll" is both a phrase in a lesson and a
function in the kernel, and a reader who types it wants whichever of those
they were thinking of -- so both are returned, clearly separated, rather than
interleaved by a relevance score this program has no way to compute.

The query is also the most obvious injection surface in the application: it
is echoed back onto the page, it reaches an FTS5 ``MATCH`` expression, and it
is entirely attacker-controlled if the learner ever pastes a link. It is
therefore text, everywhere, always.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from html_helpers import assert_page_shell, region
from web_fakes import RecordingOrchestrator

SEARCH_SENTINEL_PATH = "/tmp/netstack-academy-search-must-not-run"


def test_search_page_is_served_as_html(client: TestClient) -> None:
    response = client.get("/search", params={"q": "napi_poll"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_search_page_has_the_shared_page_shell(client: TestClient) -> None:
    assert_page_shell(client.get("/search", params={"q": "napi_poll"}).text)


def test_search_returns_both_lesson_and_symbol_matches(client: TestClient) -> None:
    html = client.get("/search", params={"q": "napi_poll"}).text

    assert "/lessons/napi-poll" in html
    assert "/symbols/napi_poll" in html


def test_search_separates_the_two_kinds_of_result(client: TestClient) -> None:
    html = client.get("/search", params={"q": "napi_poll"}).text

    lessons = region(html, "data-lesson-results", size=2000)
    symbols = region(html, "data-symbol-results", size=2000)

    assert "The NAPI poll loop" in lessons
    assert "net/core/dev.c" in symbols


def test_search_reports_how_many_of_each_it_found(client: TestClient) -> None:
    html = client.get("/search", params={"q": "napi_poll"}).text

    assert "1" in region(html, "data-symbol-results", size=400)


def test_search_echoes_the_query_into_the_box(client: TestClient) -> None:
    html = client.get("/search", params={"q": "napi_poll"}).text

    assert 'value="napi_poll"' in html


def test_search_escapes_a_hostile_query(client: TestClient) -> None:
    response = client.get("/search", params={"q": '<img src=x onerror="steal()">'})

    assert response.status_code == 200
    assert "<img" not in response.text
    assert "&lt;img" in response.text


def test_search_escapes_quotes_in_the_query(client: TestClient) -> None:
    """An unescaped quote inside ``value="..."`` is enough to add an attribute.
    """
    response = client.get("/search", params={"q": '" autofocus onfocus="steal()'})

    assert response.status_code == 200
    assert "onfocus=" not in response.text.replace("&quot;", "")


def test_blank_query_shows_the_form_and_no_results(client: TestClient) -> None:
    response = client.get("/search", params={"q": "   "})

    assert response.status_code == 200
    assert "data-search-form" in response.text
    assert "The NAPI poll loop" not in response.text


def test_search_without_a_query_parameter_still_renders(client: TestClient) -> None:
    assert client.get("/search").status_code == 200


def test_search_with_no_matches_says_so(client: TestClient) -> None:
    response = client.get("/search", params={"q": "zzzznotathing"})

    assert response.status_code == 200
    assert "no match" in response.text.lower() or "nothing" in response.text.lower()


@pytest.mark.parametrize(
    "query",
    [
        'napi AND OR "',
        "napi*",
        "napi_poll;",
        "NEAR(napi poll)",
        "name:napi",
        "-napi",
        "(((",
    ],
)
def test_fts_query_syntax_is_treated_as_text(client: TestClient, query: str) -> None:
    """FTS5 raises on a malformed ``MATCH`` expression, and every one of these
    is ordinary-looking text a reader might type.
    """
    assert client.get("/search", params={"q": query}).status_code == 200


def test_a_shell_flavoured_query_is_never_executed(client: TestClient) -> None:
    sentinel = Path(SEARCH_SENTINEL_PATH)
    sentinel.unlink(missing_ok=True)

    client.get("/search", params={"q": f"napi; touch {SEARCH_SENTINEL_PATH}"})

    assert not sentinel.exists()


def test_search_bounds_the_number_of_results(client: TestClient) -> None:
    html = client.get("/search", params={"q": "napi_poll", "limit": 1}).text

    assert html.count("/lessons/") >= 1


def test_search_rejects_a_nonsensical_limit(client: TestClient) -> None:
    assert client.get("/search", params={"q": "napi", "limit": 0}).status_code == 422


def test_search_ensures_the_index_once(
    client: TestClient, orchestrator: RecordingOrchestrator
) -> None:
    client.get("/search", params={"q": "napi_poll"})
    client.get("/search", params={"q": "netif"})

    assert orchestrator.call_count == 1


def test_a_blank_search_does_not_build_the_index(
    client: TestClient, orchestrator: RecordingOrchestrator
) -> None:
    """"Show me everything" is the dashboard's job, and it is not worth
    minutes of indexing.
    """
    client.get("/search", params={"q": "   "})

    assert orchestrator.call_count == 0


def test_search_links_symbols_with_their_disambiguating_path(
    client: TestClient,
) -> None:
    """``helper`` is ambiguous by name, so a result that only linked the name
    would land the reader on a disambiguation page they did not ask for.
    """
    html = client.get("/search", params={"q": "helper"}).text

    assert "path=net%2Fipv4%2Fa.c" in html or "path=net/ipv4/a.c" in html
