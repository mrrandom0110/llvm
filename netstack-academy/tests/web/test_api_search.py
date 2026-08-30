"""Contract for the combined search JSON API.

The page's search box calls this while the reader types, so it answers with
both halves at once -- lesson prose and kernel symbols -- and says which
fields of a lesson matched, because "why is this result here" is otherwise
unanswerable from a title alone.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from netstack_academy.web.app import MAX_SEARCH_LIMIT

from index_fixtures import DEV_C
from web_fakes import RecordingOrchestrator


def test_search_returns_both_halves(client: TestClient) -> None:
    payload = client.get("/api/search", params={"q": "napi_poll"}).json()

    assert payload["query"] == "napi_poll"
    assert [hit["slug"] for hit in payload["lessons"]] == ["napi-poll"]
    assert [hit["name"] for hit in payload["symbols"]] == ["napi_poll"]


def test_search_counts_each_half(client: TestClient) -> None:
    payload = client.get("/api/search", params={"q": "napi_poll"}).json()

    assert payload["counts"] == {
        "lessons": len(payload["lessons"]),
        "symbols": len(payload["symbols"]),
    }


def test_lesson_hits_carry_everything_a_result_row_needs(
    client: TestClient,
) -> None:
    hit = client.get("/api/search", params={"q": "napi_poll"}).json()["lessons"][0]

    assert hit["title"] == "The NAPI poll loop"
    assert hit["module_slug"] == "rx-path"
    assert hit["module_title"] == "Receive path"
    assert hit["url"] == "/lessons/napi-poll"
    assert hit["summary"]


def test_lesson_hits_say_which_fields_matched(client: TestClient) -> None:
    hit = client.get("/api/search", params={"q": "NAPI poll loop"}).json()["lessons"][0]

    assert "title" in hit["matched_fields"]


def test_lesson_search_is_case_insensitive(client: TestClient) -> None:
    payload = client.get("/api/search", params={"q": "NAPI POLL LOOP"}).json()

    assert [hit["slug"] for hit in payload["lessons"]] == ["napi-poll"]


def test_lesson_hits_are_in_curriculum_order(client: TestClient) -> None:
    """Module order, then lesson order: the same sequence the dashboard shows,
    rather than a relevance score this program has no corpus to compute.
    """
    payload = client.get("/api/search", params={"q": "sk_buff"}).json()

    assert [hit["slug"] for hit in payload["lessons"]] == [
        "napi-poll",
        "qdisc-dequeue",
    ]


def test_symbol_hits_carry_their_location_and_url(client: TestClient) -> None:
    hit = client.get("/api/search", params={"q": "napi_poll"}).json()["symbols"][0]

    assert hit["relative_path"] == DEV_C
    assert hit["url"].startswith("/symbols/napi_poll")


def test_a_blank_query_returns_both_halves_empty(client: TestClient) -> None:
    payload = client.get("/api/search", params={"q": "   "}).json()

    assert payload["lessons"] == []
    assert payload["symbols"] == []


def test_a_blank_query_never_builds_the_index(
    client: TestClient, orchestrator: RecordingOrchestrator
) -> None:
    client.get("/api/search", params={"q": ""})

    assert orchestrator.call_count == 0


def test_a_query_matching_nothing_is_not_an_error(client: TestClient) -> None:
    response = client.get("/api/search", params={"q": "zzzznotathing"})

    assert response.status_code == 200
    assert response.json()["counts"] == {"lessons": 0, "symbols": 0}


def test_search_bounds_the_results(client: TestClient) -> None:
    payload = client.get("/api/search", params={"q": "sk_buff", "limit": 1}).json()

    assert len(payload["lessons"]) == 1


@pytest.mark.parametrize("limit", [0, -5, MAX_SEARCH_LIMIT + 1, "many"])
def test_search_rejects_a_nonsensical_limit(
    client: TestClient, limit: object
) -> None:
    response = client.get("/api/search", params={"q": "sk_buff", "limit": limit})

    assert response.status_code == 422


@pytest.mark.parametrize(
    "query", ['napi AND OR "', "napi*", "NEAR(napi)", "-napi", "((("]
)
def test_fts_syntax_in_a_query_is_treated_as_text(
    client: TestClient, query: str
) -> None:
    assert client.get("/api/search", params={"q": query}).status_code == 200


def test_search_ensures_the_index_once(
    client: TestClient, orchestrator: RecordingOrchestrator
) -> None:
    client.get("/api/search", params={"q": "napi_poll"})
    client.get("/api/search", params={"q": "helper"})

    assert orchestrator.forces == [False]
