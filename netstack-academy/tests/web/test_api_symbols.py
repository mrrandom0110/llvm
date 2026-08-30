"""Contract for the symbol JSON API: search, card, and call graph.

These are the endpoints the page's JavaScript calls, so their shapes are as
much a part of the contract as the rendered HTML. Two things they must never
do: guess which of several same-named ``static`` functions was meant, and
hand out a deep link that the safe builder refused.

``IndexService`` is the only way in. The web layer does not reach past it to
the SQLite storage, which is why resolving an incoming edge's caller needs
``IndexService.symbol_by_id`` rather than a query of its own.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from netstack_academy.indexing.service import IndexService, SymbolNotFoundError
from netstack_academy.web.app import MAX_SEARCH_LIMIT

from index_fixtures import (
    CALL_SITE_COLUMN,
    CALL_SITE_LINE,
    DEV_C,
    IPV4_C,
    IPV6_C,
    NAPI_POLL_LINE,
    REFERENCE_SITE_LINE,
)
from web_fakes import RecordingOrchestrator


def _deep_link(kernel_repo: Path, relative_path: str, line: int, column: int = 1) -> str:
    absolute = (kernel_repo / relative_path).resolve().as_posix()
    return f"cursor://vscode-remote/wsl+Ubuntu{absolute}:{line}:{column}"


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------


def test_symbol_search_returns_matches(client: TestClient) -> None:
    payload = client.get("/api/symbols", params={"q": "napi_poll"}).json()

    assert payload["query"] == "napi_poll"
    assert [symbol["name"] for symbol in payload["symbols"]] == ["napi_poll"]
    assert payload["count"] == 1


def test_symbol_search_results_carry_their_source_location(
    client: TestClient,
) -> None:
    symbol = client.get("/api/symbols", params={"q": "napi_poll"}).json()["symbols"][0]

    assert symbol["relative_path"] == DEV_C
    assert symbol["line"] == NAPI_POLL_LINE
    assert symbol["kind"] == "function"
    assert symbol["url"].startswith("/symbols/napi_poll")


def test_symbol_search_reports_both_static_definitions(client: TestClient) -> None:
    payload = client.get("/api/symbols", params={"q": "helper"}).json()

    paths = sorted(symbol["relative_path"] for symbol in payload["symbols"])
    assert paths == sorted([IPV4_C, IPV6_C])


def test_symbol_search_results_are_individually_addressable(
    client: TestClient,
) -> None:
    payload = client.get("/api/symbols", params={"q": "helper"}).json()

    for symbol in payload["symbols"]:
        assert "path=" in symbol["url"]


def test_blank_symbol_search_returns_nothing(client: TestClient) -> None:
    payload = client.get("/api/symbols", params={"q": "   "}).json()

    assert payload["symbols"] == []
    assert payload["count"] == 0


def test_blank_symbol_search_does_not_build_the_index(
    client: TestClient, orchestrator: RecordingOrchestrator
) -> None:
    client.get("/api/symbols", params={"q": ""})

    assert orchestrator.call_count == 0


def test_symbol_search_honours_the_limit(client: TestClient) -> None:
    payload = client.get("/api/symbols", params={"q": "helper", "limit": 1}).json()

    assert len(payload["symbols"]) == 1


def test_symbol_search_accepts_the_maximum_limit(client: TestClient) -> None:
    response = client.get(
        "/api/symbols", params={"q": "helper", "limit": MAX_SEARCH_LIMIT}
    )

    assert response.status_code == 200


@pytest.mark.parametrize("limit", [0, -1, "abc"])
def test_symbol_search_rejects_a_nonsensical_limit(
    client: TestClient, limit: object
) -> None:
    response = client.get("/api/symbols", params={"q": "helper", "limit": limit})

    assert response.status_code == 422


def test_symbol_search_rejects_an_unbounded_limit(client: TestClient) -> None:
    """An unbounded limit is a way to ask one HTTP request to serialize a
    kernel-sized symbol table.
    """
    response = client.get(
        "/api/symbols", params={"q": "helper", "limit": MAX_SEARCH_LIMIT + 1}
    )

    assert response.status_code == 422


def test_symbol_search_ensures_the_index_once(
    client: TestClient, orchestrator: RecordingOrchestrator
) -> None:
    client.get("/api/symbols", params={"q": "napi"})
    client.get("/api/symbols", params={"q": "helper"})

    assert orchestrator.forces == [False]


# ----------------------------------------------------------------------
# Card
# ----------------------------------------------------------------------


def test_symbol_card_returns_the_definition(
    client: TestClient, kernel_head: str
) -> None:
    payload = client.get("/api/symbols/napi_poll").json()

    symbol = payload["symbol"]
    assert symbol["name"] == "napi_poll"
    assert symbol["relative_path"] == DEV_C
    assert symbol["line"] == NAPI_POLL_LINE
    assert symbol["is_static"] is False
    assert symbol["commit_hash"] == kernel_head


def test_symbol_card_includes_the_editor_deep_link(
    client: TestClient, kernel_repo: Path
) -> None:
    payload = client.get("/api/symbols/napi_poll").json()

    assert payload["deep_link"] == _deep_link(kernel_repo, DEV_C, NAPI_POLL_LINE)
    assert payload["deep_link_reason"] is None


def test_symbol_card_counts_its_edges(client: TestClient) -> None:
    counts = client.get("/api/symbols/napi_poll").json()["counts"]

    assert counts["outgoing"] == 1
    assert counts["incoming"] == 1
    assert counts["references"] == 1


def test_symbol_card_withholds_a_deep_link_for_a_missing_file(
    client: TestClient,
) -> None:
    payload = client.get("/api/symbols/gone_symbol").json()

    assert payload["deep_link"] is None
    assert payload["deep_link_reason"]


def test_symbol_card_withholds_a_deep_link_for_an_escaping_path(
    client: TestClient,
) -> None:
    payload = client.get("/api/symbols/escaped_symbol").json()

    assert payload["deep_link"] is None
    assert payload["deep_link_reason"]


def test_symbol_card_reports_ambiguity_with_its_candidates(
    client: TestClient,
) -> None:
    response = client.get("/api/symbols/helper")

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "symbol_ambiguous"
    candidates = error["details"]["candidates"]
    assert sorted(candidate["relative_path"] for candidate in candidates) == sorted(
        [IPV4_C, IPV6_C]
    )
    assert all(candidate["line"] for candidate in candidates)


def test_symbol_card_resolves_an_ambiguous_name_with_a_path(
    client: TestClient,
) -> None:
    payload = client.get("/api/symbols/helper", params={"path": IPV6_C}).json()

    assert payload["symbol"]["relative_path"] == IPV6_C
    assert payload["symbol"]["is_static"] is True


def test_symbol_card_reports_an_unknown_name(client: TestClient) -> None:
    response = client.get("/api/symbols/no_such_symbol")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "symbol_not_found"


def test_symbol_card_refuses_an_unsafe_path(client: TestClient) -> None:
    response = client.get("/api/symbols/helper", params={"path": "../../etc/passwd"})

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "unsafe_path"
    assert "passwd" not in error["message"]


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../secrets", "net/../../escape.c", "net//dev.c", "..\\secrets"],
)
def test_symbol_card_refuses_every_unsafe_path_shape(
    client: TestClient, path: str
) -> None:
    response = client.get("/api/symbols/helper", params={"path": path})

    assert response.status_code == 400


def test_symbol_card_includes_the_learners_note(
    client: TestClient, store
) -> None:
    store.upsert_symbol_note("napi_poll", "Budget is per poll.", relative_path=DEV_C)

    payload = client.get("/api/symbols/napi_poll").json()

    assert payload["note"] == "Budget is per poll."


# ----------------------------------------------------------------------
# Graph
# ----------------------------------------------------------------------


def test_graph_lists_outgoing_calls_with_their_provenance(
    client: TestClient,
) -> None:
    payload = client.get("/api/symbols/napi_poll/graph").json()

    outgoing = payload["outgoing"]
    assert [edge["name"] for edge in outgoing] == ["netif_receive_skb"]
    assert outgoing[0]["provenance"] == "semantic"
    assert outgoing[0]["confidence"] == "high"
    assert outgoing[0]["edge_type"] == "call"


def test_graph_reports_the_call_site_of_a_semantic_edge(
    client: TestClient, kernel_repo: Path
) -> None:
    edge = client.get("/api/symbols/napi_poll/graph").json()["outgoing"][0]

    assert edge["site"]["relative_path"] == DEV_C
    assert edge["site"]["line"] == CALL_SITE_LINE
    assert edge["site"]["column"] == CALL_SITE_COLUMN
    assert edge["site_deep_link"] == _deep_link(
        kernel_repo, DEV_C, CALL_SITE_LINE, CALL_SITE_COLUMN
    )


def test_graph_names_the_caller_of_an_incoming_edge(client: TestClient) -> None:
    """The stored edge only knows the caller's id. Resolving it is what makes
    an incoming call worth showing.
    """
    incoming = client.get("/api/symbols/napi_poll/graph").json()["incoming"]

    assert [edge["name"] for edge in incoming] == ["netif_receive_skb"]
    assert incoming[0]["symbol"]["relative_path"] == DEV_C


def test_graph_marks_a_heuristic_edge_as_low_confidence(
    client: TestClient,
) -> None:
    """A regex over the source cannot tell two ``static`` functions of the
    same name apart, so an edge it produced is a suggestion.
    """
    incoming = client.get("/api/symbols/napi_poll/graph").json()["incoming"]

    assert incoming[0]["provenance"] == "heuristic"
    assert incoming[0]["confidence"] == "low"
    assert incoming[0]["site"] is None
    assert incoming[0]["site_deep_link"] is None


def test_graph_keeps_references_separate_from_calls(client: TestClient) -> None:
    payload = client.get("/api/symbols/napi_poll/graph").json()

    references = payload["references"]
    assert [edge["edge_type"] for edge in references] == ["reference"]
    assert references[0]["site"]["relative_path"] == IPV4_C
    assert references[0]["site"]["line"] == REFERENCE_SITE_LINE
    assert all(edge["edge_type"] == "call" for edge in payload["outgoing"])
    assert all(edge["edge_type"] == "call" for edge in payload["incoming"])


def test_graph_counts_each_kind_of_edge(client: TestClient) -> None:
    counts = client.get("/api/symbols/napi_poll/graph").json()["counts"]

    assert counts == {"incoming": 1, "outgoing": 1, "references": 1}


def test_graph_names_the_symbol_it_describes(client: TestClient) -> None:
    payload = client.get("/api/symbols/napi_poll/graph").json()

    assert payload["symbol"]["name"] == "napi_poll"


def test_graph_of_an_unknown_symbol_is_not_found(client: TestClient) -> None:
    response = client.get("/api/symbols/no_such_symbol/graph")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "symbol_not_found"


def test_graph_of_an_ambiguous_symbol_asks_for_a_path(client: TestClient) -> None:
    response = client.get("/api/symbols/helper/graph")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "symbol_ambiguous"


def test_graph_of_a_symbol_with_no_edges_is_empty_not_missing(
    client: TestClient,
) -> None:
    payload = client.get("/api/symbols/gone_symbol/graph").json()

    assert payload["incoming"] == []
    assert payload["outgoing"] == []
    assert payload["references"] == []


# ----------------------------------------------------------------------
# The lookup the graph needs from IndexService
# ----------------------------------------------------------------------


def test_index_service_resolves_a_symbol_by_id(
    kernel_repo: Path, index_storage, indexed_generation, orchestrator
) -> None:
    service = IndexService(kernel_repo, index_storage, orchestrator)
    found = service.find_symbol("napi_poll")

    assert service.symbol_by_id(found.id) == found


def test_index_service_reports_an_unknown_symbol_id(
    kernel_repo: Path, index_storage, indexed_generation, orchestrator
) -> None:
    service = IndexService(kernel_repo, index_storage, orchestrator)

    with pytest.raises(SymbolNotFoundError):
        service.symbol_by_id(999_999)
