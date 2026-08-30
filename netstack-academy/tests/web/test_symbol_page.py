"""Contract for the symbol card page and its call graph.

A symbol card is where the course stops being prose and starts being the
kernel: it names a definition, says which file and line it lives at, opens it
in the editor at that line, and shows what calls it and what it calls.

Everything on it can be wrong in a way that wastes a reader's afternoon, so
the page is explicit about provenance instead of presenting one confident
graph. A ``semantic`` edge came from ``clangd`` and knows where the call
happens; a ``heuristic`` edge came from a regex over the source and may be a
different ``static`` function of the same name. Those are shown as what they
are.

The graph is also drawn twice on purpose: once visually, and once as a plain
list. The list is not a fallback for old browsers -- it is how the graph is
read by a screen reader, by a keyboard, and by anyone whose window is 400
pixels wide.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from netstack_academy.learning.store import LearningStore

from html_helpers import assert_page_shell, region
from index_fixtures import (
    CALL_SITE_COLUMN,
    CALL_SITE_LINE,
    DEV_C,
    IPV4_C,
    NAPI_POLL_LINE,
)
from web_fakes import RecordingOrchestrator

SYMBOL_URL = "/symbols/napi_poll"


def _expected_deep_link(
    kernel_repo: Path,
    relative_path: str,
    line: int,
    column: int = 1,
    *,
    scheme: str = "cursor",
    distro: str = "Ubuntu",
) -> str:
    absolute = (kernel_repo / relative_path).resolve().as_posix()
    return f"{scheme}://vscode-remote/wsl+{distro}{absolute}:{line}:{column}"


def test_symbol_page_is_served_as_html(client: TestClient) -> None:
    response = client.get(SYMBOL_URL)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


def test_symbol_page_has_the_shared_page_shell(client: TestClient) -> None:
    assert_page_shell(client.get(SYMBOL_URL).text)


def test_symbol_page_shows_the_source_metadata(client: TestClient) -> None:
    html = client.get(SYMBOL_URL).text

    assert "napi_poll" in html
    assert DEV_C in html
    assert str(NAPI_POLL_LINE) in html
    assert "function" in html


def test_symbol_page_shows_the_signature(client: TestClient) -> None:
    html = client.get(SYMBOL_URL).text

    assert "struct napi_struct *n" in html


def test_symbol_page_names_the_commit_the_index_was_built_from(
    client: TestClient, kernel_head: str
) -> None:
    """A line number is only meaningful together with the commit it was read
    at, and the index keeps that commit precisely so a card can say it.
    """
    html = client.get(SYMBOL_URL).text

    assert kernel_head[:12] in html


def test_symbol_page_deep_links_into_the_editor_at_the_definition_line(
    client: TestClient, kernel_repo: Path
) -> None:
    html = client.get(SYMBOL_URL).text

    assert _expected_deep_link(kernel_repo, DEV_C, NAPI_POLL_LINE) in html


def test_symbol_page_honours_the_configured_editor_scheme(
    make_client, kernel_repo: Path
) -> None:
    vscode = make_client(editor_scheme="vscode")

    html = vscode.get(SYMBOL_URL).text

    assert _expected_deep_link(kernel_repo, DEV_C, NAPI_POLL_LINE, scheme="vscode") in html


def test_symbol_page_encodes_a_wsl_distro_name_containing_spaces(
    make_client, kernel_repo: Path
) -> None:
    """The deep link is a URL. A distribution called ``Ubuntu 22.04`` is a
    perfectly ordinary WSL name and must not break it.
    """
    spaced = make_client(wsl_distro="Ubuntu 22.04")

    html = spaced.get(SYMBOL_URL).text

    assert "wsl+Ubuntu%2022.04" in html


def test_ambiguous_symbol_offers_the_candidates(client: TestClient) -> None:
    """``helper`` is ``static`` in two files. Guessing one would silently send
    a reader to the wrong definition, so the page asks instead.
    """
    response = client.get("/symbols/helper")

    assert response.status_code == 409
    assert response.headers["content-type"].startswith("text/html")
    assert IPV4_C in response.text
    assert "net/ipv6/b.c" in response.text


def test_ambiguous_candidates_link_to_a_disambiguated_card(
    client: TestClient,
) -> None:
    html = client.get("/symbols/helper").text

    assert "path=net%2Fipv4%2Fa.c" in html or f"path={IPV4_C}" in html


def test_disambiguated_symbol_resolves_to_the_requested_file(
    client: TestClient,
) -> None:
    response = client.get("/symbols/helper", params={"path": IPV4_C})

    assert response.status_code == 200
    assert IPV4_C in response.text
    assert "net/ipv6/b.c" not in response.text


def test_unknown_symbol_returns_a_not_found_page(client: TestClient) -> None:
    response = client.get("/symbols/definitely_not_a_symbol")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")
    assert "/search" in response.text


def test_unsafe_disambiguation_path_is_refused(client: TestClient) -> None:
    response = client.get("/symbols/helper", params={"path": "../../etc/passwd"})

    assert response.status_code == 400
    assert "passwd" not in response.text


def test_symbol_in_a_file_that_no_longer_exists_still_renders(
    client: TestClient,
) -> None:
    """An index built before a ``git pull`` names files that are gone. The
    card says so rather than offering a link that opens nothing.
    """
    response = client.get("/symbols/gone_symbol")

    assert response.status_code == 200
    assert "net/core/removed.c" in response.text
    assert "cursor://" not in response.text


def test_a_corrupt_stored_path_never_becomes_a_link(client: TestClient) -> None:
    """Nothing production writes a symbol whose path escapes the repository,
    but the deep-link builder is the last check before a link points outside
    the kernel tree, and the page must respect its refusal.
    """
    response = client.get("/symbols/escaped_symbol")

    assert response.status_code == 200
    assert "cursor://" not in response.text


def test_symbol_page_lists_outgoing_calls(client: TestClient) -> None:
    graph = region(client.get(SYMBOL_URL).text, "data-graph-fallback", size=3000)

    assert "netif_receive_skb" in graph


def test_symbol_page_lists_incoming_calls_by_their_caller(
    client: TestClient,
) -> None:
    """An incoming edge stores only the caller's id; a page that cannot
    resolve it shows an arrow from nowhere.
    """
    graph = region(client.get(SYMBOL_URL).text, "data-graph-fallback", size=3000)

    assert "netif_receive_skb" in graph
    assert "Incoming" in graph or "incoming" in graph


def test_symbol_page_separates_calls_from_references(client: TestClient) -> None:
    html = client.get(SYMBOL_URL).text

    assert "Reference" in html or "reference" in html
    assert IPV4_C in html


def test_symbol_page_labels_edge_provenance(client: TestClient) -> None:
    html = client.get(SYMBOL_URL).text

    assert "semantic" in html.lower()
    assert "heuristic" in html.lower()


def test_symbol_page_labels_edge_confidence(client: TestClient) -> None:
    """"Provenance" is jargon. The page also has to say what it means for how
    much a reader should trust the edge.
    """
    html = client.get(SYMBOL_URL).text.lower()

    assert "high" in html
    assert "low" in html


def test_symbol_page_shows_the_call_site_of_a_semantic_edge(
    client: TestClient, kernel_repo: Path
) -> None:
    html = client.get(SYMBOL_URL).text

    assert str(CALL_SITE_LINE) in html
    assert (
        _expected_deep_link(kernel_repo, DEV_C, CALL_SITE_LINE, CALL_SITE_COLUMN) in html
    )


def test_the_visual_graph_is_hidden_from_assistive_technology(
    client: TestClient,
) -> None:
    """The picture and the list say the same thing; announcing both is noise.
    """
    container = region(client.get(SYMBOL_URL).text, "data-symbol-graph", size=400)

    assert 'aria-hidden="true"' in container


def test_the_graph_has_a_list_fallback(client: TestClient) -> None:
    graph = region(client.get(SYMBOL_URL).text, "data-graph-fallback", size=3000)

    assert "<li" in graph


def test_symbol_page_offers_a_note_editor(client: TestClient) -> None:
    html = client.get(SYMBOL_URL).text

    assert "data-note-form" in html
    assert 'data-symbol-name="napi_poll"' in html


def test_symbol_page_shows_an_existing_symbol_note(
    client: TestClient, store: LearningStore
) -> None:
    store.upsert_symbol_note(
        "napi_poll", "Runs until the budget is spent.", relative_path=DEV_C
    )

    note = region(client.get(SYMBOL_URL).text, "data-note-form", size=1200)

    assert "Runs until the budget is spent." in note


def test_symbol_page_ensures_the_index_once(
    client: TestClient, orchestrator: RecordingOrchestrator
) -> None:
    """This is the request that genuinely needs symbols, so this is where the
    index gets built -- once, not on every page view.
    """
    client.get(SYMBOL_URL)
    client.get(SYMBOL_URL)

    assert orchestrator.call_count == 1
    assert orchestrator.forces == [False]
