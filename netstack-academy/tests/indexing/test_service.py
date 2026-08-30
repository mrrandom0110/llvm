from __future__ import annotations

from pathlib import Path

import pytest

from netstack_academy.indexing.models import EdgeInput, SymbolInput
from netstack_academy.indexing.orchestrator import IndexRunResult
from netstack_academy.indexing.service import (
    InvalidRepositoryPathError,
    IndexService,
    SymbolAmbiguousError,
    SymbolNotFoundError,
)
from netstack_academy.indexing.storage import IndexStorage


class _StubOrchestrator:
    def __init__(self, result: IndexRunResult) -> None:
        self.result = result
        self.call_count = 0

    def ensure_index(self) -> IndexRunResult:
        self.call_count += 1
        return self.result


@pytest.fixture
def populated_service(tmp_path: Path, git_repository: Path):
    storage = IndexStorage.open(tmp_path / "index.sqlite3")
    storage.replace_symbols_and_edges(
        "deadbeef",
        [
            SymbolInput(
                name="helper",
                kind="function",
                relative_path="net/ipv4/a.c",
                line=1,
                is_static=True,
            ),
            SymbolInput(
                name="helper",
                kind="function",
                relative_path="net/ipv6/b.c",
                line=20,
                is_static=True,
            ),
            SymbolInput(
                name="process",
                kind="function",
                relative_path="net/ipv4/a.c",
                line=6,
            ),
        ],
        [
            EdgeInput(
                source_index=2,
                target_index=0,
                target_name="helper",
                edge_type="call",
                provenance="heuristic",
            )
        ],
    )
    orchestrator = _StubOrchestrator(
        IndexRunResult(
            status="reindexed",
            head="deadbeef",
            symbol_count=3,
            edge_count=1,
            provider_diagnostics=(),
            reason=None,
        )
    )
    service = IndexService(git_repository, storage, orchestrator)
    try:
        yield service, orchestrator
    finally:
        storage.close()


def test_find_symbol_returns_unique_match_by_name(
    populated_service: tuple[IndexService, _StubOrchestrator],
) -> None:
    service, _ = populated_service

    symbol = service.find_symbol("process")

    assert symbol.name == "process"
    assert symbol.relative_path == "net/ipv4/a.c"


def test_find_symbol_raises_ambiguous_error_with_candidates(
    populated_service: tuple[IndexService, _StubOrchestrator],
) -> None:
    service, _ = populated_service

    with pytest.raises(SymbolAmbiguousError) as exc_info:
        service.find_symbol("helper")

    assert len(exc_info.value.candidates) == 2


def test_find_symbol_raises_not_found_for_unknown_name(
    populated_service: tuple[IndexService, _StubOrchestrator],
) -> None:
    service, _ = populated_service

    with pytest.raises(SymbolNotFoundError):
        service.find_symbol("does_not_exist")


def test_find_symbol_disambiguates_with_relative_path(
    populated_service: tuple[IndexService, _StubOrchestrator],
) -> None:
    service, _ = populated_service

    symbol = service.find_symbol("helper", relative_path="net/ipv6/b.c")

    assert symbol.relative_path == "net/ipv6/b.c"
    assert symbol.line == 20


def test_find_symbol_rejects_relative_path_with_parent_traversal(
    populated_service: tuple[IndexService, _StubOrchestrator],
) -> None:
    service, _ = populated_service

    with pytest.raises(InvalidRepositoryPathError):
        service.find_symbol("helper", relative_path="../../etc/passwd")


def test_find_symbol_rejects_absolute_path_outside_repo(
    populated_service: tuple[IndexService, _StubOrchestrator],
) -> None:
    service, _ = populated_service

    with pytest.raises(InvalidRepositoryPathError):
        service.find_symbol("helper", relative_path="/etc/passwd")


def test_find_symbol_treats_shell_metacharacters_as_plain_text(
    populated_service: tuple[IndexService, _StubOrchestrator],
) -> None:
    """The service must never interpolate untrusted input into a shell
    command; a name containing shell metacharacters is simply not found.
    """
    service, _ = populated_service

    with pytest.raises(SymbolNotFoundError):
        service.find_symbol("helper; rm -rf /")


def test_search_symbols_returns_matches_within_limit(
    populated_service: tuple[IndexService, _StubOrchestrator],
) -> None:
    service, _ = populated_service

    results = service.search_symbols("helper", limit=1)

    assert len(results) == 1


def test_get_status_reports_head_and_counts(
    populated_service: tuple[IndexService, _StubOrchestrator],
) -> None:
    service, _ = populated_service

    status = service.get_status()

    assert status.head == "deadbeef"
    assert status.symbol_count == 3
    assert status.edge_count == 1


def test_ensure_index_delegates_to_orchestrator(
    populated_service: tuple[IndexService, _StubOrchestrator],
) -> None:
    service, orchestrator = populated_service

    result = service.ensure_index()

    assert orchestrator.call_count == 1
    assert result is orchestrator.result


def test_outgoing_edges_returns_edge_views_with_provenance(
    populated_service: tuple[IndexService, _StubOrchestrator],
) -> None:
    service, _ = populated_service
    process_symbol = service.find_symbol("process")

    edges = service.outgoing_edges(process_symbol.id)

    assert len(edges) == 1
    assert edges[0].target_name == "helper"
    assert edges[0].provenance in {"heuristic", "semantic"}


def test_incoming_edges_returns_edge_views(
    populated_service: tuple[IndexService, _StubOrchestrator],
) -> None:
    service, _ = populated_service
    helper_symbol = service.find_symbol("helper", relative_path="net/ipv4/a.c")

    edges = service.incoming_edges(helper_symbol.id)

    assert len(edges) == 1
    assert edges[0].edge_type == "call"


def test_references_returns_empty_list_when_none_recorded(
    populated_service: tuple[IndexService, _StubOrchestrator],
) -> None:
    service, _ = populated_service
    process_symbol = service.find_symbol("process")

    references = service.references(process_symbol.id)

    assert references == []
