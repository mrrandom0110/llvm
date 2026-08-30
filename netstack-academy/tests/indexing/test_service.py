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
        # Tracks how many of those calls explicitly requested a forced
        # rerun, so tests can distinguish ``IndexService.ensure_index()``
        # (must never force) from ``IndexService.force_reindex()`` (must
        # always force) at the orchestrator boundary.
        self.force_call_count = 0

    def ensure_index(self, *, force: bool = False) -> IndexRunResult:
        self.call_count += 1
        if force:
            self.force_call_count += 1
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


def test_ensure_index_never_requests_a_forced_run(
    populated_service: tuple[IndexService, _StubOrchestrator],
) -> None:
    """The plain, cheap trigger a caller polls on every request must never
    silently force a rerun -- that would defeat persisted same-HEAD reuse.
    """
    service, orchestrator = populated_service

    service.ensure_index()

    assert orchestrator.force_call_count == 0


def test_force_reindex_requests_a_forced_run_from_the_orchestrator(
    populated_service: tuple[IndexService, _StubOrchestrator],
) -> None:
    """``force_reindex`` is the service's distinct, explicit trigger for
    testing provider failures or a manual refresh -- it must ask the
    orchestrator for a forced rerun rather than reusing the persisted head,
    and it must be a separate method from ``ensure_index`` rather than a
    boolean the caller has to remember to pass.
    """
    service, orchestrator = populated_service

    result = service.force_reindex()

    assert orchestrator.call_count == 1
    assert orchestrator.force_call_count == 1
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


@pytest.fixture
def service_with_call_and_reference_edges(tmp_path: Path, git_repository: Path):
    """Two symbols, each the source of one ``call`` edge and one
    ``reference`` edge to/from the other -- deliberately in both directions,
    so a leak in either ``outgoing_edges`` or ``incoming_edges`` would show
    up regardless of which endpoint a caller queries from.
    """
    storage = IndexStorage.open(tmp_path / "index.sqlite3")
    storage.replace_symbols_and_edges(
        "deadbeef",
        [
            SymbolInput(
                name="tcp_helper",
                kind="function",
                relative_path="net/ipv4/a.c",
                line=1,
            ),
            SymbolInput(
                name="tcp_process",
                kind="function",
                relative_path="net/ipv4/b.c",
                line=1,
            ),
        ],
        [
            EdgeInput(
                source_index=1,
                target_index=0,
                target_name="tcp_helper",
                edge_type="call",
                provenance="heuristic",
            ),
            EdgeInput(
                source_index=1,
                target_index=0,
                target_name="tcp_helper",
                edge_type="reference",
                provenance="semantic",
                site_relative_path="net/ipv4/b.c",
                site_line=3,
                site_column=12,
            ),
            EdgeInput(
                source_index=0,
                target_index=1,
                target_name="tcp_process",
                edge_type="call",
                provenance="heuristic",
            ),
            EdgeInput(
                source_index=0,
                target_index=1,
                target_name="tcp_process",
                edge_type="reference",
                provenance="semantic",
                site_relative_path="net/ipv4/a.c",
                site_line=7,
                site_column=4,
            ),
        ],
    )
    orchestrator = _StubOrchestrator(
        IndexRunResult(
            status="reindexed",
            head="deadbeef",
            symbol_count=2,
            edge_count=4,
            provider_diagnostics=(),
            reason=None,
        )
    )
    service = IndexService(git_repository, storage, orchestrator)
    try:
        yield service
    finally:
        storage.close()


def test_outgoing_edges_excludes_reference_type_edges(
    service_with_call_and_reference_edges: IndexService,
) -> None:
    """A ``reference`` edge anchored on the queried symbol must never appear
    in ``outgoing_edges``: that method is documented (and, per
    :meth:`IndexService.references`'s own docstring, contractually reserved)
    to report only the call graph.
    """
    service = service_with_call_and_reference_edges
    helper = service.find_symbol("tcp_helper")

    edges = service.outgoing_edges(helper.id)

    assert len(edges) == 1
    assert edges[0].edge_type == "call"
    assert edges[0].target_name == "tcp_process"


def test_incoming_edges_excludes_reference_type_edges(
    service_with_call_and_reference_edges: IndexService,
) -> None:
    """Same contract as ``outgoing_edges``, from the callee's side."""
    service = service_with_call_and_reference_edges
    helper = service.find_symbol("tcp_helper")

    edges = service.incoming_edges(helper.id)

    assert len(edges) == 1
    assert edges[0].edge_type == "call"
    assert edges[0].target_name == "tcp_helper"


def test_references_preserves_site_fields_alongside_call_edges(
    service_with_call_and_reference_edges: IndexService,
) -> None:
    """Regression guard: ``references()`` already filters to
    ``edge_type == "reference"`` today (unlike ``outgoing_edges``/
    ``incoming_edges``), and must keep doing so -- with the call/reference
    site preserved -- once the sibling methods are fixed to stop leaking
    reference edges themselves.
    """
    service = service_with_call_and_reference_edges
    helper = service.find_symbol("tcp_helper")

    references = service.references(helper.id)

    assert len(references) == 2
    assert all(reference.edge_type == "reference" for reference in references)

    by_target = {reference.target_name: reference for reference in references}
    assert (
        by_target["tcp_process"].site_relative_path,
        by_target["tcp_process"].site_line,
        by_target["tcp_process"].site_column,
    ) == ("net/ipv4/a.c", 7, 4)
    assert (
        by_target["tcp_helper"].site_relative_path,
        by_target["tcp_helper"].site_line,
        by_target["tcp_helper"].site_column,
    ) == ("net/ipv4/b.c", 3, 12)
