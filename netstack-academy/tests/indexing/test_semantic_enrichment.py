"""Contract for merging collectors and persisting semantic enrichment.

The orchestrator's existing tests only ever feed it *empty* ctags/fallback
results and a semantic provider that reports itself unavailable, so three
things are still unspecified:

- what the ctags/fallback merge actually persists when both collectors find
  something at the same location,
- that an *available* semantic provider is driven at all -- one
  ``prepareCallHierarchy`` / ``outgoingCalls`` / ``references`` round per
  indexed function, bounded by an explicit budget -- and that its results
  are merged, de-duplicated, and persisted with their call/reference sites,
- how a per-symbol provider timeout/error, provider unavailability, a
  commit change, and provider teardown are handled.

The provider double below returns the same typed outcome models
(``PrepareCallHierarchyOutcome``, ``OutgoingCallsOutcome``,
``ReferencesOutcome``) a real ``ClangdAdapter`` returns, keyed by the
1-based ``(relative_path, line)`` position it is asked about, and records
every call so the tests can prove the requests were actually made. The
sources under test are two small, real C files in a real temporary git
repository, scanned by the *real* fallback indexer -- only ctags and the
semantic provider are stubbed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from netstack_academy.indexing.ctags_parser import CtagsDefinition
from netstack_academy.indexing.ctags_runner import CtagsRunResult
from netstack_academy.indexing.fallback_indexer import index_fallback
from netstack_academy.indexing.orchestrator import IndexOrchestrator
from netstack_academy.indexing.semantic.models import (
    CallHierarchyItem,
    OutgoingCall,
    OutgoingCallsOutcome,
    PrepareCallHierarchyOutcome,
    ProviderCapabilities,
    ReferencesOutcome,
    SemanticLocation,
)
from netstack_academy.indexing.service import IndexService
from netstack_academy.indexing.storage import IndexStorage

TCP_INPUT = "net/ipv4/tcp_input.c"
TCP_UTIL = "net/ipv4/tcp_util.c"

#: ``scope`` is a ctags-only field: the regex fallback scanner always
#: reports ``None`` for it, so it proves which collector won a location.
TCP_PROCESS_DEFINITION = CtagsDefinition(
    name="tcp_process",
    kind="function",
    path=TCP_INPUT,
    line=1,
    signature="(int x)",
    scope="tcp_input",
    is_static=False,
)


class _RecordingSemanticProvider:
    """A ``SemanticProvider`` double returning pre-seeded typed outcomes.

    Deliberately has no ``close()``: provider teardown must not be required
    of every stub (see the lifecycle tests at the bottom of this module).
    """

    def __init__(
        self,
        *,
        available: bool = True,
        reason: str | None = None,
        prepare: dict[tuple[str, int], PrepareCallHierarchyOutcome] | None = None,
        outgoing: dict[tuple[str, int], OutgoingCallsOutcome] | None = None,
        references: dict[tuple[str, int], ReferencesOutcome] | None = None,
    ) -> None:
        self._available = available
        self._reason = reason
        self._prepare = dict(prepare or {})
        self._outgoing = dict(outgoing or {})
        self._references = dict(references or {})
        self.prepare_calls: list[tuple[str, int, int]] = []
        self.outgoing_items: list[CallHierarchyItem] = []
        self.reference_calls: list[tuple[str, int, int]] = []

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_name="clangd", available=self._available, reason=self._reason
        )

    def prepare_call_hierarchy(
        self, relative_path: str, *, line: int, column: int
    ) -> PrepareCallHierarchyOutcome:
        self.prepare_calls.append((relative_path, line, column))
        return self._prepare.get(
            (relative_path, line), PrepareCallHierarchyOutcome(status="ok")
        )

    def outgoing_calls(self, item: CallHierarchyItem) -> OutgoingCallsOutcome:
        self.outgoing_items.append(item)
        return self._outgoing.get(
            (item.relative_path, item.line), OutgoingCallsOutcome(status="ok")
        )

    def references(
        self, relative_path: str, *, line: int, column: int
    ) -> ReferencesOutcome:
        self.reference_calls.append((relative_path, line, column))
        return self._references.get(
            (relative_path, line), ReferencesOutcome(status="ok")
        )


class _ClosableSemanticProvider(_RecordingSemanticProvider):
    """The same double, plus the optional ``close()`` a real provider owns."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def semantic_repo(tmp_path: Path) -> Path:
    """A real two-file git repo where ``tcp_process`` calls ``tcp_helper``."""
    repo = tmp_path / "kernel"
    (repo / "net" / "ipv4").mkdir(parents=True)
    (repo / TCP_INPUT).write_text(
        "int tcp_process(int x)\n{\n    return tcp_helper(x);\n}\n",
        encoding="utf-8",
    )
    (repo / TCP_UTIL).write_text(
        "int tcp_helper(int x)\n{\n    return x + 1;\n}\n",
        encoding="utf-8",
    )
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Add tcp fixture")
    return repo


@pytest.fixture
def storage(tmp_path: Path):
    with IndexStorage.open(tmp_path / "index.sqlite3") as opened:
        yield opened


def _stub_ctags(*definitions: CtagsDefinition):
    def _run(*args: object, **kwargs: object) -> CtagsRunResult:
        return CtagsRunResult(
            status="ok", definitions=list(definitions), diagnostics=[]
        )

    return _run


def _item(name: str, relative_path: str, line: int, column: int) -> CallHierarchyItem:
    return CallHierarchyItem(
        name=name, relative_path=relative_path, line=line, column=column
    )


def _orchestrator(
    repo: Path,
    storage: IndexStorage,
    *,
    ctags_definitions: tuple[CtagsDefinition, ...] = (),
    semantic_provider: object | None = None,
    semantic_symbol_limit: int | None = None,
) -> IndexOrchestrator:
    return IndexOrchestrator(
        repo,
        storage,
        ctags_runner=_stub_ctags(*ctags_definitions),
        fallback_indexer=index_fallback,
        semantic_provider=semantic_provider,
        semantic_symbol_limit=semantic_symbol_limit,
    )


def _call_edges(storage: IndexStorage, symbol_id: int) -> list:
    return [
        edge
        for edge in storage.outgoing_edges(symbol_id)
        if edge.edge_type == "call"
    ]


# -- collector merge ---------------------------------------------------------


def test_ctags_definition_wins_over_the_fallback_symbol_at_the_same_location(
    semantic_repo: Path, storage: IndexStorage
) -> None:
    orchestrator = _orchestrator(
        semantic_repo,
        storage,
        ctags_definitions=(TCP_PROCESS_DEFINITION,),
    )

    result = orchestrator.ensure_index()

    assert result.status == "reindexed"
    # One row per location: the fallback's tcp_process is not a second symbol.
    assert storage.symbol_count() == 2

    processes = storage.find_symbols_by_name("tcp_process")
    assert len(processes) == 1
    assert processes[0].scope == "tcp_input"
    assert processes[0].signature == "(int x)"

    # The fallback-only definition is still indexed; ctags never saw it.
    helpers = storage.find_symbols_by_name("tcp_helper")
    assert len(helpers) == 1
    assert helpers[0].relative_path == TCP_UTIL
    assert helpers[0].scope is None


def test_fallback_call_edge_resolves_against_the_merged_symbol_batch(
    semantic_repo: Path, storage: IndexStorage
) -> None:
    orchestrator = _orchestrator(
        semantic_repo,
        storage,
        ctags_definitions=(TCP_PROCESS_DEFINITION,),
    )

    orchestrator.ensure_index()

    caller = storage.find_symbols_by_name("tcp_process")[0]
    callee = storage.find_symbols_by_name("tcp_helper")[0]
    edges = storage.outgoing_edges(caller.id)

    assert len(edges) == 1
    assert edges[0].target_symbol_id == callee.id
    assert edges[0].target_name == "tcp_helper"
    assert edges[0].edge_type == "call"
    assert edges[0].provenance == "heuristic"


# -- semantic enrichment ----------------------------------------------------


def test_orchestrator_drives_the_semantic_provider_for_indexed_functions(
    semantic_repo: Path, storage: IndexStorage
) -> None:
    provider = _RecordingSemanticProvider(
        prepare={
            (TCP_INPUT, 1): PrepareCallHierarchyOutcome(
                status="ok", items=(_item("tcp_process", TCP_INPUT, 1, 5),)
            ),
            (TCP_UTIL, 1): PrepareCallHierarchyOutcome(
                status="ok", items=(_item("tcp_helper", TCP_UTIL, 1, 5),)
            ),
        },
        outgoing={
            (TCP_INPUT, 1): OutgoingCallsOutcome(
                status="ok",
                calls=(
                    OutgoingCall(
                        target=_item("tcp_helper", TCP_UTIL, 1, 5),
                        call_sites=(SemanticLocation(TCP_INPUT, 3, 12),),
                    ),
                ),
            )
        },
        references={
            (TCP_UTIL, 1): ReferencesOutcome(
                status="ok", locations=(SemanticLocation(TCP_INPUT, 3, 12),)
            )
        },
    )
    orchestrator = _orchestrator(
        semantic_repo,
        storage,
        ctags_definitions=(TCP_PROCESS_DEFINITION,),
        semantic_provider=provider,
        semantic_symbol_limit=8,
    )

    result = orchestrator.ensure_index()

    assert result.status == "reindexed"
    assert sorted((path, line) for path, line, _ in provider.prepare_calls) == [
        (TCP_INPUT, 1),
        (TCP_UTIL, 1),
    ]
    assert all(column >= 1 for _, _, column in provider.prepare_calls)
    assert sorted(
        (item.relative_path, item.line) for item in provider.outgoing_items
    ) == [(TCP_INPUT, 1), (TCP_UTIL, 1)]
    assert sorted((path, line) for path, line, _ in provider.reference_calls) == [
        (TCP_INPUT, 1),
        (TCP_UTIL, 1),
    ]


def test_semantic_call_edge_is_persisted_with_its_call_site(
    semantic_repo: Path, storage: IndexStorage
) -> None:
    """The semantic edge supersedes the equivalent heuristic one.

    The fallback indexer finds the very same ``tcp_process -> tcp_helper``
    call, so keeping both would list the call twice with conflicting
    provenance; the semantic edge (which knows the call site) wins.
    """
    provider = _RecordingSemanticProvider(
        prepare={
            (TCP_INPUT, 1): PrepareCallHierarchyOutcome(
                status="ok", items=(_item("tcp_process", TCP_INPUT, 1, 5),)
            )
        },
        outgoing={
            (TCP_INPUT, 1): OutgoingCallsOutcome(
                status="ok",
                calls=(
                    OutgoingCall(
                        target=_item("tcp_helper", TCP_UTIL, 1, 5),
                        call_sites=(SemanticLocation(TCP_INPUT, 3, 12),),
                    ),
                ),
            )
        },
    )
    orchestrator = _orchestrator(
        semantic_repo,
        storage,
        ctags_definitions=(TCP_PROCESS_DEFINITION,),
        semantic_provider=provider,
        semantic_symbol_limit=8,
    )

    orchestrator.ensure_index()

    caller = storage.find_symbols_by_name("tcp_process")[0]
    callee = storage.find_symbols_by_name("tcp_helper")[0]
    edges = _call_edges(storage, caller.id)

    assert len(edges) == 1
    assert edges[0].provenance == "semantic"
    assert edges[0].target_symbol_id == callee.id
    assert edges[0].target_name == "tcp_helper"
    assert (
        edges[0].site_relative_path,
        edges[0].site_line,
        edges[0].site_column,
    ) == (TCP_INPUT, 3, 12)


def test_equivalent_semantic_edges_from_several_items_are_merged_once(
    semantic_repo: Path, storage: IndexStorage
) -> None:
    """A declaration and a definition item report the same call.

    Both prepared items must be queried (nothing is silently dropped), but
    the identical call -- and a repeated call site within one entry -- must
    collapse into a single persisted edge.
    """
    duplicate_call = OutgoingCall(
        target=_item("tcp_helper", TCP_UTIL, 1, 5),
        call_sites=(
            SemanticLocation(TCP_INPUT, 3, 12),
            SemanticLocation(TCP_INPUT, 3, 12),
        ),
    )
    provider = _RecordingSemanticProvider(
        prepare={
            (TCP_INPUT, 1): PrepareCallHierarchyOutcome(
                status="ok",
                items=(
                    _item("tcp_process", TCP_INPUT, 1, 5),
                    _item("tcp_process", TCP_INPUT, 1, 9),
                ),
            )
        },
        outgoing={
            (TCP_INPUT, 1): OutgoingCallsOutcome(
                status="ok", calls=(duplicate_call, duplicate_call)
            )
        },
    )
    orchestrator = _orchestrator(
        semantic_repo,
        storage,
        ctags_definitions=(TCP_PROCESS_DEFINITION,),
        semantic_provider=provider,
        semantic_symbol_limit=8,
    )

    orchestrator.ensure_index()

    caller = storage.find_symbols_by_name("tcp_process")[0]
    assert len(provider.outgoing_items) == 2
    assert len(_call_edges(storage, caller.id)) == 1


def test_service_references_expose_the_semantic_reference_site(
    semantic_repo: Path, storage: IndexStorage
) -> None:
    provider = _RecordingSemanticProvider(
        prepare={
            (TCP_UTIL, 1): PrepareCallHierarchyOutcome(
                status="ok", items=(_item("tcp_helper", TCP_UTIL, 1, 5),)
            )
        },
        references={
            (TCP_UTIL, 1): ReferencesOutcome(
                status="ok", locations=(SemanticLocation(TCP_INPUT, 3, 12),)
            )
        },
    )
    orchestrator = _orchestrator(
        semantic_repo,
        storage,
        ctags_definitions=(TCP_PROCESS_DEFINITION,),
        semantic_provider=provider,
        semantic_symbol_limit=8,
    )
    service = IndexService(semantic_repo, storage, orchestrator)

    service.ensure_index()

    helper = service.find_symbol("tcp_helper")
    references = service.references(helper.id)

    assert len(references) == 1
    assert references[0].edge_type == "reference"
    assert references[0].provenance == "semantic"
    assert (
        references[0].site_relative_path,
        references[0].site_line,
        references[0].site_column,
    ) == (TCP_INPUT, 3, 12)


def test_semantic_symbol_limit_bounds_the_enrichment_budget(
    semantic_repo: Path, storage: IndexStorage
) -> None:
    """Enrichment follows merged batch order: ctags definitions first."""
    provider = _RecordingSemanticProvider(
        prepare={
            (TCP_INPUT, 1): PrepareCallHierarchyOutcome(
                status="ok", items=(_item("tcp_process", TCP_INPUT, 1, 5),)
            ),
            (TCP_UTIL, 1): PrepareCallHierarchyOutcome(
                status="ok", items=(_item("tcp_helper", TCP_UTIL, 1, 5),)
            ),
        }
    )
    orchestrator = _orchestrator(
        semantic_repo,
        storage,
        ctags_definitions=(TCP_PROCESS_DEFINITION,),
        semantic_provider=provider,
        semantic_symbol_limit=1,
    )

    orchestrator.ensure_index()

    assert [(path, line) for path, line, _ in provider.prepare_calls] == [
        (TCP_INPUT, 1)
    ]
    assert [(path, line) for path, line, _ in provider.reference_calls] == [
        (TCP_INPUT, 1)
    ]


# -- degraded providers -----------------------------------------------------


def test_semantic_timeout_for_one_symbol_is_recorded_and_isolated(
    semantic_repo: Path, storage: IndexStorage
) -> None:
    provider = _RecordingSemanticProvider(
        prepare={
            (TCP_INPUT, 1): PrepareCallHierarchyOutcome(
                status="ok", items=(_item("tcp_process", TCP_INPUT, 1, 5),)
            ),
            (TCP_UTIL, 1): PrepareCallHierarchyOutcome(
                status="timeout",
                items=(),
                reason="no clangd response to request 7 within 5.000s",
            ),
        },
        outgoing={
            (TCP_INPUT, 1): OutgoingCallsOutcome(
                status="ok",
                calls=(
                    OutgoingCall(
                        target=_item("tcp_helper", TCP_UTIL, 1, 5),
                        call_sites=(SemanticLocation(TCP_INPUT, 3, 12),),
                    ),
                ),
            )
        },
    )
    orchestrator = _orchestrator(
        semantic_repo,
        storage,
        ctags_definitions=(TCP_PROCESS_DEFINITION,),
        semantic_provider=provider,
        semantic_symbol_limit=8,
    )

    result = orchestrator.ensure_index()

    assert result.status == "reindexed"
    assert any("timeout" in diagnostic for diagnostic in result.diagnostics)
    assert any(TCP_UTIL in diagnostic for diagnostic in result.diagnostics)

    # A per-symbol failure leaves the provider itself usable.
    diagnostics_by_provider = {
        diagnostic.provider_name: diagnostic
        for diagnostic in result.provider_diagnostics
    }
    assert diagnostics_by_provider["clangd"].available is True

    # Definitions and the successful symbol's semantic edge both survive.
    assert storage.symbol_count() == 2
    caller = storage.find_symbols_by_name("tcp_process")[0]
    edges = _call_edges(storage, caller.id)
    assert len(edges) == 1
    assert edges[0].provenance == "semantic"
    assert edges[0].site_line == 3


def test_semantic_reference_error_for_one_symbol_is_recorded_and_isolated(
    semantic_repo: Path, storage: IndexStorage
) -> None:
    provider = _RecordingSemanticProvider(
        prepare={
            (TCP_INPUT, 1): PrepareCallHierarchyOutcome(
                status="ok", items=(_item("tcp_process", TCP_INPUT, 1, 5),)
            ),
            (TCP_UTIL, 1): PrepareCallHierarchyOutcome(
                status="ok", items=(_item("tcp_helper", TCP_UTIL, 1, 5),)
            ),
        },
        references={
            (TCP_INPUT, 1): ReferencesOutcome(
                status="error", locations=(), reason="clangd internal error"
            ),
            (TCP_UTIL, 1): ReferencesOutcome(
                status="ok", locations=(SemanticLocation(TCP_INPUT, 3, 12),)
            ),
        },
    )
    orchestrator = _orchestrator(
        semantic_repo,
        storage,
        ctags_definitions=(TCP_PROCESS_DEFINITION,),
        semantic_provider=provider,
        semantic_symbol_limit=8,
    )

    result = orchestrator.ensure_index()

    assert result.status == "reindexed"
    assert any(TCP_INPUT in diagnostic for diagnostic in result.diagnostics)
    assert storage.symbol_count() == 2

    # The other symbol's reference edge is unaffected by the failure.
    callee = storage.find_symbols_by_name("tcp_helper")[0]
    references = [
        edge for edge in storage.references(callee.id) if edge.edge_type == "reference"
    ]
    assert len(references) == 1
    assert references[0].site_relative_path == TCP_INPUT
    assert references[0].site_line == 3

    # The heuristic call edge is still there: nothing was destroyed.
    caller = storage.find_symbols_by_name("tcp_process")[0]
    assert len(_call_edges(storage, caller.id)) == 1


def test_unavailable_semantic_provider_skips_enrichment_entirely(
    semantic_repo: Path, storage: IndexStorage
) -> None:
    provider = _RecordingSemanticProvider(
        available=False, reason="clangd not installed"
    )
    orchestrator = _orchestrator(
        semantic_repo,
        storage,
        ctags_definitions=(TCP_PROCESS_DEFINITION,),
        semantic_provider=provider,
        semantic_symbol_limit=8,
    )

    result = orchestrator.ensure_index()

    assert result.status == "reindexed"
    assert provider.prepare_calls == []
    assert provider.outgoing_items == []
    assert provider.reference_calls == []

    caller = storage.find_symbols_by_name("tcp_process")[0]
    assert [edge.provenance for edge in storage.outgoing_edges(caller.id)] == [
        "heuristic"
    ]


def test_reindexing_drops_semantic_edges_from_the_previous_commit(
    two_commit_git_repo: tuple[Path, str, str], storage: IndexStorage
) -> None:
    repo, first_head, second_head = two_commit_git_repo
    tcp_input = "net/ipv4/tcp_input.c"
    provider = _RecordingSemanticProvider(
        prepare={
            (tcp_input, 1): PrepareCallHierarchyOutcome(
                status="ok", items=(_item("tcp_input", tcp_input, 1, 5),)
            )
        },
        outgoing={
            (tcp_input, 1): OutgoingCallsOutcome(
                status="ok",
                calls=(
                    OutgoingCall(
                        target=_item("kfree_skb", "net/core/skbuff.c", 100, 5),
                        call_sites=(SemanticLocation(tcp_input, 3, 12),),
                    ),
                ),
            )
        },
    )
    first_orchestrator = _orchestrator(
        repo, storage, semantic_provider=provider, semantic_symbol_limit=8
    )

    first = first_orchestrator.ensure_index()

    assert first.head == first_head
    assert storage.edge_count() == 1

    subprocess.run(
        ["git", "checkout", second_head],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    second_orchestrator = _orchestrator(
        repo,
        storage,
        semantic_provider=_RecordingSemanticProvider(
            available=False, reason="clangd stopped"
        ),
        semantic_symbol_limit=8,
    )

    second = second_orchestrator.ensure_index()

    assert second.status == "reindexed"
    assert storage.current_head() == second_head
    assert storage.edge_count() == 0

    symbol = storage.find_symbols_by_name("tcp_input")[0]
    assert symbol.commit_hash == second_head
    assert storage.references(symbol.id) == []


# -- provider lifecycle -----------------------------------------------------


def _closable_provider() -> _ClosableSemanticProvider:
    return _ClosableSemanticProvider(
        prepare={
            (TCP_INPUT, 1): PrepareCallHierarchyOutcome(
                status="ok", items=(_item("tcp_process", TCP_INPUT, 1, 5),)
            )
        },
        outgoing={
            (TCP_INPUT, 1): OutgoingCallsOutcome(
                status="ok",
                calls=(
                    OutgoingCall(
                        target=_item("tcp_helper", TCP_UTIL, 1, 5),
                        call_sites=(SemanticLocation(TCP_INPUT, 3, 12),),
                    ),
                ),
            )
        },
    )


def test_orchestrator_close_releases_the_semantic_provider_once(
    semantic_repo: Path, storage: IndexStorage
) -> None:
    provider = _closable_provider()
    orchestrator = _orchestrator(
        semantic_repo,
        storage,
        ctags_definitions=(TCP_PROCESS_DEFINITION,),
        semantic_provider=provider,
        semantic_symbol_limit=8,
    )

    result = orchestrator.ensure_index()

    # Enrichment must not close a provider it may need again on the next run.
    assert result.status == "reindexed"
    assert provider.close_calls == 0

    orchestrator.close()
    orchestrator.close()

    assert provider.close_calls == 1


def test_orchestrator_close_tolerates_a_provider_without_close(
    semantic_repo: Path, storage: IndexStorage
) -> None:
    provider = _RecordingSemanticProvider()
    orchestrator = _orchestrator(
        semantic_repo,
        storage,
        ctags_definitions=(TCP_PROCESS_DEFINITION,),
        semantic_provider=provider,
        semantic_symbol_limit=8,
    )

    orchestrator.ensure_index()
    orchestrator.close()

    assert not hasattr(provider, "close")


def test_composition_closes_the_provider_after_enrichment(
    semantic_repo: Path, storage: IndexStorage
) -> None:
    from netstack_academy.indexing.composition import run_indexing_session

    provider = _closable_provider()
    requested_repos: list[Path] = []

    def factory(kernel_repo: Path) -> _ClosableSemanticProvider:
        requested_repos.append(Path(kernel_repo))
        return provider

    result = run_indexing_session(
        semantic_repo,
        storage,
        semantic_provider_factory=factory,
        ctags_runner=_stub_ctags(TCP_PROCESS_DEFINITION),
        fallback_indexer=index_fallback,
        semantic_symbol_limit=8,
    )

    assert result.status == "reindexed"
    assert requested_repos == [semantic_repo]
    assert provider.close_calls == 1

    # The semantic edge landed, so the close happened *after* enrichment.
    caller = storage.find_symbols_by_name("tcp_process")[0]
    assert [edge.provenance for edge in _call_edges(storage, caller.id)] == ["semantic"]


def test_composition_closes_the_provider_when_reindexing_fails(
    semantic_repo: Path, storage: IndexStorage
) -> None:
    from netstack_academy.indexing.composition import run_indexing_session

    provider = _closable_provider()

    def failing_fallback(*args: object, **kwargs: object) -> object:
        raise RuntimeError("simulated fallback failure")

    result = run_indexing_session(
        semantic_repo,
        storage,
        semantic_provider_factory=lambda kernel_repo: provider,
        ctags_runner=_stub_ctags(TCP_PROCESS_DEFINITION),
        fallback_indexer=failing_fallback,
        semantic_symbol_limit=8,
    )

    assert result.status == "failed"
    assert provider.close_calls == 1
    assert storage.current_head() is None


# -- default semantic symbol budget ------------------------------------------
#
# ``IndexOrchestrator`` itself keeps ``semantic_symbol_limit: int | None =
# None`` (unbounded) as its own constructor default -- every orchestrator
# test above passes an explicit, small limit precisely because unbounded is
# the wrong default for a real run. ``run_indexing_session`` is the
# composition root real callers (and, eventually, an HTTP layer) actually
# go through, so *it* is where an unbounded default is a footgun: omitting
# ``semantic_symbol_limit`` should not silently mean "enrich every function
# in the kernel tree, one synchronous clangd round trip at a time". These
# tests pin a finite, documented, conservative default there, while
# preserving ``None`` as an explicit, opt-in escape hatch for unbounded
# enrichment (e.g. a small tree, or a deliberately patient batch run).


class _RecordingOrchestrator:
    """Stands in for :class:`IndexOrchestrator` to capture the
    ``semantic_symbol_limit`` composition wires it up with, without needing
    to actually run enrichment against hundreds of synthetic symbols to
    observe where a numeric default truncates.
    """

    last_kwargs: dict[str, object] | None = None

    def __init__(self, kernel_repo: Path, storage: IndexStorage, **kwargs: object) -> None:
        type(self).last_kwargs = kwargs

    def ensure_index(self):
        from netstack_academy.indexing.orchestrator import IndexRunResult

        return IndexRunResult(
            status="reindexed", head="deadbeef", symbol_count=0, edge_count=0
        )

    def close(self) -> None:
        pass


def test_default_semantic_symbol_limit_is_a_conservative_positive_constant() -> None:
    """The finding requires a *finite*, *documented*, *exact* default: pin
    the concrete value here so a future change to it is a deliberate,
    reviewed edit to this test rather than a silent drift. 200 symbols, at
    up to three synchronous provider round trips each (per
    ``IndexOrchestrator``'s own enrichment budget documentation), is a
    conservative bound for one ``ensure_index()`` call against a live
    ``clangd`` session.
    """
    from netstack_academy.indexing.composition import DEFAULT_SEMANTIC_SYMBOL_LIMIT

    assert DEFAULT_SEMANTIC_SYMBOL_LIMIT == 200
    assert isinstance(DEFAULT_SEMANTIC_SYMBOL_LIMIT, int)
    assert DEFAULT_SEMANTIC_SYMBOL_LIMIT > 0


def test_run_indexing_session_defaults_to_the_finite_symbol_budget_when_omitted(
    monkeypatch: pytest.MonkeyPatch, semantic_repo: Path, storage: IndexStorage
) -> None:
    """Omitting ``semantic_symbol_limit`` entirely must not mean unbounded:
    it must mean the documented finite default.
    """
    from netstack_academy.indexing import composition

    monkeypatch.setattr(composition, "IndexOrchestrator", _RecordingOrchestrator)
    _RecordingOrchestrator.last_kwargs = None

    composition.run_indexing_session(
        semantic_repo,
        storage,
        semantic_provider_factory=lambda kernel_repo: object(),
        ctags_runner=_stub_ctags(TCP_PROCESS_DEFINITION),
        fallback_indexer=index_fallback,
    )

    assert _RecordingOrchestrator.last_kwargs is not None
    assert (
        _RecordingOrchestrator.last_kwargs["semantic_symbol_limit"]
        == composition.DEFAULT_SEMANTIC_SYMBOL_LIMIT
    )


def test_run_indexing_session_still_honors_explicit_none_as_unbounded(
    monkeypatch: pytest.MonkeyPatch, semantic_repo: Path, storage: IndexStorage
) -> None:
    """``None`` remains a valid, explicit opt-in for unbounded enrichment --
    distinct from simply omitting the argument -- so a caller that has
    deliberately chosen "no bound" is not silently capped by the new
    default.
    """
    from netstack_academy.indexing import composition

    monkeypatch.setattr(composition, "IndexOrchestrator", _RecordingOrchestrator)
    _RecordingOrchestrator.last_kwargs = None

    composition.run_indexing_session(
        semantic_repo,
        storage,
        semantic_provider_factory=lambda kernel_repo: object(),
        ctags_runner=_stub_ctags(TCP_PROCESS_DEFINITION),
        fallback_indexer=index_fallback,
        semantic_symbol_limit=None,
    )

    assert _RecordingOrchestrator.last_kwargs is not None
    assert _RecordingOrchestrator.last_kwargs["semantic_symbol_limit"] is None
