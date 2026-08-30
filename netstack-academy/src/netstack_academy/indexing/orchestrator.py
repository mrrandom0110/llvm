"""Commit-aware orchestration of the kernel symbol index.

``IndexOrchestrator.ensure_index()`` is the single entry point that decides,
on every call, whether the on-disk index still matches the kernel repo's
current ``HEAD``:

- If the repository itself is unavailable (missing path, not a git repo,
  ``git`` timeout, ...), the run is reported as ``"failed"`` *without*
  touching ``storage`` at all -- there is nothing to reuse or replace.
- If this orchestrator instance has already verified the index against the
  current ``HEAD`` (tracked in-memory, per instance, since the last time
  *this* instance successfully reindexed), the run is reported as
  ``"reused"`` without invoking any provider -- ctags, the fallback
  indexer, and the semantic provider are all skipped entirely.
- Otherwise a full reindex is attempted: ctags (authoritative definitions,
  when available) and the regex fallback indexer (heuristic call edges --
  always run, since ctags alone has no call graph) are both invoked, their
  output is merged into a single ``SymbolInput``/``EdgeInput`` batch, the
  optional semantic provider's capabilities are recorded as a diagnostic,
  and the batch is committed via ``storage.replace_symbols_and_edges`` in
  one atomic transaction.

Any exception raised anywhere in the reindex pipeline -- a provider
misbehaving, an unexpected ``storage`` failure, or anything else -- is
caught and turned into a ``"failed"`` result. Because
``replace_symbols_and_edges`` only ever mutates ``storage`` after building
the entire new generation, and because collection (ctags/fallback/semantic)
never touches ``storage`` at all, a failure anywhere in this pipeline always
leaves the previously indexed generation -- the "last good index" -- fully
intact, including ``storage.current_head()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from ..repo_inspector import inspect_repository
from .ctags_runner import CtagsRunResult
from .ctags_runner import run_ctags as _default_ctags_runner
from .fallback_indexer import FallbackEdge, FallbackIndexResult
from .fallback_indexer import index_fallback as _default_fallback_indexer
from .models import EdgeInput, SymbolInput
from .semantic.provider import SemanticProvider
from .storage import IndexStorage

IndexRunStatus = Literal["reused", "reindexed", "failed"]

CtagsRunnerCallable = Callable[..., CtagsRunResult]
FallbackIndexerCallable = Callable[..., FallbackIndexResult]


@dataclass(frozen=True, slots=True)
class ProviderDiagnostic:
    """A single provider's availability, surfaced from one ``ensure_index`` run."""

    provider_name: str
    available: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class IndexRunResult:
    """The outcome of one ``IndexOrchestrator.ensure_index()`` call."""

    status: IndexRunStatus
    head: str | None
    symbol_count: int
    edge_count: int
    provider_diagnostics: tuple[ProviderDiagnostic, ...] = field(default_factory=tuple)
    reason: str | None = None


def _ctags_symbol_key(relative_path: str, line: int) -> tuple[str, int]:
    return (relative_path, line)


def _merge_collectors(
    ctags_result: CtagsRunResult,
    fallback_result: FallbackIndexResult,
) -> tuple[list[SymbolInput], list[EdgeInput]]:
    """Merge ctags definitions and fallback symbols/edges into one batch.

    Ctags definitions are preferred whenever ctags succeeded: they carry
    richer, tool-verified metadata (signature, scope) than the regex
    fallback scanner. Fallback symbols only fill in a ``(relative_path,
    line)`` location that ctags did not already report -- this keeps the
    fallback indexer useful even when ctags is available (e.g. ctags
    misconfiguration, restricted roots) while never overwriting a ctags
    definition. The fallback indexer's heuristic call edges are always
    merged in, since neither ctags nor this merge step have any other
    source of call-graph edges; each edge's provenance is preserved as
    ``"heuristic"``.
    """

    symbols: list[SymbolInput] = []
    index_by_location: dict[tuple[str, int], int] = {}

    def _add_symbol(symbol_input: SymbolInput) -> int:
        key = _ctags_symbol_key(symbol_input.relative_path, symbol_input.line)
        existing_index = index_by_location.get(key)
        if existing_index is not None:
            return existing_index
        new_index = len(symbols)
        symbols.append(symbol_input)
        index_by_location[key] = new_index
        return new_index

    if ctags_result.status == "ok":
        for definition in ctags_result.definitions:
            _add_symbol(
                SymbolInput(
                    name=definition.name,
                    kind=definition.kind or "function",
                    relative_path=definition.path,
                    line=definition.line,
                    column=None,
                    signature=definition.signature,
                    scope=definition.scope,
                    is_static=definition.is_static,
                )
            )

    for fallback_symbol in fallback_result.symbols:
        _add_symbol(
            SymbolInput(
                name=fallback_symbol.name,
                kind=fallback_symbol.kind,
                relative_path=fallback_symbol.relative_path,
                line=fallback_symbol.line,
                column=None,
                signature=fallback_symbol.signature,
                scope=None,
                is_static=fallback_symbol.is_static,
            )
        )

    by_name_and_path: dict[tuple[str, str], int] = {}
    by_name_global: dict[str, list[int]] = {}
    for symbol_index, symbol_input in enumerate(symbols):
        by_name_and_path[(symbol_input.name, symbol_input.relative_path)] = symbol_index
        by_name_global.setdefault(symbol_input.name, []).append(symbol_index)

    edges: list[EdgeInput] = [
        edge
        for fallback_edge in fallback_result.edges
        if (edge := _resolve_edge(fallback_edge, by_name_and_path, by_name_global))
        is not None
    ]

    return symbols, edges


def _resolve_edge(
    fallback_edge: FallbackEdge,
    by_name_and_path: dict[tuple[str, str], int],
    by_name_global: dict[str, list[int]],
) -> EdgeInput | None:
    source_index = by_name_and_path.get(
        (fallback_edge.source_name, fallback_edge.source_relative_path)
    )
    if source_index is None:
        # The caller itself was not merged into the symbol batch (should
        # not normally happen, since fallback symbols are always merged),
        # so there is no valid source to attach this edge to.
        return None

    target_index: int | None = None
    if fallback_edge.target_relative_path is not None:
        target_index = by_name_and_path.get(
            (fallback_edge.target_name, fallback_edge.target_relative_path)
        )
    else:
        candidates = by_name_global.get(fallback_edge.target_name, [])
        if len(candidates) == 1:
            target_index = candidates[0]

    return EdgeInput(
        source_index=source_index,
        target_index=target_index,
        target_name=fallback_edge.target_name,
        edge_type="call",
        provenance="heuristic",
    )


class IndexOrchestrator:
    """Decides when to reindex and drives ctags/fallback/semantic providers."""

    def __init__(
        self,
        kernel_repo: Path,
        storage: IndexStorage,
        *,
        ctags_runner: CtagsRunnerCallable = _default_ctags_runner,
        fallback_indexer: FallbackIndexerCallable = _default_fallback_indexer,
        semantic_provider: SemanticProvider | None = None,
    ) -> None:
        self._kernel_repo = Path(kernel_repo)
        self._storage = storage
        self._ctags_runner = ctags_runner
        self._fallback_indexer = fallback_indexer
        self._semantic_provider = semantic_provider

        # In-memory, per-instance bookkeeping of the last HEAD *this*
        # instance successfully reindexed. Deliberately not derived from
        # ``storage.current_head()``: a freshly constructed orchestrator
        # always re-verifies by running the pipeline at least once, even if
        # ``storage`` already holds a matching generation from a previous
        # instance/run, so that a provider regression is caught on the next
        # call rather than silently masked by a stale on-disk match.
        self._verified_head: str | None = None
        self._last_symbol_count = 0
        self._last_edge_count = 0

    def ensure_index(self) -> IndexRunResult:
        repository_state = inspect_repository(self._kernel_repo)
        if not repository_state.available:
            return IndexRunResult(
                status="failed",
                head=None,
                symbol_count=0,
                edge_count=0,
                provider_diagnostics=(),
                reason=repository_state.reason,
            )

        head = repository_state.head
        if head is not None and head == self._verified_head:
            return IndexRunResult(
                status="reused",
                head=head,
                symbol_count=self._last_symbol_count,
                edge_count=self._last_edge_count,
                provider_diagnostics=(),
                reason=None,
            )

        try:
            return self._reindex(head)
        except Exception as exc:  # noqa: BLE001 - never let a provider crash the caller
            return IndexRunResult(
                status="failed",
                head=head,
                symbol_count=0,
                edge_count=0,
                provider_diagnostics=(),
                reason=f"Reindexing failed: {exc}",
            )

    def _reindex(self, head: str | None) -> IndexRunResult:
        provider_diagnostics: list[ProviderDiagnostic] = []

        ctags_result = self._ctags_runner(self._kernel_repo)
        provider_diagnostics.append(_ctags_diagnostic(ctags_result))

        fallback_result = self._fallback_indexer(self._kernel_repo)

        if self._semantic_provider is not None:
            capabilities = self._semantic_provider.capabilities()
            provider_diagnostics.append(
                ProviderDiagnostic(
                    provider_name=capabilities.provider_name,
                    available=capabilities.available,
                    reason=capabilities.reason,
                )
            )

        symbols, edges = _merge_collectors(ctags_result, fallback_result)

        replace_result = self._storage.replace_symbols_and_edges(head, symbols, edges)

        self._verified_head = head
        self._last_symbol_count = replace_result.symbol_count
        self._last_edge_count = replace_result.edge_count

        return IndexRunResult(
            status="reindexed",
            head=head,
            symbol_count=replace_result.symbol_count,
            edge_count=replace_result.edge_count,
            provider_diagnostics=tuple(provider_diagnostics),
            reason=None,
        )


def _ctags_diagnostic(ctags_result: CtagsRunResult) -> ProviderDiagnostic:
    available = ctags_result.status == "ok"
    reason = None if available else (ctags_result.diagnostics[0] if ctags_result.diagnostics else ctags_result.status)
    return ProviderDiagnostic(provider_name="ctags", available=available, reason=reason)
