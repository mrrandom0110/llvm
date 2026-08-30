"""Commit-aware orchestration of the kernel symbol index.

``IndexOrchestrator.ensure_index()`` is the single entry point that decides,
on every call, whether the on-disk index still matches the kernel repo's
current ``HEAD``:

- If the repository itself is unavailable (missing path, not a git repo,
  ``git`` timeout, ...), the run is reported as ``"failed"`` *without*
  touching ``storage`` at all -- there is nothing to reuse or replace.
- Otherwise, unless ``force=True`` was requested, ``ensure_index`` compares
  the repository's current ``HEAD`` against ``storage.current_head()`` --
  the *persisted* generation, read straight from ``storage`` rather than
  from any in-memory, per-instance bookkeeping. When they match, the run is
  reported as ``"reused"`` without invoking any provider -- ctags, the
  fallback indexer, and the semantic provider are all skipped entirely.
  Because this comparison is against ``storage`` itself, reuse works across
  a newly constructed ``IndexOrchestrator``/session pointed at the same
  on-disk index, not only across repeated calls on one long-lived instance.
- ``force=True`` skips that comparison unconditionally and always runs the
  full pipeline below, even when the persisted ``HEAD`` already matches --
  the explicit escape hatch for testing provider failures or a manual
  "refresh now" trigger.
- Otherwise (no persisted match, or ``force=True``) a full reindex is
  attempted: ctags (authoritative definitions, when available) and the
  regex fallback indexer (heuristic call edges -- always run, since ctags
  alone has no call graph) are both invoked, their output is merged into a
  single ``SymbolInput``/``EdgeInput`` batch, an *available* semantic
  provider is asked to enrich that batch, and the result is committed via
  ``storage.replace_symbols_and_edges`` in one atomic transaction.

Semantic enrichment
-------------------

Enrichment happens after the merge and *before* the single
``replace_symbols_and_edges`` call, so heuristic and semantic edges are
committed together as one generation -- and a reindex at a new ``HEAD``
therefore drops the previous generation's semantic edges atomically along
with everything else.

It runs only when a semantic provider was injected *and* reports
``capabilities().available``; an unavailable provider (no ``clangd``, a
dead session) is never asked for a single position. Each indexed function
costs up to three synchronous provider round trips, so
``semantic_symbol_limit`` bounds how many symbols are enriched per run --
omitting it selects the conservative :data:`DEFAULT_SEMANTIC_SYMBOL_LIMIT`
(200); passing ``None`` explicitly opts into unbounded enrichment instead.
Every operation is isolated: a ``"timeout"``/``"error"``/``"unavailable"``
outcome (or an outright exception) for one symbol is recorded in
``IndexRunResult.diagnostics`` and the run continues, keeping all merged
definitions and every other symbol's edges.

Any exception raised anywhere in the reindex pipeline -- a provider
misbehaving, an unexpected ``storage`` failure, or anything else -- is
caught and turned into a ``"failed"`` result. Because
``replace_symbols_and_edges`` only ever mutates ``storage`` after building
the entire new generation, and because collection (ctags/fallback/semantic)
never touches ``storage`` at all, a failure anywhere in this pipeline always
leaves the previously indexed generation -- the "last good index" -- fully
intact, including ``storage.current_head()``. This holds equally for a
``force=True`` run: forcing only skips the persisted-reuse check, never the
atomicity of the commit itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from ..repo_inspector import inspect_repository
from .ctags_runner import CtagsRunResult, default_index_roots
from .ctags_runner import run_ctags as _default_ctags_runner
from .fallback_indexer import FallbackEdge, FallbackIndexResult
from .fallback_indexer import index_fallback as _default_fallback_indexer
from .models import EdgeInput, SymbolInput
from .semantic.provider import SemanticProvider
from .storage import IndexStorage

IndexRunStatus = Literal["reused", "reindexed", "failed"]

CtagsRunnerCallable = Callable[..., CtagsRunResult]
FallbackIndexerCallable = Callable[..., FallbackIndexResult]

#: Conservative default cap on how many functions one ``ensure_index()`` run
#: enriches with the semantic provider, applied whenever a caller does not
#: pass an explicit ``semantic_symbol_limit``
#: (:func:`~netstack_academy.indexing.composition.run_indexing_session`, the
#: real composition root, uses this constant as its own default). Each
#: enriched symbol costs up to three synchronous provider round trips
#: (``prepareCallHierarchy``, ``outgoingCalls``, ``references``), so 200
#: symbols bounds one run to roughly 600 round trips against a live
#: ``clangd`` session -- a conservative amount of synchronous provider
#: traffic for a default. Passing ``semantic_symbol_limit=None`` explicitly
#: remains a supported, opt-in escape hatch for unbounded enrichment (e.g.
#: a small tree, or a deliberately patient batch run).
DEFAULT_SEMANTIC_SYMBOL_LIMIT = 200


@dataclass(frozen=True, slots=True)
class ProviderDiagnostic:
    """A single provider's availability, surfaced from one ``ensure_index`` run."""

    provider_name: str
    available: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class IndexRunResult:
    """The outcome of one ``IndexOrchestrator.ensure_index()`` call.

    ``provider_diagnostics`` describes each provider's *availability*;
    ``diagnostics`` collects human-readable messages about partial failures
    inside an otherwise successful run (e.g. one symbol timing out during
    semantic enrichment), each naming the symbol location it concerns.
    """

    status: IndexRunStatus
    head: str | None
    symbol_count: int
    edge_count: int
    provider_diagnostics: tuple[ProviderDiagnostic, ...] = field(default_factory=tuple)
    reason: str | None = None
    diagnostics: tuple[str, ...] = field(default_factory=tuple)


def _ctags_symbol_key(relative_path: str, line: int) -> tuple[str, int]:
    return (relative_path, line)


class _SymbolIndexResolver:
    """Maps a symbol name (optionally with a path) to its batch index.

    Never guesses: a name that occurs in several files only resolves when
    the caller supplies the path it lives in.
    """

    def __init__(self, symbols: list[SymbolInput]) -> None:
        self._by_name_and_path: dict[tuple[str, str], int] = {}
        self._by_name: dict[str, list[int]] = {}
        for symbol_index, symbol_input in enumerate(symbols):
            self._by_name_and_path.setdefault(
                (symbol_input.name, symbol_input.relative_path), symbol_index
            )
            self._by_name.setdefault(symbol_input.name, []).append(symbol_index)

    def resolve_at(self, name: str, relative_path: str) -> int | None:
        return self._by_name_and_path.get((name, relative_path))

    def resolve_unique(self, name: str) -> int | None:
        candidates = self._by_name.get(name, [])
        return candidates[0] if len(candidates) == 1 else None


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

    resolver = _SymbolIndexResolver(symbols)

    edges: list[EdgeInput] = [
        edge
        for fallback_edge in fallback_result.edges
        if (edge := _resolve_edge(fallback_edge, resolver)) is not None
    ]

    return symbols, edges


def _resolve_edge(
    fallback_edge: FallbackEdge, resolver: _SymbolIndexResolver
) -> EdgeInput | None:
    source_index = resolver.resolve_at(
        fallback_edge.source_name, fallback_edge.source_relative_path
    )
    if source_index is None:
        # The caller itself was not merged into the symbol batch (should
        # not normally happen, since fallback symbols are always merged),
        # so there is no valid source to attach this edge to.
        return None

    if fallback_edge.target_relative_path is not None:
        # The fallback indexer already committed to a file for this target
        # (a same-file `static`, or the single non-static candidate); it is
        # never re-resolved to a different file here.
        target_index = resolver.resolve_at(
            fallback_edge.target_name, fallback_edge.target_relative_path
        )
    else:
        target_index = resolver.resolve_unique(fallback_edge.target_name)

    return EdgeInput(
        source_index=source_index,
        target_index=target_index,
        target_name=fallback_edge.target_name,
        edge_type="call",
        provenance="heuristic",
    )


def _edge_key(edge: EdgeInput) -> tuple[object, ...]:
    return (
        edge.source_index,
        edge.target_index,
        edge.target_name,
        edge.edge_type,
        edge.provenance,
        edge.site_relative_path,
        edge.site_line,
        edge.site_column,
    )


def _merge_edge_batches(
    heuristic_edges: list[EdgeInput], semantic_edges: list[EdgeInput]
) -> list[EdgeInput]:
    """Combine both batches, dropping duplicate and superseded edges.

    Two de-duplications happen here. Within a batch, edges that agree on
    every field (endpoints, type, provenance and site) are the same fact
    observed twice -- a repeated call site, a repeated ``outgoingCalls``
    entry, or two prepared items reporting the same call -- and collapse
    into one row. Across batches, a heuristic call edge is dropped when a
    semantic call edge exists for the same caller and callee name: they
    describe the same call, and the semantic one additionally knows where
    the call happens, so listing both would report the call twice with
    conflicting provenance.
    """
    superseded = {
        (edge.source_index, edge.target_name)
        for edge in semantic_edges
        if edge.edge_type == "call"
    }

    merged: list[EdgeInput] = []
    seen: set[tuple[object, ...]] = set()

    for edge in (*heuristic_edges, *semantic_edges):
        if (
            edge.provenance == "heuristic"
            and edge.edge_type == "call"
            and (edge.source_index, edge.target_name) in superseded
        ):
            continue
        key = _edge_key(edge)
        if key in seen:
            continue
        seen.add(key)
        merged.append(edge)

    return merged


def _semantic_diagnostic(
    provider_name: str,
    operation: str,
    symbol: SymbolInput,
    status: str,
    reason: str | None,
) -> str:
    location = f"{symbol.relative_path}:{symbol.line}"
    detail = f": {reason}" if reason else ""
    return f"{provider_name} {operation} {status} at {location}{detail}"


def _semantic_edges_for_symbol(
    provider: SemanticProvider,
    *,
    provider_name: str,
    symbol: SymbolInput,
    symbol_index: int,
    resolver: _SymbolIndexResolver,
    diagnostics: list[str],
) -> list[EdgeInput]:
    """Ask the provider about one symbol, collecting its semantic edges.

    ``prepareCallHierarchy``/``outgoingCalls`` and ``references`` are
    independent: a failure of one is recorded and the other is still
    attempted, so a symbol whose call hierarchy cannot be prepared can
    still contribute reference edges.
    """
    edges: list[EdgeInput] = []
    line = symbol.line
    column = symbol.column or 1

    prepared = provider.prepare_call_hierarchy(
        symbol.relative_path, line=line, column=column
    )
    if prepared.status != "ok":
        diagnostics.append(
            _semantic_diagnostic(
                provider_name,
                "prepareCallHierarchy",
                symbol,
                prepared.status,
                prepared.reason,
            )
        )
    else:
        for item in prepared.items:
            outgoing = provider.outgoing_calls(item)
            if outgoing.status != "ok":
                diagnostics.append(
                    _semantic_diagnostic(
                        provider_name,
                        "outgoingCalls",
                        symbol,
                        outgoing.status,
                        outgoing.reason,
                    )
                )
                continue
            for call in outgoing.calls:
                target_index = resolver.resolve_at(
                    call.target.name, call.target.relative_path
                )
                if target_index is None:
                    target_index = resolver.resolve_unique(call.target.name)

                if not call.call_sites:
                    edges.append(
                        EdgeInput(
                            source_index=symbol_index,
                            target_index=target_index,
                            target_name=call.target.name,
                            edge_type="call",
                            provenance="semantic",
                        )
                    )
                    continue

                for call_site in call.call_sites:
                    edges.append(
                        EdgeInput(
                            source_index=symbol_index,
                            target_index=target_index,
                            target_name=call.target.name,
                            edge_type="call",
                            provenance="semantic",
                            site_relative_path=call_site.relative_path,
                            site_line=call_site.line,
                            site_column=call_site.column,
                        )
                    )

    referenced = provider.references(symbol.relative_path, line=line, column=column)
    if referenced.status != "ok":
        diagnostics.append(
            _semantic_diagnostic(
                provider_name, "references", symbol, referenced.status, referenced.reason
            )
        )
    else:
        for location in referenced.locations:
            # A reference edge is anchored on the *referenced* symbol: a use
            # site is a position in a file and need not fall inside another
            # indexed definition, so there is no second endpoint to resolve.
            edges.append(
                EdgeInput(
                    source_index=symbol_index,
                    target_index=None,
                    target_name=symbol.name,
                    edge_type="reference",
                    provenance="semantic",
                    site_relative_path=location.relative_path,
                    site_line=location.line,
                    site_column=location.column,
                )
            )

    return edges


def _enrich_with_semantic_edges(
    provider: SemanticProvider,
    *,
    provider_name: str,
    symbols: list[SymbolInput],
    edges: list[EdgeInput],
    symbol_limit: int | None,
    diagnostics: list[str],
) -> list[EdgeInput]:
    """Add semantic call/reference edges for up to ``symbol_limit`` functions.

    Symbols are visited in merged batch order (ctags definitions first,
    then fallback-only ones), so the budget is spent on the definitions the
    authoritative collector found.
    """
    resolver = _SymbolIndexResolver(symbols)
    semantic_edges: list[EdgeInput] = []
    enriched = 0

    for symbol_index, symbol in enumerate(symbols):
        if symbol.kind != "function":
            continue
        if symbol_limit is not None and enriched >= symbol_limit:
            break
        enriched += 1

        try:
            semantic_edges.extend(
                _semantic_edges_for_symbol(
                    provider,
                    provider_name=provider_name,
                    symbol=symbol,
                    symbol_index=symbol_index,
                    resolver=resolver,
                    diagnostics=diagnostics,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one symbol must not fail the run
            diagnostics.append(
                _semantic_diagnostic(
                    provider_name, "enrichment", symbol, "error", str(exc)
                )
            )

    return _merge_edge_batches(edges, semantic_edges)


class _Unset:
    """Sentinel type distinguishing "argument omitted" from "explicit ``None``".

    Used only for ``semantic_symbol_limit`` below: omitting the argument
    must select :data:`DEFAULT_SEMANTIC_SYMBOL_LIMIT`, while explicitly
    passing ``None`` must still mean "unbounded" -- a plain ``None``
    default cannot represent both.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "<unset>"


_UNSET_SYMBOL_LIMIT = _Unset()


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
        semantic_symbol_limit: int | None | _Unset = _UNSET_SYMBOL_LIMIT,
    ) -> None:
        self._kernel_repo = Path(kernel_repo)
        self._storage = storage
        self._ctags_runner = ctags_runner
        self._fallback_indexer = fallback_indexer
        self._semantic_provider = semantic_provider
        # Each enriched symbol costs up to three synchronous provider round
        # trips, so the budget is explicit. Omitting the argument selects
        # the conservative ``DEFAULT_SEMANTIC_SYMBOL_LIMIT``; passing
        # ``None`` explicitly means "no bound", which is only sensible for
        # a small tree or a fast provider.
        self._semantic_symbol_limit = (
            DEFAULT_SEMANTIC_SYMBOL_LIMIT
            if semantic_symbol_limit is _UNSET_SYMBOL_LIMIT
            else semantic_symbol_limit
        )
        self._closed = False

    def ensure_index(self, *, force: bool = False) -> IndexRunResult:
        """Reindex if needed, or reuse the persisted index at the same ``HEAD``.

        Unless ``force=True``, reuse is decided by comparing the
        repository's current ``HEAD`` against ``storage.current_head()`` --
        the generation already persisted on disk -- rather than any
        in-memory state private to this instance. A brand new
        ``IndexOrchestrator`` constructed around the same ``storage`` file
        therefore reuses just as a repeated call on one long-lived instance
        does. ``force=True`` skips that comparison and always reruns the
        full pipeline, e.g. to test a provider failure or to serve a manual
        "refresh now" request; a failure during a forced run still leaves
        the previously persisted generation untouched (see this module's
        docstring).
        """
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
        if (
            not force
            and head is not None
            and head == self._storage.current_head()
        ):
            return IndexRunResult(
                status="reused",
                head=head,
                symbol_count=self._storage.symbol_count(),
                edge_count=self._storage.edge_count(),
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

    def close(self) -> None:
        """Release an owned semantic provider, if it has anything to release.

        ``close()`` is optional on the provider contract (a stub, or a
        provider with nothing to tear down, may simply not have it), so it
        is looked up at runtime. Idempotent, and never raises: teardown
        failures must not surface as indexing failures.
        """
        if self._closed:
            return
        self._closed = True

        close = getattr(self._semantic_provider, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:  # noqa: BLE001 - teardown must never raise
            pass

    def _reindex(self, head: str | None) -> IndexRunResult:
        provider_diagnostics: list[ProviderDiagnostic] = []
        diagnostics: list[str] = []

        # Both collectors are handed the *same* curated, network-focused
        # root list computed once here, rather than each independently
        # falling back to its own notion of "everything" (ctags' own
        # default happens to already be this list, but the fallback
        # indexer's own default is a whole-repo "."). Centralizing it
        # guarantees the two collectors can never silently disagree about
        # what "the index" covers.
        roots = default_index_roots()

        ctags_result = self._ctags_runner(self._kernel_repo, roots=roots)
        provider_diagnostics.append(_ctags_diagnostic(ctags_result))

        fallback_result = self._fallback_indexer(self._kernel_repo, roots=roots)

        symbols, edges = _merge_collectors(ctags_result, fallback_result)

        if self._semantic_provider is not None:
            capabilities = self._semantic_provider.capabilities()
            provider_diagnostics.append(
                ProviderDiagnostic(
                    provider_name=capabilities.provider_name,
                    available=capabilities.available,
                    reason=capabilities.reason,
                )
            )
            if capabilities.available:
                edges = _enrich_with_semantic_edges(
                    self._semantic_provider,
                    provider_name=capabilities.provider_name,
                    symbols=symbols,
                    edges=edges,
                    symbol_limit=self._semantic_symbol_limit,
                    diagnostics=diagnostics,
                )

        replace_result = self._storage.replace_symbols_and_edges(head, symbols, edges)

        return IndexRunResult(
            status="reindexed",
            head=head,
            symbol_count=replace_result.symbol_count,
            edge_count=replace_result.edge_count,
            provider_diagnostics=tuple(provider_diagnostics),
            reason=None,
            diagnostics=tuple(diagnostics),
        )


def _ctags_diagnostic(ctags_result: CtagsRunResult) -> ProviderDiagnostic:
    available = ctags_result.status == "ok"
    reason = None if available else (ctags_result.diagnostics[0] if ctags_result.diagnostics else ctags_result.status)
    return ProviderDiagnostic(provider_name="ctags", available=available, reason=reason)
