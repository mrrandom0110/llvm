"""Application service facade over the commit-aware symbol index.

``IndexService`` is the single, typed contract the rest of the application
(and, eventually, an HTTP API layer) should use to read the index and to
trigger reindexing -- it never exposes ``storage`` or a raw SQL connection
to its callers.

Two safety properties are load-bearing here, both proven by
``tests/indexing/test_service.py``:

- **Path containment**: any caller-supplied ``relative_path`` disambiguation
  filter is validated with the purely lexical
  :func:`~netstack_academy.indexing.paths.assert_safe_relative_path` *before*
  it is ever used in a storage lookup. A path containing ``..`` traversal or
  an absolute path raises :class:`InvalidRepositoryPathError` immediately;
  storage is never touched with an unsafe path.
- **No shell/subprocess interpolation**: every lookup method (``find_symbol``,
  ``search_symbols``, ...) passes caller-supplied strings only into
  parameterized SQL queries (via :mod:`netstack_academy.indexing.storage`)
  or pure string comparisons -- never into a shell command or subprocess
  argument list. A name containing shell metacharacters is simply treated as
  literal text and, if it matches nothing, reported as
  :class:`SymbolNotFoundError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from .models import Edge, Symbol
from .orchestrator import IndexRunResult
from .paths import PathEscapesRepositoryError, assert_safe_relative_path
from .identity import resolve_symbol
from .storage import IndexStorage


class SymbolNotFoundError(LookupError):
    """Raised when a symbol lookup matches no symbol at all."""


class SymbolAmbiguousError(ValueError):
    """Raised when a symbol lookup matches more than one symbol.

    Carries the ambiguous ``candidates`` so a caller (or an HTTP layer) can
    present them for disambiguation, rather than guessing.
    """

    def __init__(
        self, message: str, *, candidates: Sequence["SymbolView"] = ()
    ) -> None:
        super().__init__(message)
        self.candidates: tuple["SymbolView", ...] = tuple(candidates)


class InvalidRepositoryPathError(ValueError):
    """Raised when a caller-supplied relative path escapes the repository.

    Covers both ``..`` parent traversal and absolute paths outside the
    kernel repository -- both are rejected lexically, before any storage
    lookup, regardless of whether the path currently exists on disk.
    """


@dataclass(frozen=True, slots=True)
class SymbolView:
    """A stable, typed, read-only view of a persisted symbol."""

    id: int
    name: str
    kind: str
    relative_path: str
    line: int
    column: int | None
    signature: str | None
    scope: str | None
    is_static: bool
    commit_hash: str


@dataclass(frozen=True, slots=True)
class EdgeView:
    """A stable, typed, read-only view of a persisted call/reference edge."""

    id: int
    source_symbol_id: int
    target_symbol_id: int | None
    target_name: str
    edge_type: str
    provenance: str
    commit_hash: str


@dataclass(frozen=True, slots=True)
class IndexStatusView:
    """A snapshot of the currently indexed generation, read directly from storage."""

    head: str | None
    symbol_count: int
    edge_count: int


@runtime_checkable
class _Orchestrator(Protocol):
    """The minimal shape :class:`IndexService` depends on for reindexing.

    Kept as a narrow structural protocol (rather than importing
    :class:`~netstack_academy.indexing.orchestrator.IndexOrchestrator`
    directly) so tests can inject a stub orchestrator without constructing
    a real one.
    """

    def ensure_index(self) -> IndexRunResult: ...


def _symbol_to_view(symbol: Symbol) -> SymbolView:
    return SymbolView(
        id=symbol.id,
        name=symbol.name,
        kind=symbol.kind,
        relative_path=symbol.relative_path,
        line=symbol.line,
        column=symbol.column,
        signature=symbol.signature,
        scope=symbol.scope,
        is_static=symbol.is_static,
        commit_hash=symbol.commit_hash,
    )


def _edge_to_view(edge: Edge) -> EdgeView:
    return EdgeView(
        id=edge.id,
        source_symbol_id=edge.source_symbol_id,
        target_symbol_id=edge.target_symbol_id,
        target_name=edge.target_name,
        edge_type=edge.edge_type,
        provenance=edge.provenance,
        commit_hash=edge.commit_hash,
    )


class IndexService:
    """Typed application-facing facade over :class:`IndexStorage`.

    ``kernel_repo`` is retained for context (and for a future HTTP layer
    that may need it for deep-link construction) but is never required for
    the lookups below -- they only ever need already-persisted, relative
    paths and the purely lexical path-safety check.
    """

    def __init__(
        self,
        kernel_repo: Path,
        storage: IndexStorage,
        orchestrator: _Orchestrator,
    ) -> None:
        self._kernel_repo = Path(kernel_repo)
        self._storage = storage
        self._orchestrator = orchestrator

    def get_status(self) -> IndexStatusView:
        return IndexStatusView(
            head=self._storage.current_head(),
            symbol_count=self._storage.symbol_count(),
            edge_count=self._storage.edge_count(),
        )

    def ensure_index(self) -> IndexRunResult:
        """Delegate verbatim to the injected orchestrator."""
        return self._orchestrator.ensure_index()

    def find_symbol(self, name: str, *, relative_path: str | None = None) -> SymbolView:
        if relative_path is not None:
            try:
                assert_safe_relative_path(relative_path)
            except PathEscapesRepositoryError as exc:
                raise InvalidRepositoryPathError(str(exc)) from exc

        resolution = resolve_symbol(self._storage, name, relative_path=relative_path)

        if resolution.status == "found":
            assert resolution.symbol is not None
            return _symbol_to_view(resolution.symbol)

        if resolution.status == "ambiguous":
            raise SymbolAmbiguousError(
                resolution.reason or f"Multiple symbols named {name!r} found",
                candidates=[_symbol_to_view(candidate) for candidate in resolution.candidates],
            )

        raise SymbolNotFoundError(resolution.reason or f"No symbol named {name!r} found")

    def search_symbols(self, query: str, *, limit: int = 50) -> list[SymbolView]:
        matches = self._storage.search_symbols(query, limit=limit)
        return [_symbol_to_view(match) for match in matches]

    def outgoing_edges(self, symbol_id: int) -> list[EdgeView]:
        return [_edge_to_view(edge) for edge in self._storage.outgoing_edges(symbol_id)]

    def incoming_edges(self, symbol_id: int) -> list[EdgeView]:
        return [_edge_to_view(edge) for edge in self._storage.incoming_edges(symbol_id)]

    def references(self, symbol_id: int) -> list[EdgeView]:
        """Non-call reference edges (``edge_type == "reference"``) touching this symbol.

        Call-graph edges (``edge_type == "call"``) are exposed separately via
        :meth:`outgoing_edges`/:meth:`incoming_edges`; ``references`` is
        reserved for the semantic provider's ``textDocument/references``
        -derived edges, which use ``edge_type="reference"``.
        """
        combined = self._storage.references(symbol_id)
        return [_edge_to_view(edge) for edge in combined if edge.edge_type == "reference"]
