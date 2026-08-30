"""Immutable data models shared across the symbol index package.

``SymbolInput``/``EdgeInput`` are the pre-insert representations produced by
collectors (ctags, the regex fallback indexer, a future semantic provider);
``Symbol``/``Edge`` are the post-insert representations read back out of
storage. Edges reference symbols by their position (``source_index`` /
``target_index``) within the same list passed to
``IndexStorage.replace_symbols_and_edges``, since database ids do not exist
until insert time and collectors never need a live database connection to
build a consistent batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

EdgeType = Literal["call", "reference"]
EdgeProvenance = Literal["heuristic", "semantic"]
ResolutionStatus = Literal["found", "not_found", "ambiguous"]


@dataclass(frozen=True, slots=True)
class SymbolInput:
    """A symbol definition observed by a collector, not yet persisted."""

    name: str
    kind: str
    relative_path: str
    line: int
    column: int | None = None
    signature: str | None = None
    scope: str | None = None
    is_static: bool = False


@dataclass(frozen=True, slots=True)
class EdgeInput:
    """A call/reference edge observed by a collector, not yet persisted.

    ``source_index``/``target_index`` are indices into the ``symbols`` list
    passed to the same ``replace_symbols_and_edges`` call. ``target_index``
    is ``None`` when the target could not be resolved to a known symbol
    (the edge is retained with only ``target_name`` for diagnostics).
    """

    source_index: int
    target_index: int | None
    target_name: str
    edge_type: EdgeType
    provenance: EdgeProvenance


@dataclass(frozen=True, slots=True)
class Symbol:
    """A symbol definition as persisted in storage."""

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
class Edge:
    """A call/reference edge as persisted in storage."""

    id: int
    source_symbol_id: int
    target_symbol_id: int | None
    target_name: str
    edge_type: EdgeType
    provenance: EdgeProvenance
    commit_hash: str


@dataclass(frozen=True, slots=True)
class SymbolResolution:
    """The outcome of resolving a symbol by name (and optional path).

    Invariants: ``found`` implies exactly one symbol and no candidates;
    ``ambiguous`` implies ``symbol is None`` and ``len(candidates) > 1``;
    ``not_found`` implies ``symbol is None`` and no candidates.
    """

    status: ResolutionStatus
    symbol: Symbol | None
    candidates: tuple[Symbol, ...] = field(default_factory=tuple)
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ReplaceIndexResult:
    """The outcome of a successful ``replace_symbols_and_edges`` call."""

    commit_hash: str
    symbol_count: int
    edge_count: int
