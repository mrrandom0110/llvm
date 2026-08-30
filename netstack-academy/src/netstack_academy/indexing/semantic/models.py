"""Immutable data models for the optional semantic (clangd) provider.

Positions here are always the app's 1-based ``(line, column)`` convention
(matching the rest of the ``indexing`` package, e.g. ``ctags_parser``'s
``CtagsDefinition.line``) -- the 0-based LSP convention never leaks past
``clangd_adapter``. Likewise, ``relative_path`` is always a kernel-repo
-relative POSIX path (see ``indexing/paths.py``), never a raw ``file://``
URI or absolute filesystem path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ProviderOutcomeStatus = Literal["ok", "unavailable", "timeout", "error"]


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Static description of a semantic provider's availability."""

    provider_name: str
    available: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticLocation:
    """A single source location, 1-based, relative-path form."""

    relative_path: str
    line: int
    column: int


@dataclass(frozen=True, slots=True)
class CallHierarchyItem:
    """A callable symbol as returned by ``textDocument/prepareCallHierarchy``
    (or supplied back to ``callHierarchy/outgoingCalls``)."""

    name: str
    relative_path: str
    line: int
    column: int


@dataclass(frozen=True, slots=True)
class OutgoingCall:
    """One ``callHierarchy/outgoingCalls`` entry: a callee plus the call
    site(s), within the queried item's own file, that invoke it."""

    target: CallHierarchyItem
    call_sites: tuple[SemanticLocation, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class PrepareCallHierarchyOutcome:
    """Result of :meth:`SemanticProvider.prepare_call_hierarchy`.

    ``status="ok"`` always has ``reason is None``; any other status leaves
    ``items`` empty and populates ``reason`` with a human-readable
    explanation.
    """

    status: ProviderOutcomeStatus
    items: tuple[CallHierarchyItem, ...] = field(default_factory=tuple)
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class OutgoingCallsOutcome:
    """Result of :meth:`SemanticProvider.outgoing_calls`."""

    status: ProviderOutcomeStatus
    calls: tuple[OutgoingCall, ...] = field(default_factory=tuple)
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ReferencesOutcome:
    """Result of :meth:`SemanticProvider.references`."""

    status: ProviderOutcomeStatus
    locations: tuple[SemanticLocation, ...] = field(default_factory=tuple)
    reason: str | None = None
