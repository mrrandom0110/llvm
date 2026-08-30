"""The optional semantic-provider contract.

``IndexOrchestrator`` (a future slice) depends on this ``Protocol``, not on
``ClangdAdapter`` directly, so that a semantic provider is always optional
and swappable: passing ``None`` (no semantic provider) or any other object
satisfying this shape both work identically from the orchestrator's point of
view.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    CallHierarchyItem,
    OutgoingCallsOutcome,
    PrepareCallHierarchyOutcome,
    ProviderCapabilities,
    ReferencesOutcome,
)


@runtime_checkable
class SemanticProvider(Protocol):
    """A best-effort, optional source of semantically-derived call/reference edges."""

    def capabilities(self) -> ProviderCapabilities:
        """Describe whether this provider is currently usable, without raising."""
        ...

    def prepare_call_hierarchy(
        self, relative_path: str, *, line: int, column: int
    ) -> PrepareCallHierarchyOutcome:
        """Resolve the callable symbol at a 1-based ``(line, column)`` position."""
        ...

    def outgoing_calls(self, item: CallHierarchyItem) -> OutgoingCallsOutcome:
        """List the calls made from within ``item``."""
        ...

    def references(
        self, relative_path: str, *, line: int, column: int
    ) -> ReferencesOutcome:
        """List references to the symbol at a 1-based ``(line, column)`` position."""
        ...
