"""Production entry point for the optional ``clangd`` semantic provider.

This is the only module that decides *which* binary to launch: a fixed
executable name and a fixed argv list, both module constants, so no caller
can inject an arbitrary command (there is no argv or command-string
parameter to pass, and :class:`StdioLspTransport` refuses command strings
anyway).

The semantic provider is optional by design -- ``IndexOrchestrator`` treats
it as best-effort and only ever asks it for ``capabilities()`` -- so a
missing or unstartable ``clangd`` must never raise here. It produces an
:class:`UnavailableSemanticProvider` instead, which satisfies the same
``SemanticProvider`` shape and reports every operation as ``"unavailable"``.

Both providers returned by :func:`create_clangd_provider` expose ``close()``.
Callers that start a provider own it: the successful path holds a live
``clangd`` process until ``close()`` is called.
"""

from __future__ import annotations

from pathlib import Path

from .clangd_adapter import DEFAULT_TIMEOUT_SECONDS, ClangdAdapter
from .models import (
    CallHierarchyItem,
    OutgoingCallsOutcome,
    PrepareCallHierarchyOutcome,
    ProviderCapabilities,
    ReferencesOutcome,
)
from .provider import SemanticProvider
from .transport import StdioLspTransport, TransportClosedError

CLANGD_EXECUTABLE = "clangd"

#: The exact, fixed argv used to launch ``clangd``. Kept to long-stable
#: flags only: an unrecognized option makes clangd exit at startup, which
#: would silently degrade the whole semantic provider.
CLANGD_ARGV: tuple[str, ...] = (
    CLANGD_EXECUTABLE,
    "--background-index",
    "--pch-storage=memory",
    "--log=error",
)


class UnavailableSemanticProvider:
    """A ``SemanticProvider`` that reports every operation as unavailable.

    Used when ``clangd`` could not be started at all, so callers get the same
    typed outcomes they would get from a live provider that lost its server,
    instead of an exception or a ``None`` they must special-case.
    """

    def __init__(
        self, reason: str, *, provider_name: str = CLANGD_EXECUTABLE
    ) -> None:
        self._reason = reason
        self._provider_name = provider_name

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_name=self._provider_name, available=False, reason=self._reason
        )

    def prepare_call_hierarchy(
        self, relative_path: str, *, line: int, column: int
    ) -> PrepareCallHierarchyOutcome:
        return PrepareCallHierarchyOutcome(
            status="unavailable", items=(), reason=self._reason
        )

    def outgoing_calls(self, item: CallHierarchyItem) -> OutgoingCallsOutcome:
        return OutgoingCallsOutcome(
            status="unavailable", calls=(), reason=self._reason
        )

    def references(
        self, relative_path: str, *, line: int, column: int
    ) -> ReferencesOutcome:
        return ReferencesOutcome(
            status="unavailable", locations=(), reason=self._reason
        )

    def close(self) -> None:
        """Present so provider teardown is uniform; there is nothing to release."""


def create_clangd_provider(
    kernel_repo: Path, *, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> SemanticProvider:
    """Start ``clangd`` for ``kernel_repo``, or return an unavailable provider.

    Never raises: a missing binary, a binary that cannot be executed, or a
    handshake that does not complete all degrade to a provider reporting
    ``available=False`` with a reason.
    """
    repo = Path(kernel_repo)

    try:
        transport = StdioLspTransport(list(CLANGD_ARGV), cwd=repo)
    except TransportClosedError as exc:
        return UnavailableSemanticProvider(f"{CLANGD_EXECUTABLE} unavailable: {exc}")
    except OSError as exc:  # defensive: the transport already maps these
        return UnavailableSemanticProvider(f"{CLANGD_EXECUTABLE} unavailable: {exc}")

    return ClangdAdapter(transport, kernel_repo=repo, timeout=timeout)
