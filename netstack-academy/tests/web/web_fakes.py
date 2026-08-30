"""Test doubles for the web slice: a clock, an orchestrator, a server runner.

Everything else these tests touch is real -- a real git repository, a real
curriculum on disk, a real learning database, a real symbol index. Only
three things are faked, and each for the same reason: a test cannot
otherwise control it.

``FakeClock`` makes "due tomorrow" assertable. ``RecordingOrchestrator``
stands in for the collection pipeline behind
:class:`~netstack_academy.indexing.service.IndexService`, because the real
one shells out to ``ctags``/``clangd`` -- and, more importantly, because
*how many times it is called* is the whole point of the laziness contract.
``RecordingServer`` stands in for ``uvicorn.run``, which would otherwise
block forever inside a test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from netstack_academy.indexing.orchestrator import IndexRunResult, ProviderDiagnostic


class FakeClock:
    """A monotonic, manually advanced UTC clock."""

    def __init__(self, start: datetime | None = None) -> None:
        self.now = start if start is not None else datetime(
            2026, 3, 1, 12, 0, tzinfo=timezone.utc
        )

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **timedelta_kwargs: float) -> datetime:
        self.now = self.now + timedelta(**timedelta_kwargs)
        return self.now


def reused_result(
    head: str, *, symbol_count: int = 0, edge_count: int = 0
) -> IndexRunResult:
    """What the orchestrator reports when the persisted index still matches."""
    return IndexRunResult(
        status="reused",
        head=head,
        symbol_count=symbol_count,
        edge_count=edge_count,
        provider_diagnostics=(
            ProviderDiagnostic(provider_name="ctags", available=True),
            ProviderDiagnostic(provider_name="clangd", available=True),
        ),
    )


def reindexed_result(
    head: str, *, symbol_count: int = 0, edge_count: int = 0
) -> IndexRunResult:
    return IndexRunResult(
        status="reindexed",
        head=head,
        symbol_count=symbol_count,
        edge_count=edge_count,
        provider_diagnostics=(
            ProviderDiagnostic(provider_name="ctags", available=True),
            ProviderDiagnostic(
                provider_name="clangd",
                available=False,
                reason="clangd executable not found",
            ),
        ),
        diagnostics=("clangd references timeout at net/core/dev.c:12",),
    )


def failed_result(reason: str = "Repository path not found") -> IndexRunResult:
    return IndexRunResult(
        status="failed",
        head=None,
        symbol_count=0,
        edge_count=0,
        provider_diagnostics=(),
        reason=reason,
    )


@dataclass
class RecordingOrchestrator:
    """The narrow ``ensure_index(force=...)`` contract ``IndexService`` needs.

    ``forces`` records one entry per call, so a test can prove both *that*
    the pipeline ran and *how* it was asked to run.
    """

    result: IndexRunResult
    forces: list[bool] = field(default_factory=list)

    def ensure_index(self, *, force: bool = False) -> IndexRunResult:
        self.forces.append(force)
        return self.result

    @property
    def call_count(self) -> int:
        return len(self.forces)


class ExplodingOrchestrator:
    """An orchestrator that must never run.

    Used by the tests that pin which requests are allowed to trigger
    indexing: a page that quietly kicks off a kernel-sized reindex is a
    performance bug that only shows up in production.
    """

    def ensure_index(self, *, force: bool = False) -> IndexRunResult:
        raise AssertionError(
            f"indexing must not be triggered by this request (force={force!r})"
        )


@dataclass
class RecordingSessionRunner:
    """Stands in for ``indexing.composition.run_indexing_session``."""

    result: IndexRunResult
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self,
        kernel_repo: Path,
        storage: Any,
        *,
        force: bool = False,
        **kwargs: Any,
    ) -> IndexRunResult:
        self.calls.append(
            {
                "kernel_repo": Path(kernel_repo),
                "storage": storage,
                "force": force,
                "kwargs": dict(kwargs),
            }
        )
        return self.result

    @property
    def call_count(self) -> int:
        return len(self.calls)


@dataclass
class RecordingServer:
    """Stands in for ``uvicorn.run``: records the bind address and returns."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, app: Any, *, host: str, port: int) -> None:
        self.calls.append({"app": app, "host": host, "port": port})

    @property
    def call_count(self) -> int:
        return len(self.calls)


class ExplodingServer:
    """A server runner that must never be reached."""

    def __call__(self, app: Any, *, host: str, port: int) -> None:
        raise AssertionError(f"the server must not be started on {host}:{port}")
