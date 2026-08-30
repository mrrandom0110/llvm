"""Per-process laziness in front of the symbol index.

Building a symbol index over a kernel tree is minutes of ``ctags`` and
``clangd``, so the web layer never does it merely because a page was
rendered. :class:`LazyIndex` makes that policy explicit and keeps it in one
place: reads that only need what is already persisted go straight through,
and the collection pipeline runs at most once per process -- when something
actually needs symbols, or when a learner asks for it after a ``git pull``.

Three consequences are worth stating, because each is a behaviour a caller
can observe:

**A failure is not an answer.** If ``ctags`` was missing on the first
attempt, ``ensured`` stays ``False`` and the next request that needs symbols
tries again, rather than serving an empty index for the life of the process.

**The last run is remembered.** Which providers were available is only
knowable from the run that consulted them, and it is what explains a call
graph with no semantic edges. Nothing else in the system keeps that.

**Status is free.** :meth:`status` reads the persisted generation out of
storage and cannot trigger a run, which is what lets a dashboard say "your
index is one commit behind" without becoming the thing that rebuilds it.
"""

from __future__ import annotations

from collections.abc import Sequence

from netstack_academy.indexing.orchestrator import IndexRunResult
from netstack_academy.indexing.service import IndexService, IndexStatusView, SymbolView


class LazyIndex:
    """A once-per-process ``ensure`` in front of an :class:`IndexService`."""

    def __init__(self, service: IndexService) -> None:
        self._service = service
        self._ensured = False
        self._last_result: IndexRunResult | None = None

    @property
    def service(self) -> IndexService:
        """The wrapped service, for the reads that need no ensuring."""
        return self._service

    @property
    def ensured(self) -> bool:
        """Whether a run has already succeeded in this process."""
        return self._ensured

    @property
    def last_result(self) -> IndexRunResult | None:
        """The most recent run, or ``None`` when none has happened yet."""
        return self._last_result

    def status(self) -> IndexStatusView:
        """What is persisted right now. Never triggers a run."""
        return self._service.get_status()

    def ensure(self, *, force: bool = False) -> IndexRunResult:
        """Make sure symbols are available, running the pipeline if needed.

        Without ``force`` this is a no-op once a run has succeeded: the
        orchestrator underneath would itself decide to reuse a matching
        persisted generation, but it has to inspect the repository's ``HEAD``
        to find that out, and paying for that on every request that touches a
        symbol is what this memo exists to avoid.

        ``force=True`` always reruns -- it is the "I just pulled" button --
        and re-arms the memo from the result.
        """
        if self._ensured and not force:
            assert self._last_result is not None  # set together with _ensured
            return self._last_result

        result = (
            self._service.force_reindex() if force else self._service.ensure_index()
        )
        self._last_result = result
        self._ensured = result.status != "failed"
        return result

    def search_symbols(self, query: str, *, limit: int) -> Sequence[SymbolView]:
        """Search, ensuring the index first.

        This is the whole of what
        :class:`~netstack_academy.learning.services.LearningService` wants
        from a symbol index, so a ``LazyIndex`` can be handed to it directly
        and a combined lesson/symbol search inherits the same laziness.
        """
        self.ensure()
        return self._service.search_symbols(query, limit=limit)
