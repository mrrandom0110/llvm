"""Composition root for one indexing run that owns its semantic provider.

A live semantic provider is a real ``clangd`` process (see
``semantic/factory.py``): whoever starts it must stop it. ``IndexOrchestrator``
deliberately does not decide *when* to start one on its own behalf -- that
policy question (start it at all? start it now, or only once a reindex is
actually happening?) belongs one layer up, in this module.

:func:`run_indexing_session` takes a *factory* rather than a provider
instance so that the underlying ``clangd`` process is never started for a
run that turns out to be a no-op: the factory is forwarded to
``IndexOrchestrator`` unchanged, which calls it itself, lazily, from inside
``_reindex`` -- i.e. only once the persisted-``HEAD`` reuse check has
already decided a real reindex is happening, and at most once per session.
A ``"reused"`` run therefore never spawns ``clangd`` at all. Either way,
the session still owns what gets created: ``orchestrator.close()`` in a
``finally`` closes whatever provider ended up on the orchestrator --
lazily created or (on the reused path) never created at all -- after
enrichment on the success path, and equally when the run reports
``"failed"``. Because ``IndexOrchestrator.ensure_index`` already converts
every pipeline failure into a ``"failed"`` result rather than an exception,
the previously indexed generation is left intact in both cases.
"""

from __future__ import annotations

from pathlib import Path

from .ctags_runner import run_ctags as _default_ctags_runner
from .fallback_indexer import index_fallback as _default_fallback_indexer
from .orchestrator import (
    DEFAULT_SEMANTIC_SYMBOL_LIMIT,
    UNSET_SYMBOL_LIMIT,
    CtagsRunnerCallable,
    FallbackIndexerCallable,
    IndexOrchestrator,
    IndexRunResult,
    SemanticProviderFactory,
    UnsetType,
)
from .storage import IndexStorage

# ``DEFAULT_SEMANTIC_SYMBOL_LIMIT``, ``SemanticProviderFactory``,
# ``UnsetType``, and ``UNSET_SYMBOL_LIMIT`` are re-exported (via the import
# above) so callers/tests can reference them as, e.g.,
# ``composition.DEFAULT_SEMANTIC_SYMBOL_LIMIT`` without reaching into
# :mod:`.orchestrator` directly -- this is the composition root real callers
# go through, so its own public names are the ones worth depending on.


def run_indexing_session(
    kernel_repo: Path,
    storage: IndexStorage,
    *,
    semantic_provider_factory: SemanticProviderFactory,
    ctags_runner: CtagsRunnerCallable = _default_ctags_runner,
    fallback_indexer: FallbackIndexerCallable = _default_fallback_indexer,
    semantic_symbol_limit: int | None | UnsetType = UNSET_SYMBOL_LIMIT,
    force: bool = False,
) -> IndexRunResult:
    """Index ``kernel_repo`` once, starting a semantic provider only if needed.

    ``semantic_provider_factory`` is handed to :class:`IndexOrchestrator`
    verbatim rather than called here: the orchestrator itself decides
    whether it is ever invoked, lazily and at most once, from inside its
    own reindex path. A session whose persisted index already matches the
    repository's current ``HEAD`` (``force=False``, the default) therefore
    never starts ``clangd`` at all.

    ``force`` mirrors :meth:`IndexOrchestrator.ensure_index`'s own
    keyword-only flag: ``True`` always reruns the full collection pipeline
    -- and, on that path, still starts the semantic provider lazily rather
    than up front -- even though the persisted ``HEAD`` already matches.
    This is the composition root's own explicit "refresh now" escape
    hatch, so a caller does not need to reach past ``run_indexing_session``
    into ``IndexOrchestrator`` directly to force a rerun.

    ``semantic_symbol_limit`` defaults to :data:`DEFAULT_SEMANTIC_SYMBOL_LIMIT`
    when omitted -- this is the composition root real callers go through, so
    an unbounded-by-default enrichment budget would be a footgun. Passing
    ``semantic_symbol_limit=None`` explicitly remains a supported opt-in for
    unbounded enrichment (e.g. a small tree, or a deliberately patient batch
    run); it is forwarded to :class:`IndexOrchestrator` verbatim, distinct
    from simply omitting the argument.
    """
    repo = Path(kernel_repo)

    resolved_symbol_limit = (
        DEFAULT_SEMANTIC_SYMBOL_LIMIT
        if semantic_symbol_limit is UNSET_SYMBOL_LIMIT
        else semantic_symbol_limit
    )

    orchestrator = IndexOrchestrator(
        repo,
        storage,
        ctags_runner=ctags_runner,
        fallback_indexer=fallback_indexer,
        semantic_provider_factory=semantic_provider_factory,
        semantic_symbol_limit=resolved_symbol_limit,
    )
    # ``force`` is only ever passed through when actually requested, rather
    # than always forwarding ``force=force``: ``IndexOrchestrator.ensure_index``
    # already defaults to ``force=False``, so an omitted-``force`` call
    # behaves identically either way for the real orchestrator, while still
    # working against any test double whose ``ensure_index()`` predates
    # this parameter and does not accept it at all.
    ensure_index_kwargs = {"force": True} if force else {}
    try:
        return orchestrator.ensure_index(**ensure_index_kwargs)
    finally:
        orchestrator.close()
