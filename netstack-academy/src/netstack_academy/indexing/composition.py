"""Composition root for one indexing run that owns its semantic provider.

A live semantic provider is a real ``clangd`` process (see
``semantic/factory.py``): whoever starts it must stop it. ``IndexOrchestrator``
deliberately does not, because an injected provider may outlive many
``ensure_index()`` calls -- so something above it has to own that lifetime.
That is this module.

:func:`run_indexing_session` takes a *factory* rather than a provider
instance precisely so the session owns what it creates: it starts the
provider, indexes once, and closes the provider in a ``finally`` -- after
enrichment on the success path, and equally when the run reports
``"failed"``. Because ``IndexOrchestrator.ensure_index`` already converts
every pipeline failure into a ``"failed"`` result rather than an exception,
the previously indexed generation is left intact in both cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .ctags_runner import run_ctags as _default_ctags_runner
from .fallback_indexer import index_fallback as _default_fallback_indexer
from .orchestrator import (
    DEFAULT_SEMANTIC_SYMBOL_LIMIT,
    CtagsRunnerCallable,
    FallbackIndexerCallable,
    IndexOrchestrator,
    IndexRunResult,
    _Unset,
    _UNSET_SYMBOL_LIMIT,
)
from .semantic.provider import SemanticProvider
from .storage import IndexStorage

#: Re-exported (via the import above) so callers/tests can reference the
#: composition root's own default as ``composition.DEFAULT_SEMANTIC_SYMBOL_LIMIT``
#: without reaching into :mod:`.orchestrator` directly.

#: Called with the kernel repository to index; must not raise for a missing
#: or unstartable provider (``create_clangd_provider`` returns an
#: unavailable provider instead), so a session never fails to start.
SemanticProviderFactory = Callable[[Path], SemanticProvider]


def run_indexing_session(
    kernel_repo: Path,
    storage: IndexStorage,
    *,
    semantic_provider_factory: SemanticProviderFactory,
    ctags_runner: CtagsRunnerCallable = _default_ctags_runner,
    fallback_indexer: FallbackIndexerCallable = _default_fallback_indexer,
    semantic_symbol_limit: int | None | _Unset = _UNSET_SYMBOL_LIMIT,
) -> IndexRunResult:
    """Index ``kernel_repo`` once with a freshly created semantic provider.

    ``semantic_symbol_limit`` defaults to :data:`DEFAULT_SEMANTIC_SYMBOL_LIMIT`
    when omitted -- this is the composition root real callers go through, so
    an unbounded-by-default enrichment budget would be a footgun. Passing
    ``semantic_symbol_limit=None`` explicitly remains a supported opt-in for
    unbounded enrichment (e.g. a small tree, or a deliberately patient batch
    run); it is forwarded to :class:`IndexOrchestrator` verbatim, distinct
    from simply omitting the argument.
    """
    repo = Path(kernel_repo)
    provider = semantic_provider_factory(repo)

    resolved_symbol_limit = (
        DEFAULT_SEMANTIC_SYMBOL_LIMIT
        if semantic_symbol_limit is _UNSET_SYMBOL_LIMIT
        else semantic_symbol_limit
    )

    orchestrator = IndexOrchestrator(
        repo,
        storage,
        ctags_runner=ctags_runner,
        fallback_indexer=fallback_indexer,
        semantic_provider=provider,
        semantic_symbol_limit=resolved_symbol_limit,
    )
    try:
        return orchestrator.ensure_index()
    finally:
        orchestrator.close()
