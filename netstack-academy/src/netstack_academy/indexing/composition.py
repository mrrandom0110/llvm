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
    CtagsRunnerCallable,
    FallbackIndexerCallable,
    IndexOrchestrator,
    IndexRunResult,
)
from .semantic.provider import SemanticProvider
from .storage import IndexStorage

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
    semantic_symbol_limit: int | None = None,
) -> IndexRunResult:
    """Index ``kernel_repo`` once with a freshly created semantic provider."""
    repo = Path(kernel_repo)
    provider = semantic_provider_factory(repo)

    orchestrator = IndexOrchestrator(
        repo,
        storage,
        ctags_runner=ctags_runner,
        fallback_indexer=fallback_indexer,
        semantic_provider=provider,
        semantic_symbol_limit=semantic_symbol_limit,
    )
    try:
        return orchestrator.ensure_index()
    finally:
        orchestrator.close()
