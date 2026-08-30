"""RED tests for lazy semantic-provider startup (Task 2, final finding).

Today, ``run_indexing_session`` calls ``semantic_provider_factory(repo)``
*unconditionally*, before ``IndexOrchestrator`` has any chance to decide
whether the persisted index even needs reindexing:

    provider = semantic_provider_factory(repo)
    orchestrator = IndexOrchestrator(..., semantic_provider=provider, ...)
    return orchestrator.ensure_index()

So every call to ``run_indexing_session`` spawns/initializes a real
``clangd`` session (via ``create_clangd_provider``) even when
``storage.current_head()`` already equals the repository's current
``HEAD`` and ``ensure_index()`` is about to report ``"reused"`` without
touching any provider at all. That is wasted process spawning and startup
handshaking on every poll of an unchanged tree.

The fix this module specifies: give ``IndexOrchestrator`` an optional
``semantic_provider_factory`` constructor argument that it calls itself,
lazily, from inside ``_reindex`` -- i.e. only once the persisted-``HEAD``
reuse check has already decided a real reindex is happening. Direct
``semantic_provider`` injection (today's contract, exercised extensively by
``test_semantic_enrichment.py`` and ``test_orchestrator.py``) must keep
working unchanged; passing both a provider *and* a factory at once is
ambiguous and must be rejected. ``run_indexing_session`` then forwards its
factory straight to the orchestrator instead of calling it itself, and
gains a ``force`` passthrough so an explicit "refresh now" caller can still
reach the always-reindex path (and, on that path, still starts the
provider lazily -- exactly once -- rather than up front).

None of this is implemented yet: every test below is RED against the
current ``orchestrator.py``/``composition.py`` on this branch. No
production code is modified by this change.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from netstack_academy.indexing.composition import run_indexing_session
from netstack_academy.indexing.ctags_runner import CtagsRunResult
from netstack_academy.indexing.fallback_indexer import FallbackIndexResult
from netstack_academy.indexing.orchestrator import IndexOrchestrator
from netstack_academy.indexing.semantic.models import ProviderCapabilities
from netstack_academy.indexing.storage import IndexStorage


def _ok_ctags(*args: object, **kwargs: object) -> CtagsRunResult:
    return CtagsRunResult(status="ok", definitions=[], diagnostics=[])


def _ok_fallback(*args: object, **kwargs: object) -> FallbackIndexResult:
    return FallbackIndexResult(symbols=[], edges=[], diagnostics=[])


def _exploding_ctags(*args: object, **kwargs: object) -> CtagsRunResult:
    raise AssertionError(
        "ctags_runner must not run when the persisted index is reused"
    )


def _exploding_fallback(*args: object, **kwargs: object) -> FallbackIndexResult:
    raise AssertionError(
        "fallback_indexer must not run when the persisted index is reused"
    )


class _CountingSemanticProvider:
    """A minimal ``SemanticProvider`` double that only ever reports itself
    unavailable -- enrichment specifics are already covered by
    ``test_semantic_enrichment.py``; this module only cares *whether* and
    *how many times* a provider gets created and closed.
    """

    def __init__(self) -> None:
        self.close_calls = 0

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_name="clangd", available=False, reason="stub: unavailable"
        )

    def prepare_call_hierarchy(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("not exercised: provider reports unavailable")

    def outgoing_calls(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("not exercised: provider reports unavailable")

    def references(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("not exercised: provider reports unavailable")

    def close(self) -> None:
        self.close_calls += 1


class _RecordingFactory:
    """Records every call and the provider(s) it minted, so a test can
    assert both "called exactly once" and "never called".
    """

    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.created: list[_CountingSemanticProvider] = []

    def __call__(self, kernel_repo: Path) -> _CountingSemanticProvider:
        self.calls.append(Path(kernel_repo))
        provider = _CountingSemanticProvider()
        self.created.append(provider)
        return provider

    @property
    def call_count(self) -> int:
        return len(self.calls)


class _ExplodingFactory:
    """A factory double that fails the test immediately if ever invoked --
    the strongest possible assertion that reuse never reaches provider
    creation at all.
    """

    def __call__(self, kernel_repo: Path) -> _CountingSemanticProvider:
        raise AssertionError(
            "semantic_provider_factory must not be called when the "
            "persisted index is reused"
        )


# -- IndexOrchestrator: optional, lazy provider factory ----------------------


def test_orchestrator_accepts_a_semantic_provider_factory_and_does_not_call_it_eagerly(
    tmp_path: Path, git_repository: Path
) -> None:
    """Merely constructing the orchestrator with a factory must not create
    a provider -- creation happens lazily, only once a reindex actually
    runs.
    """
    storage = IndexStorage.open(tmp_path / "index.sqlite3")
    factory = _RecordingFactory()

    IndexOrchestrator(
        git_repository,
        storage,
        ctags_runner=_ok_ctags,
        fallback_indexer=_ok_fallback,
        semantic_provider_factory=factory,
    )

    assert factory.call_count == 0
    storage.close()


def test_orchestrator_creates_the_factory_provider_exactly_once_while_reindexing(
    tmp_path: Path, git_repository: Path
) -> None:
    storage = IndexStorage.open(tmp_path / "index.sqlite3")
    factory = _RecordingFactory()
    orchestrator = IndexOrchestrator(
        git_repository,
        storage,
        ctags_runner=_ok_ctags,
        fallback_indexer=_ok_fallback,
        semantic_provider_factory=factory,
    )

    result = orchestrator.ensure_index()

    assert result.status == "reindexed"
    assert factory.call_count == 1
    assert factory.calls[0] == Path(git_repository)
    storage.close()


def test_orchestrator_never_calls_provider_factory_when_persisted_head_already_matches(
    tmp_path: Path, git_repository: Path
) -> None:
    """The central claim of this finding, at the ``IndexOrchestrator``
    level: a brand new orchestrator instance pointed at storage whose
    persisted ``HEAD`` already matches the repository must report
    ``"reused"`` without ever touching ctags, the fallback indexer, or the
    semantic provider factory -- mirroring
    ``test_ensure_index_reuses_persisted_head_across_new_orchestrator_instance``
    in ``test_orchestrator.py``, but for the new lazy-factory constructor
    argument instead of a directly injected provider.
    """
    db_path = tmp_path / "index.sqlite3"
    storage = IndexStorage.open(db_path)
    seeding_factory = _RecordingFactory()
    seeding_orchestrator = IndexOrchestrator(
        git_repository,
        storage,
        ctags_runner=_ok_ctags,
        fallback_indexer=_ok_fallback,
        semantic_provider_factory=seeding_factory,
    )
    first_result = seeding_orchestrator.ensure_index()
    assert first_result.status == "reindexed"
    assert seeding_factory.call_count == 1
    seeding_orchestrator.close()
    storage.close()

    reopened_storage = IndexStorage.open(db_path)
    exploding_factory = _ExplodingFactory()
    reused_orchestrator = IndexOrchestrator(
        git_repository,
        reopened_storage,
        ctags_runner=_exploding_ctags,
        fallback_indexer=_exploding_fallback,
        semantic_provider_factory=exploding_factory,
    )

    result = reused_orchestrator.ensure_index()

    assert result.status == "reused"
    assert result.head == first_result.head
    reopened_storage.close()


def test_orchestrator_rejects_simultaneous_semantic_provider_and_factory(
    tmp_path: Path, git_repository: Path
) -> None:
    """Passing both a directly-injected provider and a factory is
    ambiguous -- which one wins is not specified anywhere, so it must be
    rejected at construction time rather than silently picking one.
    """
    storage = IndexStorage.open(tmp_path / "index.sqlite3")
    provider = _CountingSemanticProvider()
    factory = _RecordingFactory()

    with pytest.raises(ValueError):
        IndexOrchestrator(
            git_repository,
            storage,
            ctags_runner=_ok_ctags,
            fallback_indexer=_ok_fallback,
            semantic_provider=provider,
            semantic_provider_factory=factory,
        )

    assert factory.call_count == 0
    storage.close()


def test_orchestrator_still_supports_direct_semantic_provider_injection_without_a_factory(
    tmp_path: Path, git_repository: Path
) -> None:
    """Regression guard (green today, must stay green): a directly
    injected ``semantic_provider`` -- the existing, extensively tested
    contract in ``test_semantic_enrichment.py`` -- must keep working
    unchanged once ``semantic_provider_factory`` exists as a sibling
    argument.
    """
    storage = IndexStorage.open(tmp_path / "index.sqlite3")
    provider = _CountingSemanticProvider()
    orchestrator = IndexOrchestrator(
        git_repository,
        storage,
        ctags_runner=_ok_ctags,
        fallback_indexer=_ok_fallback,
        semantic_provider=provider,
    )

    result = orchestrator.ensure_index()

    assert result.status == "reindexed"
    diagnostics_by_provider = {
        diagnostic.provider_name: diagnostic
        for diagnostic in result.provider_diagnostics
    }
    assert diagnostics_by_provider["clangd"].available is False
    storage.close()


# -- run_indexing_session: no eager factory call, force passthrough ---------


def test_run_indexing_session_never_calls_provider_factory_when_index_is_reused(
    tmp_path: Path, git_repository: Path
) -> None:
    """The composition root's version of the same claim: today,
    ``run_indexing_session`` calls ``semantic_provider_factory(repo)``
    unconditionally at the top of the function, before ``IndexOrchestrator``
    is even constructed -- so it spawns/initializes ``clangd`` even when the
    persisted index is about to be reused untouched. Once the factory is
    forwarded to ``IndexOrchestrator`` instead (see the orchestrator tests
    above), a second, same-``HEAD`` session must skip ctags, the fallback
    indexer, and the semantic provider factory entirely.
    """
    db_path = tmp_path / "index.sqlite3"
    storage = IndexStorage.open(db_path)
    seeding_factory = _RecordingFactory()

    first = run_indexing_session(
        git_repository,
        storage,
        semantic_provider_factory=seeding_factory,
        ctags_runner=_ok_ctags,
        fallback_indexer=_ok_fallback,
    )

    assert first.status == "reindexed"
    assert seeding_factory.call_count == 1

    exploding_factory = _ExplodingFactory()
    second = run_indexing_session(
        git_repository,
        storage,
        semantic_provider_factory=exploding_factory,
        ctags_runner=_exploding_ctags,
        fallback_indexer=_exploding_fallback,
    )

    assert second.status == "reused"
    assert second.head == first.head
    storage.close()


def test_run_indexing_session_exposes_force_to_bypass_persisted_reuse(
    tmp_path: Path, git_repository: Path
) -> None:
    """``run_indexing_session`` has no ``force`` parameter today, so a
    caller cannot ask for an explicit "refresh now" without reaching past
    the composition root into ``IndexOrchestrator`` directly. Exposing it
    here makes the composition root's own public API usable for that case.
    """
    storage = IndexStorage.open(tmp_path / "index.sqlite3")
    first = run_indexing_session(
        git_repository,
        storage,
        semantic_provider_factory=_RecordingFactory(),
        ctags_runner=_ok_ctags,
        fallback_indexer=_ok_fallback,
    )
    assert first.status == "reindexed"

    second = run_indexing_session(
        git_repository,
        storage,
        semantic_provider_factory=_RecordingFactory(),
        ctags_runner=_ok_ctags,
        fallback_indexer=_ok_fallback,
        force=True,
    )

    assert second.status == "reindexed"
    storage.close()


def test_run_indexing_session_force_still_lazily_starts_the_provider_exactly_once(
    tmp_path: Path, git_repository: Path
) -> None:
    """Combines the two claims above: even when the persisted ``HEAD``
    already matches (the case that must normally skip the factory
    entirely), ``force=True`` must still route through the lazy,
    exactly-once provider startup inside ``IndexOrchestrator._reindex`` --
    not call the factory before/outside of that reindex, and not call it
    more than once.
    """
    db_path = tmp_path / "index.sqlite3"
    storage = IndexStorage.open(db_path)
    seeding_factory = _RecordingFactory()

    first = run_indexing_session(
        git_repository,
        storage,
        semantic_provider_factory=seeding_factory,
        ctags_runner=_ok_ctags,
        fallback_indexer=_ok_fallback,
    )
    assert first.status == "reindexed"

    forced_factory = _RecordingFactory()
    forced_result = run_indexing_session(
        git_repository,
        storage,
        semantic_provider_factory=forced_factory,
        ctags_runner=_ok_ctags,
        fallback_indexer=_ok_fallback,
        force=True,
    )

    assert forced_result.status == "reindexed"
    assert forced_factory.call_count == 1
    assert forced_factory.created[0].close_calls == 1
    storage.close()


# Note: "closes the (now lazily created) provider exactly once on
# success/failure" is deliberately *not* re-tested here for
# ``run_indexing_session``. Under today's eager-factory implementation the
# provider already gets created and closed exactly once on both the success
# and failure paths -- ``test_composition_closes_the_provider_after_enrichment``
# and ``test_composition_closes_the_provider_when_reindexing_fails`` in
# ``test_semantic_enrichment.py`` already cover this and pass today (green),
# and nothing about making creation lazy changes that close-once contract, so
# adding a duplicate assertion here would not be RED -- it would just repeat
# already-covered, already-passing coverage.
