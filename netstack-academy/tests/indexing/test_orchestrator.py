from __future__ import annotations

from pathlib import Path

import pytest

from netstack_academy.indexing.ctags_runner import CtagsRunResult
from netstack_academy.indexing.fallback_indexer import FallbackIndexResult
from netstack_academy.indexing.orchestrator import IndexOrchestrator
from netstack_academy.indexing.semantic.models import ProviderCapabilities
from netstack_academy.indexing.storage import IndexStorage


class _StubCtagsRunner:
    def __init__(self, result: CtagsRunResult) -> None:
        self.result = result
        self.call_count = 0

    def __call__(self, *args: object, **kwargs: object) -> CtagsRunResult:
        self.call_count += 1
        return self.result


class _StubFallbackIndexer:
    def __init__(self, result: FallbackIndexResult) -> None:
        self.result = result
        self.call_count = 0

    def __call__(self, *args: object, **kwargs: object) -> FallbackIndexResult:
        self.call_count += 1
        return self.result


class _StubSemanticProvider:
    def __init__(self, available: bool = False, reason: str | None = "clangd not installed") -> None:
        self._available = available
        self._reason = reason

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_name="clangd", available=self._available, reason=self._reason
        )

    def prepare_call_hierarchy(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("not exercised in orchestrator tests")

    def outgoing_calls(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("not exercised in orchestrator tests")

    def references(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("not exercised in orchestrator tests")


def _empty_ctags_result(status: str = "ok") -> CtagsRunResult:
    return CtagsRunResult(status=status, definitions=[], diagnostics=[])


def _empty_fallback_result() -> FallbackIndexResult:
    return FallbackIndexResult(symbols=[], edges=[], diagnostics=[])


def test_ensure_index_reindexes_on_first_run(
    tmp_path: Path, git_repository: Path
) -> None:
    storage = IndexStorage.open(tmp_path / "index.sqlite3")
    ctags = _StubCtagsRunner(_empty_ctags_result())
    fallback = _StubFallbackIndexer(_empty_fallback_result())
    orchestrator = IndexOrchestrator(
        git_repository,
        storage,
        ctags_runner=ctags,
        fallback_indexer=fallback,
        semantic_provider=_StubSemanticProvider(),
    )

    result = orchestrator.ensure_index()

    assert result.status == "reindexed"
    assert storage.current_head() == result.head
    storage.close()


def test_ensure_index_reuses_database_when_head_unchanged(
    tmp_path: Path, git_repository: Path
) -> None:
    storage = IndexStorage.open(tmp_path / "index.sqlite3")
    ctags = _StubCtagsRunner(_empty_ctags_result())
    fallback = _StubFallbackIndexer(_empty_fallback_result())
    orchestrator = IndexOrchestrator(
        git_repository,
        storage,
        ctags_runner=ctags,
        fallback_indexer=fallback,
        semantic_provider=_StubSemanticProvider(),
    )

    first = orchestrator.ensure_index()
    second = orchestrator.ensure_index()

    assert first.status == "reindexed"
    assert second.status == "reused"
    assert second.head == first.head
    assert ctags.call_count == 1
    assert fallback.call_count == 1
    storage.close()


def test_ensure_index_falls_back_when_ctags_unavailable(
    tmp_path: Path, git_repository: Path
) -> None:
    storage = IndexStorage.open(tmp_path / "index.sqlite3")
    ctags = _StubCtagsRunner(_empty_ctags_result(status="unavailable"))
    fallback = _StubFallbackIndexer(_empty_fallback_result())
    orchestrator = IndexOrchestrator(
        git_repository,
        storage,
        ctags_runner=ctags,
        fallback_indexer=fallback,
        semantic_provider=_StubSemanticProvider(),
    )

    result = orchestrator.ensure_index()

    assert result.status == "reindexed"
    assert fallback.call_count == 1
    diagnostics_by_provider = {d.provider_name: d for d in result.provider_diagnostics}
    assert diagnostics_by_provider["ctags"].available is False


def test_ensure_index_records_semantic_provider_capabilities(
    tmp_path: Path, git_repository: Path
) -> None:
    storage = IndexStorage.open(tmp_path / "index.sqlite3")
    orchestrator = IndexOrchestrator(
        git_repository,
        storage,
        ctags_runner=_StubCtagsRunner(_empty_ctags_result()),
        fallback_indexer=_StubFallbackIndexer(_empty_fallback_result()),
        semantic_provider=_StubSemanticProvider(available=False, reason="clangd not installed"),
    )

    result = orchestrator.ensure_index()

    diagnostics_by_provider = {d.provider_name: d for d in result.provider_diagnostics}
    assert diagnostics_by_provider["clangd"].available is False
    assert diagnostics_by_provider["clangd"].reason == "clangd not installed"


def test_ensure_index_reports_failure_for_unavailable_repository(
    tmp_path: Path,
) -> None:
    storage = IndexStorage.open(tmp_path / "index.sqlite3")
    missing_repo = tmp_path / "does-not-exist"
    orchestrator = IndexOrchestrator(
        missing_repo,
        storage,
        ctags_runner=_StubCtagsRunner(_empty_ctags_result()),
        fallback_indexer=_StubFallbackIndexer(_empty_fallback_result()),
        semantic_provider=_StubSemanticProvider(),
    )

    result = orchestrator.ensure_index()

    assert result.status == "failed"
    assert result.reason is not None
    assert storage.current_head() is None
    storage.close()


def test_ensure_index_leaves_last_good_index_intact_when_reindexing_raises(
    tmp_path: Path, two_commit_git_repo: tuple[Path, str, str]
) -> None:
    repo, first_head, second_head = two_commit_git_repo
    storage = IndexStorage.open(tmp_path / "index.sqlite3")

    good_ctags = _StubCtagsRunner(_empty_ctags_result())
    good_fallback = _StubFallbackIndexer(_empty_fallback_result())
    orchestrator = IndexOrchestrator(
        repo,
        storage,
        ctags_runner=good_ctags,
        fallback_indexer=good_fallback,
        semantic_provider=_StubSemanticProvider(),
    )
    first_result = orchestrator.ensure_index()
    assert first_result.status == "reindexed"
    assert storage.current_head() == first_head

    def _raise(*args: object, **kwargs: object) -> FallbackIndexResult:
        raise RuntimeError("simulated indexing failure")

    broken_orchestrator = IndexOrchestrator(
        repo,
        storage,
        ctags_runner=_StubCtagsRunner(_empty_ctags_result()),
        fallback_indexer=_raise,
        semantic_provider=_StubSemanticProvider(),
    )

    second_result = broken_orchestrator.ensure_index()

    assert second_result.status == "failed"
    assert storage.current_head() == first_head
    storage.close()


def test_ensure_index_reindexes_when_head_changes(
    tmp_path: Path, two_commit_git_repo: tuple[Path, str, str]
) -> None:
    repo, first_head, second_head = two_commit_git_repo
    storage = IndexStorage.open(tmp_path / "index.sqlite3")
    orchestrator = IndexOrchestrator(
        repo,
        storage,
        ctags_runner=_StubCtagsRunner(_empty_ctags_result()),
        fallback_indexer=_StubFallbackIndexer(_empty_fallback_result()),
        semantic_provider=_StubSemanticProvider(),
    )

    first_result = orchestrator.ensure_index()
    assert first_result.head == first_head

    import subprocess

    subprocess.run(
        ["git", "checkout", second_head],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    second_result = orchestrator.ensure_index()

    assert second_result.status == "reindexed"
    assert second_result.head == second_head
    assert storage.current_head() == second_head
    storage.close()
