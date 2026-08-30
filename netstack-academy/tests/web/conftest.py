"""Fixtures for the web slice.

Everything is composed by hand and injected: a real curriculum loaded from
real Markdown, a real learning database, a real symbol index over a real git
repository, and real :class:`~netstack_academy.settings.Settings`. That is
the point of building the app from an explicit context object rather than
from process-wide state -- a test can serve a three-lesson course over a
six-symbol index in milliseconds, and two tests can disagree about the
editor scheme without fighting over environment variables.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from netstack_academy.curriculum.loader import load_curriculum
from netstack_academy.indexing.service import IndexService
from netstack_academy.indexing.storage import IndexStorage
from netstack_academy.learning.store import LearningStore
from netstack_academy.settings import Settings
from netstack_academy.web.app import create_web_app
from netstack_academy.web.context import AcademyContext

from academy_content import write_academy_content
from index_fixtures import index_kernel_repo, write_kernel_repo
from web_fakes import FakeClock, RecordingOrchestrator, reused_result


@pytest.fixture
def kernel_repo(tmp_path: Path) -> Path:
    """A tiny real git repository standing in for the kernel tree."""
    repo = tmp_path / "kernel"
    repo.mkdir()
    write_kernel_repo(repo)
    return repo


@pytest.fixture
def kernel_head(kernel_repo: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=kernel_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.fixture
def content_root(tmp_path: Path) -> Path:
    return write_academy_content(tmp_path / "content")


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    """Where the learner's state and the index live: never inside the repo."""
    return tmp_path / "state"


@pytest.fixture
def settings(kernel_repo: Path, content_root: Path, state_dir: Path) -> Settings:
    return Settings(
        kernel_repo=kernel_repo,
        editor_scheme="cursor",
        wsl_distro="Ubuntu",
        content_root=content_root,
        state_dir=state_dir,
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(state_dir: Path, clock: FakeClock):
    with LearningStore.open(state_dir / "learning.sqlite3", clock=clock) as opened:
        yield opened


@pytest.fixture
def curriculum(content_root: Path):
    result = load_curriculum(content_root, strict=True)
    assert result.curriculum is not None
    return result.curriculum


@pytest.fixture
def index_storage(state_dir: Path):
    with IndexStorage.open(state_dir / "index.sqlite3") as opened:
        yield opened


@pytest.fixture
def indexed_generation(index_storage: IndexStorage, kernel_head: str):
    """The persisted symbol generation, committed at the repository's HEAD."""
    return index_kernel_repo(index_storage, head=kernel_head)


@pytest.fixture
def orchestrator(kernel_head: str, indexed_generation) -> RecordingOrchestrator:
    return RecordingOrchestrator(
        reused_result(
            kernel_head,
            symbol_count=indexed_generation.symbol_count,
            edge_count=indexed_generation.edge_count,
        )
    )


@pytest.fixture
def index_service(
    kernel_repo: Path,
    index_storage: IndexStorage,
    orchestrator: RecordingOrchestrator,
) -> IndexService:
    return IndexService(kernel_repo, index_storage, orchestrator)


@pytest.fixture
def context(
    settings: Settings,
    curriculum,
    store: LearningStore,
    index_service: IndexService,
) -> AcademyContext:
    return AcademyContext.build(
        settings=settings,
        curriculum=curriculum,
        store=store,
        index_service=index_service,
    )


@pytest.fixture
def client(context: AcademyContext) -> TestClient:
    return TestClient(create_web_app(context))


@pytest.fixture
def make_client(
    settings: Settings,
    curriculum,
    store: LearningStore,
    kernel_repo: Path,
    index_storage: IndexStorage,
    orchestrator: RecordingOrchestrator,
    indexed_generation,
):
    """Build a client with a different editor scheme or orchestrator.

    Returns a callable taking ``orchestrator=`` plus any
    :class:`~netstack_academy.settings.Settings` field as a keyword override.
    """

    def _make(*, orchestrator_override=None, **settings_overrides) -> TestClient:
        used_settings = (
            settings.model_copy(update=settings_overrides)
            if settings_overrides
            else settings
        )
        used_orchestrator = (
            orchestrator_override if orchestrator_override is not None else orchestrator
        )
        service = IndexService(kernel_repo, index_storage, used_orchestrator)
        built = AcademyContext.build(
            settings=used_settings,
            curriculum=curriculum,
            store=store,
            index_service=service,
        )
        return TestClient(create_web_app(built))

    return _make
