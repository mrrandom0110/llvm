"""Contract for :mod:`netstack_academy.web.runtime`.

The runtime is the production composition root: it decides where the
learner's database and the symbol index live, loads the course, wires the
services the app needs, and owns the shutdown of everything it opened.

Three properties are worth more than the rest.

**State never lands inside the kernel repository.** The index and the
learner's notes are this program's data, not the kernel's. Writing them
under the checkout would put untracked databases in ``git status``, risk
their loss to ``git clean``, and make ``HEAD`` -- the thing the index is
keyed on -- depend on files the index itself wrote.

**Indexing is lazy.** Building a symbol index over a kernel tree is minutes
of work, so it happens when something actually needs symbols, at most once
per process, and never merely because a page was rendered.

**Whatever the runtime opens, the runtime closes.** Two SQLite connections
and, on a real run, a ``clangd`` process.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import netstack_academy
from netstack_academy.curriculum.loader import CurriculumValidationError, load_curriculum
from netstack_academy.indexing.service import IndexService
from netstack_academy.settings import Settings
from netstack_academy.web.app import create_web_app
from netstack_academy.web.context import AcademyContext
from netstack_academy.web.index_access import LazyIndex
from netstack_academy.web.runtime import (
    INDEX_DB_NAME,
    LEARNING_DB_NAME,
    STATE_DIR_NAME,
    AcademyRuntime,
    StateDirectoryInsideRepositoryError,
    packaged_content_root,
    resolve_content_root,
    resolve_state_dir,
)

from academy_content import write_invalid_content
from web_fakes import (
    ExplodingOrchestrator,
    RecordingOrchestrator,
    RecordingSessionRunner,
    failed_result,
    reindexed_result,
    reused_result,
)


@pytest.fixture
def session_runner(kernel_head: str) -> RecordingSessionRunner:
    return RecordingSessionRunner(reused_result(kernel_head))


@pytest.fixture
def runtime(settings: Settings, session_runner: RecordingSessionRunner):
    with AcademyRuntime.open(settings, session_runner=session_runner) as opened:
        yield opened


# ----------------------------------------------------------------------
# Settings: where the runtime is told to put things
# ----------------------------------------------------------------------


def test_settings_read_the_state_directory_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("STATE_DIR", str(tmp_path / "elsewhere"))

    assert Settings().state_dir == tmp_path / "elsewhere"


def test_settings_read_the_content_root_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CONTENT_ROOT", str(tmp_path / "course"))

    assert Settings().content_root == tmp_path / "course"


def test_settings_leave_both_locations_unset_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset means "use the packaged course and the per-user state dir",
    which is what a learner who configured nothing should get.
    """
    monkeypatch.delenv("STATE_DIR", raising=False)
    monkeypatch.delenv("CONTENT_ROOT", raising=False)

    settings = Settings()

    assert settings.state_dir is None
    assert settings.content_root is None


# ----------------------------------------------------------------------
# State directory resolution
# ----------------------------------------------------------------------


def test_configured_state_directory_is_used_as_given(
    settings: Settings, state_dir: Path
) -> None:
    assert resolve_state_dir(settings) == state_dir.resolve()


def test_state_directory_expands_a_home_relative_path(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))

    resolved = resolve_state_dir(
        settings.model_copy(update={"state_dir": Path("~/academy-state")})
    )

    assert resolved == home / "academy-state"


def test_default_state_directory_follows_xdg_state_home(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))

    resolved = resolve_state_dir(settings.model_copy(update={"state_dir": None}))

    assert resolved == tmp_path / "xdg" / STATE_DIR_NAME


def test_default_state_directory_falls_back_to_local_state_under_home(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))

    resolved = resolve_state_dir(settings.model_copy(update={"state_dir": None}))

    assert resolved == home / ".local" / "state" / STATE_DIR_NAME


def test_resolving_the_state_directory_creates_nothing(
    settings: Settings, state_dir: Path
) -> None:
    resolve_state_dir(settings)

    assert not state_dir.exists()


def test_state_directory_inside_the_kernel_repository_is_refused(
    settings: Settings, kernel_repo: Path
) -> None:
    """The index is keyed on the repository's ``HEAD``; letting it live
    inside the repository makes the key depend on the thing it describes,
    and puts a multi-gigabyte database one ``git clean -xdf`` from deletion.
    """
    inside = settings.model_copy(update={"state_dir": kernel_repo / ".academy"})

    with pytest.raises(StateDirectoryInsideRepositoryError):
        resolve_state_dir(inside)


def test_refusing_a_state_directory_writes_nothing_into_the_repository(
    settings: Settings,
    kernel_repo: Path,
    session_runner: RecordingSessionRunner,
) -> None:
    before = sorted(path.name for path in kernel_repo.iterdir())
    inside = settings.model_copy(update={"state_dir": kernel_repo / ".academy"})

    with pytest.raises(StateDirectoryInsideRepositoryError):
        AcademyRuntime.open(inside, session_runner=session_runner)

    assert sorted(path.name for path in kernel_repo.iterdir()) == before


# ----------------------------------------------------------------------
# Content root resolution
# ----------------------------------------------------------------------


def test_configured_content_root_is_used(
    settings: Settings, content_root: Path
) -> None:
    assert resolve_content_root(settings) == content_root.resolve()


def test_unconfigured_content_root_falls_back_to_the_packaged_course(
    settings: Settings,
) -> None:
    unset = settings.model_copy(update={"content_root": None})

    assert resolve_content_root(unset) == packaged_content_root()


def test_packaged_content_root_lives_inside_the_installed_package() -> None:
    package_directory = Path(netstack_academy.__file__).resolve().parent

    assert packaged_content_root() == package_directory / "content"


def test_packaged_course_is_valid_content() -> None:
    """The course this program ships has to load under the same strict rules
    an author's local content does; a shipped course that fails validation
    leaves a fresh install with nothing to read.
    """
    result = load_curriculum(packaged_content_root())

    assert result.errors == ()
    assert result.curriculum is not None
    assert result.curriculum.modules != ()


# ----------------------------------------------------------------------
# Opening and closing a runtime
# ----------------------------------------------------------------------


def test_runtime_creates_both_databases_in_the_state_directory(
    runtime: AcademyRuntime, state_dir: Path
) -> None:
    assert runtime.state_dir == state_dir.resolve()
    assert (state_dir / LEARNING_DB_NAME).is_file()
    assert (state_dir / INDEX_DB_NAME).is_file()


def test_runtime_databases_are_outside_the_kernel_repository(
    runtime: AcademyRuntime, kernel_repo: Path
) -> None:
    for database in (
        runtime.state_dir / LEARNING_DB_NAME,
        runtime.state_dir / INDEX_DB_NAME,
    ):
        with pytest.raises(ValueError):
            database.relative_to(kernel_repo.resolve())


def test_runtime_loads_the_configured_course(runtime: AcademyRuntime) -> None:
    assert [module.slug for module in runtime.curriculum.modules] == [
        "rx-path",
        "tx-path",
    ]


def test_runtime_context_can_serve_the_dashboard(runtime: AcademyRuntime) -> None:
    client = TestClient(create_web_app(runtime.context))

    response = client.get("/")

    assert response.status_code == 200
    assert "Receive path" in response.text


def test_runtime_context_is_wired_to_the_runtime_collaborators(
    runtime: AcademyRuntime,
) -> None:
    context = runtime.context

    assert isinstance(context, AcademyContext)
    assert context.store is runtime.store
    assert isinstance(context.index, LazyIndex)
    assert isinstance(context.index.service, IndexService)


def test_runtime_rejects_invalid_content(
    settings: Settings,
    tmp_path: Path,
    session_runner: RecordingSessionRunner,
) -> None:
    broken = write_invalid_content(tmp_path / "broken-content")
    invalid = settings.model_copy(update={"content_root": broken})

    with pytest.raises(CurriculumValidationError) as excinfo:
        AcademyRuntime.open(invalid, session_runner=session_runner)

    assert excinfo.value.errors != ()


def test_runtime_close_closes_the_learning_database(
    settings: Settings, session_runner: RecordingSessionRunner
) -> None:
    opened = AcademyRuntime.open(settings, session_runner=session_runner)

    opened.close()

    with pytest.raises(sqlite3.ProgrammingError):
        opened.store.list_progress()


def test_runtime_close_closes_the_index_database(
    settings: Settings, session_runner: RecordingSessionRunner
) -> None:
    opened = AcademyRuntime.open(settings, session_runner=session_runner)

    opened.close()

    with pytest.raises(sqlite3.ProgrammingError):
        opened.storage.symbol_count()


def test_runtime_close_is_idempotent(
    settings: Settings, session_runner: RecordingSessionRunner
) -> None:
    """Shutdown runs from a ``finally`` and from a context manager exit; the
    second one must not turn a clean stop into a traceback.
    """
    opened = AcademyRuntime.open(settings, session_runner=session_runner)

    opened.close()
    opened.close()


def test_runtime_context_manager_closes_on_exit(
    settings: Settings, session_runner: RecordingSessionRunner
) -> None:
    with AcademyRuntime.open(settings, session_runner=session_runner) as opened:
        store = opened.store

    with pytest.raises(sqlite3.ProgrammingError):
        store.list_progress()


def test_reopening_a_runtime_keeps_the_learners_state(
    settings: Settings, session_runner: RecordingSessionRunner
) -> None:
    with AcademyRuntime.open(settings, session_runner=session_runner) as first:
        first.store.start_lesson("lesson-napi-poll")

    with AcademyRuntime.open(settings, session_runner=session_runner) as second:
        assert second.store.get_progress("lesson-napi-poll").status == "in_progress"


# ----------------------------------------------------------------------
# Laziness
# ----------------------------------------------------------------------


def test_opening_a_runtime_does_not_index(
    runtime: AcademyRuntime, session_runner: RecordingSessionRunner
) -> None:
    """Starting the server must be instant. Indexing a kernel tree is not.
    """
    assert session_runner.call_count == 0
    assert runtime.index.ensured is False


def test_runtime_ensure_index_runs_the_session_once(
    runtime: AcademyRuntime, session_runner: RecordingSessionRunner
) -> None:
    runtime.index.ensure()
    runtime.index.ensure()

    assert session_runner.call_count == 1
    assert session_runner.calls[0]["force"] is False
    assert runtime.index.ensured is True


def test_runtime_ensure_index_passes_the_kernel_repository_and_storage(
    runtime: AcademyRuntime,
    session_runner: RecordingSessionRunner,
    kernel_repo: Path,
) -> None:
    runtime.index.ensure()

    call = session_runner.calls[0]
    assert call["kernel_repo"] == kernel_repo
    assert call["storage"] is runtime.storage


def test_runtime_force_reindex_always_runs_the_session(
    runtime: AcademyRuntime, session_runner: RecordingSessionRunner
) -> None:
    runtime.index.ensure()
    runtime.index.ensure(force=True)

    assert [call["force"] for call in session_runner.calls] == [False, True]


def test_a_failed_run_is_retried_by_the_next_ensure(
    settings: Settings, kernel_head: str
) -> None:
    """A failure is not an answer. If ``clangd`` was missing the first time,
    the next request that needs symbols should try again rather than serve an
    empty index for the life of the process.
    """
    runner = RecordingSessionRunner(failed_result("ctags not found"))

    with AcademyRuntime.open(settings, session_runner=runner) as opened:
        first = opened.index.ensure()
        opened.index.ensure()

        assert first.status == "failed"
        assert opened.index.ensured is False
        assert runner.call_count == 2


# ----------------------------------------------------------------------
# LazyIndex on its own
# ----------------------------------------------------------------------


def test_lazy_index_status_never_triggers_indexing(
    kernel_repo: Path, index_storage, indexed_generation
) -> None:
    lazy = LazyIndex(IndexService(kernel_repo, index_storage, ExplodingOrchestrator()))

    status = lazy.status()

    assert status.symbol_count == indexed_generation.symbol_count
    assert lazy.ensured is False


def test_lazy_index_reports_the_last_run(
    kernel_repo: Path, index_storage, indexed_generation, kernel_head: str
) -> None:
    """The dashboard shows provider availability, and the only place that is
    known is the result of the run that actually consulted them.
    """
    orchestrator = RecordingOrchestrator(reindexed_result(kernel_head))
    lazy = LazyIndex(IndexService(kernel_repo, index_storage, orchestrator))

    assert lazy.last_result is None

    lazy.ensure()

    assert lazy.last_result is not None
    assert lazy.last_result.status == "reindexed"
    assert [
        diagnostic.provider_name for diagnostic in lazy.last_result.provider_diagnostics
    ] == ["ctags", "clangd"]


def test_lazy_index_exposes_the_service_it_wraps(
    kernel_repo: Path, index_storage, indexed_generation
) -> None:
    service = IndexService(kernel_repo, index_storage, ExplodingOrchestrator())

    assert LazyIndex(service).service is service


def test_reopened_content_root_is_reloaded_not_cached(
    settings: Settings, content_root: Path, session_runner: RecordingSessionRunner
) -> None:
    """Content is authored while the program is not running; a second start
    must see the new lesson.
    """
    with AcademyRuntime.open(settings, session_runner=session_runner) as first:
        assert first.curriculum.lesson_by_slug("bql-draft") is not None

    (content_root / "10-tx" / "bql-draft.md").unlink()

    with AcademyRuntime.open(settings, session_runner=session_runner) as second:
        assert second.curriculum.lesson_by_slug("bql-draft") is None
