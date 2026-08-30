"""The production composition root: where things live, and who closes them.

:class:`AcademyRuntime` decides where the learner's database and the symbol
index go, loads the course, wires the services the app needs, and owns the
shutdown of everything it opened. It is the counterpart to
:class:`~netstack_academy.web.context.AcademyContext`, which owns no
lifetimes at all.

Three properties are worth more than the rest.

**State never lands inside the kernel repository.** The index and the
learner's notes are this program's data, not the kernel's. Writing them
under the checkout would put untracked databases in ``git status``, put a
potentially multi-gigabyte index one ``git clean -xdf`` from deletion, and
make ``HEAD`` -- the thing the index is keyed on -- depend on files the
index itself wrote. :func:`resolve_state_dir` refuses it, and refuses it
before anything is created, so a rejected configuration leaves no trace.

**Indexing is lazy.** Building a symbol index over a kernel tree is minutes
of work. Opening a runtime therefore indexes nothing; see
:class:`~netstack_academy.web.index_access.LazyIndex` for when it happens.

**Whatever the runtime opens, the runtime closes.** Two SQLite connections
here, and -- on a real run -- a ``clangd`` process, which
:func:`~netstack_academy.indexing.composition.run_indexing_session` starts
and stops within the single run that needed it, so no provider outlives the
session that owns it.

Resolution order for the state directory, first match winning:

1. ``STATE_DIR`` (or an explicitly constructed ``Settings.state_dir``),
2. ``$XDG_STATE_HOME/netstack-academy``,
3. ``~/.local/state/netstack-academy``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

import netstack_academy
from netstack_academy.curriculum.loader import load_curriculum
from netstack_academy.curriculum.models import Curriculum
from netstack_academy.indexing.composition import run_indexing_session
from netstack_academy.indexing.orchestrator import IndexRunResult, SemanticProviderFactory
from netstack_academy.indexing.semantic.factory import create_clangd_provider
from netstack_academy.indexing.service import IndexService
from netstack_academy.indexing.storage import IndexStorage
from netstack_academy.learning.store import LearningStore
from netstack_academy.settings import Settings

from .context import AcademyContext
from .index_access import LazyIndex

#: Directory this program owns under the user's state home.
STATE_DIR_NAME = "netstack-academy"

#: The learner's progress, notes, quiz attempts and review cards.
LEARNING_DB_NAME = "learning.sqlite3"

#: The symbol index for the configured kernel checkout.
INDEX_DB_NAME = "index.sqlite3"

#: Where the packaged course lives inside the installed package.
CONTENT_DIR_NAME = "content"


class StateDirectoryInsideRepositoryError(ValueError):
    """Raised when the configured state directory is inside the kernel repo."""


class SessionRunner(Protocol):
    """The one call this module makes to run the indexing pipeline.

    Narrower than
    :func:`~netstack_academy.indexing.composition.run_indexing_session`'s
    full signature on purpose: a test can substitute a recorder that proves
    *whether* and *how* the pipeline ran without owning a ``clangd``.
    """

    def __call__(
        self,
        kernel_repo: Path,
        storage: IndexStorage,
        *,
        force: bool = False,
        **kwargs: Any,
    ) -> IndexRunResult: ...


def packaged_content_root() -> Path:
    """The course that ships with this build.

    A fresh install with nothing configured has to have something to read,
    so the course is package data rather than a separate download.
    """
    return Path(netstack_academy.__file__).resolve().parent / CONTENT_DIR_NAME


def resolve_content_root(settings: Settings) -> Path:
    """The configured course, or the packaged one."""
    if settings.content_root is not None:
        return Path(settings.content_root).expanduser().resolve()
    return packaged_content_root()


def resolve_state_dir(settings: Settings) -> Path:
    """Where this program's databases go. Creates nothing.

    Resolution is separate from creation so that a caller can check a
    configuration -- or reject one -- without leaving a directory behind.
    """
    if settings.state_dir is not None:
        state_dir = Path(settings.state_dir).expanduser().resolve()
    else:
        state_dir = _default_state_dir()

    _reject_state_dir_inside_repository(state_dir, Path(settings.kernel_repo))
    return state_dir


def _default_state_dir() -> Path:
    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        base = Path(xdg_state_home).expanduser()
    else:
        base = Path.home() / ".local" / "state"
    return (base / STATE_DIR_NAME).resolve()


def _reject_state_dir_inside_repository(state_dir: Path, kernel_repo: Path) -> None:
    try:
        repository = kernel_repo.expanduser().resolve()
    except OSError:
        # An unresolvable kernel repo is a problem for indexing, not for
        # deciding where state goes; nothing can be inside a path that
        # cannot be named.
        return

    if state_dir == repository or state_dir.is_relative_to(repository):
        raise StateDirectoryInsideRepositoryError(
            f"The state directory {state_dir} is inside the kernel repository "
            f"{repository}; the index is keyed on the repository's HEAD and must "
            "not live inside the tree it describes."
        )


class _SessionOrchestrator:
    """Adapts a session runner to the ``ensure_index(force=...)`` protocol.

    :class:`~netstack_academy.indexing.service.IndexService` depends on that
    one method; the composition root it should reach is
    :func:`~netstack_academy.indexing.composition.run_indexing_session`,
    which owns the semantic provider's lifetime for the length of one run.
    This is the two-line adapter between them.
    """

    def __init__(
        self,
        kernel_repo: Path,
        storage: IndexStorage,
        *,
        session_runner: SessionRunner,
        semantic_provider_factory: SemanticProviderFactory,
    ) -> None:
        self._kernel_repo = kernel_repo
        self._storage = storage
        self._session_runner = session_runner
        self._semantic_provider_factory = semantic_provider_factory

    def ensure_index(self, *, force: bool = False) -> IndexRunResult:
        return self._session_runner(
            self._kernel_repo,
            self._storage,
            semantic_provider_factory=self._semantic_provider_factory,
            force=force,
        )


class AcademyRuntime:
    """One opened installation: resolved paths, open databases, a context."""

    def __init__(
        self,
        *,
        settings: Settings,
        state_dir: Path,
        content_root: Path,
        curriculum: Curriculum,
        store: LearningStore,
        storage: IndexStorage,
        context: AcademyContext,
    ) -> None:
        self.settings = settings
        self.state_dir = state_dir
        self.content_root = content_root
        self.curriculum = curriculum
        self.store = store
        self.storage = storage
        self.context = context
        self._closed = False

    @classmethod
    def open(
        cls,
        settings: Settings,
        *,
        session_runner: SessionRunner = run_indexing_session,
        semantic_provider_factory: SemanticProviderFactory = create_clangd_provider,
    ) -> "AcademyRuntime":
        """Resolve, validate, and open everything one installation needs.

        The order is deliberate: paths are resolved and the course is
        validated *before* any database is created, so neither a state
        directory inside the kernel tree nor a course that fails validation
        leaves a half-built installation on disk.
        """
        state_dir = resolve_state_dir(settings)
        content_root = resolve_content_root(settings)

        # Strict: a half-valid course is never served, because a lesson
        # silently missing its quiz is indistinguishable to a learner from a
        # lesson that has none. Raises CurriculumValidationError carrying
        # every problem found, not just the first.
        result = load_curriculum(content_root, strict=True)
        assert result.curriculum is not None  # strict=True raises otherwise

        store = LearningStore.open(state_dir / LEARNING_DB_NAME)
        try:
            storage = IndexStorage.open(state_dir / INDEX_DB_NAME)
        except Exception:
            store.close()
            raise

        orchestrator = _SessionOrchestrator(
            Path(settings.kernel_repo),
            storage,
            session_runner=session_runner,
            semantic_provider_factory=semantic_provider_factory,
        )
        context = AcademyContext.build(
            settings=settings,
            curriculum=result.curriculum,
            store=store,
            index_service=IndexService(
                Path(settings.kernel_repo), storage, orchestrator
            ),
        )

        return cls(
            settings=settings,
            state_dir=state_dir,
            content_root=content_root,
            curriculum=result.curriculum,
            store=store,
            storage=storage,
            context=context,
        )

    @property
    def index(self) -> LazyIndex:
        return self.context.index

    def close(self) -> None:
        """Close both databases. Idempotent.

        Shutdown runs from a ``finally`` in the CLI *and* from a context
        manager exit, so the second call must not turn a clean stop into a
        traceback.
        """
        if self._closed:
            return
        self._closed = True
        self.store.close()
        self.storage.close()

    def __enter__(self) -> "AcademyRuntime":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
