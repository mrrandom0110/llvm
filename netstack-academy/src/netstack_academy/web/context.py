"""Everything the app is allowed to know, in one injected object.

The application is built from an :class:`AcademyContext` rather than from
module-level state, because every single thing in it -- which kernel
checkout, which course, whose progress database, which editor scheme -- is
something a test, a second instance, or a second checkout needs to differ
on. Reading configuration per request from the process environment would
make all of that untestable and would let anything else in the process
change this app's behaviour by exporting a variable.

The context owns no lifetimes. It is handed an already-open store and an
already-wired index service, and whoever opened them closes them; see
:class:`~netstack_academy.web.runtime.AcademyRuntime` for the production
composition root that does.
"""

from __future__ import annotations

from dataclasses import dataclass

from netstack_academy.curriculum.models import Curriculum
from netstack_academy.indexing.service import IndexService
from netstack_academy.learning.services import LearningService
from netstack_academy.learning.store import LearningStore
from netstack_academy.settings import Settings

from .index_access import LazyIndex


@dataclass(frozen=True, slots=True)
class AcademyContext:
    """The collaborators one running application serves from."""

    settings: Settings
    curriculum: Curriculum
    store: LearningStore
    index: LazyIndex
    learning: LearningService

    @classmethod
    def build(
        cls,
        *,
        settings: Settings,
        curriculum: Curriculum,
        store: LearningStore,
        index_service: IndexService,
    ) -> "AcademyContext":
        """Wire a context from the four things that have to come from outside.

        The :class:`~netstack_academy.web.index_access.LazyIndex` and the
        :class:`~netstack_academy.learning.services.LearningService` are
        derived rather than injected: they are composition, not
        configuration. The service is given the lazy index as its symbol
        search, so a combined lesson/symbol search inherits the same
        "index only when someone needs symbols" policy the rest of the app
        follows.
        """
        index = LazyIndex(index_service)
        return cls(
            settings=settings,
            curriculum=curriculum,
            store=store,
            index=index,
            learning=LearningService(curriculum, store, symbol_index=index),
        )
