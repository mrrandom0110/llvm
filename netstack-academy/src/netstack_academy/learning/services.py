"""The read side of the academy: dashboard, module, lesson and search views.

This is what a web handler, a CLI or a test calls to get something
displayable. It composes three things that know nothing about each other --
a loaded :class:`~netstack_academy.curriculum.models.Curriculum`, the
learner's :class:`~netstack_academy.learning.store.LearningStore`, and
(optionally) a symbol index -- into flat, frozen view objects.

Three rules shape the module:

**No web framework.** Nothing here imports FastAPI, Starlette or any
serving machinery. Teaching logic that drags a web framework into every
caller cannot be reused from a CLI or a background job, and cannot be
tested without one. The views are plain dataclasses; turning one into JSON
or into a template context is the caller's business.

**The answer key never reaches a view.** A ``LessonView`` carries
``quiz: tuple[PublicQuizQuestion, ...]`` and does *not* hold a reference to
the underlying :class:`~netstack_academy.curriculum.models.Lesson`. Holding
one would put ``view.lesson.quiz[0].answer`` a single attribute access away
from any template that iterates what it is given, which is exactly the leak
:func:`~netstack_academy.curriculum.models.public_quiz` exists to prevent.
The content fields are therefore copied through explicitly; the answer key
and the per-question explanations are the only things withheld.

**The symbol index is optional and narrow.** The service depends on a
single method, ``search_symbols(query, limit=...)``, so it works against
the real index, a fake, or nothing at all. With no index configured a
search still returns its curriculum hits and an empty ``symbols`` tuple:
a missing cross-reference degrades the answer, it does not fail it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from netstack_academy.curriculum.models import (
    Curriculum,
    Lab,
    Lesson,
    MasteryGate,
    Module,
    PublicQuizQuestion,
    StructureReference,
    SymbolReference,
    public_quiz,
)

from .store import LearningStore, LessonProgress
from .store import LessonStatus as ProgressStatus

#: Default cap on both lesson hits and symbol hits for one search.
DEFAULT_SEARCH_LIMIT = 20

#: Lesson text searched, in the order reported by ``LessonHit.matched_fields``.
SEARCHED_FIELDS: tuple[str, ...] = ("title", "summary", "objectives", "body")


# Both of these deliberately shadow builtins inside this module's namespace
# (``ModuleNotFoundError`` is a builtin subclass of ``ImportError``). The
# names are the domain's -- a module and a lesson are the two things a URL
# can name -- and nothing in this module catches the import-time flavour.
class ModuleNotFoundError(LookupError):
    """Raised when no module in the curriculum matches the requested slug."""


class LessonNotFoundError(LookupError):
    """Raised when no lesson in the curriculum matches the requested key."""


class SymbolSearch(Protocol):
    """The whole of what the learning layer needs from a symbol index."""

    def search_symbols(self, query: str, *, limit: int) -> Sequence[Any]:
        ...


@dataclass(frozen=True, slots=True)
class LessonLink:
    """A pointer from one lesson to another, with enough to render it."""

    lesson_id: str
    slug: str
    title: str
    module_slug: str
    completed: bool


@dataclass(frozen=True, slots=True)
class LessonSummary:
    """A lesson as it appears in a list: dashboard row, module row."""

    id: str
    slug: str
    title: str
    order: int
    module_id: str
    module_slug: str
    status: str
    summary: str
    progress_status: ProgressStatus
    completed_at: datetime | None = None
    is_unlocked: bool = True
    review_due: bool = False


@dataclass(frozen=True, slots=True)
class ModuleView:
    """One module and its lessons, with the learner's progress folded in."""

    id: str
    slug: str
    title: str
    order: int
    summary: str
    lessons: tuple[LessonSummary, ...] = ()
    lesson_count: int = 0
    completed_count: int = 0
    in_progress_count: int = 0
    percent_complete: float = 0.0


@dataclass(frozen=True, slots=True)
class DashboardView:
    """Everything the landing page shows."""

    modules: tuple[ModuleView, ...] = ()
    lesson_count: int = 0
    completed_count: int = 0
    in_progress_count: int = 0
    not_started_count: int = 0
    percent_complete: float = 0.0
    due_review_count: int = 0
    next_lesson: LessonSummary | None = None


@dataclass(frozen=True, slots=True)
class LessonView:
    """One lesson page: content, the learner's state, and nothing secret."""

    id: str
    slug: str
    title: str
    order: int
    module_id: str
    module_slug: str
    module_title: str
    status: str
    summary: str
    objectives: tuple[str, ...]
    body_html: str
    prerequisites: tuple[LessonLink, ...] = ()
    unlocks: tuple[LessonLink, ...] = ()
    is_unlocked: bool = True
    packet_stage: str | None = None
    execution_context: str | None = None
    ownership: str | None = None
    locking: str | None = None
    rcu: str | None = None
    structures: tuple[StructureReference, ...] = ()
    config_caveats: tuple[str, ...] = ()
    version_caveats: tuple[str, ...] = ()
    tracepoints: tuple[str, ...] = ()
    source_symbols: tuple[SymbolReference, ...] = ()
    lab: Lab | None = None
    quiz: tuple[PublicQuizQuestion, ...] = ()
    mastery_gate: MasteryGate | None = None
    attempt_count: int = 0
    best_score: float | None = None
    meets_mastery_gate: bool = False
    note: str | None = None
    progress_status: ProgressStatus = "not_started"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    review_level: int | None = None
    review_due_at: datetime | None = None
    review_due: bool = False


@dataclass(frozen=True, slots=True)
class LessonHit:
    """A curriculum search hit."""

    lesson_id: str
    slug: str
    title: str
    module_slug: str
    module_title: str
    summary: str
    status: str
    matched_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SearchResults:
    """Curriculum hits and symbol hits for one query, plus the query itself."""

    query: str
    lessons: tuple[LessonHit, ...] = ()
    symbols: tuple[Any, ...] = ()


class LearningService:
    """Read-side composition of curriculum, learner state and symbol index."""

    def __init__(
        self,
        curriculum: Curriculum,
        store: LearningStore,
        *,
        symbol_index: SymbolSearch | None = None,
    ) -> None:
        self._curriculum = curriculum
        self._store = store
        self._symbol_index = symbol_index

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def dashboard(self) -> DashboardView:
        """The whole curriculum with the learner's progress folded in."""
        progress = self._progress_map()
        due = self._due_lesson_ids()

        modules = tuple(
            self._module_view(module, progress, due)
            for module in self._curriculum.modules
        )
        lessons = tuple(
            summary for module in modules for summary in module.lessons
        )

        completed = _count(lessons, "completed")
        in_progress = _count(lessons, "in_progress")
        return DashboardView(
            modules=modules,
            lesson_count=len(lessons),
            completed_count=completed,
            in_progress_count=in_progress,
            not_started_count=_count(lessons, "not_started"),
            percent_complete=_percent(completed, len(lessons)),
            due_review_count=len(due),
            next_lesson=_next_lesson(lessons),
        )

    # ------------------------------------------------------------------
    # Module page
    # ------------------------------------------------------------------

    def module_view(self, slug: str) -> ModuleView:
        module = self._curriculum.module_by_slug(slug)
        if module is None:
            raise ModuleNotFoundError(f"No module with slug {slug!r}")
        return self._module_view(module, self._progress_map(), self._due_lesson_ids())

    # ------------------------------------------------------------------
    # Lesson page
    # ------------------------------------------------------------------

    def lesson_view(self, key: str) -> LessonView:
        """One lesson, addressed by either its id or its slug.

        Both work because a deep link may carry either: ids are what other
        lessons' ``prerequisites`` name, slugs are what a URL reads well
        with.
        """
        lesson = self._curriculum.lesson_by_id(key) or self._curriculum.lesson_by_slug(
            key
        )
        if lesson is None:
            raise LessonNotFoundError(f"No lesson with id or slug {key!r}")

        module = self._curriculum.module_by_id(lesson.module_id)
        if module is None:
            raise ModuleNotFoundError(
                f"Lesson {lesson.id!r} names unknown module {lesson.module_id!r}"
            )

        progress = self._progress_map()
        due = self._due_lesson_ids()
        state = _progress_of(progress, lesson.id)
        card = self._store.review_card(lesson.id)
        note = self._store.get_lesson_note(lesson.id)
        attempts = self._store.list_quiz_attempts(lesson.id)
        best_score = max((attempt.score for attempt in attempts), default=None)
        review_level = card.level if card is not None else None

        return LessonView(
            id=lesson.id,
            slug=lesson.slug,
            title=lesson.title,
            order=lesson.order,
            module_id=module.id,
            module_slug=module.slug,
            module_title=module.title,
            status=lesson.status,
            summary=lesson.summary,
            objectives=lesson.objectives,
            body_html=lesson.body_html,
            prerequisites=self._prerequisite_links(lesson, progress),
            unlocks=self._unlock_links(lesson, progress),
            is_unlocked=_is_unlocked(lesson, progress),
            packet_stage=lesson.packet_stage,
            execution_context=lesson.execution_context,
            ownership=lesson.ownership,
            locking=lesson.locking,
            rcu=lesson.rcu,
            structures=lesson.structures,
            config_caveats=lesson.config_caveats,
            version_caveats=lesson.version_caveats,
            tracepoints=lesson.tracepoints,
            source_symbols=lesson.source_symbols,
            lab=lesson.lab,
            quiz=public_quiz(lesson),
            mastery_gate=lesson.mastery_gate,
            attempt_count=len(attempts),
            best_score=best_score,
            meets_mastery_gate=_meets_gate(lesson.mastery_gate, best_score, review_level),
            note=note.body if note is not None else None,
            progress_status=state.status,
            started_at=state.started_at,
            completed_at=state.completed_at,
            review_level=review_level,
            review_due_at=card.next_due if card is not None else None,
            review_due=lesson.id in due,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, *, limit: int = DEFAULT_SEARCH_LIMIT) -> SearchResults:
        """Search curriculum text and, when configured, the symbol index.

        A blank query returns nothing rather than everything, and does not
        spend a symbol-index round trip: "show me all lessons" is what the
        dashboard is for.
        """
        needle = query.strip()
        if not needle:
            return SearchResults(query=query)

        folded = needle.casefold()
        hits: list[LessonHit] = []
        for module in self._curriculum.modules:
            for lesson in module.lessons:
                matched = _matched_fields(lesson, folded)
                if not matched:
                    continue
                hits.append(
                    LessonHit(
                        lesson_id=lesson.id,
                        slug=lesson.slug,
                        title=lesson.title,
                        module_slug=module.slug,
                        module_title=module.title,
                        summary=lesson.summary,
                        status=lesson.status,
                        matched_fields=matched,
                    )
                )

        # Hits are collected in curriculum order and then truncated, so the
        # limit changes how many results a caller sees and never which ones.
        bounded = hits[: max(limit, 0)]

        symbols: tuple[Any, ...] = ()
        if self._symbol_index is not None:
            symbols = tuple(self._symbol_index.search_symbols(needle, limit=limit))

        return SearchResults(query=query, lessons=tuple(bounded), symbols=symbols)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _progress_map(self) -> dict[str, LessonProgress]:
        """One query for the whole page rather than one per lesson."""
        return {
            progress.lesson_id: progress for progress in self._store.list_progress()
        }

    def _due_lesson_ids(self) -> frozenset[str]:
        """Due cards, narrowed to lessons the curriculum still has.

        Content is reorganised while a learner's state stays put, so a card
        can outlive its lesson. Counting one would put a number on the
        dashboard that no page can ever satisfy.
        """
        known = {lesson.id for lesson in self._curriculum.lessons}
        return frozenset(
            card.lesson_id for card in self._store.due_reviews() if card.lesson_id in known
        )

    def _module_view(
        self,
        module: Module,
        progress: dict[str, LessonProgress],
        due: frozenset[str],
    ) -> ModuleView:
        lessons = tuple(
            _summarize(lesson, module, progress, due) for lesson in module.lessons
        )
        completed = _count(lessons, "completed")
        return ModuleView(
            id=module.id,
            slug=module.slug,
            title=module.title,
            order=module.order,
            summary=module.summary,
            lessons=lessons,
            lesson_count=len(lessons),
            completed_count=completed,
            in_progress_count=_count(lessons, "in_progress"),
            percent_complete=_percent(completed, len(lessons)),
        )

    def _prerequisite_links(
        self, lesson: Lesson, progress: dict[str, LessonProgress]
    ) -> tuple[LessonLink, ...]:
        links = []
        for prerequisite_id in lesson.prerequisites:
            required = self._curriculum.lesson_by_id(prerequisite_id)
            if required is None:
                # A validated curriculum cannot contain this, but a service
                # handed a hand-built one should still render the page.
                continue
            links.append(self._link(required, progress))
        return tuple(links)

    def _unlock_links(
        self, lesson: Lesson, progress: dict[str, LessonProgress]
    ) -> tuple[LessonLink, ...]:
        """Lessons that name this one as a prerequisite, in curriculum order."""
        return tuple(
            self._link(candidate, progress)
            for candidate in self._curriculum.lessons
            if lesson.id in candidate.prerequisites
        )

    def _link(
        self, lesson: Lesson, progress: dict[str, LessonProgress]
    ) -> LessonLink:
        module = self._curriculum.module_by_id(lesson.module_id)
        return LessonLink(
            lesson_id=lesson.id,
            slug=lesson.slug,
            title=lesson.title,
            module_slug=module.slug if module is not None else "",
            completed=_progress_of(progress, lesson.id).status == "completed",
        )


def _progress_of(
    progress: dict[str, LessonProgress], lesson_id: str
) -> LessonProgress:
    return progress.get(
        lesson_id, LessonProgress(lesson_id=lesson_id, status="not_started")
    )


def _summarize(
    lesson: Lesson,
    module: Module,
    progress: dict[str, LessonProgress],
    due: frozenset[str],
) -> LessonSummary:
    state = _progress_of(progress, lesson.id)
    return LessonSummary(
        id=lesson.id,
        slug=lesson.slug,
        title=lesson.title,
        order=lesson.order,
        module_id=module.id,
        module_slug=module.slug,
        status=lesson.status,
        summary=lesson.summary,
        progress_status=state.status,
        completed_at=state.completed_at,
        is_unlocked=_is_unlocked(lesson, progress),
        review_due=lesson.id in due,
    )


def _is_unlocked(lesson: Lesson, progress: dict[str, LessonProgress]) -> bool:
    return all(
        _progress_of(progress, prerequisite_id).status == "completed"
        for prerequisite_id in lesson.prerequisites
    )


def _count(lessons: Sequence[LessonSummary], status: str) -> int:
    return sum(1 for lesson in lessons if lesson.progress_status == status)


def _percent(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return round(100.0 * part / whole, 1)


def _next_lesson(lessons: Sequence[LessonSummary]) -> LessonSummary | None:
    """The first published, unfinished, unlocked lesson in curriculum order.

    Drafts are skipped: they are visible to their author, but sending a
    learner to unfinished material is not a suggestion worth making. A
    lesson whose prerequisites are outstanding is skipped for the same
    reason -- it is not something the learner can do next.
    """
    for lesson in lessons:
        if lesson.status != "published":
            continue
        if lesson.progress_status == "completed":
            continue
        if not lesson.is_unlocked:
            continue
        return lesson
    return None


def _meets_gate(
    gate: MasteryGate | None, best_score: float | None, review_level: int | None
) -> bool:
    if gate is None:
        return False
    if best_score is None or best_score < gate.min_quiz_score:
        return False
    return (review_level or 0) >= gate.required_review_level


def _matched_fields(lesson: Lesson, folded_query: str) -> tuple[str, ...]:
    """Which searched fields contain the query, in ``SEARCHED_FIELDS`` order.

    Only authored prose is searched -- title, summary, objectives and the
    Markdown body. Identifier-ish fields (symbols, tracepoints, structures)
    are deliberately left to the symbol index, which answers questions
    about identifiers far better than a substring scan over lesson
    metadata would.
    """
    haystacks = {
        "title": lesson.title,
        "summary": lesson.summary,
        "objectives": "\n".join(lesson.objectives),
        "body": lesson.body_markdown,
    }
    return tuple(
        name
        for name in SEARCHED_FIELDS
        if folded_query in haystacks[name].casefold()
    )
