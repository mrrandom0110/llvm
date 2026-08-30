"""JSON shapes for the API.

These are the payloads the page's JavaScript reads, so they are as much a
part of the contract as the rendered HTML, and they are built here rather
than inline in handlers so that the same symbol appears identically whether
it arrived from a search, a card, or the far end of a call edge.

Two decisions run through the whole module.

**Provenance travels with every edge, and so does its confidence.** A call
edge found by ``clangd`` knows where the call happens; one guessed by a
regex over the source cannot even tell two same-named ``static`` functions
apart. Presenting them identically would be the single most misleading thing
this program could do, so each edge carries both the provenance it came from
and the plain-language confidence that follows from it.

**A deep link is a capability, not a formatting concern.** Links are built
through :func:`~netstack_academy.web.links.editor_deep_link`, which refuses
anything outside the kernel repository, and the refusal is reported as
``deep_link: null`` plus a reason rather than omitted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from urllib.parse import quote, urlencode

from netstack_academy.indexing.orchestrator import IndexRunResult, ProviderDiagnostic
from netstack_academy.indexing.service import EdgeView, SymbolView
from netstack_academy.learning.quiz import QuestionResult
from netstack_academy.learning.review import MAX_LEITNER_LEVEL
from netstack_academy.learning.services import (
    DashboardView,
    LessonHit,
    LessonSummary,
    LessonView,
    ModuleView,
    SearchResults,
)
from netstack_academy.learning.store import LessonProgress, QuizAttempt, ReviewCard
from netstack_academy.repo_inspector import RepositoryState
from netstack_academy.settings import Settings

from .index_access import LazyIndex
from .links import editor_deep_link

#: How much an edge is worth believing, given where it came from. ``clangd``
#: resolved a real translation unit; the fallback indexer matched a regex.
CONFIDENCE_BY_PROVENANCE: Mapping[str, str] = {
    "semantic": "high",
    "heuristic": "low",
}

#: Used when a future provenance reaches here without a mapping above --
#: better to under-claim than to present a guess as a fact.
DEFAULT_CONFIDENCE = "low"


def lesson_url(slug: str) -> str:
    """The page for one lesson. Slugs are the URL identity, not ids."""
    return f"/lessons/{quote(slug, safe='')}"


def symbol_url(name: str, relative_path: str | None = None) -> str:
    """The page for one symbol.

    The path is always included when known: two ``static`` functions can
    share a name, so a link without one is a link to a question rather than
    to a symbol.
    """
    url = f"/symbols/{quote(name, safe='')}"
    if relative_path:
        url = f"{url}?{urlencode({'path': relative_path})}"
    return url


def symbol_payload(symbol: SymbolView) -> dict[str, Any]:
    """One symbol's identity and where it is defined."""
    return {
        "id": symbol.id,
        "name": symbol.name,
        "kind": symbol.kind,
        "relative_path": symbol.relative_path,
        "line": symbol.line,
        "column": symbol.column,
        "signature": symbol.signature,
        "scope": symbol.scope,
        "is_static": symbol.is_static,
        "commit_hash": symbol.commit_hash,
        "url": symbol_url(symbol.name, symbol.relative_path),
    }


def candidate_payload(symbol: SymbolView) -> dict[str, Any]:
    """Just enough of a symbol to disambiguate it in a picker."""
    return {
        "name": symbol.name,
        "relative_path": symbol.relative_path,
        "line": symbol.line,
        "is_static": symbol.is_static,
        "url": symbol_url(symbol.name, symbol.relative_path),
    }


def edge_payload(
    settings: Settings,
    edge: EdgeView,
    *,
    name: str,
    endpoint: SymbolView | None,
) -> dict[str, Any]:
    """One call or reference edge, labelled by where it came from.

    ``name`` is what to display at the far end -- the callee for an outgoing
    call, the caller for an incoming one -- and ``endpoint`` is that symbol
    when the index could resolve it. A heuristic edge frequently cannot be
    resolved to a definition, which is exactly why its confidence is low.
    """
    site: dict[str, Any] | None = None
    site_deep_link: str | None = None
    if edge.site_relative_path and edge.site_line:
        site = {
            "relative_path": edge.site_relative_path,
            "line": edge.site_line,
            "column": edge.site_column,
        }
        site_deep_link, _ = editor_deep_link(
            settings, edge.site_relative_path, edge.site_line, edge.site_column
        )

    return {
        "id": edge.id,
        "name": name,
        "edge_type": edge.edge_type,
        "provenance": edge.provenance,
        "confidence": CONFIDENCE_BY_PROVENANCE.get(
            edge.provenance, DEFAULT_CONFIDENCE
        ),
        "site": site,
        "site_deep_link": site_deep_link,
        "symbol": symbol_payload(endpoint) if endpoint is not None else None,
    }


def provider_payload(diagnostic: ProviderDiagnostic) -> dict[str, Any]:
    return {
        "provider_name": diagnostic.provider_name,
        "available": diagnostic.available,
        "reason": diagnostic.reason,
    }


def run_payload(result: IndexRunResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "head": result.head,
        "symbol_count": result.symbol_count,
        "edge_count": result.edge_count,
        "reason": result.reason,
        "diagnostics": list(result.diagnostics),
    }


def index_status_payload(
    index: LazyIndex, repository: RepositoryState
) -> dict[str, Any]:
    """What is indexed, what the repository is at, and whether they agree.

    ``stale`` is the answer to the only question a learner actually asks of
    this endpoint -- "am I reading a graph of the code I have checked out?"
    -- and it is deliberately ``False`` when the repository cannot be
    inspected at all, because then there is nothing to compare against and
    claiming staleness would be a guess.
    """
    status = index.status()
    last_result = index.last_result
    return {
        "indexed_head": status.head,
        "symbol_count": status.symbol_count,
        "edge_count": status.edge_count,
        "repository_available": repository.available,
        "repository_head": repository.head,
        "repository_reason": repository.reason,
        "stale": bool(repository.available and status.head != repository.head),
        "ensured": index.ensured,
        "last_run": run_payload(last_result) if last_result is not None else None,
        "providers": [
            provider_payload(diagnostic)
            for diagnostic in (
                last_result.provider_diagnostics if last_result is not None else ()
            )
        ],
    }


def index_run_response(index: LazyIndex, result: IndexRunResult) -> dict[str, Any]:
    """The outcome of one ensure/force request.

    A failed run is reported here with a 200: "ctags is not installed" is
    information the dashboard should show, and turning it into a 500 would
    make the caller guess whether the request or the machine was at fault.
    """
    payload = run_payload(result)
    payload["ensured"] = index.ensured
    payload["providers"] = [
        provider_payload(diagnostic) for diagnostic in result.provider_diagnostics
    ]
    return payload


def _moment(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def progress_payload(progress: LessonProgress) -> dict[str, Any]:
    return {
        "lesson_id": progress.lesson_id,
        "status": progress.status,
        "started_at": _moment(progress.started_at),
        "completed_at": _moment(progress.completed_at),
    }


def lesson_summary_payload(summary: LessonSummary) -> dict[str, Any]:
    return {
        "id": summary.id,
        "slug": summary.slug,
        "title": summary.title,
        "order": summary.order,
        "module_id": summary.module_id,
        "module_slug": summary.module_slug,
        "status": summary.status,
        "summary": summary.summary,
        "progress_status": summary.progress_status,
        "completed_at": _moment(summary.completed_at),
        "is_unlocked": summary.is_unlocked,
        "review_due": summary.review_due,
        "url": lesson_url(summary.slug),
    }


def module_payload(module: ModuleView) -> dict[str, Any]:
    return {
        "id": module.id,
        "slug": module.slug,
        "title": module.title,
        "order": module.order,
        "summary": module.summary,
        "lesson_count": module.lesson_count,
        "completed_count": module.completed_count,
        "in_progress_count": module.in_progress_count,
        "percent_complete": module.percent_complete,
        "lessons": [lesson_summary_payload(lesson) for lesson in module.lessons],
        "url": f"/modules/{quote(module.slug, safe='')}",
    }


def dashboard_payload(dashboard: DashboardView) -> dict[str, Any]:
    return {
        "lesson_count": dashboard.lesson_count,
        "completed_count": dashboard.completed_count,
        "in_progress_count": dashboard.in_progress_count,
        "not_started_count": dashboard.not_started_count,
        "percent_complete": dashboard.percent_complete,
        "due_review_count": dashboard.due_review_count,
        "next_lesson": (
            lesson_summary_payload(dashboard.next_lesson)
            if dashboard.next_lesson is not None
            else None
        ),
        "modules": [module_payload(module) for module in dashboard.modules],
    }


def note_payload(body: str) -> dict[str, Any]:
    """A saved note, verbatim.

    The learner's own text goes back exactly as it was written. Escaping
    belongs to the template at render time, not to storage or transport: a
    note that arrives HTML-escaped can never be edited without accumulating
    entities, and one that arrives pre-rendered cannot be escaped correctly
    by whatever displays it next.
    """
    return {"body": body}


def review_card_payload(card: ReviewCard) -> dict[str, Any]:
    """Where a card sits on the ladder and when it comes back.

    ``max_level`` travels with it because "level 3" means nothing without
    the height of the ladder, and the client should not hard-code it.
    """
    return {
        "lesson_id": card.lesson_id,
        "level": card.level,
        "max_level": MAX_LEITNER_LEVEL,
        "next_due": _moment(card.next_due),
        "last_reviewed_at": _moment(card.last_reviewed_at),
    }


def quiz_result_payload(result: QuestionResult) -> dict[str, Any]:
    """One graded question, *after* an attempt was recorded.

    This is the only shape in the API that carries ``correct_option_id`` and
    ``explanation``. It is built from a grade, which only exists once a
    submission has been made, so there is no code path that produces it for
    a learner who has not answered yet -- that is what
    :func:`~netstack_academy.curriculum.models.public_quiz` is for.
    """
    return {
        "question_id": result.question_id,
        "response": result.response,
        "correct": result.correct,
        "correct_option_id": result.correct_option_id,
        "explanation": result.explanation,
    }


def quiz_attempt_payload(
    attempt: QuizAttempt,
    results: Sequence[QuestionResult],
    lesson: LessonView,
) -> dict[str, Any]:
    """The graded submission.

    ``score`` is read back from the recorded attempt rather than from the
    grade the handler computed, so the number returned to the learner is
    provably the number that was persisted. ``lesson`` supplies the
    across-attempts figures -- best score, attempt count, and whether the
    mastery gate is now met -- so this endpoint and the lesson page cannot
    disagree about a gate.
    """
    return {
        "lesson_id": attempt.lesson_id,
        "score": attempt.score,
        "correct_count": attempt.correct_count,
        "question_count": attempt.question_count,
        "submitted_at": _moment(attempt.created_at),
        "results": [quiz_result_payload(result) for result in results],
        "attempt_count": lesson.attempt_count,
        "best_score": lesson.best_score,
        "meets_mastery_gate": lesson.meets_mastery_gate,
    }


def lesson_hit_payload(hit: LessonHit) -> dict[str, Any]:
    """One curriculum search hit.

    ``matched_fields`` is the part worth keeping: "why is this result here"
    is otherwise unanswerable from a title, and a reader who searched for a
    function name wants to know it was the body that matched.
    """
    return {
        "lesson_id": hit.lesson_id,
        "slug": hit.slug,
        "title": hit.title,
        "module_slug": hit.module_slug,
        "module_title": hit.module_title,
        "summary": hit.summary,
        "status": hit.status,
        "matched_fields": list(hit.matched_fields),
        "url": lesson_url(hit.slug),
    }


def search_payload(
    results: SearchResults, *, symbols_unavailable_reason: str | None = None
) -> dict[str, Any]:
    """Both halves of one search, and the reason one of them may be short.

    The symbol half degrades rather than failing: a reader whose index has
    not been built yet still gets every lesson hit, an empty ``symbols``
    list, and ``symbols_unavailable_reason`` explaining what to fix. An
    error here would take the working half down with the broken one.
    """
    lessons = [lesson_hit_payload(hit) for hit in results.lessons]
    symbols = [symbol_payload(symbol) for symbol in results.symbols]
    return {
        "query": results.query,
        "lessons": lessons,
        "symbols": symbols,
        "counts": {"lessons": len(lessons), "symbols": len(symbols)},
        "symbols_unavailable_reason": symbols_unavailable_reason,
    }
