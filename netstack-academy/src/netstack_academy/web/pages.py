"""View models for the server-rendered pages.

:mod:`netstack_academy.web.payloads` turns the domain into JSON; this turns
it into what a template needs. They are separate because the two audiences
want different things from the same data. A JSON client wants every field and
will compute the rest; a template wants the computation already done, because
a conditional in Jinja is harder to read and impossible to test than the same
conditional here.

Three rules shape everything below.

**Every page is complete without JavaScript.** The lesson body, the lab, the
context table, the quiz and the call graph's list form are all built here and
rendered by the server. The script adds saving and drawing; it is not what
makes a page legible. So nothing here defers a value to the client.

**Untrusted text is escaped once, here.** Notes and search queries are the
only free text in the application, and they arrive as
:class:`~markupsafe.Markup` from :func:`~netstack_academy.web.escaping.untrusted_text`
so a template cannot forget to escape them and cannot double-escape them
either. The raw strings are deliberately *not* passed alongside: a template
that has no access to the unescaped value cannot render it by mistake.

**A link is only emitted when it can be built.** Deep links come from
:func:`~netstack_academy.web.links.editor_deep_link`, which refuses a path
that escapes the repository or names a file that is gone, and a refusal
arrives here as a reason to display rather than as a link to suppress
downstream. For the same reason an unsafe stored path is never rendered *at
all* -- not as a link, not as text -- since printing it would put a path from
outside the kernel tree on the page whether or not it was clickable. That
check is :func:`displayable_path`, registered as a template filter rather
than applied once here, because a path reaches a page from more places than
this module assembles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from netstack_academy.curriculum.models import Lesson
from netstack_academy.indexing.paths import is_safe_relative_path
from netstack_academy.indexing.service import SymbolView
from netstack_academy.learning.review import MAX_LEITNER_LEVEL
from netstack_academy.learning.services import (
    DashboardView,
    LessonView,
    ModuleView,
    SearchResults,
)
from netstack_academy.repo_inspector import RepositoryState

from .context import AcademyContext
from .escaping import untrusted_text
from .links import editor_deep_link
from .payloads import api_symbol_url, graph_payload, lesson_url, symbol_url

#: Shown instead of a stored path that is not repository-relative. Nothing
#: this program writes produces such a row, but the index is a file that
#: outlives the process that wrote it, and the path itself must not reach the
#: page: it names somewhere outside the kernel tree by definition.
UNSAFE_PATH_LABEL = "(path outside the kernel repository)"

#: How much of a commit hash a page shows. Twelve hex digits is what kernel
#: changelogs abbreviate to, and a line number is only meaningful together
#: with the commit it was read at.
SHORT_HASH_LENGTH = 12


@dataclass(frozen=True, slots=True)
class Crumb:
    """One breadcrumb. ``url`` is ``None`` for the page you are already on."""

    label: str
    url: str | None = None


def short_hash(value: str | None) -> str | None:
    return None if not value else value[:SHORT_HASH_LENGTH]


def displayable_path(relative_path: str | None) -> str:
    """A stored path, or a label standing in for one that is not shown.

    Registered as a template filter, because a path reaches a page from more
    places than this module builds: a symbol's own location, a call site, a
    reference site. Every one of them is a string that came out of a database
    file, so every one of them goes through here.
    """
    if not relative_path or not is_safe_relative_path(relative_path):
        return UNSAFE_PATH_LABEL
    return relative_path


# ----------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------


def dashboard_page(
    dashboard: DashboardView,
    index_status: dict[str, Any],
    repository: RepositoryState,
) -> dict[str, Any]:
    """The landing page: the course, the learner's place in it, the machinery.

    The index status is included because every symbol link on every lesson
    page depends on an index that may be missing, stale, or built without
    ``clangd``, and a reader who is told that up front can act on it. A
    reader who instead meets a deep link that opens the wrong line has to
    work out why on their own.
    """
    due_lessons = tuple(
        lesson
        for module in dashboard.modules
        for lesson in module.lessons
        if lesson.review_due
    )
    return {
        "dashboard": dashboard,
        "due_lessons": due_lessons,
        "index": index_status,
        "repository": repository,
        "repository_short_head": short_hash(repository.head),
        "indexed_short_head": short_hash(index_status["indexed_head"]),
    }


# ----------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------


def module_page(module: ModuleView) -> dict[str, Any]:
    return {
        "module": module,
        "crumbs": (Crumb("Dashboard", "/"), Crumb(module.title)),
    }


# ----------------------------------------------------------------------
# Lesson
# ----------------------------------------------------------------------


def lesson_page(lesson: LessonView, next_lesson: Lesson | None) -> dict[str, Any]:
    """One lesson, and the one after it.

    ``next_lesson`` comes from curriculum order rather than from
    ``lesson.unlocks``: "what do I read next" is a question about the
    sequence the course was written in, and only some lessons gate another.

    The three ``has_*`` flags exist so the template can omit a section
    header for a section with nothing in it -- a draft lesson carries
    identity and nothing else, and a heading over an empty list reads like
    missing content rather than absent content.
    """
    source_symbols = tuple(
        {
            "name": reference.name,
            "path": reference.path,
            "url": symbol_url(reference.name, reference.path),
        }
        for reference in lesson.source_symbols
    )
    return {
        "lesson": lesson,
        "next_lesson": (
            None
            if next_lesson is None
            else {"title": next_lesson.title, "url": lesson_url(next_lesson.slug)}
        ),
        "source_symbols": source_symbols,
        # Escaped here, and the raw note is not passed: see the module
        # docstring.
        "note_text": untrusted_text(lesson.note),
        "crumbs": (
            Crumb("Dashboard", "/"),
            Crumb(lesson.module_title, f"/modules/{lesson.module_slug}"),
            Crumb(lesson.title),
        ),
        "has_kernel_context": any(
            (
                lesson.packet_stage,
                lesson.execution_context,
                lesson.ownership,
                lesson.locking,
                lesson.rcu,
            )
        ),
        "has_caveats": bool(lesson.config_caveats or lesson.version_caveats),
        # "Box 3" means nothing without the height of the ladder, and the
        # template should not hard-code it any more than a client should.
        "review_max_level": MAX_LEITNER_LEVEL,
        "mastery_percent": (
            round(lesson.mastery_gate.min_quiz_score * 100)
            if lesson.mastery_gate is not None
            else None
        ),
        "best_percent": (
            round(lesson.best_score * 100) if lesson.best_score is not None else None
        ),
    }


# ----------------------------------------------------------------------
# Symbol
# ----------------------------------------------------------------------


def symbol_page(context: AcademyContext, symbol: SymbolView) -> dict[str, Any]:
    """One symbol card, its call graph, and its references.

    The graph is assembled server-side in its list form. That list is not a
    fallback for a browser that cannot draw: it is how the graph is read by
    a screen reader, by a keyboard, and by anyone whose window is narrow, so
    it has to exist before any script runs. It comes from the same
    :func:`~netstack_academy.web.payloads.graph_payload` the JSON endpoint
    uses, so the list and the drawing cannot disagree.
    """
    deep_link, deep_link_reason = editor_deep_link(
        context.settings, symbol.relative_path, symbol.line, symbol.column
    )
    graph = graph_payload(context.settings, context.index.service, symbol.id)
    note = context.store.get_symbol_note(
        symbol.name, relative_path=symbol.relative_path
    )
    return {
        "symbol": symbol,
        "short_commit": short_hash(symbol.commit_hash),
        "deep_link": deep_link,
        "deep_link_reason": deep_link_reason,
        **graph,
        "note_text": untrusted_text(note.body if note is not None else None),
        # Two ``static`` functions can share a name, so both the note and the
        # graph are addressed by name *and* file. Built with the same encoder
        # the page's own links use, so an odd path cannot break out of the
        # query string.
        "note_url": api_symbol_url("note", symbol.name, symbol.relative_path),
        "graph_url": api_symbol_url("graph", symbol.name, symbol.relative_path),
        "crumbs": (
            Crumb("Dashboard", "/"),
            Crumb("Search", "/search"),
            Crumb(symbol.name),
        ),
    }


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------


def search_page(
    results: SearchResults, *, symbols_unavailable_reason: str | None = None
) -> dict[str, Any]:
    """Both halves of one search, separated rather than interleaved.

    Interleaving would need a relevance score across two corpora this
    program has no way to compare -- "how well does a lesson match compared
    with a function name" has no answer -- so lessons keep curriculum order
    and symbols keep the index's own ranking.
    """
    lessons = tuple(
        {
            "title": hit.title,
            "summary": hit.summary,
            "module_title": hit.module_title,
            "module_slug": hit.module_slug,
            "status": hit.status,
            "matched_fields": hit.matched_fields,
            "url": lesson_url(hit.slug),
        }
        for hit in results.lessons
    )
    symbols = tuple(
        {
            "name": symbol.name,
            "kind": symbol.kind,
            "relative_path": symbol.relative_path,
            "line": symbol.line,
            "url": symbol_url(symbol.name, symbol.relative_path),
        }
        for symbol in results.symbols
    )
    return {
        # The query reaches the template escaped and only escaped: it is
        # echoed into an attribute, and it is the one value on any page that
        # an attacker fully controls through a link.
        "query_text": untrusted_text(results.query),
        "has_query": bool(results.query.strip()),
        "lessons": lessons,
        "symbols": symbols,
        "symbols_unavailable_reason": symbols_unavailable_reason,
    }


__all__ = [
    "SHORT_HASH_LENGTH",
    "UNSAFE_PATH_LABEL",
    "Crumb",
    "dashboard_page",
    "displayable_path",
    "lesson_page",
    "module_page",
    "search_page",
    "short_hash",
    "symbol_page",
]
