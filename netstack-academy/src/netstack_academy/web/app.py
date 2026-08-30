"""The local academy application, bound to 127.0.0.1 and nothing else.

This app is composed from an explicit
:class:`~netstack_academy.web.context.AcademyContext` rather than from
module-level state, so a test, a second instance, or a second kernel
checkout can differ on every input it has.

It is a *local* application in a strong sense. There is no authentication
anywhere in it, and every endpoint that mutates state does so on behalf of
whoever can reach the port. That is a reasonable design for a single-user
tool reading a kernel checkout on the same machine, and it is only
reasonable while the port is unreachable from anywhere else, so the bind
address is fixed at :data:`LOOPBACK_HOST` (``127.0.0.1``) and
:func:`is_loopback_host` exists to refuse anything else. For the same
reason the interactive API console is disabled: a schema browser on an
unauthenticated API is a map of available mutations for anything else that
can reach the port.

Three habits keep the surface small:

- **Mutations are never GETs.** This app is full of links, and a link that
  writes will eventually be followed by a prefetcher or a crawler.
- **Only ``/api`` speaks JSON.** A human who follows a stale link gets a
  page they can read and navigate away from.
- **Nothing here executes anything.** Lessons *display* kernel lab
  commands; a course full of shell is content, not a command to run.

Two more rules govern the learning endpoints specifically. Grading is
server-side against the loaded lesson, so a submission contributes which
option was chosen and nothing else, and the answer key and its explanations
appear only in the response to an attempt that has already been recorded --
there is no code path that produces them for a learner who has not answered.
And search degrades rather than fails: when the symbol index cannot be
built, the lesson half of a result still answers, with a reason attached,
because an unbuilt index is not a reason to break the working half.

Every handler is ``async`` even though the work inside is synchronous. That
is deliberate: a ``def`` handler is run in a threadpool, which would mean two
requests touching the one SQLite connection from two threads at once. Serving
them on the event-loop thread instead serialises database access without a
lock, and the queries here are millisecond-scale reads of a local file, so
there is nothing worth yielding for.
"""

from __future__ import annotations

from ipaddress import ip_address
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Body, FastAPI, Query, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict

from netstack_academy.curriculum.models import Lesson
from netstack_academy.indexing.service import (
    InvalidRepositoryPathError,
    SymbolAmbiguousError,
    SymbolNotFoundError,
    SymbolView,
)
from netstack_academy.indexing.paths import is_safe_relative_path
from netstack_academy.learning.quiz import (
    UnknownQuizOptionError,
    UnknownQuizQuestionError,
    grade_quiz,
)
from netstack_academy.learning.services import (
    DEFAULT_SEARCH_LIMIT as _LEARNING_DEFAULT_SEARCH_LIMIT,
)
from netstack_academy.learning.store import (
    InvalidStatusTransitionError,
    StateImportError,
    UnsafeNotePathError,
)
from netstack_academy.repo_inspector import inspect_repository

from . import errors
from .context import AcademyContext
from .errors import install_error_handlers
from .links import editor_deep_link
from .payloads import (
    candidate_payload,
    dashboard_payload,
    edge_payload,
    index_run_response,
    index_status_payload,
    note_payload,
    progress_payload,
    quiz_attempt_payload,
    review_card_payload,
    search_payload,
    symbol_payload,
)

#: The only address this application is ever bound to.
LOOPBACK_HOST = "127.0.0.1"

#: A fixed, unprivileged port, so the bookmark a learner saves keeps working.
DEFAULT_PORT = 8765

#: Upper bound on how many results one request may ask for. An unbounded
#: limit is a way to ask a single HTTP request to serialize a kernel-sized
#: symbol table.
MAX_SEARCH_LIMIT = 100

#: What a search returns when the caller does not say how many results it
#: wants. Taken from the learning layer rather than declared again here, so
#: the two halves of a combined search cannot drift into different defaults,
#: and re-exported because it is part of this module's HTTP contract.
DEFAULT_SEARCH_LIMIT = _LEARNING_DEFAULT_SEARCH_LIMIT

PACKAGE_ROOT = Path(__file__).resolve().parent
TEMPLATE_ROOT = PACKAGE_ROOT / "templates"
STATIC_ROOT = PACKAGE_ROOT / "static"

#: Host names that mean "this machine" but are not IP literals.
_LOOPBACK_NAMES = frozenset({"localhost"})

#: Sent on every response. None of it replaces the escaping the templates
#: already do; it is the second line, for the case where something does get
#: through.
_SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    # No 'unsafe-inline' and no 'unsafe-eval': behaviour lives in
    # /static/js, so injected markup has nowhere to run even if it survives
    # sanitisation, and neither does anything a note or a search query
    # smuggles in. 'self' throughout also means an offline session renders
    # exactly the same page as an online one.
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'none'; "
        "frame-ancestors 'none'; "
        "form-action 'self'"
    ),
}


def is_loopback_host(host: str) -> bool:
    """Whether ``host`` names this machine and only this machine.

    ``0.0.0.0`` and ``::`` are the two that matter: they read like defaults
    but mean "every interface", which is precisely what this application
    must never be published on.
    """
    if not host:
        return False

    candidate = host.strip()
    if candidate.casefold() in _LOOPBACK_NAMES:
        return True

    # A bracketed IPv6 literal is how a host:port string carries one.
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]

    try:
        return ip_address(candidate).is_loopback
    except ValueError:
        return False


class ProgressRequest(BaseModel):
    """A move along ``not_started -> in_progress -> completed``.

    ``not_started`` is deliberately not accepted: this endpoint records
    progress, and there is no such thing as un-reading a lesson. Resetting
    is what state import is for.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["in_progress", "completed"]


class NoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    body: str


class QuizSubmission(BaseModel):
    """Which option was chosen for each question, and nothing else.

    No score, no answer: grading happens server-side against the loaded
    lesson, so nothing in this payload can influence its own result. A body
    that carries a ``score`` is refused by ``extra="forbid"`` rather than
    quietly ignored -- a client that thinks it grades its own quizzes should
    hear about it.
    """

    model_config = ConfigDict(extra="forbid")

    responses: dict[str, str]


class ReviewRequest(BaseModel):
    """The outcome of one spaced-review answer.

    A single boolean is the whole input: the next level and the next due
    date follow from it and the clock, with no difficulty estimate and no
    randomness, which is what makes "what is due today" have one answer.
    """

    model_config = ConfigDict(extra="forbid")

    correct: bool


def create_web_app(context: AcademyContext) -> FastAPI:
    """Build the application that serves ``context``."""
    app = FastAPI(
        title="netstack-academy",
        # Disabled together: leaving ``openapi_url`` alive would still
        # publish the schema even with both consoles gone.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.context = context

    templates = Jinja2Templates(directory=str(TEMPLATE_ROOT))

    def render_not_found(request: Request) -> Response:
        return templates.TemplateResponse(
            request, "not_found.html", {"path": request.url.path}, status_code=404
        )

    install_error_handlers(app, render_not_found=render_not_found)

    @app.middleware("http")
    async def _apply_security_headers(request: Request, call_next: Any) -> Response:
        response = await call_next(request)
        for header, value in _SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        return response

    app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    @app.get("/", response_class=Response)
    async def dashboard(request: Request) -> Response:
        return templates.TemplateResponse(
            request, "dashboard.html", {"dashboard": context.learning.dashboard()}
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    @app.get("/api/index/status")
    async def index_status() -> dict[str, Any]:
        # Deliberately reads the persisted generation and the repository's
        # HEAD without ensuring anything: a dashboard must be able to say
        # "your index is stale" without being the thing that rebuilds it.
        return index_status_payload(
            context.index, inspect_repository(context.settings.kernel_repo)
        )

    @app.post("/api/index/ensure")
    async def index_ensure(force: bool = Query(default=False)) -> dict[str, Any]:
        result = context.index.ensure(force=force)
        return index_run_response(context.index, result)

    # ------------------------------------------------------------------
    # Symbols
    # ------------------------------------------------------------------

    @app.get("/api/symbols")
    async def symbol_search(
        q: str = Query(default=""),
        limit: Annotated[int, Query(ge=1, le=MAX_SEARCH_LIMIT)] = DEFAULT_SEARCH_LIMIT,
    ) -> dict[str, Any]:
        query = q.strip()
        if not query:
            # "Show me everything" is what the dashboard is for, and it is
            # not worth building an index to answer.
            return {"query": q, "symbols": [], "count": 0}

        context.index.ensure()
        matches = context.index.service.search_symbols(query, limit=limit)
        return {
            "query": q,
            "symbols": [symbol_payload(symbol) for symbol in matches],
            "count": len(matches),
        }

    @app.get("/api/symbols/{name}")
    async def symbol_card(
        name: str, path: str | None = Query(default=None)
    ) -> dict[str, Any]:
        symbol = _resolve_symbol(name, path)
        service = context.index.service
        link, reason = editor_deep_link(
            context.settings, symbol.relative_path, symbol.line, symbol.column
        )
        note = context.store.get_symbol_note(
            symbol.name, relative_path=symbol.relative_path
        )
        return {
            "symbol": symbol_payload(symbol),
            "deep_link": link,
            "deep_link_reason": reason,
            "counts": {
                "outgoing": len(service.outgoing_edges(symbol.id)),
                "incoming": len(service.incoming_edges(symbol.id)),
                "references": len(service.references(symbol.id)),
            },
            "note": note.body if note is not None else None,
        }

    @app.get("/api/symbols/{name}/graph")
    async def symbol_graph(
        name: str, path: str | None = Query(default=None)
    ) -> dict[str, Any]:
        symbol = _resolve_symbol(name, path)
        service = context.index.service
        settings = context.settings

        outgoing = [
            edge_payload(
                settings,
                edge,
                name=edge.target_name,
                endpoint=_symbol_or_none(edge.target_symbol_id),
            )
            for edge in service.outgoing_edges(symbol.id)
        ]

        incoming = []
        for edge in service.incoming_edges(symbol.id):
            # The stored edge only knows the caller's id; resolving it is
            # what makes an incoming call worth showing at all.
            caller = _symbol_or_none(edge.source_symbol_id)
            incoming.append(
                edge_payload(
                    settings,
                    edge,
                    name=caller.name if caller is not None else edge.target_name,
                    endpoint=caller,
                )
            )

        # A reference is a position in a file, not a second definition, so
        # there is no endpoint symbol to resolve -- the site is the point.
        references = [
            edge_payload(settings, edge, name=edge.target_name, endpoint=None)
            for edge in service.references(symbol.id)
        ]

        return {
            "symbol": symbol_payload(symbol),
            "outgoing": outgoing,
            "incoming": incoming,
            "references": references,
            "counts": {
                "incoming": len(incoming),
                "outgoing": len(outgoing),
                "references": len(references),
            },
        }

    # ------------------------------------------------------------------
    # Learner state
    # ------------------------------------------------------------------

    @app.get("/api/progress")
    async def progress() -> dict[str, Any]:
        return dashboard_payload(context.learning.dashboard())

    @app.post("/api/lessons/{key}/progress")
    async def record_progress(key: str, submitted: ProgressRequest) -> dict[str, Any]:
        lesson = _require_lesson(key)
        try:
            if submitted.status == "in_progress":
                updated = context.store.start_lesson(lesson.id)
            else:
                updated = context.store.complete_lesson(lesson.id)
        except InvalidStatusTransitionError as exc:
            raise errors.invalid_transition(str(exc)) from exc
        return progress_payload(updated)

    @app.put("/api/lessons/{key}/note")
    async def put_lesson_note(key: str, submitted: NoteRequest) -> dict[str, Any]:
        lesson = _require_lesson(key)
        if not submitted.body.strip():
            raise errors.invalid_note_body()
        note = context.store.upsert_lesson_note(lesson.id, submitted.body)
        return {"lesson_id": lesson.id, **note_payload(note.body)}

    @app.delete("/api/lessons/{key}/note")
    async def delete_lesson_note(key: str) -> dict[str, Any]:
        lesson = _require_lesson(key)
        return {
            "lesson_id": lesson.id,
            "deleted": context.store.delete_lesson_note(lesson.id),
        }

    @app.put("/api/symbols/{name}/note")
    async def put_symbol_note(
        name: str,
        submitted: NoteRequest,
        path: str | None = Query(default=None),
    ) -> dict[str, Any]:
        # The path is checked before the body: an unsafe path is refused
        # whether or not the note itself would have been acceptable, and the
        # store must never be asked to write one.
        _require_safe_note_path(path)
        if not submitted.body.strip():
            raise errors.invalid_note_body()
        try:
            note = context.store.upsert_symbol_note(
                name, submitted.body, relative_path=path
            )
        except UnsafeNotePathError as exc:
            raise errors.unsafe_path() from exc
        return {
            "symbol": name,
            "relative_path": path,
            **note_payload(note.body),
        }

    @app.delete("/api/symbols/{name}/note")
    async def delete_symbol_note(
        name: str, path: str | None = Query(default=None)
    ) -> dict[str, Any]:
        _require_safe_note_path(path)
        return {
            "symbol": name,
            "relative_path": path,
            "deleted": context.store.delete_symbol_note(name, relative_path=path),
        }

    @app.post("/api/lessons/{key}/quiz")
    async def submit_quiz(key: str, submitted: QuizSubmission) -> dict[str, Any]:
        lesson = _require_lesson(key)

        # Graded twice, deliberately. This call validates the submission and
        # produces the per-question results the response needs; the store
        # grades again inside ``record_quiz_attempt`` because the score it
        # persists must come from the answer key it read, not from a number
        # this handler passed in. Doing it in this order also means a
        # divergent submission is refused before the store is touched at
        # all, so a rejected attempt leaves no row behind.
        try:
            grade = grade_quiz(lesson, submitted.responses)
        except (UnknownQuizQuestionError, UnknownQuizOptionError) as exc:
            # The submission and the content have diverged -- a stale page, a
            # renamed question, or tampering. Scoring it as merely "wrong"
            # would hide that, and echoing the offending id back would
            # reflect caller input for no benefit.
            raise errors.invalid_quiz_response(
                "The submission does not match this lesson's quiz; reload the "
                "lesson and submit again."
            ) from exc

        attempt = context.store.record_quiz_attempt(lesson, submitted.responses)
        return quiz_attempt_payload(
            attempt, grade.results, context.learning.lesson_view(lesson.id)
        )

    @app.post("/api/lessons/{key}/review")
    async def record_review(key: str, submitted: ReviewRequest) -> dict[str, Any]:
        lesson = _require_lesson(key)
        card = context.store.record_review(lesson.id, correct=submitted.correct)
        return review_card_payload(card)

    # ------------------------------------------------------------------
    # Portable state
    # ------------------------------------------------------------------

    @app.get("/api/state/export")
    async def export_state() -> dict[str, Any]:
        # Returned as the store produced it, with no envelope: this document
        # is the input to /api/state/import, and a wrapper would mean the two
        # endpoints no longer round-trip through each other.
        return context.store.export_state()

    @app.post("/api/state/import")
    async def import_state(
        document: Annotated[dict[str, Any], Body()],
    ) -> dict[str, Any]:
        # A restore, not a merge, and validated in full before the first
        # write -- so a problem in the last record cannot leave the earlier
        # ones applied. The store owns both properties; this is the
        # translation of its refusal into a status code.
        #
        # The store's reason travels in ``details.reason`` while the
        # top-level message stays fixed, which is what makes it safe to
        # forward: those messages name the record and field that failed --
        # in a document of hundreds of records, the difference between a
        # fixable error and an unactionable one -- and what they quote is
        # the caller's own document, not anything from this machine.
        try:
            context.store.import_state(document)
        except (StateImportError, UnsafeNotePathError) as exc:
            raise errors.invalid_state_document(str(exc)) from exc
        return {"imported": True}

    # ------------------------------------------------------------------
    # Combined search
    # ------------------------------------------------------------------

    @app.get("/api/search")
    async def search(
        q: str = Query(default=""),
        limit: Annotated[int, Query(ge=1, le=MAX_SEARCH_LIMIT)] = DEFAULT_SEARCH_LIMIT,
    ) -> dict[str, Any]:
        # The service composes both halves, so a blank query returns early
        # and never reaches the symbol index -- searching for nothing is not
        # a reason to spend minutes indexing a kernel tree.
        results = context.learning.search(q, limit=limit)
        return search_payload(
            results, symbols_unavailable_reason=_symbols_unavailable_reason()
        )

    # ------------------------------------------------------------------
    # Shared resolution helpers
    # ------------------------------------------------------------------

    def _resolve_symbol(name: str, path: str | None) -> SymbolView:
        """Resolve a symbol, refusing to guess and refusing unsafe paths.

        The path is validated here rather than relying on
        ``IndexService``'s own check, for one reason:
        :class:`~netstack_academy.indexing.service.InvalidRepositoryPathError`
        quotes the offending path in its message, and this layer must not
        echo caller input back. The service check still runs underneath as
        the real guarantee; this one only shapes the response.
        """
        if path is not None and not is_safe_relative_path(path):
            raise errors.unsafe_path()

        context.index.ensure()
        try:
            return context.index.service.find_symbol(name, relative_path=path)
        except SymbolAmbiguousError as exc:
            raise errors.symbol_ambiguous(
                name, [candidate_payload(candidate) for candidate in exc.candidates]
            ) from exc
        except SymbolNotFoundError as exc:
            raise errors.symbol_not_found(name) from exc
        except InvalidRepositoryPathError as exc:
            raise errors.unsafe_path() from exc

    def _require_safe_note_path(path: str | None) -> None:
        """Refuse a note path that is not repository-relative.

        The store checks this too, and its check is the real guarantee. This
        one exists so the refusal becomes a typed 400 whose message does not
        quote the path back, and so the store is never asked to write one.
        """
        if path is not None and not is_safe_relative_path(path):
            raise errors.unsafe_path()

    def _symbols_unavailable_reason() -> str | None:
        """Why the symbol half of a search may be short, or ``None``.

        Only a *failed* run is worth reporting. An index that is merely
        empty or one commit behind still answers, and calling that
        unavailable would put a warning on a working search.
        """
        last_result = context.index.last_result
        if last_result is None or last_result.status != "failed":
            return None
        return last_result.reason or "The symbol index could not be built."

    def _symbol_or_none(symbol_id: int | None) -> SymbolView | None:
        """An edge endpoint, when the index can still name it.

        A heuristic edge often has no resolved endpoint at all, and an id
        from a superseded generation no longer exists; both render as an
        unresolved far end rather than as an error.
        """
        if symbol_id is None:
            return None
        try:
            return context.index.service.symbol_by_id(symbol_id)
        except SymbolNotFoundError:
            return None

    def _require_lesson(key: str) -> Lesson:
        """The lesson a URL names, by id or by slug.

        Both work because a deep link may carry either: ids are what other
        lessons' ``prerequisites`` name, slugs are what a URL reads well
        with.
        """
        lesson = context.curriculum.lesson_by_id(
            key
        ) or context.curriculum.lesson_by_slug(key)
        if lesson is None:
            raise errors.lesson_not_found(key)
        return lesson

    return app


__all__ = [
    "DEFAULT_PORT",
    "DEFAULT_SEARCH_LIMIT",
    "LOOPBACK_HOST",
    "MAX_SEARCH_LIMIT",
    "STATIC_ROOT",
    "TEMPLATE_ROOT",
    "create_web_app",
    "is_loopback_host",
]
