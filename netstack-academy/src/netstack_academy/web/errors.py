"""One error shape, one registry of codes, for the whole JSON API.

Every failure the API can report leaves as::

    {"error": {"code": "...", "message": "...", "details": {...}}}

with ``details`` present only when there is something structured to say.
The page's JavaScript has to tell "you need to pick a file" apart from "that
lesson is gone" without reading prose, so ``code`` -- not the message, and
not the status alone -- is the part clients are allowed to branch on.
:data:`ERROR_CODES` is the closed set of values it can take.

Four statuses carry all the meaning:

- **400** the input is unsafe or names something that cannot exist,
- **404** it could have existed but does not,
- **409** it conflicts with the state the server found (an ambiguous symbol,
  a status move the machine forbids),
- **422** the body or query failed validation.

Two rules apply to every message built here.

**Nothing is echoed back.** A message that quotes a rejected path turns an
error response into a reflection surface, and one that quotes a filesystem
path tells a caller something about the machine they did not ask about. The
messages below are therefore fixed strings; the untrusted value that caused
the error is deliberately dropped rather than interpolated. That matters
most for ``unsafe_path``, whose whole purpose is to refuse a path the
caller supplied.

**FastAPI's own 422 does not get through.** Its default body is
``{"detail": [...]}``, a second error shape for the case clients hit most,
so :func:`install_error_handlers` replaces it with the envelope above.

The same failures reach the reader through pages, where an error object is
not a useful answer. The status codes and the codes above are unchanged
there; only the body differs, and :func:`install_error_handlers` takes the
renderers that produce it. Every :class:`ApiError` therefore carries enough
to render either form, which is why an ambiguous symbol's candidates travel
in ``details`` rather than being formatted into its message.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

#: Every ``error.code`` this API can return. Closed on purpose: a client may
#: enumerate it, and a new failure mode has to be named here before it can
#: be reported, which is the cheapest way to stop one from arriving as an
#: unlabelled 500.
ERROR_CODES: frozenset[str] = frozenset(
    {
        "unsafe_path",
        "invalid_quiz_response",
        "invalid_note_body",
        "invalid_state_document",
        "invalid_request",
        "validation_error",
        "symbol_not_found",
        "symbol_ambiguous",
        "lesson_not_found",
        "module_not_found",
        "invalid_transition",
        "not_found",
        "method_not_allowed",
        "unsupported_media_type",
    }
)

#: Deliberately vague: naming the path back to the caller would echo
#: untrusted input, and the caller already knows what it sent.
UNSAFE_PATH_MESSAGE = (
    "The requested file path is not a safe repository-relative path."
)

_STATUS_CODES: Mapping[int, str] = {
    404: "not_found",
    405: "method_not_allowed",
    415: "unsupported_media_type",
    422: "validation_error",
}


class ApiError(Exception):
    """A failure with a status, a stable code, and nothing untrusted in it."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"Error code {code!r} is not declared in ERROR_CODES")
        super().__init__(f"{status_code} {code}: {message}")
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def unsafe_path() -> ApiError:
    return ApiError(400, "unsafe_path", UNSAFE_PATH_MESSAGE)


def symbol_not_found(name: str) -> ApiError:
    """A name that matched nothing.

    The symbol name is safe to repeat: it came from the path segment the
    caller is looking at, and unlike a path it names nothing on disk.
    """
    return ApiError(
        404,
        "symbol_not_found",
        f"No symbol named {name!r} is present in the current index.",
    )


def symbol_ambiguous(name: str, candidates: list[dict[str, Any]]) -> ApiError:
    """Several definitions share this name, so the caller has to choose.

    The candidates travel in ``details`` rather than in the message: they
    are what a client needs to build a disambiguation list, and they carry
    only repository-relative paths that the index already contains.
    """
    return ApiError(
        409,
        "symbol_ambiguous",
        f"Several symbols are named {name!r}; name the file to choose one.",
        {"candidates": candidates},
    )


def lesson_not_found(key: str) -> ApiError:
    return ApiError(
        404, "lesson_not_found", f"No lesson with id or slug {key!r} exists."
    )


def module_not_found(slug: str) -> ApiError:
    return ApiError(404, "module_not_found", f"No module with slug {slug!r} exists.")


def invalid_transition(message: str) -> ApiError:
    return ApiError(409, "invalid_transition", message)


def invalid_quiz_response(message: str) -> ApiError:
    return ApiError(400, "invalid_quiz_response", message)


def invalid_note_body() -> ApiError:
    return ApiError(
        422,
        "invalid_note_body",
        "A note body must contain something; delete the note instead of emptying it.",
    )


def invalid_state_document(reason: str) -> ApiError:
    return ApiError(
        422,
        "invalid_state_document",
        "The state document was rejected; nothing was changed.",
        {"reason": reason},
    )


def error_payload(
    code: str, message: str, details: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The one envelope. ``details`` is omitted rather than sent as null."""
    error: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        error["details"] = dict(details)
    return {"error": error}


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content=error_payload(code, message, details)
    )


def _validation_fields(exception: RequestValidationError) -> dict[str, Any]:
    """Which fields failed, by location only.

    Pydantic's own error entries carry the offending ``input`` alongside the
    location, and that input is whatever the caller sent -- so only the
    dotted location is reported. It is enough to point a client at the field
    it got wrong, and it cannot reflect a payload back.
    """
    locations = []
    for error in exception.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        if location and location not in locations:
            locations.append(location)
    return {"fields": locations}


def wants_json(request: Request) -> bool:
    """Only ``/api`` speaks JSON; everything else is a page for a human."""
    return request.url.path.startswith("/api")


def install_error_handlers(
    app: FastAPI,
    *,
    render_not_found: Callable[[Request], Response],
    render_page_error: Callable[[Request, ApiError], Response] | None = None,
) -> None:
    """Route every failure through the envelope, or through a page.

    Both renderers are injected because the templates belong to
    :mod:`netstack_academy.web.app`, not here.

    ``render_not_found`` handles a URL that matched no route at all.
    ``render_page_error`` handles a typed failure raised by a *page* handler
    -- an ambiguous symbol, a lesson that does not exist, a disambiguation
    path that is not repository-relative. Those need the same status codes as
    the API and a different body: someone who followed a stale link should
    get something they can read and navigate away from, and an ambiguous
    symbol in particular needs its candidates rendered as links rather than
    as JSON. Left unset, every :class:`ApiError` answers in the envelope,
    which is the right default for a JSON-only application.
    """

    @app.exception_handler(ApiError)
    async def _handle_api_error(request: Request, exc: ApiError) -> Response:
        if render_page_error is not None and not wants_json(request):
            return render_page_error(request, exc)
        return error_response(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> Response:
        failure = ApiError(
            422,
            "validation_error",
            "The request did not match what this endpoint accepts.",
            _validation_fields(exc),
        )
        if render_page_error is not None and not wants_json(request):
            # A page can fail validation too -- ``/search?limit=0`` is a link
            # someone can construct -- and answering a browser with an error
            # object would be a dead end.
            return render_page_error(request, failure)
        return error_response(
            failure.status_code, failure.code, failure.message, failure.details
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> Response:
        if exc.status_code == 404 and not wants_json(request):
            return render_not_found(request)

        code = _STATUS_CODES.get(exc.status_code, "invalid_request")
        # ``exc.detail`` is Starlette's own wording for the generic cases
        # ("Not Found", "Method Not Allowed"). Anything raised with a
        # caller-derived detail goes through ``ApiError`` instead, so there
        # is nothing untrusted to launder here.
        return error_response(exc.status_code, code, str(exc.detail))
