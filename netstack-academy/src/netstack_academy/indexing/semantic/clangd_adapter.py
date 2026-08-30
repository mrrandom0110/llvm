"""Synchronous ``clangd`` semantic provider adapter.

``ClangdAdapter`` speaks the LSP subset needed for call-graph enrichment
(``textDocument/prepareCallHierarchy``, ``callHierarchy/outgoingCalls``,
``textDocument/references``) over an injected :class:`LspTransport`. It
never starts a process or opens a socket itself, and never accepts a shell
command string -- callers own how (and whether) a real ``clangd`` binary is
launched and wired up to a transport (see ``semantic/factory.py``); this
adapter only needs something that implements ``send``/``recv``/``close``.
That is what makes it testable against ``tests/indexing/lsp_fakes.py``'s
in-memory ``FakeLspTransport`` with no real ``clangd`` process anywhere in
the test run.

Session lifecycle, in the order a real ``clangd`` requires it:

1. ``__init__`` sends ``initialize`` (with ``rootUri`` for the kernel repo),
   blocks -- bounded by ``timeout`` -- for the response *carrying that
   request's own id*, then sends the ``initialized`` notification.
2. Before the first semantic request touching a file, the file is opened
   with a ``textDocument/didOpen`` notification carrying its source text.
   Each file is opened at most once per session.
3. :meth:`close` sends ``shutdown``, waits (bounded) for its response, sends
   ``exit``, and closes the transport.

Any step may fail without raising: the failure is recorded and surfaced as
``capabilities().available is False`` plus a typed, degraded outcome from
every subsequent call.

Response correlation: a server may interleave notifications
(``window/logMessage``, ``$/progress``, ``textDocument/publishDiagnostics``),
server-initiated requests, and responses to earlier requests with the
response we are waiting for. Every read therefore accumulates into a
persistent buffer that survives across calls, is drained with
``jsonrpc.decode_messages``, and yields only the frame that is a *response*
(no ``method``) whose ``id`` matches the request in flight; everything else
is skipped, and any frame decoded but not consumed stays queued for the next
call rather than being discarded. Requests are still issued one at a time.

Position convention: callers of this adapter always use the same 1-based
``(line, column)`` convention as the rest of ``indexing`` (ctags, the
fallback indexer); LSP's 0-based positions are an internal, private detail
of this module, converted at the edges. Likewise, callers only ever see
kernel-repo-relative POSIX paths, never raw ``file://`` URIs -- URI
construction/parsing is also entirely internal to this module.
"""

from __future__ import annotations

import itertools
import os
import time
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse

from ..paths import (
    PathEscapesRepositoryError,
    normalize_relative_path,
    to_absolute_path,
)
from . import jsonrpc
from .models import (
    CallHierarchyItem,
    OutgoingCall,
    OutgoingCallsOutcome,
    PrepareCallHierarchyOutcome,
    ProviderCapabilities,
    ReferencesOutcome,
    SemanticLocation,
)
from .transport import LspTransport, TransportClosedError, TransportTimeoutError

DEFAULT_TIMEOUT_SECONDS = 5.0

#: Upper bound on how long :meth:`ClangdAdapter.close` waits for the
#: ``shutdown`` response before giving up and sending ``exit`` anyway. A
#: server that is wedged must never wedge the caller's teardown.
SHUTDOWN_TIMEOUT_SECONDS = 2.0

_PROVIDER_NAME = "clangd"

_LANGUAGE_ID = "c"

_INITIAL_DOCUMENT_VERSION = 1

# Deliberately minimal: this adapter only ever asks for call hierarchy and
# references, and never registers dynamic capabilities or handles progress.
_CLIENT_CAPABILITIES = {
    "textDocument": {
        "synchronization": {"dynamicRegistration": False},
        "callHierarchy": {"dynamicRegistration": False},
        "references": {"dynamicRegistration": False},
    },
    "window": {"workDoneProgress": False},
}


class _DegradedOutcome(Exception):
    """Internal signal: stop processing and report ``status``/``reason``.

    Every public method catches this (and any other unexpected exception,
    as a last-resort safety net) and converts it into the appropriate typed
    outcome instead of letting it propagate -- callers of this adapter
    should never need a try/except around it.
    """

    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


class ClangdAdapter:
    """Drives a subset of the LSP over ``transport`` on behalf of one kernel repo."""

    def __init__(
        self,
        transport: LspTransport,
        *,
        kernel_repo: Path,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._transport = transport
        self._kernel_repo = Path(kernel_repo)
        self._timeout = timeout
        self._request_ids = itertools.count(1)
        self._unavailable_reason: str | None = None
        self._closed = False
        self._receive_buffer = b""
        self._pending_messages: list[dict] = []
        self._open_document_uris: set[str] = set()
        self._initialize_session()

    def capabilities(self) -> ProviderCapabilities:
        if self._unavailable_reason is not None:
            return ProviderCapabilities(
                provider_name=_PROVIDER_NAME,
                available=False,
                reason=self._unavailable_reason,
            )
        return ProviderCapabilities(
            provider_name=_PROVIDER_NAME, available=True, reason=None
        )

    def prepare_call_hierarchy(
        self, relative_path: str, *, line: int, column: int
    ) -> PrepareCallHierarchyOutcome:
        try:
            self._require_available()
            self._ensure_document_open(relative_path)
            request = self._build_request(
                "textDocument/prepareCallHierarchy",
                {
                    "textDocument": {"uri": self._to_uri(relative_path)},
                    "position": self._to_lsp_position(line, column),
                },
            )
            message = self._send_and_receive(request)
            items = tuple(
                self._parse_call_hierarchy_item(entry)
                for entry in (message.get("result") or [])
            )
            return PrepareCallHierarchyOutcome(status="ok", items=items, reason=None)
        except _DegradedOutcome as exc:
            return PrepareCallHierarchyOutcome(
                status=exc.status, items=(), reason=exc.reason
            )
        except Exception as exc:  # defensive: never raise out of the adapter
            return PrepareCallHierarchyOutcome(
                status="error", items=(), reason=str(exc)
            )

    def outgoing_calls(self, item: CallHierarchyItem) -> OutgoingCallsOutcome:
        try:
            self._require_available()
            self._ensure_document_open(item.relative_path)
            request = self._build_request(
                "callHierarchy/outgoingCalls",
                {"item": self._call_hierarchy_item_to_lsp(item)},
            )
            message = self._send_and_receive(request)
            calls = tuple(
                self._parse_outgoing_call(entry, source_relative_path=item.relative_path)
                for entry in (message.get("result") or [])
            )
            return OutgoingCallsOutcome(status="ok", calls=calls, reason=None)
        except _DegradedOutcome as exc:
            return OutgoingCallsOutcome(status=exc.status, calls=(), reason=exc.reason)
        except Exception as exc:  # defensive: never raise out of the adapter
            return OutgoingCallsOutcome(status="error", calls=(), reason=str(exc))

    def references(
        self, relative_path: str, *, line: int, column: int
    ) -> ReferencesOutcome:
        try:
            self._require_available()
            self._ensure_document_open(relative_path)
            request = self._build_request(
                "textDocument/references",
                {
                    "textDocument": {"uri": self._to_uri(relative_path)},
                    "position": self._to_lsp_position(line, column),
                    "context": {"includeDeclaration": True},
                },
            )
            message = self._send_and_receive(request)
            locations = tuple(
                self._parse_location(entry) for entry in (message.get("result") or [])
            )
            return ReferencesOutcome(status="ok", locations=locations, reason=None)
        except _DegradedOutcome as exc:
            return ReferencesOutcome(
                status=exc.status, locations=(), reason=exc.reason
            )
        except Exception as exc:  # defensive: never raise out of the adapter
            return ReferencesOutcome(status="error", locations=(), reason=str(exc))

    def close(self) -> None:
        """End the session: ``shutdown``, ``exit``, then close the transport.

        Safe to call more than once, and safe when the session never came up
        (no ``shutdown``/``exit`` is sent to a server that never answered
        ``initialize``). The transport is always closed, whatever happens.
        """
        if self._closed:
            return
        self._closed = True

        if self._unavailable_reason is None:
            self._request_shutdown()
            self._suppress(self._send_notification, "exit")

        self._unavailable_reason = self._unavailable_reason or "clangd session closed"
        self._suppress(self._transport.close)

    # -- session lifecycle ---------------------------------------------------

    def _initialize_session(self) -> None:
        request = self._build_request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": self._root_uri(),
                "capabilities": _CLIENT_CAPABILITIES,
            },
        )

        try:
            self._send_and_receive(request)
            self._send_notification("initialized", {})
        except _DegradedOutcome as exc:
            self._unavailable_reason = f"clangd initialization failed: {exc.reason}"
        except Exception as exc:  # defensive: construction must never raise
            self._unavailable_reason = f"clangd initialization failed: {exc}"

    def _request_shutdown(self) -> None:
        request = self._build_request("shutdown")
        try:
            self._send_and_receive(
                request, timeout=min(self._timeout, SHUTDOWN_TIMEOUT_SECONDS)
            )
        except _DegradedOutcome:
            pass  # a server that will not shut down cleanly is still exited
        except Exception:
            pass

    def _ensure_document_open(self, relative_path: str) -> None:
        """Send ``textDocument/didOpen`` the first time a file is needed."""
        try:
            uri = self._to_uri(relative_path)
            absolute = to_absolute_path(relative_path, kernel_repo=self._kernel_repo)
        except PathEscapesRepositoryError as exc:
            raise _DegradedOutcome("error", str(exc)) from exc

        if uri in self._open_document_uris:
            return

        try:
            text = absolute.read_text(encoding="utf-8")
        except OSError as exc:
            raise _DegradedOutcome(
                "error", f"cannot read {relative_path!r}: {exc}"
            ) from exc

        self._send_notification(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": _LANGUAGE_ID,
                    "version": _INITIAL_DOCUMENT_VERSION,
                    "text": text,
                }
            },
        )
        self._open_document_uris.add(uri)

    @staticmethod
    def _suppress(action: Callable[..., object], *args: object) -> None:
        try:
            action(*args)
        except Exception:  # teardown must never raise
            pass

    # -- transport plumbing --------------------------------------------------

    def _next_id(self) -> int:
        return next(self._request_ids)

    def _require_available(self) -> None:
        if self._unavailable_reason is not None:
            raise _DegradedOutcome("unavailable", self._unavailable_reason)

    def _build_request(self, method: str, params: dict | None = None) -> dict:
        request: dict = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
        }
        if params is not None:
            request["params"] = params
        return request

    def _send_notification(self, method: str, params: dict | None = None) -> None:
        notification: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            notification["params"] = params
        self._send(notification)

    def _send(self, payload: dict) -> None:
        try:
            self._transport.send(jsonrpc.encode_message(payload))
        except TransportClosedError as exc:
            raise _DegradedOutcome("unavailable", str(exc)) from exc
        except OSError as exc:
            raise _DegradedOutcome("unavailable", str(exc)) from exc

    def _send_and_receive(self, request: dict, *, timeout: float | None = None) -> dict:
        self._send(request)
        return self._await_response(
            request["id"], timeout=self._timeout if timeout is None else timeout
        )

    def _await_response(self, request_id: int, *, timeout: float) -> dict:
        """Block until the response with ``request_id`` arrives, or time out.

        Notifications, server-initiated requests, and responses to other
        requests are skipped; the whole wait is bounded by a single deadline
        so a chatty server cannot extend it indefinitely.
        """
        deadline = time.monotonic() + timeout

        while True:
            message = self._take_response(request_id)
            if message is not None:
                return self._unwrap_response(message)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _DegradedOutcome(
                    "timeout",
                    f"no clangd response to request {request_id} within {timeout:.3f}s",
                )

            self._receive_buffer += self._recv(remaining)

    def _recv(self, timeout: float) -> bytes:
        try:
            return self._transport.recv(timeout)
        except TransportTimeoutError as exc:
            raise _DegradedOutcome("timeout", str(exc)) from exc
        except TransportClosedError as exc:
            raise _DegradedOutcome("unavailable", str(exc)) from exc
        except OSError as exc:
            raise _DegradedOutcome("unavailable", str(exc)) from exc

    def _take_response(self, request_id: int) -> dict | None:
        """Pop the response for ``request_id`` from what has already arrived.

        Frames decoded but not consumed stay queued in ``_pending_messages``,
        and a trailing partial frame stays in ``_receive_buffer``, so nothing
        read on behalf of one request is lost to the next one.
        """
        self._pending_messages.extend(self._decode_buffered())

        while self._pending_messages:
            message = self._pending_messages.pop(0)
            if self._is_response_to(message, request_id):
                return message
            # Anything else -- a notification, a server-initiated request, or
            # a late response to a request we already gave up on -- is not
            # ours to return, and is dropped.

        return None

    def _decode_buffered(self) -> list[dict]:
        try:
            messages, self._receive_buffer = jsonrpc.decode_messages(
                self._receive_buffer
            )
        except jsonrpc.JsonRpcFramingError as exc:
            # The stream is no longer parseable at this offset; keeping the
            # bytes would just replay the same failure forever.
            self._receive_buffer = b""
            raise _DegradedOutcome("error", f"malformed response frame: {exc}") from exc
        return messages

    @staticmethod
    def _is_response_to(message: dict, request_id: int) -> bool:
        # A frame carrying a "method" is a request or notification from the
        # server, never a response to us -- even if it reuses our id.
        return "method" not in message and message.get("id") == request_id

    @staticmethod
    def _unwrap_response(message: dict) -> dict:
        error = message.get("error")
        if error is not None:
            reason = error.get("message") if isinstance(error, dict) else None
            raise _DegradedOutcome("error", str(reason if reason is not None else error))
        return message

    # -- path / position / URI conversion ------------------------------------

    def _root_uri(self) -> str:
        return f"file://{self._kernel_repo.resolve().as_posix()}"

    def _to_uri(self, relative_path: str) -> str:
        absolute = to_absolute_path(relative_path, kernel_repo=self._kernel_repo)
        return f"file://{absolute.as_posix()}"

    def _uri_to_relative_path(self, uri: str) -> str:
        parsed = urlparse(uri)
        absolute = Path(unquote(parsed.path))
        return normalize_relative_path(absolute, kernel_repo=self._kernel_repo)

    @staticmethod
    def _to_lsp_position(line: int, column: int) -> dict:
        return {"line": line - 1, "character": column - 1}

    @staticmethod
    def _lsp_position_to_1_based(position: dict) -> tuple[int, int]:
        return position["line"] + 1, position["character"] + 1

    # -- LSP payload <-> model conversion ------------------------------------

    def _parse_call_hierarchy_item(self, entry: dict) -> CallHierarchyItem:
        line, column = self._lsp_position_to_1_based(entry["range"]["start"])
        return CallHierarchyItem(
            name=entry["name"],
            relative_path=self._uri_to_relative_path(entry["uri"]),
            line=line,
            column=column,
        )

    def _call_hierarchy_item_to_lsp(self, item: CallHierarchyItem) -> dict:
        position = self._to_lsp_position(item.line, item.column)
        lsp_range = {
            "start": position,
            "end": {"line": position["line"], "character": position["character"] + 1},
        }
        return {
            "name": item.name,
            "kind": 12,  # SymbolKind.Function
            "uri": self._to_uri(item.relative_path),
            "range": lsp_range,
            "selectionRange": lsp_range,
        }

    def _parse_outgoing_call(
        self, entry: dict, *, source_relative_path: str
    ) -> OutgoingCall:
        target = self._parse_call_hierarchy_item(entry["to"])
        call_sites = tuple(
            self._parse_location_from_range(source_relative_path, call_range)
            for call_range in entry.get("fromRanges", [])
        )
        return OutgoingCall(target=target, call_sites=call_sites)

    def _parse_location(self, entry: dict) -> SemanticLocation:
        relative_path = self._uri_to_relative_path(entry["uri"])
        return self._parse_location_from_range(relative_path, entry["range"])

    @staticmethod
    def _parse_location_from_range(relative_path: str, lsp_range: dict) -> SemanticLocation:
        line, column = ClangdAdapter._lsp_position_to_1_based(lsp_range["start"])
        return SemanticLocation(relative_path=relative_path, line=line, column=column)
