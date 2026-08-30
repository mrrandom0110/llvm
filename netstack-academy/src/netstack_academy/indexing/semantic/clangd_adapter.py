"""Synchronous ``clangd`` semantic provider adapter.

``ClangdAdapter`` speaks the LSP subset needed for call-graph enrichment
(``textDocument/prepareCallHierarchy``, ``callHierarchy/outgoingCalls``,
``textDocument/references``) over an injected :class:`LspTransport`. It
never starts a process or opens a socket itself, and never accepts a shell
command string -- callers own how (and whether) a real ``clangd`` binary is
launched and wired up to a transport; this adapter only needs something
that implements ``send``/``recv``/``close``. That is what makes it testable
against ``tests/indexing/lsp_fakes.py``'s in-memory ``FakeLspTransport``
with no real ``clangd`` process, socket, or subprocess anywhere in the test
run, while still permitting a real transport to be plugged in later.

Position convention: callers of this adapter always use the same 1-based
``(line, column)`` convention as the rest of ``indexing`` (ctags, the
fallback indexer); LSP's 0-based positions are an internal, private detail
of this module, converted at the edges. Likewise, callers only ever see
kernel-repo-relative POSIX paths, never raw ``file://`` URIs -- URI
construction/parsing is also entirely internal to this module.

This adapter is deliberately synchronous and single-request-in-flight: each
public method sends exactly one request and then blocks for exactly one
response. It does not correlate the JSON-RPC ``id`` of the response against
the request it just sent (there is, by construction, never more than one
request outstanding), and it does not handle server-initiated requests or
notifications arriving interleaved with responses. Both are acceptable
simplifications for the bounded, one-shot call-hierarchy/reference lookups
this adapter performs, but would need to be revisited before layering
request pipelining on top of a real ``clangd`` connection.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..paths import normalize_relative_path, to_absolute_path
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

_PROVIDER_NAME = "clangd"


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
        self._kernel_repo = kernel_repo
        self._timeout = timeout
        self._request_ids = itertools.count(1)
        self._unavailable_reason: str | None = None
        self._send_initialize()

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

    # -- transport plumbing --------------------------------------------------

    def _next_id(self) -> int:
        return next(self._request_ids)

    def _send_initialize(self) -> None:
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {"processId": None, "rootUri": None, "capabilities": {}},
        }
        try:
            self._transport.send(jsonrpc.encode_message(request))
        except (TransportClosedError, OSError) as exc:
            self._unavailable_reason = f"clangd transport unavailable: {exc}"

    def _build_request(self, method: str, params: dict) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params,
        }

    def _send_and_receive(self, request: dict) -> dict:
        if self._unavailable_reason is not None:
            raise _DegradedOutcome("unavailable", self._unavailable_reason)

        try:
            self._transport.send(jsonrpc.encode_message(request))
        except TransportClosedError as exc:
            raise _DegradedOutcome("unavailable", str(exc)) from exc
        except OSError as exc:
            raise _DegradedOutcome("unavailable", str(exc)) from exc

        try:
            raw = self._transport.recv(self._timeout)
        except TransportTimeoutError as exc:
            raise _DegradedOutcome("timeout", str(exc)) from exc
        except TransportClosedError as exc:
            raise _DegradedOutcome("unavailable", str(exc)) from exc
        except OSError as exc:
            raise _DegradedOutcome("unavailable", str(exc)) from exc

        try:
            messages, _ = jsonrpc.decode_messages(raw)
        except jsonrpc.JsonRpcFramingError as exc:
            raise _DegradedOutcome("error", f"malformed response frame: {exc}") from exc

        if not messages:
            raise _DegradedOutcome(
                "error", "no complete JSON-RPC message frame in transport response"
            )

        message = messages[0]
        error = message.get("error")
        if error is not None:
            reason = error.get("message") if isinstance(error, dict) else None
            raise _DegradedOutcome("error", str(reason if reason is not None else error))

        return message

    # -- path / position / URI conversion ------------------------------------

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
