"""In-memory/fake LSP transport used to unit-test the clangd adapter without
spawning a real ``clangd`` process or touching the network/filesystem.

This module is test support code only; it is intentionally kept out of
``src/`` because it has no production purpose.
"""

from __future__ import annotations

import json

from netstack_academy.indexing.semantic.jsonrpc import encode_message
from netstack_academy.indexing.semantic.transport import (
    TransportClosedError,
    TransportTimeoutError,
)


class FakeLspTransport:
    """A scriptable stand-in for a real clangd stdio/socket transport.

    Tests script responses (or raw byte chunks, to exercise partial-frame
    reassembly) and then drive a :class:`ClangdAdapter` against this fake,
    asserting on both the outgoing requests recorded in ``sent_messages`` and
    the values the adapter derives from scripted responses.
    """

    def __init__(self) -> None:
        self.sent_messages: list[dict] = []
        self.closed = False
        self._recv_queue: list[bytes] = []
        self._raise_timeout_next = False
        self._raise_closed_on_send = False

    def script_response(self, result: object = None, *, error: dict | None = None) -> None:
        """Queue a properly framed response correlated to the last sent request id."""
        if not self.sent_messages:
            raise AssertionError("cannot script a response before a request was sent")
        request_id = self.sent_messages[-1]["id"]
        payload: dict = {"jsonrpc": "2.0", "id": request_id}
        if error is not None:
            payload["error"] = error
        else:
            payload["result"] = result
        self._recv_queue.append(encode_message(payload))

    def script_raw_chunks(self, *chunks: bytes) -> None:
        """Queue raw bytes to be returned verbatim on successive ``recv`` calls."""
        self._recv_queue.extend(chunks)

    def fail_next_recv_with_timeout(self) -> None:
        self._raise_timeout_next = True

    def fail_send_with_closed(self) -> None:
        self._raise_closed_on_send = True

    def send(self, data: bytes) -> None:
        if self._raise_closed_on_send:
            raise TransportClosedError("transport is closed")
        _header, _, body = data.partition(b"\r\n\r\n")
        self.sent_messages.append(json.loads(body.decode("utf-8")))

    def recv(self, timeout: float) -> bytes:
        if self._raise_timeout_next:
            self._raise_timeout_next = False
            raise TransportTimeoutError("timed out waiting for clangd response")
        if not self._recv_queue:
            raise TransportTimeoutError("no scripted response available")
        return self._recv_queue.pop(0)

    def close(self) -> None:
        self.closed = True
