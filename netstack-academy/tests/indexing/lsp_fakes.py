"""Fakes for testing the clangd integration without a real ``clangd`` binary.

Two independent fakes live here:

- :class:`FakeLspTransport` -- an in-memory :class:`LspTransport` that
  emulates a *correct* language server. It decodes the frames the adapter
  sends, answers every request with a response carrying **that request's own
  JSON-RPC id**, never answers notifications, and can interleave server
  notifications, server-initiated requests, foreign/stale responses, and
  arbitrarily split byte chunks into the stream. Tests script it by *method*
  (``queue_response("textDocument/references", result=...)``) rather than by
  "the last id seen", so a scripted response is always correlated the way a
  real server would correlate it.
- :class:`FakeClangdProcess`, :class:`RecordingPopen` and
  :class:`FakeClangdServer` -- a pipe-backed stand-in for
  ``subprocess.Popen`` plus a tiny threaded server, used to exercise the
  real ``StdioLspTransport`` (framing over OS pipes, process ownership,
  termination) without launching any external executable.

This module is test support code only; it is intentionally kept out of
``src/`` because it has no production purpose.

Note on the previous design: ``script_response`` used to derive the response
id from ``sent_messages[-1]["id"]`` at scripting time. Because every test
scripts a response *before* triggering the request it answers, that id was
always the id of the *previous* request (the construction-time
``initialize``), so the fake systematically handed the adapter mismatched
response ids -- forcing the adapter to ignore ids entirely. That is a defect
in the fake, not a protocol property, and it is gone: responses here are
generated at send time from the real request.
"""

from __future__ import annotations

import os
import select
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from netstack_academy.indexing.semantic.jsonrpc import decode_messages, encode_message
from netstack_academy.indexing.semantic.transport import (
    TransportClosedError,
    TransportTimeoutError,
)

#: Results the fake server returns for lifecycle methods unless a test
#: scripts something else (or suppresses the reply entirely).
DEFAULT_SERVER_RESULTS: dict[str, Any] = {
    "initialize": {
        "capabilities": {"callHierarchyProvider": True, "referencesProvider": True}
    },
    "shutdown": None,
}

#: A server message factory: either a literal payload, or a callable that
#: receives the client request currently being answered (so a test can build
#: a server message that reuses the client's own request id).
ServerMessage = dict | Callable[[dict], dict]


class FakeLspTransport:
    """A scriptable, protocol-correct stand-in for a real clangd transport.

    Tests script per-method responses, optionally interleave extra server
    traffic, and optionally force the byte stream to be delivered in small
    chunks (``chunk_size``) to exercise partial-frame reassembly.
    """

    def __init__(self, *, chunk_size: int | None = None) -> None:
        self.sent_messages: list[dict] = []
        self.closed = False
        self.recv_calls = 0
        self.chunk_size = chunk_size
        self._outgoing = bytearray()
        self._scripted: dict[str, list[dict]] = {}
        self._suppressed: set[str] = set()
        self._prefix: list[ServerMessage] = []
        self._trailing: list[bytes] = []
        self._raise_timeout_next = False
        self._raise_closed_on_send = False

    # -- scripting -----------------------------------------------------------

    def queue_response(
        self, method: str, *, result: object = None, error: dict | None = None
    ) -> None:
        """Answer the next request for ``method`` with ``result`` (or ``error``)."""
        payload = {"error": error} if error is not None else {"result": result}
        self._scripted.setdefault(method, []).append(payload)

    def suppress_response(self, method: str) -> None:
        """Make the server never answer ``method`` (so ``recv`` times out)."""
        self._suppressed.add(method)

    def queue_server_message(self, message: ServerMessage) -> None:
        """Emit ``message`` just before the next generated response.

        ``message`` may be a callable, which is invoked with the client
        request being answered -- useful for building a server-initiated
        request that deliberately reuses the client's own id.
        """
        self._prefix.append(message)

    def queue_trailing_raw(self, data: bytes) -> None:
        """Append raw bytes immediately after the next generated response."""
        self._trailing.append(data)

    def emit_message(self, payload: dict) -> None:
        """Frame and append ``payload`` to the stream right now."""
        self._outgoing += encode_message(payload)

    def emit_raw(self, data: bytes) -> None:
        """Append raw (possibly partial-frame) bytes to the stream right now."""
        self._outgoing += data

    def fail_next_recv_with_timeout(self) -> None:
        self._raise_timeout_next = True

    def fail_send_with_closed(self) -> None:
        self._raise_closed_on_send = True

    # -- inspection ----------------------------------------------------------

    @property
    def sent_methods(self) -> list[str]:
        return [
            message["method"] for message in self.sent_messages if "method" in message
        ]

    def messages_for(self, method: str) -> list[dict]:
        return [
            message
            for message in self.sent_messages
            if message.get("method") == method
        ]

    def pending_bytes(self) -> int:
        return len(self._outgoing)

    # -- LspTransport --------------------------------------------------------

    def send(self, data: bytes) -> None:
        if self._raise_closed_on_send or self.closed:
            raise TransportClosedError("transport is closed")

        messages, remaining = decode_messages(data)
        if remaining:
            raise AssertionError(
                f"adapter sent an incomplete JSON-RPC frame: {remaining!r}"
            )

        for message in messages:
            self.sent_messages.append(message)
            self._respond_to(message)

    def recv(self, timeout: float) -> bytes:
        self.recv_calls += 1
        if self._raise_timeout_next:
            self._raise_timeout_next = False
            raise TransportTimeoutError("timed out waiting for clangd response")
        if self.closed:
            raise TransportClosedError("transport is closed")
        if not self._outgoing:
            raise TransportTimeoutError("no server bytes available")

        size = (
            len(self._outgoing)
            if self.chunk_size is None
            else min(self.chunk_size, len(self._outgoing))
        )
        chunk = bytes(self._outgoing[:size])
        del self._outgoing[:size]
        return chunk

    def close(self) -> None:
        self.closed = True

    # -- server behaviour ----------------------------------------------------

    def _respond_to(self, message: dict) -> None:
        if "id" not in message:
            return  # a notification is never answered by a real server

        for extra in self._prefix:
            payload = extra(message) if callable(extra) else extra
            self._outgoing += encode_message(payload)
        self._prefix.clear()

        response_payload = self._payload_for(str(message.get("method", "")))
        if response_payload is None:
            return

        self._outgoing += encode_message(
            {"jsonrpc": "2.0", "id": message["id"], **response_payload}
        )

        for trailing in self._trailing:
            self._outgoing += trailing
        self._trailing.clear()

    def _payload_for(self, method: str) -> dict | None:
        scripted = self._scripted.get(method)
        if scripted:
            return scripted.pop(0)
        if method in self._suppressed:
            return None
        if method in DEFAULT_SERVER_RESULTS:
            return {"result": DEFAULT_SERVER_RESULTS[method]}
        return None


class FakeClangdProcess:
    """A pipe-backed stand-in for the handle ``subprocess.Popen`` returns.

    Real OS pipes are used for stdin/stdout so that any reasonable transport
    implementation works against it -- ``select``, a reader thread, or plain
    blocking reads all see a genuine file descriptor.
    """

    def __init__(
        self, argv: Sequence[str], *, ignore_terminate: bool = False, **popen_kwargs: Any
    ) -> None:
        self.argv = list(argv)
        self.popen_kwargs = dict(popen_kwargs)
        self.cwd = popen_kwargs.get("cwd")
        self.ignore_terminate = ignore_terminate

        stdin_read, stdin_write = os.pipe()
        stdout_read, stdout_write = os.pipe()
        # Client (transport) ends, named as ``Popen`` names them.
        self.stdin = os.fdopen(stdin_write, "wb", buffering=0)
        self.stdout = os.fdopen(stdout_read, "rb", buffering=0)
        self.stderr = None
        # Server (test) ends.
        self.server_input = os.fdopen(stdin_read, "rb", buffering=0)
        self.server_output = os.fdopen(stdout_write, "wb", buffering=0)

        self.pid = 4242
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    # -- Popen surface -------------------------------------------------------

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls += 1
        if self.returncode is None:
            raise subprocess.TimeoutExpired(cmd=self.argv, timeout=timeout or 0)
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.ignore_terminate:
            return
        self._exit(-15)

    def kill(self) -> None:
        self.kill_calls += 1
        self._exit(-9)

    # -- server-side helpers -------------------------------------------------

    def write_to_client(self, data: bytes) -> None:
        self.server_output.write(data)

    def read_from_client(self, size: int = 65536, timeout: float | None = None) -> bytes:
        """Read whatever the transport has written to the process's stdin.

        With ``timeout`` set, raises :class:`TimeoutError` instead of blocking
        forever, so a transport that never flushes fails fast and loudly.
        """
        if timeout is not None:
            ready, _, _ = select.select([self.server_input], [], [], timeout)
            if not ready:
                raise TimeoutError("transport wrote nothing to the process stdin")
        return self.server_input.read(size) or b""

    def close_server_output(self) -> None:
        """Simulate the server closing its stdout (EOF for the transport)."""
        _close_quietly(self.server_output)

    def close_server_input(self) -> None:
        _close_quietly(self.server_input)

    def cleanup(self) -> None:
        for handle in (self.stdin, self.stdout, self.server_input, self.server_output):
            _close_quietly(handle)

    def _exit(self, returncode: int) -> None:
        self.returncode = returncode
        self.close_server_output()


def _close_quietly(handle: Any) -> None:
    try:
        handle.close()
    except (OSError, ValueError):
        pass


@dataclass
class PopenCall:
    """One recorded ``subprocess.Popen(...)`` invocation."""

    argv: Any
    kwargs: dict = field(default_factory=dict)


class RecordingPopen:
    """A ``subprocess.Popen`` replacement that records how it was called.

    Set ``error`` to make every launch fail (a missing or non-executable
    binary), ``ignore_terminate`` to simulate a process that survives
    ``terminate()``, and ``serve=True`` to attach a :class:`FakeClangdServer`
    to each launched process so a real handshake can complete.
    """

    def __init__(
        self,
        *,
        error: BaseException | None = None,
        ignore_terminate: bool = False,
        serve: bool = False,
        results: dict[str, Any] | None = None,
    ) -> None:
        self.calls: list[PopenCall] = []
        self.processes: list[FakeClangdProcess] = []
        self.servers: list[FakeClangdServer] = []
        self._error = error
        self._ignore_terminate = ignore_terminate
        self._serve = serve
        self._results = results

    def __call__(self, argv: Any = None, **kwargs: Any) -> FakeClangdProcess:
        if argv is None:
            argv = kwargs.get("args")
        self.calls.append(PopenCall(argv=argv, kwargs=dict(kwargs)))
        if self._error is not None:
            raise self._error

        process = FakeClangdProcess(
            argv, ignore_terminate=self._ignore_terminate, **kwargs
        )
        self.processes.append(process)
        if self._serve:
            self.servers.append(
                FakeClangdServer(process, results=self._results).start()
            )
        return process

    def shutdown(self) -> None:
        for server in self.servers:
            server.stop()
        for process in self.processes:
            process.cleanup()


class FakeClangdServer:
    """A minimal, correct LSP server driven over a :class:`FakeClangdProcess`.

    Runs on a daemon thread so a transport can complete a real handshake
    against it: every request is answered with a response carrying the
    request's own id; notifications are recorded but never answered.
    """

    def __init__(
        self, process: FakeClangdProcess, *, results: dict[str, Any] | None = None
    ) -> None:
        self._process = process
        self._results = {**DEFAULT_SERVER_RESULTS, **(results or {})}
        self.received: list[dict] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> "FakeClangdServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._process.close_server_input()
        self._thread.join(timeout=5.0)

    @property
    def received_methods(self) -> list[str]:
        with self._lock:
            return [
                message["method"] for message in self.received if "method" in message
            ]

    def wait_for_method(self, method: str, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if method in self.received_methods:
                return True
            time.sleep(0.01)
        return False

    def _serve(self) -> None:
        buffer = b""
        while True:
            try:
                chunk = self._process.read_from_client()
            except (OSError, ValueError):
                return
            if not chunk:
                return

            buffer += chunk
            try:
                messages, buffer = decode_messages(buffer)
            except Exception:  # a malformed frame ends this fake session
                return

            for message in messages:
                with self._lock:
                    self.received.append(message)
                self._answer(message)

    def _answer(self, message: dict) -> None:
        if "id" not in message:
            return
        method = str(message.get("method", ""))
        result = self._results.get(method, [])
        try:
            self._process.write_to_client(
                encode_message({"jsonrpc": "2.0", "id": message["id"], "result": result})
            )
        except (OSError, ValueError):
            pass
