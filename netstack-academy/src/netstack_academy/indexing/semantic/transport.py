"""Transport abstraction for a synchronous LSP (``clangd``) client, plus the
one concrete implementation that owns a real ``clangd`` subprocess.

``ClangdAdapter`` only ever talks to an :class:`LspTransport` -- it never
constructs a subprocess argv, shell command string, or socket itself. That
keeps the adapter fully testable against an in-memory fake
(``tests/indexing/lsp_fakes.py``'s ``FakeLspTransport``) and keeps the choice
of *how* a real ``clangd`` process is started (and with what argv -- never an
arbitrary shell string) with the caller that constructs the transport.

:class:`StdioLspTransport` is that concrete implementation: it launches a
caller-supplied **argv list** (a command *string* is rejected outright, so
there is no path by which a shell could ever interpret metacharacters),
keeps exactly one process for the life of the transport, bounds every read
by the caller's timeout, and always reaps the process on ``close``.

``subprocess`` is imported at module level (not ``from subprocess import
Popen``) so tests can monkeypatch ``transport.subprocess.Popen`` directly,
matching the ``ctags_runner``/``repo_inspector`` convention. Reads use
``select`` on the process's stdout pipe, which is a POSIX (Linux/WSL)
assumption -- the same platform assumption the rest of this app already
makes.
"""

from __future__ import annotations

import os
import select
import subprocess
from pathlib import Path
from typing import IO, Callable, Protocol, Sequence, runtime_checkable


class TransportTimeoutError(Exception):
    """Raised by :meth:`LspTransport.recv` when no response arrives in time."""


class TransportClosedError(Exception):
    """Raised when ``send``/``recv`` is attempted on a closed/unavailable transport."""


@runtime_checkable
class LspTransport(Protocol):
    """The minimal surface :class:`ClangdAdapter` needs from a transport.

    A real implementation would frame/deliver bytes over a subprocess's
    stdio pipes or a socket; ``FakeLspTransport`` implements the same
    protocol purely in-memory for tests.
    """

    def send(self, data: bytes) -> None:
        """Write a fully framed JSON-RPC message. Raises :class:`TransportClosedError`
        if the transport is closed/unavailable."""
        ...

    def recv(self, timeout: float) -> bytes:
        """Block for up to ``timeout`` seconds and return newly available bytes.

        Raises :class:`TransportTimeoutError` if nothing arrives in time, or
        :class:`TransportClosedError` if the transport is closed.
        """
        ...

    def close(self) -> None:
        """Release any underlying resources. Safe to call more than once."""
        ...


#: How long ``close`` waits for the process to die before escalating.
TERMINATE_TIMEOUT_SECONDS = 5.0

_READ_CHUNK_BYTES = 65536


class StdioLspTransport:
    """Owns one ``clangd`` subprocess and frames bytes over its stdio pipes.

    ``argv`` must be a sequence of arguments -- a command string raises
    :class:`TypeError` rather than being split or handed to a shell, and the
    process is always launched without ``shell=True``. A binary that cannot
    be launched at all (missing, not executable, ...) raises
    :class:`TransportClosedError` instead of leaking an ``OSError``, so
    callers have a single failure type to degrade on.
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        terminate_timeout: float = TERMINATE_TIMEOUT_SECONDS,
    ) -> None:
        if isinstance(argv, (str, bytes)):
            raise TypeError(
                "argv must be a sequence of arguments, never a command string "
                f"(got {argv!r})"
            )

        arguments = list(argv)
        if not arguments:
            raise ValueError("argv must contain at least an executable name")
        if not all(isinstance(argument, str) for argument in arguments):
            raise TypeError(f"argv must contain only strings (got {arguments!r})")

        self._argv = arguments
        self._cwd = Path(cwd)
        self._terminate_timeout = terminate_timeout
        self._closed = False

        try:
            self._process = subprocess.Popen(
                arguments,
                cwd=self._cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # Nothing drains stderr, so it must not be a pipe that could
                # fill up and wedge the server (clangd logs are verbose).
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except (OSError, ValueError) as exc:
            raise TransportClosedError(
                f"could not start {arguments[0]!r}: {exc}"
            ) from exc

        self._stdin = self._process.stdin
        self._stdout = self._process.stdout
        if self._stdin is None or self._stdout is None:  # defensive
            self.close()
            raise TransportClosedError(
                f"{arguments[0]!r} was started without usable stdio pipes"
            )

    @property
    def argv(self) -> tuple[str, ...]:
        return tuple(self._argv)

    @property
    def pid(self) -> int | None:
        return getattr(self._process, "pid", None)

    def send(self, data: bytes) -> None:
        if self._closed:
            raise TransportClosedError("transport is closed")

        view = memoryview(data)
        try:
            while view:
                written = self._stdin.write(view)
                if not written:
                    raise TransportClosedError("clangd accepted no bytes on stdin")
                view = view[written:]
            self._stdin.flush()
        except TransportClosedError:
            raise
        except ValueError as exc:  # write on an already-closed pipe
            raise TransportClosedError(f"clangd stdin is closed: {exc}") from exc
        except OSError as exc:
            raise TransportClosedError(f"writing to clangd failed: {exc}") from exc

    def recv(self, timeout: float) -> bytes:
        if self._closed:
            raise TransportClosedError("transport is closed")

        try:
            readable, _, _ = select.select([self._stdout], [], [], max(timeout, 0.0))
        except (OSError, ValueError) as exc:
            raise TransportClosedError(f"clangd stdout is closed: {exc}") from exc

        if not readable:
            returncode = self._process.poll()
            if returncode is not None:
                raise TransportClosedError(f"clangd exited with code {returncode}")
            raise TransportTimeoutError(
                f"no clangd output within {max(timeout, 0.0):.3f}s"
            )

        try:
            chunk = os.read(self._stdout.fileno(), _READ_CHUNK_BYTES)
        except (OSError, ValueError) as exc:
            raise TransportClosedError(f"reading from clangd failed: {exc}") from exc

        if not chunk:
            raise TransportClosedError("clangd closed its stdout")

        return chunk

    def close(self) -> None:
        """Close the pipes and reap the process. Safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True

        for handle in (self._stdin, self._stdout, self._process.stderr):
            _close_quietly(handle)

        if self._process.poll() is not None:
            return

        _call_quietly(self._process.terminate)
        if self._reap():
            return

        _call_quietly(self._process.kill)
        self._reap()

    def _reap(self) -> bool:
        """Wait (bounded) for the process; ``False`` if it is still alive."""
        try:
            self._process.wait(timeout=self._terminate_timeout)
        except subprocess.TimeoutExpired:
            return False
        except OSError:
            return True
        return True


def _close_quietly(handle: IO[bytes] | None) -> None:
    if handle is None:
        return
    try:
        handle.close()
    except (OSError, ValueError):
        pass


def _call_quietly(action: Callable[[], None]) -> None:
    try:
        action()
    except (OSError, ValueError):
        pass
