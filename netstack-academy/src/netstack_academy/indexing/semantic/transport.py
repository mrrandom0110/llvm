"""Transport abstraction for a synchronous LSP (``clangd``) client.

``ClangdAdapter`` only ever talks to an :class:`LspTransport` -- it never
constructs a subprocess argv, shell command string, or socket itself. That
keeps the adapter fully testable against an in-memory fake
(``tests/indexing/lsp_fakes.py``'s ``FakeLspTransport``) and keeps the choice
of *how* a real ``clangd`` process is started (and with what argv -- never an
arbitrary shell string) entirely with the caller that constructs the real
transport implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


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
