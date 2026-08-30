"""Pure ``Content-Length``-framed JSON-RPC codec, as used by the Language
Server Protocol (and thus ``clangd``) over stdio/socket transports.

This module has no knowledge of sockets, subprocesses, or ``clangd``
specifically -- it only encodes/decodes byte buffers. That keeps it trivially
unit-testable and reusable by both a real transport and
``tests/indexing/lsp_fakes.py``'s in-memory fake.

Framing is::

    Content-Length: <byte length of body>\\r\\n
    \\r\\n
    <body bytes, itself UTF-8 encoded JSON>

The declared length is a *byte* length, not a character length -- this
matters as soon as the payload contains any non-ASCII text.
"""

from __future__ import annotations

import json

_HEADER_BODY_SEPARATOR = b"\r\n\r\n"
_HEADER_LINE_SEPARATOR = b"\r\n"
_CONTENT_LENGTH_FIELD = b"content-length"


class JsonRpcFramingError(ValueError):
    """Raised when a message frame's header is structurally malformed.

    This is distinct from an *incomplete* frame (not enough bytes have
    arrived yet, which is a normal, expected condition when reading from a
    streaming transport): a malformed header -- e.g. a non-numeric
    ``Content-Length`` value, or a header block with no ``Content-Length``
    field at all -- can never be fixed by reading more bytes, so it is
    reported as an explicit error instead of silently waiting forever.
    """


def encode_message(payload: dict) -> bytes:
    """Frame ``payload`` as a single LSP ``Content-Length``-delimited message."""
    body = json.dumps(payload).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def decode_messages(buffer: bytes) -> tuple[list[dict], bytes]:
    """Decode as many complete frames as are present at the start of ``buffer``.

    Returns ``(messages, remaining)`` where ``messages`` is every fully
    decoded JSON payload, in order, and ``remaining`` is whatever bytes are
    left over after the last complete frame (an empty, partial header, or a
    header-plus-partial-body). Incomplete frames never raise -- callers are
    expected to accumulate more bytes and retry. A structurally malformed
    header (e.g. a non-numeric ``Content-Length``) raises
    :class:`JsonRpcFramingError` immediately, since more bytes cannot help.
    """
    messages: list[dict] = []
    remaining = buffer

    while True:
        separator_index = remaining.find(_HEADER_BODY_SEPARATOR)
        if separator_index == -1:
            break

        header_block = remaining[:separator_index]
        content_length = _parse_content_length(header_block)

        body_start = separator_index + len(_HEADER_BODY_SEPARATOR)
        body_end = body_start + content_length
        if len(remaining) < body_end:
            break

        body = remaining[body_start:body_end]
        messages.append(json.loads(body.decode("utf-8")))
        remaining = remaining[body_end:]

    return messages, remaining


def _parse_content_length(header_block: bytes) -> int:
    for line in header_block.split(_HEADER_LINE_SEPARATOR):
        if not line:
            continue
        name, _, value = line.partition(b":")
        if name.strip().lower() != _CONTENT_LENGTH_FIELD:
            continue
        try:
            return int(value.strip())
        except ValueError as exc:
            raise JsonRpcFramingError(
                f"Malformed Content-Length header: {line!r}"
            ) from exc

    raise JsonRpcFramingError(
        f"Message header has no Content-Length field: {header_block!r}"
    )
