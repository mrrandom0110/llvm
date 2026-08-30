from __future__ import annotations

import json

import pytest

from netstack_academy.indexing.semantic.jsonrpc import (
    JsonRpcFramingError,
    decode_messages,
    encode_message,
)


def test_encode_message_produces_lsp_header_and_body() -> None:
    payload = {"jsonrpc": "2.0", "id": 1, "method": "ping"}

    framed = encode_message(payload)

    header, _, body = framed.partition(b"\r\n\r\n")
    assert header.startswith(b"Content-Length: ")
    assert json.loads(body.decode("utf-8")) == payload


def test_encode_message_uses_byte_length_not_character_length() -> None:
    payload = {"jsonrpc": "2.0", "id": 1, "method": "café"}

    framed = encode_message(payload)

    header, _, body = framed.partition(b"\r\n\r\n")
    declared_length = int(header.split(b":")[1].strip())
    assert declared_length == len(body)
    assert declared_length != len(json.dumps(payload))


def test_decode_messages_parses_single_complete_frame() -> None:
    payload = {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
    framed = encode_message(payload)

    messages, remaining = decode_messages(framed)

    assert messages == [payload]
    assert remaining == b""


def test_decode_messages_parses_multiple_concatenated_frames_in_order() -> None:
    first = {"jsonrpc": "2.0", "id": 1, "result": "first"}
    second = {"jsonrpc": "2.0", "id": 2, "result": "second"}
    buffer = encode_message(first) + encode_message(second)

    messages, remaining = decode_messages(buffer)

    assert messages == [first, second]
    assert remaining == b""


def test_decode_messages_waits_for_more_data_on_incomplete_body() -> None:
    payload = {"jsonrpc": "2.0", "id": 1, "result": "value"}
    framed = encode_message(payload)
    partial = framed[:-5]

    messages, remaining = decode_messages(partial)

    assert messages == []
    assert remaining == partial


def test_decode_messages_waits_for_more_data_on_incomplete_header() -> None:
    partial_header = b"Content-Length: 12"

    messages, remaining = decode_messages(partial_header)

    assert messages == []
    assert remaining == partial_header


def test_decode_messages_roundtrips_through_encode_message() -> None:
    payload = {"jsonrpc": "2.0", "id": 42, "method": "textDocument/references"}

    messages, remaining = decode_messages(encode_message(payload))

    assert messages == [payload]
    assert remaining == b""


def test_decode_messages_raises_explicit_error_for_malformed_content_length() -> None:
    corrupt = b"Content-Length: not-a-number\r\n\r\n{}"

    with pytest.raises(JsonRpcFramingError):
        decode_messages(corrupt)


def test_decode_messages_returns_remaining_bytes_after_last_complete_frame() -> None:
    payload = {"jsonrpc": "2.0", "id": 1, "result": "value"}
    framed = encode_message(payload)
    trailing_partial = b"Content-Length: 5\r\n\r\n{\"a\""
    buffer = framed + trailing_partial

    messages, remaining = decode_messages(buffer)

    assert messages == [payload]
    assert remaining == trailing_partial
