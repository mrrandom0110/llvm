"""Lifecycle contract for :class:`ClangdAdapter` against a real clangd session.

These tests pin down the parts of the LSP session a real ``clangd`` process
requires and that an in-memory fake previously let us skip: a completed
``initialize``/``initialized`` handshake against the kernel repo root, a
``textDocument/didOpen`` before any semantic request touching a file,
response correlation by JSON-RPC id in the presence of interleaved server
traffic, byte-accurate reassembly of partial/multiple frames across ``recv``
calls, and a bounded ``shutdown``/``exit``/close sequence that stays safe
when the handshake never succeeded.
"""

from __future__ import annotations

import time
from pathlib import Path

from netstack_academy.indexing.semantic.clangd_adapter import ClangdAdapter
from netstack_academy.indexing.semantic.jsonrpc import encode_message
from netstack_academy.indexing.semantic.models import CallHierarchyItem
from netstack_academy.indexing.semantic.transport import LspTransport

from lsp_fakes import FakeLspTransport

FIXTURE_PATH = "net/ipv4/tcp_input.c"

LOG_NOTIFICATION = {
    "jsonrpc": "2.0",
    "method": "window/logMessage",
    "params": {"type": 3, "message": "indexing kernel sources"},
}

FOREIGN_RESPONSE = {"jsonrpc": "2.0", "id": 987654, "result": [{"bogus": True}]}


def _uri(kernel_repo: Path, relative_path: str = FIXTURE_PATH) -> str:
    return f"file://{(kernel_repo / relative_path).resolve().as_posix()}"


def _call_hierarchy_result(
    kernel_repo: Path,
    *,
    name: str = "tcp_input",
    relative_path: str = FIXTURE_PATH,
    line: int = 41,
    character: int = 4,
) -> list[dict]:
    return [
        {
            "name": name,
            "uri": _uri(kernel_repo, relative_path),
            "range": {
                "start": {"line": line, "character": character},
                "end": {"line": line, "character": character + len(name)},
            },
        }
    ]


def _references_result(
    kernel_repo: Path, *, line: int = 99, character: int = 2
) -> list[dict]:
    return [
        {
            "uri": _uri(kernel_repo),
            "range": {
                "start": {"line": line, "character": character},
                "end": {"line": line, "character": character + 9},
            },
        }
    ]


def _started_adapter(transport: FakeLspTransport, kernel_repo: Path) -> ClangdAdapter:
    adapter = ClangdAdapter(transport, kernel_repo=kernel_repo)
    assert adapter.capabilities().available is True
    return adapter


# -- handshake ---------------------------------------------------------------


def test_fake_transport_satisfies_the_transport_protocol() -> None:
    assert isinstance(FakeLspTransport(), LspTransport)


def test_initialize_declares_the_kernel_repo_as_root_uri(git_repository: Path) -> None:
    transport = FakeLspTransport()

    _started_adapter(transport, git_repository)

    initialize = transport.messages_for("initialize")[0]
    assert initialize["params"]["rootUri"] == (
        f"file://{git_repository.resolve().as_posix()}"
    )
    assert "capabilities" in initialize["params"]
    assert "id" in initialize


def test_initialized_notification_follows_the_initialize_response(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()

    _started_adapter(transport, git_repository)

    methods = transport.sent_methods
    assert methods.index("initialize") < methods.index("initialized")
    initialized = transport.messages_for("initialized")[0]
    assert "id" not in initialized  # a notification, never a request


def test_handshake_ignores_notifications_while_awaiting_the_initialize_response(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    transport.queue_server_message(LOG_NOTIFICATION)
    transport.queue_server_message(LOG_NOTIFICATION)

    adapter = ClangdAdapter(transport, kernel_repo=git_repository)

    assert adapter.capabilities().available is True
    assert "initialized" in transport.sent_methods


def test_handshake_ignores_a_response_carrying_a_foreign_id(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    transport.queue_server_message(FOREIGN_RESPONSE)

    adapter = ClangdAdapter(transport, kernel_repo=git_repository)

    assert adapter.capabilities().available is True
    assert "initialized" in transport.sent_methods


def test_handshake_ignores_a_server_request_reusing_the_client_request_id(
    git_repository: Path,
) -> None:
    """A message with a ``method`` is a request, never our response -- even
    when the server (wrongly, but harmlessly) numbers it like ours."""
    transport = FakeLspTransport()
    transport.queue_server_message(
        lambda request: {
            "jsonrpc": "2.0",
            "id": request["id"],
            "method": "window/workDoneProgress/create",
            "params": {"token": "clangd-index"},
        }
    )

    adapter = ClangdAdapter(transport, kernel_repo=git_repository)

    assert adapter.capabilities().available is True
    assert "initialized" in transport.sent_methods


def test_handshake_timeout_leaves_the_provider_unavailable_without_raising(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    transport.suppress_response("initialize")

    adapter = ClangdAdapter(transport, kernel_repo=git_repository, timeout=0.1)

    capabilities = adapter.capabilities()
    assert capabilities.available is False
    assert capabilities.reason is not None
    assert "initialized" not in transport.sent_methods


def test_requests_after_a_failed_handshake_report_unavailable(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    transport.suppress_response("initialize")
    adapter = ClangdAdapter(transport, kernel_repo=git_repository, timeout=0.1)

    prepared = adapter.prepare_call_hierarchy(FIXTURE_PATH, line=1, column=1)
    referenced = adapter.references(FIXTURE_PATH, line=1, column=1)
    calls = adapter.outgoing_calls(
        CallHierarchyItem(
            name="tcp_input", relative_path=FIXTURE_PATH, line=1, column=1
        )
    )

    assert prepared.status == "unavailable"
    assert referenced.status == "unavailable"
    assert calls.status == "unavailable"
    assert transport.messages_for("textDocument/didOpen") == []


def test_handshake_send_failure_never_raises_out_of_the_constructor(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    transport.fail_send_with_closed()

    adapter = ClangdAdapter(transport, kernel_repo=git_repository)

    assert adapter.capabilities().available is False
    assert adapter.capabilities().reason is not None


# -- textDocument/didOpen ----------------------------------------------------


def test_document_is_opened_before_the_first_semantic_request(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    adapter = _started_adapter(transport, git_repository)
    transport.queue_response("textDocument/prepareCallHierarchy", result=[])

    adapter.prepare_call_hierarchy(FIXTURE_PATH, line=42, column=5)

    methods = transport.sent_methods
    assert methods.index("textDocument/didOpen") < methods.index(
        "textDocument/prepareCallHierarchy"
    )


def test_did_open_carries_uri_language_version_and_source_text(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    adapter = _started_adapter(transport, git_repository)
    transport.queue_response("textDocument/prepareCallHierarchy", result=[])

    adapter.prepare_call_hierarchy(FIXTURE_PATH, line=42, column=5)

    document = transport.messages_for("textDocument/didOpen")[0]["params"][
        "textDocument"
    ]
    assert document["uri"] == _uri(git_repository)
    assert document["languageId"] == "c"
    assert isinstance(document["version"], int)
    assert document["version"] >= 1
    assert document["text"] == (git_repository / FIXTURE_PATH).read_text(
        encoding="utf-8"
    )


def test_did_open_preserves_non_ascii_source_text(git_repository: Path) -> None:
    source = git_repository / "net" / "ipv4" / "udp_caf\u00e9.c"
    source.write_text("/* r\u00e9ception caf\u00e9 */\nint caf\u00e9(void);\n", encoding="utf-8")
    transport = FakeLspTransport()
    adapter = _started_adapter(transport, git_repository)
    transport.queue_response("textDocument/prepareCallHierarchy", result=[])

    adapter.prepare_call_hierarchy("net/ipv4/udp_caf\u00e9.c", line=2, column=5)

    document = transport.messages_for("textDocument/didOpen")[0]["params"][
        "textDocument"
    ]
    assert document["text"] == source.read_text(encoding="utf-8")


def test_each_file_is_opened_only_once(git_repository: Path) -> None:
    transport = FakeLspTransport()
    adapter = _started_adapter(transport, git_repository)
    transport.queue_response("textDocument/prepareCallHierarchy", result=[])
    transport.queue_response("textDocument/references", result=[])

    adapter.prepare_call_hierarchy(FIXTURE_PATH, line=42, column=5)
    adapter.references(FIXTURE_PATH, line=42, column=5)

    assert len(transport.messages_for("textDocument/didOpen")) == 1


def test_distinct_files_are_each_opened(git_repository: Path) -> None:
    other = git_repository / "net" / "ipv4" / "udp.c"
    other.write_text("int udp_rcv(int x)\n{\n    return x;\n}\n", encoding="utf-8")
    transport = FakeLspTransport()
    adapter = _started_adapter(transport, git_repository)
    transport.queue_response("textDocument/prepareCallHierarchy", result=[])
    transport.queue_response("textDocument/prepareCallHierarchy", result=[])

    adapter.prepare_call_hierarchy(FIXTURE_PATH, line=1, column=1)
    adapter.prepare_call_hierarchy("net/ipv4/udp.c", line=1, column=1)

    opened = [
        message["params"]["textDocument"]["uri"]
        for message in transport.messages_for("textDocument/didOpen")
    ]
    assert opened == [_uri(git_repository), _uri(git_repository, "net/ipv4/udp.c")]


def test_outgoing_calls_opens_the_document_of_its_item(git_repository: Path) -> None:
    transport = FakeLspTransport()
    adapter = _started_adapter(transport, git_repository)
    transport.queue_response("callHierarchy/outgoingCalls", result=[])

    adapter.outgoing_calls(
        CallHierarchyItem(
            name="tcp_input", relative_path=FIXTURE_PATH, line=42, column=5
        )
    )

    methods = transport.sent_methods
    assert methods.index("textDocument/didOpen") < methods.index(
        "callHierarchy/outgoingCalls"
    )


def test_missing_file_degrades_without_opening_or_requesting(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    adapter = _started_adapter(transport, git_repository)

    outcome = adapter.prepare_call_hierarchy(
        "net/ipv4/does_not_exist.c", line=1, column=1
    )

    assert outcome.status == "error"
    assert outcome.reason is not None
    assert outcome.items == ()
    assert transport.messages_for("textDocument/didOpen") == []
    assert transport.messages_for("textDocument/prepareCallHierarchy") == []


def test_path_outside_the_repository_degrades_without_sending_anything(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    adapter = _started_adapter(transport, git_repository)

    outcome = adapter.references("../outside.c", line=1, column=1)

    assert outcome.status == "error"
    assert outcome.reason is not None
    assert outcome.locations == ()
    assert transport.messages_for("textDocument/didOpen") == []
    assert transport.messages_for("textDocument/references") == []


# -- response correlation ----------------------------------------------------


def test_request_skips_interleaved_notifications_and_foreign_responses(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    adapter = _started_adapter(transport, git_repository)
    transport.queue_server_message(LOG_NOTIFICATION)
    transport.queue_server_message(FOREIGN_RESPONSE)
    transport.queue_server_message(LOG_NOTIFICATION)
    transport.queue_response(
        "textDocument/prepareCallHierarchy", result=_call_hierarchy_result(git_repository)
    )

    outcome = adapter.prepare_call_hierarchy(FIXTURE_PATH, line=42, column=5)

    assert outcome.status == "ok"
    assert [item.name for item in outcome.items] == ["tcp_input"]


def test_request_skips_a_server_request_reusing_its_own_id(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    adapter = _started_adapter(transport, git_repository)
    transport.queue_server_message(
        lambda request: {
            "jsonrpc": "2.0",
            "id": request["id"],
            "method": "workspace/configuration",
            "params": {"items": []},
        }
    )
    transport.queue_response(
        "textDocument/references", result=_references_result(git_repository)
    )

    outcome = adapter.references(FIXTURE_PATH, line=42, column=5)

    assert outcome.status == "ok"
    assert outcome.locations[0].line == 100


def test_every_frame_in_a_single_chunk_is_consumed(git_repository: Path) -> None:
    transport = FakeLspTransport()
    adapter = _started_adapter(transport, git_repository)
    transport.queue_server_message(LOG_NOTIFICATION)
    transport.queue_server_message(FOREIGN_RESPONSE)
    transport.queue_response(
        "textDocument/references", result=_references_result(git_repository)
    )
    recv_calls_before = transport.recv_calls

    outcome = adapter.references(FIXTURE_PATH, line=42, column=5)

    assert outcome.status == "ok"
    # All three frames arrived together; a second read would mean the adapter
    # threw away everything it had not yet used.
    assert transport.recv_calls - recv_calls_before == 1


def test_partial_frames_are_reassembled_across_recv_calls(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport(chunk_size=3)
    adapter = _started_adapter(transport, git_repository)
    transport.queue_response(
        "textDocument/prepareCallHierarchy", result=_call_hierarchy_result(git_repository)
    )

    outcome = adapter.prepare_call_hierarchy(FIXTURE_PATH, line=42, column=5)

    assert outcome.status == "ok"
    assert outcome.items[0].line == 42
    assert outcome.items[0].column == 5


def test_leftover_partial_frame_survives_until_the_next_request(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    adapter = _started_adapter(transport, git_repository)
    log_frame = encode_message(LOG_NOTIFICATION)
    split_at = len(log_frame) // 2
    transport.queue_response("textDocument/prepareCallHierarchy", result=[])
    transport.queue_trailing_raw(log_frame[:split_at])

    first = adapter.prepare_call_hierarchy(FIXTURE_PATH, line=42, column=5)
    assert first.status == "ok"

    transport.emit_raw(log_frame[split_at:])
    transport.queue_response(
        "textDocument/references", result=_references_result(git_repository)
    )
    second = adapter.references(FIXTURE_PATH, line=42, column=5)

    assert second.status == "ok"
    assert second.locations[0].line == 100


def test_multibyte_utf8_split_across_chunks_is_decoded(git_repository: Path) -> None:
    transport = FakeLspTransport(chunk_size=5)
    adapter = _started_adapter(transport, git_repository)
    transport.queue_response(
        "textDocument/prepareCallHierarchy",
        result=_call_hierarchy_result(git_repository, name="caf\u00e9_handler"),
    )

    outcome = adapter.prepare_call_hierarchy(FIXTURE_PATH, line=42, column=5)

    assert outcome.status == "ok"
    assert outcome.items[0].name == "caf\u00e9_handler"


# -- shutdown ----------------------------------------------------------------


def test_close_shuts_down_then_exits_then_closes_the_transport(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    adapter = _started_adapter(transport, git_repository)

    adapter.close()

    methods = transport.sent_methods
    assert methods[-2:] == ["shutdown", "exit"]
    assert "id" in transport.messages_for("shutdown")[0]
    assert "id" not in transport.messages_for("exit")[0]
    assert transport.closed is True


def test_close_is_idempotent(git_repository: Path) -> None:
    transport = FakeLspTransport()
    adapter = _started_adapter(transport, git_repository)

    adapter.close()
    adapter.close()

    assert len(transport.messages_for("shutdown")) == 1
    assert len(transport.messages_for("exit")) == 1
    assert transport.closed is True


def test_close_after_a_failed_handshake_is_safe(git_repository: Path) -> None:
    transport = FakeLspTransport()
    transport.suppress_response("initialize")
    adapter = ClangdAdapter(transport, kernel_repo=git_repository, timeout=0.1)

    adapter.close()

    assert transport.messages_for("shutdown") == []
    assert transport.closed is True


def test_close_is_bounded_when_the_shutdown_response_never_arrives(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    adapter = ClangdAdapter(transport, kernel_repo=git_repository, timeout=0.1)
    assert adapter.capabilities().available is True
    transport.suppress_response("shutdown")

    started = time.monotonic()
    adapter.close()
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    assert "exit" in transport.sent_methods
    assert transport.closed is True


def test_close_tolerates_a_transport_that_fails_to_send(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    adapter = _started_adapter(transport, git_repository)
    transport.fail_send_with_closed()

    adapter.close()

    assert transport.closed is True


def test_requests_after_close_report_unavailable(git_repository: Path) -> None:
    transport = FakeLspTransport()
    adapter = _started_adapter(transport, git_repository)
    adapter.close()

    outcome = adapter.prepare_call_hierarchy(FIXTURE_PATH, line=42, column=5)

    assert outcome.status == "unavailable"
    assert outcome.items == ()
