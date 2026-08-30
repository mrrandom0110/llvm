from __future__ import annotations

from pathlib import Path

from netstack_academy.indexing.semantic.clangd_adapter import ClangdAdapter
from netstack_academy.indexing.semantic.models import CallHierarchyItem
from netstack_academy.indexing.semantic.provider import SemanticProvider

from lsp_fakes import FakeLspTransport


def test_clangd_adapter_satisfies_semantic_provider_protocol(
    git_repository: Path,
) -> None:
    adapter = ClangdAdapter(FakeLspTransport(), kernel_repo=git_repository)

    assert isinstance(adapter, SemanticProvider)


def test_prepare_call_hierarchy_sends_zero_indexed_position(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    adapter = ClangdAdapter(transport, kernel_repo=git_repository)
    transport.queue_response("textDocument/prepareCallHierarchy", result=[])

    adapter.prepare_call_hierarchy("net/ipv4/tcp_input.c", line=42, column=5)

    request = transport.sent_messages[-1]
    assert request["method"] == "textDocument/prepareCallHierarchy"
    assert request["params"]["position"]["line"] == 41
    assert request["params"]["position"]["character"] == 4


def test_prepare_call_hierarchy_maps_response_uri_to_relative_path(
    git_repository: Path,
) -> None:
    target = (git_repository / "net" / "ipv4" / "tcp_input.c").resolve()
    transport = FakeLspTransport()
    adapter = ClangdAdapter(transport, kernel_repo=git_repository)
    transport.queue_response(
        "textDocument/prepareCallHierarchy",
        result=[
            {
                "name": "tcp_input",
                "uri": f"file://{target.as_posix()}",
                "range": {
                    "start": {"line": 41, "character": 4},
                    "end": {"line": 41, "character": 14},
                },
            }
        ]
    )

    outcome = adapter.prepare_call_hierarchy(
        "net/ipv4/tcp_input.c", line=42, column=5
    )

    assert outcome.status == "ok"
    assert len(outcome.items) == 1
    item = outcome.items[0]
    assert item.name == "tcp_input"
    assert item.relative_path == "net/ipv4/tcp_input.c"
    assert item.line == 42
    assert item.column == 5


def test_outgoing_calls_maps_response_to_relative_locations(
    git_repository: Path,
) -> None:
    target = (git_repository / "net" / "ipv4" / "tcp_input.c").resolve()
    transport = FakeLspTransport()
    adapter = ClangdAdapter(transport, kernel_repo=git_repository)
    transport.queue_response(
        "callHierarchy/outgoingCalls",
        result=[
            {
                "to": {
                    "name": "helper",
                    "uri": f"file://{target.as_posix()}",
                    "range": {
                        "start": {"line": 9, "character": 0},
                        "end": {"line": 9, "character": 6},
                    },
                },
                "fromRanges": [
                    {
                        "start": {"line": 41, "character": 4},
                        "end": {"line": 41, "character": 10},
                    }
                ],
            }
        ]
    )
    item = CallHierarchyItem(
        name="tcp_input", relative_path="net/ipv4/tcp_input.c", line=42, column=5
    )

    outcome = adapter.outgoing_calls(item)

    assert outcome.status == "ok"
    assert len(outcome.calls) == 1
    call = outcome.calls[0]
    assert call.target.name == "helper"
    assert call.target.relative_path == "net/ipv4/tcp_input.c"
    assert call.target.line == 10
    assert len(call.call_sites) == 1
    assert call.call_sites[0].line == 42


def test_outgoing_calls_sends_call_hierarchy_item_verbatim(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    adapter = ClangdAdapter(transport, kernel_repo=git_repository)
    transport.queue_response("callHierarchy/outgoingCalls", result=[])
    item = CallHierarchyItem(
        name="tcp_input", relative_path="net/ipv4/tcp_input.c", line=42, column=5
    )

    adapter.outgoing_calls(item)

    request = transport.sent_messages[-1]
    assert request["method"] == "callHierarchy/outgoingCalls"


def test_references_maps_response_locations_to_relative_paths(
    git_repository: Path,
) -> None:
    target = (git_repository / "net" / "ipv4" / "tcp_input.c").resolve()
    transport = FakeLspTransport()
    adapter = ClangdAdapter(transport, kernel_repo=git_repository)
    transport.queue_response(
        "textDocument/references",
        result=[
            {
                "uri": f"file://{target.as_posix()}",
                "range": {
                    "start": {"line": 99, "character": 2},
                    "end": {"line": 99, "character": 12},
                },
            }
        ]
    )

    outcome = adapter.references("net/ipv4/tcp_input.c", line=42, column=5)

    assert outcome.status == "ok"
    assert len(outcome.locations) == 1
    assert outcome.locations[0].relative_path == "net/ipv4/tcp_input.c"
    assert outcome.locations[0].line == 100
    assert outcome.locations[0].column == 3


def test_references_request_uses_one_indexed_to_zero_indexed_conversion(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    adapter = ClangdAdapter(transport, kernel_repo=git_repository)
    transport.queue_response("textDocument/references", result=[])

    adapter.references("net/ipv4/tcp_input.c", line=1, column=1)

    request = transport.sent_messages[-1]
    assert request["params"]["position"]["line"] == 0
    assert request["params"]["position"]["character"] == 0


def test_prepare_call_hierarchy_degrades_gracefully_on_timeout(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    adapter = ClangdAdapter(transport, kernel_repo=git_repository)
    transport.fail_next_recv_with_timeout()

    outcome = adapter.prepare_call_hierarchy("net/ipv4/tcp_input.c", line=1, column=1)

    assert outcome.status == "timeout"
    assert outcome.items == ()
    assert outcome.reason is not None


def test_prepare_call_hierarchy_degrades_gracefully_on_jsonrpc_error_response(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    adapter = ClangdAdapter(transport, kernel_repo=git_repository)
    transport.queue_response(
        "textDocument/prepareCallHierarchy",
        error={"code": -32601, "message": "method not found"},
    )

    outcome = adapter.prepare_call_hierarchy("net/ipv4/tcp_input.c", line=1, column=1)

    assert outcome.status == "error"
    assert outcome.items == ()
    assert "method not found" in (outcome.reason or "")


def test_references_reports_unavailable_when_transport_is_closed(
    git_repository: Path,
) -> None:
    transport = FakeLspTransport()
    adapter = ClangdAdapter(transport, kernel_repo=git_repository)
    transport.fail_send_with_closed()

    outcome = adapter.references("net/ipv4/tcp_input.c", line=1, column=1)

    assert outcome.status == "unavailable"
    assert outcome.locations == ()


def test_adapter_never_requires_a_real_clangd_process(
    git_repository: Path,
) -> None:
    """The adapter must be fully exercisable against an in-memory fake; no
    subprocess, socket, or external ``clangd`` binary is touched by these
    tests.
    """
    transport = FakeLspTransport()
    adapter = ClangdAdapter(transport, kernel_repo=git_repository)
    transport.queue_response("textDocument/prepareCallHierarchy", result=[])

    outcome = adapter.prepare_call_hierarchy("net/ipv4/tcp_input.c", line=1, column=1)

    assert outcome.status == "ok"


def test_capabilities_reports_provider_name() -> None:
    transport = FakeLspTransport()
    adapter = ClangdAdapter(transport, kernel_repo=Path("/tmp"))

    capabilities = adapter.capabilities()

    assert capabilities.provider_name == "clangd"
