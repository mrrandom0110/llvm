"""Contract for the real stdio transport that owns a ``clangd`` subprocess.

``StdioLspTransport`` is the only place in the semantic stack that touches a
process. These tests hold it to the same rules the rest of the indexing
pipeline follows for external tools (see ``test_ctags_runner.py``): a fixed
argv list -- never a shell command string -- launched with the kernel repo as
cwd, bounded reads that can never hang the caller, a single long-lived
process reused across the whole session, and a close path that always
reaps the process and can be called more than once.

``subprocess`` is patched at ``transport.subprocess.Popen`` (module-level
import, matching the ``ctags_runner``/``repo_inspector`` convention), and the
fake it is replaced with hands out real OS pipes, so any implementation
strategy -- ``select``, a reader thread, or plain blocking reads -- is
exercised for real.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Iterator

import pytest

from netstack_academy.indexing.semantic import transport as transport_module
from netstack_academy.indexing.semantic.jsonrpc import decode_messages, encode_message
from netstack_academy.indexing.semantic.transport import (
    LspTransport,
    StdioLspTransport,
    TransportClosedError,
    TransportTimeoutError,
)

from lsp_fakes import RecordingPopen

CLANGD_ARGV = ["clangd", "--log=error"]

PING = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
PONG = {"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}}


@pytest.fixture
def fake_popen(monkeypatch: pytest.MonkeyPatch) -> Iterator[RecordingPopen]:
    recorder = RecordingPopen()
    monkeypatch.setattr(transport_module.subprocess, "Popen", recorder)
    yield recorder
    recorder.shutdown()


def _install_failing_popen(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> RecordingPopen:
    recorder = RecordingPopen(error=error)
    monkeypatch.setattr(transport_module.subprocess, "Popen", recorder)
    return recorder


def _recv_frame(transport: StdioLspTransport, *, timeout: float = 2.0) -> dict:
    """Read until one complete frame has been reassembled from the stream."""
    buffer = b""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        buffer += transport.recv(0.5)
        messages, _ = decode_messages(buffer)
        if messages:
            return messages[0]
    raise AssertionError("no complete frame arrived before the deadline")


# -- launch ------------------------------------------------------------------


def test_stdio_transport_satisfies_the_transport_protocol(
    fake_popen: RecordingPopen, git_repository: Path
) -> None:
    transport = StdioLspTransport(CLANGD_ARGV, cwd=git_repository)

    assert isinstance(transport, LspTransport)


def test_launches_a_fixed_argv_list_without_a_shell(
    fake_popen: RecordingPopen, git_repository: Path
) -> None:
    StdioLspTransport(CLANGD_ARGV, cwd=git_repository)

    call = fake_popen.calls[0]
    assert list(call.argv) == CLANGD_ARGV
    assert all(isinstance(argument, str) for argument in call.argv)
    assert not call.kwargs.get("shell", False)


def test_launches_with_the_kernel_repo_as_cwd(
    fake_popen: RecordingPopen, git_repository: Path
) -> None:
    StdioLspTransport(CLANGD_ARGV, cwd=git_repository)

    assert Path(fake_popen.calls[0].kwargs["cwd"]) == git_repository


def test_launches_with_piped_stdin_and_stdout(
    fake_popen: RecordingPopen, git_repository: Path
) -> None:
    StdioLspTransport(CLANGD_ARGV, cwd=git_repository)

    kwargs = fake_popen.calls[0].kwargs
    assert kwargs["stdin"] == subprocess.PIPE
    assert kwargs["stdout"] == subprocess.PIPE
    # clangd is chatty on stderr; it must never be inherited into our own.
    assert kwargs.get("stderr") in (subprocess.PIPE, subprocess.DEVNULL)


def test_rejects_a_shell_command_string_instead_of_an_argv_list(
    fake_popen: RecordingPopen, git_repository: Path
) -> None:
    with pytest.raises(TypeError):
        StdioLspTransport("clangd --log=error", cwd=git_repository)

    assert fake_popen.calls == []


def test_rejects_an_empty_argv(
    fake_popen: RecordingPopen, git_repository: Path
) -> None:
    with pytest.raises(ValueError):
        StdioLspTransport([], cwd=git_repository)

    assert fake_popen.calls == []


def test_missing_executable_degrades_to_a_transport_error(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    _install_failing_popen(monkeypatch, FileNotFoundError("no such file: clangd"))

    with pytest.raises(TransportClosedError) as failure:
        StdioLspTransport(CLANGD_ARGV, cwd=git_repository)

    assert "clangd" in str(failure.value)


def test_unlaunchable_executable_degrades_to_a_transport_error(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    _install_failing_popen(monkeypatch, PermissionError("permission denied"))

    with pytest.raises(TransportClosedError):
        StdioLspTransport(CLANGD_ARGV, cwd=git_repository)


# -- bounded, framed I/O -----------------------------------------------------


def test_send_writes_framed_bytes_to_the_process_stdin(
    fake_popen: RecordingPopen, git_repository: Path
) -> None:
    transport = StdioLspTransport(CLANGD_ARGV, cwd=git_repository)
    process = fake_popen.processes[0]

    transport.send(encode_message(PING))

    messages, remaining = decode_messages(process.read_from_client(timeout=2.0))
    assert messages == [PING]
    assert remaining == b""


def test_recv_returns_bytes_written_by_the_server(
    fake_popen: RecordingPopen, git_repository: Path
) -> None:
    transport = StdioLspTransport(CLANGD_ARGV, cwd=git_repository)
    fake_popen.processes[0].write_to_client(encode_message(PONG))

    assert _recv_frame(transport) == PONG


def test_recv_raises_timeout_without_blocking_past_the_deadline(
    fake_popen: RecordingPopen, git_repository: Path
) -> None:
    transport = StdioLspTransport(CLANGD_ARGV, cwd=git_repository)

    started = time.monotonic()
    with pytest.raises(TransportTimeoutError):
        transport.recv(0.05)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0


def test_recv_reports_closed_when_the_process_closes_its_output(
    fake_popen: RecordingPopen, git_repository: Path
) -> None:
    transport = StdioLspTransport(CLANGD_ARGV, cwd=git_repository)
    fake_popen.processes[0].close_server_output()

    with pytest.raises(TransportClosedError):
        transport.recv(0.5)


def test_one_process_is_owned_for_the_whole_session(
    fake_popen: RecordingPopen, git_repository: Path
) -> None:
    transport = StdioLspTransport(CLANGD_ARGV, cwd=git_repository)
    process = fake_popen.processes[0]

    transport.send(encode_message(PING))
    transport.send(encode_message({**PING, "id": 2}))

    assert len(fake_popen.calls) == 1
    buffer = b""
    messages: list[dict] = []
    while len(messages) < 2:
        buffer += process.read_from_client(timeout=2.0)
        messages, buffer = decode_messages(buffer)
    assert [message["id"] for message in messages] == [1, 2]


# -- close -------------------------------------------------------------------


def test_close_terminates_the_process(
    fake_popen: RecordingPopen, git_repository: Path
) -> None:
    transport = StdioLspTransport(CLANGD_ARGV, cwd=git_repository)
    process = fake_popen.processes[0]

    transport.close()

    assert process.terminate_calls == 1
    assert process.poll() is not None


def test_close_is_idempotent(
    fake_popen: RecordingPopen, git_repository: Path
) -> None:
    transport = StdioLspTransport(CLANGD_ARGV, cwd=git_repository)
    process = fake_popen.processes[0]

    transport.close()
    transport.close()

    assert process.terminate_calls == 1


def test_close_kills_a_process_that_ignores_terminate(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    recorder = RecordingPopen(ignore_terminate=True)
    monkeypatch.setattr(transport_module.subprocess, "Popen", recorder)
    transport = StdioLspTransport(CLANGD_ARGV, cwd=git_repository)
    process = recorder.processes[0]

    transport.close()

    assert process.terminate_calls == 1
    assert process.kill_calls >= 1
    assert process.poll() is not None
    recorder.shutdown()


def test_send_after_close_reports_closed(
    fake_popen: RecordingPopen, git_repository: Path
) -> None:
    transport = StdioLspTransport(CLANGD_ARGV, cwd=git_repository)
    transport.close()

    with pytest.raises(TransportClosedError):
        transport.send(encode_message(PING))


def test_recv_after_close_reports_closed(
    fake_popen: RecordingPopen, git_repository: Path
) -> None:
    transport = StdioLspTransport(CLANGD_ARGV, cwd=git_repository)
    transport.close()

    with pytest.raises(TransportClosedError):
        transport.recv(0.5)
