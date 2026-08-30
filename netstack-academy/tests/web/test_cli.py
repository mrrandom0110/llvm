"""Contract for :mod:`netstack_academy.cli`.

Three commands, and nothing that runs what a caller typed. ``serve`` starts
the local app, ``validate-content`` answers "is this course loadable" for an
author mid-edit, and ``index`` builds or refreshes the symbol index from a
terminal rather than from a request.

The command surface is deliberately closed: every argument is a path, a
port, or a flag, and there is no option that takes a shell command, a
program to run, or a template to expand. A learning tool that displays
kernel lab commands is a program whose *content* is full of shell; the one
thing it must never do is execute it.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

from netstack_academy.cli import main
from netstack_academy.web.app import DEFAULT_PORT, LOOPBACK_HOST

from academy_content import LAB_SENTINEL_PATH, write_invalid_content
from web_fakes import (
    ExplodingServer,
    RecordingServer,
    RecordingSessionRunner,
    failed_result,
    reindexed_result,
    reused_result,
)

CLI_SENTINEL_PATH = "/tmp/netstack-academy-cli-must-not-run"


@pytest.fixture
def cli_env(
    monkeypatch: pytest.MonkeyPatch,
    kernel_repo: Path,
    content_root: Path,
    state_dir: Path,
) -> None:
    """Configure the CLI the way a learner's shell would."""
    monkeypatch.setenv("KERNEL_REPO", str(kernel_repo))
    monkeypatch.setenv("CONTENT_ROOT", str(content_root))
    monkeypatch.setenv("STATE_DIR", str(state_dir))
    monkeypatch.setenv("EDITOR_SCHEME", "cursor")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.delenv("TEST_SYMBOL_PATH", raising=False)


@pytest.fixture
def server() -> RecordingServer:
    return RecordingServer()


@pytest.fixture
def session_runner(kernel_head: str) -> RecordingSessionRunner:
    return RecordingSessionRunner(reindexed_result(kernel_head, symbol_count=6, edge_count=4))


# ----------------------------------------------------------------------
# serve
# ----------------------------------------------------------------------


def test_serve_binds_loopback_by_default(
    cli_env: None, server: RecordingServer
) -> None:
    exit_code = main(["serve"], server_runner=server)

    assert exit_code == 0
    assert server.call_count == 1
    assert server.calls[0]["host"] == LOOPBACK_HOST
    assert server.calls[0]["port"] == DEFAULT_PORT


def test_serve_accepts_an_alternative_port(
    cli_env: None, server: RecordingServer
) -> None:
    main(["serve", "--port", "9123"], server_runner=server)

    assert server.calls[0]["port"] == 9123


def test_serve_hands_the_web_app_to_the_server(
    cli_env: None, server: RecordingServer
) -> None:
    main(["serve"], server_runner=server)

    app = server.calls[0]["app"]
    assert app.docs_url is None
    assert app.state.context is not None


def test_serve_prints_the_local_url(
    cli_env: None, server: RecordingServer, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["serve", "--port", "9123"], server_runner=server)

    assert "http://127.0.0.1:9123" in capsys.readouterr().out


def test_serve_closes_the_runtime_when_the_server_returns(
    cli_env: None, server: RecordingServer
) -> None:
    """Serving ends when uvicorn returns, and the two SQLite connections it
    was holding have to go with it.
    """
    main(["serve"], server_runner=server)

    store = server.calls[0]["app"].state.context.store
    with pytest.raises(sqlite3.ProgrammingError):
        store.list_progress()


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::", "example.com"])
def test_serve_refuses_a_non_loopback_bind_address(
    cli_env: None, host: str
) -> None:
    """There is no authentication in this program. Binding anywhere routable
    would publish a learner's notes and an unauthenticated import endpoint to
    the local network.
    """
    exploding = ExplodingServer()

    with pytest.raises(SystemExit) as excinfo:
        main(["serve", "--host", host], server_runner=exploding)

    assert excinfo.value.code == 2


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_serve_accepts_loopback_spellings(
    cli_env: None, server: RecordingServer, host: str
) -> None:
    assert main(["serve", "--host", host], server_runner=server) == 0
    assert server.calls[0]["host"] == host


def test_serve_rejects_a_non_numeric_port(cli_env: None) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["serve", "--port", "9123; touch /tmp/x"], server_runner=ExplodingServer())

    assert excinfo.value.code == 2


# ----------------------------------------------------------------------
# validate-content
# ----------------------------------------------------------------------


def test_validate_content_accepts_a_valid_course(
    cli_env: None, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["validate-content"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "rx-path" in output or "2 module" in output


def test_validate_content_reports_every_error_and_fails(
    cli_env: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = write_invalid_content(tmp_path / "broken-content")

    exit_code = main(["validate-content", "--content-root", str(broken)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "half-written.md" in captured.out + captured.err


def test_validate_content_does_not_touch_the_state_directory(
    cli_env: None, state_dir: Path
) -> None:
    """Asking whether content is valid is a read of the filesystem. It should
    not create a database, and it should work on a machine where the kernel
    repository is not even checked out.
    """
    main(["validate-content"])

    assert not state_dir.exists()


def test_validate_content_reports_a_missing_content_root(
    cli_env: None, tmp_path: Path
) -> None:
    assert main(["validate-content", "--content-root", str(tmp_path / "nope")]) == 1


# ----------------------------------------------------------------------
# index
# ----------------------------------------------------------------------


def test_index_runs_the_pipeline_once(
    cli_env: None, session_runner: RecordingSessionRunner
) -> None:
    exit_code = main(["index"], session_runner=session_runner)

    assert exit_code == 0
    assert session_runner.call_count == 1
    assert session_runner.calls[0]["force"] is False


def test_index_force_reruns_the_pipeline(
    cli_env: None, session_runner: RecordingSessionRunner
) -> None:
    main(["index", "--force"], session_runner=session_runner)

    assert session_runner.calls[0]["force"] is True


def test_index_reports_the_outcome(
    cli_env: None,
    session_runner: RecordingSessionRunner,
    kernel_head: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["index"], session_runner=session_runner)

    output = capsys.readouterr().out
    assert "reindexed" in output
    assert kernel_head[:12] in output
    assert "6" in output


def test_index_reports_a_reused_generation(
    cli_env: None, kernel_head: str, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = RecordingSessionRunner(reused_result(kernel_head, symbol_count=6))

    assert main(["index"], session_runner=runner) == 0
    assert "reused" in capsys.readouterr().out


def test_index_fails_loudly_when_the_run_fails(
    cli_env: None, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = RecordingSessionRunner(failed_result("Path is not a git repository"))

    exit_code = main(["index"], session_runner=runner)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "not a git repository" in captured.out + captured.err


def test_index_closes_the_databases_it_opened(
    cli_env: None, session_runner: RecordingSessionRunner, state_dir: Path
) -> None:
    """A second ``index`` run in the same shell must not find a stale lock.
    """
    main(["index"], session_runner=session_runner)
    assert main(["index"], session_runner=session_runner) == 0


# ----------------------------------------------------------------------
# The command surface itself
# ----------------------------------------------------------------------


def test_an_unknown_command_is_refused(cli_env: None) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["reindex-everything"])

    assert excinfo.value.code == 2


def test_an_unknown_option_is_refused(cli_env: None) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["serve", "--command", "id"], server_runner=ExplodingServer())

    assert excinfo.value.code == 2


def test_no_arguments_prints_usage_without_serving(cli_env: None) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])

    assert excinfo.value.code == 2


def test_validating_content_spawns_no_processes(
    cli_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loading Markdown is file reads and YAML parsing. Nothing in it needs a
    child process, so nothing in it gets one.
    """

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"no process may be spawned: {args!r}")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(subprocess, "Popen", _forbidden)
    monkeypatch.setattr(subprocess, "call", _forbidden)
    monkeypatch.setattr(os, "system", _forbidden)

    assert main(["validate-content"]) == 0


def test_a_hostile_content_root_is_a_path_not_a_command(cli_env: None) -> None:
    sentinel = Path(CLI_SENTINEL_PATH)
    sentinel.unlink(missing_ok=True)

    exit_code = main(["validate-content", "--content-root", f"; touch {CLI_SENTINEL_PATH}"])

    assert exit_code == 1
    assert not sentinel.exists()


def test_lab_commands_are_never_executed_by_the_cli(
    cli_env: None, session_runner: RecordingSessionRunner
) -> None:
    """The course tells a learner to run ``touch`` in a lab. Loading that
    course must not run it.
    """
    sentinel = Path(LAB_SENTINEL_PATH)
    sentinel.unlink(missing_ok=True)

    main(["validate-content"])
    main(["index"], session_runner=session_runner)

    assert not sentinel.exists()
