from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from netstack_academy.indexing import ctags_runner
from netstack_academy.indexing.ctags_runner import (
    DEFAULT_INDEX_ROOTS,
    check_ctags_binary,
    default_index_roots,
    run_ctags,
)


def _fake_completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ctags"], returncode=returncode, stdout=stdout, stderr=""
    )


def test_default_index_roots_include_net_and_include_net() -> None:
    roots = default_index_roots()

    assert "net" in roots
    assert "include/net" in roots


def test_default_index_roots_include_network_relevant_linux_headers() -> None:
    roots = default_index_roots()

    assert any(root.startswith("include/linux/") for root in roots)


def test_default_index_roots_include_selected_drivers_net_paths() -> None:
    roots = default_index_roots()

    assert any(root.startswith("drivers/net") for root in roots)


def test_default_index_roots_exclude_unrelated_kernel_trees() -> None:
    roots = default_index_roots()

    assert "fs" not in roots
    assert "arch" not in roots
    assert "drivers" not in roots
    assert "include/linux" not in roots


def test_default_index_roots_constant_is_subset_of_full_root_list() -> None:
    assert set(DEFAULT_INDEX_ROOTS).issubset(set(default_index_roots()))


def test_check_ctags_binary_recognizes_universal_ctags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ctags_runner.subprocess,
        "run",
        lambda *a, **k: _fake_completed("Universal Ctags 6.1.0, Copyright...\n"),
    )

    availability = check_ctags_binary()

    assert availability.available is True
    assert availability.is_universal is True
    assert availability.reason is None


def test_check_ctags_binary_rejects_emacs_ctags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ctags_runner.subprocess,
        "run",
        lambda *a, **k: _fake_completed("ctags (GNU Emacs 27.1)\n"),
    )

    availability = check_ctags_binary()

    assert availability.is_universal is False
    assert availability.reason is not None
    assert "emacs" in availability.reason.lower()


def test_check_ctags_binary_reports_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_missing(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(ctags_runner.subprocess, "run", raise_missing)

    availability = check_ctags_binary()

    assert availability.available is False
    assert availability.reason is not None
    assert "not found" in availability.reason.lower() or "missing" in availability.reason.lower()


def test_check_ctags_binary_reports_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="ctags --version", timeout=kwargs.get("timeout"))

    monkeypatch.setattr(ctags_runner.subprocess, "run", raise_timeout)

    availability = check_ctags_binary()

    assert availability.available is False
    assert availability.reason is not None
    assert "timeout" in availability.reason.lower()


def test_check_ctags_binary_uses_finite_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.update(kwargs)
        return _fake_completed("Universal Ctags 6.1.0\n")

    monkeypatch.setattr(ctags_runner.subprocess, "run", fake_run)

    check_ctags_binary()

    timeout = captured.get("timeout")
    assert timeout is not None
    assert isinstance(timeout, (int, float))
    assert timeout > 0


def test_run_ctags_reports_unavailable_without_invoking_indexing_subprocess(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(ctags_runner.subprocess, "run", fake_run)

    result = run_ctags(git_repository)

    assert result.status == "unavailable"
    assert result.definitions == []
    assert len(calls) == 1  # only the --version probe, never the indexing run


def test_run_ctags_reports_incompatible_for_emacs_ctags(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return _fake_completed("ctags (GNU Emacs 27.1)\n")

    monkeypatch.setattr(ctags_runner.subprocess, "run", fake_run)

    result = run_ctags(git_repository)

    assert result.status == "incompatible"
    assert len(calls) == 1


def test_run_ctags_reports_timeout_for_indexing_subprocess(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "--version" in args:
            return _fake_completed("Universal Ctags 6.1.0\n")
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(ctags_runner.subprocess, "run", fake_run)

    result = run_ctags(git_repository)

    assert result.status == "timeout"
    assert result.definitions == []


def test_run_ctags_parses_successful_output(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    import json

    tag_line = json.dumps(
        {
            "_type": "tag",
            "name": "tcp_input",
            "path": "net/ipv4/tcp_input.c",
            "line": 1,
            "kind": "function",
        }
    )

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "--version" in args:
            return _fake_completed("Universal Ctags 6.1.0\n")
        return _fake_completed(tag_line + "\n")

    monkeypatch.setattr(ctags_runner.subprocess, "run", fake_run)

    result = run_ctags(git_repository)

    assert result.status == "ok"
    assert [d.name for d in result.definitions] == ["tcp_input"]


def test_run_ctags_never_raises_for_unexpected_os_error(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "--version" in args:
            return _fake_completed("Universal Ctags 6.1.0\n")
        raise PermissionError("permission denied")

    monkeypatch.setattr(ctags_runner.subprocess, "run", fake_run)

    result = run_ctags(git_repository)

    assert result.status == "error"
    assert result.definitions == []


def test_run_ctags_only_passes_roots_that_exist_on_disk(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    captured_args: list[str] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "--version" in args:
            return _fake_completed("Universal Ctags 6.1.0\n")
        captured_args.extend(args)
        return _fake_completed("")

    monkeypatch.setattr(ctags_runner.subprocess, "run", fake_run)

    # git_repository fixture only has a "net" directory; other default roots
    # (include/net, drivers/net, ...) do not exist and must be skipped rather
    # than passed through to the ctags invocation.
    run_ctags(git_repository)

    assert "net" in captured_args
    assert "include/net" not in captured_args


def test_run_ctags_invokes_indexing_subprocess_with_repo_as_cwd(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    captured_kwargs: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "--version" in args:
            return _fake_completed("Universal Ctags 6.1.0\n")
        captured_kwargs.update(kwargs)
        return _fake_completed("")

    monkeypatch.setattr(ctags_runner.subprocess, "run", fake_run)

    run_ctags(git_repository)

    assert captured_kwargs.get("cwd") == git_repository


def test_run_ctags_skips_indexing_subprocess_when_no_configured_roots_exist(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    """When none of the caller-supplied ``roots`` exist on disk, there are
    no path operands to give ctags. Invoking ``ctags -R`` with no path
    operands still recursively scans ``cwd``, defeating the whole purpose
    of a curated root list, so the indexing subprocess must never be
    launched at all in this case -- only the ``--version`` probe runs, and
    the result degrades to a benign, empty outcome instead of an error.
    """
    calls: list[list[str]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if "--version" in args:
            return _fake_completed("Universal Ctags 6.1.0\n")
        return _fake_completed("")

    monkeypatch.setattr(ctags_runner.subprocess, "run", fake_run)

    result = run_ctags(git_repository, roots=["does/not/exist"])

    assert len(calls) == 1  # only the --version probe; no recursive fallback scan
    assert result.status not in {"error", "timeout"}
    assert result.definitions == []


def test_run_ctags_requests_signature_field(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    """Ctags definitions are meant to carry a ``signature`` (see
    ``CtagsDefinition.signature`` / ``ctags_parser``), which Universal
    Ctags only emits when the ``S`` fields flag is requested. Without it,
    every parsed definition's ``signature`` is silently ``None``.
    """
    captured_args: list[str] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "--version" in args:
            return _fake_completed("Universal Ctags 6.1.0\n")
        captured_args.extend(args)
        return _fake_completed("")

    monkeypatch.setattr(ctags_runner.subprocess, "run", fake_run)

    run_ctags(git_repository)

    fields_args = [arg for arg in captured_args if arg.startswith("--fields=")]
    assert fields_args, "expected a --fields= argument in the ctags invocation"
    assert any("S" in arg for arg in fields_args)


def test_run_ctags_uses_the_same_caller_supplied_timeout_for_probe_and_index(
    monkeypatch: pytest.MonkeyPatch, git_repository: Path
) -> None:
    """``run_ctags(..., timeout=...)`` bounds the indexing subprocess, but
    the ``--version`` probe it runs first (via ``check_ctags_binary``) is
    hard-coded to ``CTAGS_VERSION_TIMEOUT_SECONDS`` regardless of what the
    caller asked for. A caller with a tight overall budget must have that
    budget honored coherently by *both* subprocess invocations, not just
    the second one.
    """
    captured_timeouts: list[object] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured_timeouts.append(kwargs.get("timeout"))
        if "--version" in args:
            return _fake_completed("Universal Ctags 6.1.0\n")
        return _fake_completed("")

    monkeypatch.setattr(ctags_runner.subprocess, "run", fake_run)

    run_ctags(git_repository, timeout=1.0)

    assert captured_timeouts == [1.0, 1.0]
