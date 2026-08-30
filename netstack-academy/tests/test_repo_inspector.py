from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from netstack_academy import repo_inspector
from netstack_academy.repo_inspector import inspect_repository


def test_repo_inspector_returns_head_for_valid_git_repository(
    git_repository: Path,
) -> None:
    expected_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=git_repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    state = inspect_repository(git_repository)

    assert state.available is True
    assert state.head == expected_head
    assert state.reason is None


def test_repo_inspector_reports_unavailable_when_path_missing(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does-not-exist"

    state = inspect_repository(missing)

    assert state.available is False
    assert state.head is None
    assert state.reason is not None
    assert "not found" in state.reason.lower() or "missing" in state.reason.lower()


def test_repo_inspector_reports_unavailable_when_path_not_git_repository(
    non_git_directory: Path,
) -> None:
    state = inspect_repository(non_git_directory)

    assert state.available is False
    assert state.head is None
    assert state.reason is not None
    assert "git" in state.reason.lower()


def test_repo_inspector_reports_unavailable_when_path_is_file_not_directory(
    tmp_path: Path,
) -> None:
    target_file = tmp_path / "not-a-directory.txt"
    target_file.write_text("plain file\n", encoding="utf-8")

    state = inspect_repository(target_file)

    assert state.available is False
    assert state.head is None
    assert state.reason is not None
    assert "directory" in state.reason.lower() or "file" in state.reason.lower()


def test_repo_inspector_reports_unavailable_when_git_executable_missing(
    monkeypatch: pytest.MonkeyPatch,
    git_repository: Path,
) -> None:
    monkeypatch.setenv("PATH", "")

    state = inspect_repository(git_repository)

    assert state.available is False
    assert state.head is None
    assert state.reason is not None
    assert "git" in state.reason.lower()


def test_repo_inspector_git_subprocess_uses_finite_timeout(
    monkeypatch: pytest.MonkeyPatch,
    git_repository: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.update(kwargs)
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="deadbeef\n",
            stderr="",
        )

    monkeypatch.setattr(repo_inspector.subprocess, "run", fake_run)

    inspect_repository(git_repository)

    timeout = captured.get("timeout")
    assert timeout is not None
    assert isinstance(timeout, (int, float))
    assert timeout > 0


def test_repo_inspector_reports_unavailable_when_git_subprocess_times_out(
    monkeypatch: pytest.MonkeyPatch,
    git_repository: Path,
) -> None:
    def raise_timeout(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="git rev-parse HEAD", timeout=kwargs.get("timeout"))

    monkeypatch.setattr(repo_inspector.subprocess, "run", raise_timeout)

    state = inspect_repository(git_repository)

    assert state.available is False
    assert state.head is None
    assert state.reason is not None
    assert "timeout" in state.reason.lower()
