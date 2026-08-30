from __future__ import annotations

import subprocess
from pathlib import Path

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
