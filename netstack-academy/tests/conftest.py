from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def git_repository(tmp_path: Path) -> Path:
    """Create a temporary directory initialized as a real git repository."""
    repo = tmp_path / "kernel"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    sample = repo / "net" / "ipv4"
    sample.mkdir(parents=True)
    source = sample / "tcp_input.c"
    source.write_text("/* netstack academy fixture */\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Add tcp_input fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return repo


@pytest.fixture
def non_git_directory(tmp_path: Path) -> Path:
    """Create a directory that is not a git repository."""
    directory = tmp_path / "plain"
    directory.mkdir()
    (directory / "README.txt").write_text("not a git repo\n", encoding="utf-8")
    return directory
