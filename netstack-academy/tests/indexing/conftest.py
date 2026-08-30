from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _run_git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _run_git("init", cwd=repo)
    _run_git("config", "user.email", "test@example.com", cwd=repo)
    _run_git("config", "user.name", "Test User", cwd=repo)


def _commit_all(repo: Path, message: str) -> str:
    _run_git("add", ".", cwd=repo)
    _run_git("commit", "-m", message, cwd=repo)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def duplicate_static_c_repo(tmp_path: Path) -> Path:
    """A tiny real git repository containing two files that each define a
    ``static`` function named ``helper``, plus one non-static, globally unique
    function that both call. Used to prove the fallback indexer and symbol
    identity resolution never cross-link same-named static definitions.
    """
    repo = tmp_path / "kernel"
    _init_repo(repo)

    ipv4_dir = repo / "net" / "ipv4"
    ipv6_dir = repo / "net" / "ipv6"
    ipv4_dir.mkdir(parents=True)
    ipv6_dir.mkdir(parents=True)

    (ipv4_dir / "a.c").write_text(
        "static int helper(int x)\n"
        "{\n"
        "    return x + 1;\n"
        "}\n"
        "\n"
        "int process(int x)\n"
        "{\n"
        "    return helper(x) + shared_util(x);\n"
        "}\n",
        encoding="utf-8",
    )

    (ipv6_dir / "b.c").write_text(
        "static int helper(int x)\n"
        "{\n"
        "    return x - 1;\n"
        "}\n"
        "\n"
        "int process6(int x)\n"
        "{\n"
        "    return helper(x);\n"
        "}\n",
        encoding="utf-8",
    )

    (repo / "net" / "util.c").write_text(
        "int shared_util(int x)\n"
        "{\n"
        "    return x * 2;\n"
        "}\n",
        encoding="utf-8",
    )

    outside_dir = repo / "unrelated"
    outside_dir.mkdir(parents=True)
    (outside_dir / "outside.c").write_text(
        "static int helper(int x)\n"
        "{\n"
        "    return x * 100;\n"
        "}\n",
        encoding="utf-8",
    )

    _commit_all(repo, "Add duplicate static helper fixture")
    return repo


@pytest.fixture
def two_commit_git_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """A tiny real git repository with two commits, returning
    ``(repo_path, first_head, second_head)`` so tests can exercise
    commit-aware behaviour without relying on timing.
    """
    repo = tmp_path / "kernel"
    _init_repo(repo)

    net_dir = repo / "net" / "ipv4"
    net_dir.mkdir(parents=True)
    source = net_dir / "tcp_input.c"
    source.write_text(
        "int tcp_input(int x)\n{\n    return x;\n}\n",
        encoding="utf-8",
    )
    first_head = _commit_all(repo, "Initial tcp_input")

    source.write_text(
        "int tcp_input(int x)\n{\n    return x + 1;\n}\n",
        encoding="utf-8",
    )
    second_head = _commit_all(repo, "Tweak tcp_input")

    return repo, first_head, second_head
