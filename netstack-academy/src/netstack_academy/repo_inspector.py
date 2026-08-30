from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

GIT_INSPECTION_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class RepositoryState:
    available: bool
    head: str | None
    reason: str | None


def inspect_repository(path: Path) -> RepositoryState:
    if not path.exists():
        return RepositoryState(
            available=False,
            head=None,
            reason="Repository path not found",
        )

    if not path.is_dir():
        return RepositoryState(
            available=False,
            head=None,
            reason="Repository path is not a directory",
        )

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=GIT_INSPECTION_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return RepositoryState(
            available=False,
            head=None,
            reason="Git inspection timeout",
        )
    except FileNotFoundError:
        return RepositoryState(
            available=False,
            head=None,
            reason="Git executable not found",
        )

    if result.returncode != 0:
        return RepositoryState(
            available=False,
            head=None,
            reason="Path is not a git repository",
        )

    return RepositoryState(
        available=True,
        head=result.stdout.strip(),
        reason=None,
    )
