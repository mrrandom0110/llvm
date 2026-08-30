"""Safe, normalized repository-relative path handling for the symbol index.

Two distinct notions of "safe path" are needed by the indexing pipeline:

- :func:`normalize_relative_path` / :func:`to_absolute_path` resolve a path
  against the filesystem (following symlinks) and require containment under
  the kernel repository root *right now*. These are used when converting a
  freshly observed filesystem path into canonical form.
- :func:`is_safe_relative_path` / :func:`assert_safe_relative_path` are
  purely lexical, filesystem-free checks used to validate a relative path
  that is already stored in the database (or supplied by an API caller as a
  disambiguation filter), since the file may no longer exist at HEAD.
"""

from __future__ import annotations

from pathlib import Path


class PathEscapesRepositoryError(ValueError):
    """Raised when a path is not safely contained under the kernel repo root."""


def is_safe_relative_path(candidate: str) -> bool:
    """Purely lexical check: no leading ``/``, no ``..`` segment, non-empty."""
    if not candidate:
        return False

    posix_candidate = candidate.replace("\\", "/")
    if posix_candidate.startswith("/"):
        return False

    segments = posix_candidate.split("/")
    if any(segment in ("", "..") for segment in segments):
        return False

    return True


def assert_safe_relative_path(candidate: str) -> None:
    """Raise :class:`PathEscapesRepositoryError` unless ``candidate`` is safe."""
    if not is_safe_relative_path(candidate):
        raise PathEscapesRepositoryError(
            f"Path is not a safe repository-relative path: {candidate!r}"
        )


def normalize_relative_path(path: str | Path, *, kernel_repo: Path) -> str:
    """Resolve ``path`` (following symlinks) to a canonical POSIX-relative form.

    Accepts a path relative to ``kernel_repo`` or an absolute path. Raises
    :class:`PathEscapesRepositoryError` if the resolved, real filesystem
    location does not fall under ``kernel_repo``.
    """
    repo_root = kernel_repo.resolve()

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate

    resolved = candidate.resolve()

    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise PathEscapesRepositoryError(
            f"Path escapes the kernel repository root: {path!r}"
        ) from exc

    return relative.as_posix()


def to_absolute_path(relative_path: str, *, kernel_repo: Path) -> Path:
    """Inverse of :func:`normalize_relative_path`, with the same containment guarantee."""
    assert_safe_relative_path(relative_path)

    repo_root = kernel_repo.resolve()
    absolute = (repo_root / relative_path).resolve()

    try:
        absolute.relative_to(repo_root)
    except ValueError as exc:
        raise PathEscapesRepositoryError(
            f"Path escapes the kernel repository root: {relative_path!r}"
        ) from exc

    return absolute
