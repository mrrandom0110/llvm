from __future__ import annotations

from pathlib import Path

import pytest

from netstack_academy.indexing.paths import (
    PathEscapesRepositoryError,
    assert_safe_relative_path,
    is_safe_relative_path,
    normalize_relative_path,
    to_absolute_path,
)


def test_normalize_relative_path_accepts_relative_path_under_repo(
    git_repository: Path,
) -> None:
    relative = normalize_relative_path(
        "net/ipv4/tcp_input.c", kernel_repo=git_repository
    )

    assert relative == "net/ipv4/tcp_input.c"


def test_normalize_relative_path_accepts_absolute_path_under_repo(
    git_repository: Path,
) -> None:
    absolute = git_repository / "net" / "ipv4" / "tcp_input.c"

    relative = normalize_relative_path(absolute, kernel_repo=git_repository)

    assert relative == "net/ipv4/tcp_input.c"


def test_normalize_relative_path_collapses_dot_segments(
    git_repository: Path,
) -> None:
    relative = normalize_relative_path(
        "net/./ipv4/../ipv4/tcp_input.c", kernel_repo=git_repository
    )

    assert relative == "net/ipv4/tcp_input.c"


def test_normalize_relative_path_returns_posix_separators(
    git_repository: Path,
) -> None:
    relative = normalize_relative_path(
        "net/ipv4/tcp_input.c", kernel_repo=git_repository
    )

    assert "\\" not in relative
    assert relative == relative.replace("\\", "/")


def test_normalize_relative_path_rejects_parent_traversal_outside_repo(
    git_repository: Path,
) -> None:
    with pytest.raises(PathEscapesRepositoryError):
        normalize_relative_path(
            "net/../../etc/passwd", kernel_repo=git_repository
        )


def test_normalize_relative_path_rejects_absolute_path_outside_repo(
    git_repository: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.c"
    outside.write_text("outside\n", encoding="utf-8")

    with pytest.raises(PathEscapesRepositoryError):
        normalize_relative_path(outside, kernel_repo=git_repository)


def test_normalize_relative_path_resolves_symlink_pointing_inside_repo(
    git_repository: Path,
) -> None:
    real_file = git_repository / "net" / "ipv4" / "tcp_input.c"
    link = git_repository / "shortcut.c"
    link.symlink_to(real_file)

    relative = normalize_relative_path(link, kernel_repo=git_repository)

    assert relative == "net/ipv4/tcp_input.c"


def test_normalize_relative_path_rejects_symlink_escaping_repo(
    git_repository: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.c"
    outside.write_text("outside\n", encoding="utf-8")
    link = git_repository / "escape.c"
    link.symlink_to(outside)

    with pytest.raises(PathEscapesRepositoryError):
        normalize_relative_path(link, kernel_repo=git_repository)


def test_to_absolute_path_roundtrips_with_normalize_relative_path(
    git_repository: Path,
) -> None:
    target = git_repository / "net" / "ipv4" / "tcp_input.c"

    relative = normalize_relative_path(target, kernel_repo=git_repository)
    absolute = to_absolute_path(relative, kernel_repo=git_repository)

    assert absolute == target.resolve()


def test_to_absolute_path_rejects_relative_path_containing_parent_traversal(
    git_repository: Path,
) -> None:
    with pytest.raises(PathEscapesRepositoryError):
        to_absolute_path("../outside.c", kernel_repo=git_repository)


def test_to_absolute_path_rejects_absolute_relative_path_argument(
    git_repository: Path,
) -> None:
    with pytest.raises(PathEscapesRepositoryError):
        to_absolute_path("/etc/passwd", kernel_repo=git_repository)


def test_is_safe_relative_path_accepts_well_formed_relative_path() -> None:
    assert is_safe_relative_path("net/ipv4/tcp_input.c") is True


def test_is_safe_relative_path_does_not_touch_the_filesystem(
    tmp_path: Path,
) -> None:
    """Unlike ``normalize_relative_path``, this check must work purely on the
    string form so that database rows referencing a past commit's files can
    still be validated after HEAD has moved and the file no longer exists.
    """
    missing_but_well_formed = "net/ipv4/does_not_exist_on_disk.c"

    assert is_safe_relative_path(missing_but_well_formed) is True


def test_is_safe_relative_path_rejects_absolute_paths() -> None:
    assert is_safe_relative_path("/etc/passwd") is False


def test_is_safe_relative_path_rejects_parent_traversal() -> None:
    assert is_safe_relative_path("net/../../etc/passwd") is False
    assert is_safe_relative_path("../outside.c") is False


def test_is_safe_relative_path_rejects_empty_string() -> None:
    assert is_safe_relative_path("") is False


def test_assert_safe_relative_path_raises_for_unsafe_input() -> None:
    with pytest.raises(PathEscapesRepositoryError):
        assert_safe_relative_path("../outside.c")


def test_assert_safe_relative_path_returns_none_for_safe_input() -> None:
    assert assert_safe_relative_path("net/ipv4/tcp_input.c") is None
