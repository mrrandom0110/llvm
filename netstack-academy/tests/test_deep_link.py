from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pytest

from netstack_academy.deep_link import build_editor_deep_link


def test_deep_link_builds_cursor_scheme_for_file_under_kernel_repo(
    git_repository: Path,
) -> None:
    target = git_repository / "net" / "ipv4" / "tcp_input.c"
    link = build_editor_deep_link(
        file_path=target,
        line=12,
        column=3,
        kernel_repo=git_repository,
        wsl_distro="Ubuntu",
        editor_scheme="cursor",
    )
    assert (
        link
        == f"cursor://vscode-remote/wsl+Ubuntu{target.resolve().as_posix()}:12:3"
    )


def test_deep_link_builds_vscode_scheme_when_configured(
    git_repository: Path,
) -> None:
    target = git_repository / "net" / "ipv4" / "tcp_input.c"
    link = build_editor_deep_link(
        file_path=target,
        line=1,
        column=1,
        kernel_repo=git_repository,
        wsl_distro="Ubuntu",
        editor_scheme="vscode",
    )
    assert (
        link
        == f"vscode://vscode-remote/wsl+Ubuntu{target.resolve().as_posix()}:1:1"
    )


def test_deep_link_url_encodes_distro_and_path_segments(
    tmp_path: Path,
) -> None:
    kernel_repo = tmp_path / "kernel repo"
    kernel_repo.mkdir()
    target = kernel_repo / "net stack" / "tcp input.c"
    target.parent.mkdir()
    target.write_text("payload\n", encoding="utf-8")

    link = build_editor_deep_link(
        file_path=target,
        line=4,
        column=2,
        kernel_repo=kernel_repo,
        wsl_distro="Ubuntu 22.04",
        editor_scheme="cursor",
    )

    encoded_distro = quote("Ubuntu 22.04", safe="")
    encoded_path = quote(target.resolve().as_posix(), safe="/")
    assert (
        link
        == f"cursor://vscode-remote/wsl+{encoded_distro}{encoded_path}:4:2"
    )


def test_deep_link_rejects_path_outside_kernel_repo(
    tmp_path: Path,
) -> None:
    kernel_repo = tmp_path / "kernel"
    kernel_repo.mkdir()
    outside = tmp_path / "outside.c"
    outside.write_text("outside\n", encoding="utf-8")

    with pytest.raises(ValueError, match="outside|kernel"):
        build_editor_deep_link(
            file_path=outside,
            line=1,
            column=1,
            kernel_repo=kernel_repo,
            wsl_distro="Ubuntu",
            editor_scheme="cursor",
        )


def test_deep_link_rejects_parent_directory_traversal(
    git_repository: Path,
) -> None:
    traversal = git_repository / "net" / ".." / ".." / "etc" / "passwd"

    with pytest.raises(ValueError, match="traversal|outside|kernel"):
        build_editor_deep_link(
            file_path=traversal,
            line=1,
            column=1,
            kernel_repo=git_repository,
            wsl_distro="Ubuntu",
            editor_scheme="cursor",
        )


def test_deep_link_rejects_non_positive_line(git_repository: Path) -> None:
    target = git_repository / "net" / "ipv4" / "tcp_input.c"

    with pytest.raises(ValueError, match="line"):
        build_editor_deep_link(
            file_path=target,
            line=0,
            column=1,
            kernel_repo=git_repository,
            wsl_distro="Ubuntu",
            editor_scheme="cursor",
        )


def test_deep_link_rejects_non_positive_column(git_repository: Path) -> None:
    target = git_repository / "net" / "ipv4" / "tcp_input.c"

    with pytest.raises(ValueError, match="column"):
        build_editor_deep_link(
            file_path=target,
            line=1,
            column=0,
            kernel_repo=git_repository,
            wsl_distro="Ubuntu",
            editor_scheme="cursor",
        )


def test_deep_link_accepts_resolved_symlink_under_kernel_repo(
    git_repository: Path,
) -> None:
    real_file = git_repository / "net" / "ipv4" / "tcp_input.c"
    link_path = git_repository / "shortcut.c"
    link_path.symlink_to(real_file)

    link = build_editor_deep_link(
        file_path=link_path,
        line=7,
        column=5,
        kernel_repo=git_repository,
        wsl_distro="Ubuntu",
        editor_scheme="cursor",
    )

    assert (
        link
        == f"cursor://vscode-remote/wsl+Ubuntu{real_file.resolve().as_posix()}:7:5"
    )
