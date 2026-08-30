from __future__ import annotations

from pathlib import Path
from urllib.parse import quote


def build_editor_deep_link(
    *,
    file_path: Path,
    line: int,
    column: int,
    kernel_repo: Path,
    wsl_distro: str,
    editor_scheme: str,
) -> str:
    if line <= 0:
        raise ValueError("line must be a positive integer")
    if column <= 0:
        raise ValueError("column must be a positive integer")

    repo_root = kernel_repo.resolve()
    resolved_file = file_path.resolve()

    try:
        resolved_file.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("file path is outside kernel repository") from exc

    if not resolved_file.is_file():
        raise ValueError("file path is outside kernel repository")

    encoded_distro = quote(wsl_distro, safe="")
    encoded_path = quote(resolved_file.as_posix(), safe="/")
    return (
        f"{editor_scheme}://vscode-remote/wsl+{encoded_distro}{encoded_path}"
        f":{line}:{column}"
    )
