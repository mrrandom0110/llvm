"""Editor deep links, or an honest reason there isn't one.

:func:`~netstack_academy.deep_link.build_editor_deep_link` refuses to build a
link that leaves the kernel repository or points at a file that is not there,
and it signals that by raising. A page cannot fail for that reason -- a symbol
whose file was deleted since the index was built is still worth showing -- so
:func:`editor_deep_link` turns the refusal into a ``(None, reason)`` pair the
template and the JSON payload can both render.

The stored path is checked lexically before anything touches the filesystem.
Nothing this program writes produces an index row whose path escapes the
repository, but the index is a file on disk that outlives the process that
wrote it, and this builder is the last thing between a corrupt row and a
link that opens something outside the kernel tree.
"""

from __future__ import annotations

from pathlib import Path

from netstack_academy.deep_link import build_editor_deep_link
from netstack_academy.indexing.paths import is_safe_relative_path
from netstack_academy.settings import Settings

#: Returned for a path that is not repository-relative at all.
ESCAPING_PATH_REASON = (
    "The indexed path is not inside the kernel repository, so no editor link "
    "can be built for it."
)

#: Returned for a path that is fine but names a file that is not there.
MISSING_FILE_REASON = (
    "The file is not present in the kernel repository at its current HEAD, so "
    "no editor link can be built for it."
)


def editor_deep_link(
    settings: Settings,
    relative_path: str | None,
    line: int | None,
    column: int | None = None,
) -> tuple[str | None, str | None]:
    """Build a link to ``relative_path:line:column``, or explain the refusal.

    Returns ``(link, None)`` on success and ``(None, reason)`` otherwise. A
    missing path or a non-positive line is treated the same way as an unsafe
    one: there is nothing to point at.
    """
    if not relative_path or not is_safe_relative_path(relative_path):
        return None, ESCAPING_PATH_REASON
    if line is None or line <= 0:
        return None, MISSING_FILE_REASON

    kernel_repo = Path(settings.kernel_repo)
    try:
        link = build_editor_deep_link(
            file_path=kernel_repo / relative_path,
            line=line,
            # The index records a column only when a provider reported one;
            # column 1 is the start of the line, which is where an editor
            # would land anyway.
            column=column if column and column > 0 else 1,
            kernel_repo=kernel_repo,
            wsl_distro=settings.wsl_distro,
            editor_scheme=settings.editor_scheme,
        )
    except ValueError:
        # The builder does not distinguish "outside the repository" from
        # "not on disk", and its own message is not worth forwarding: it can
        # name an absolute path, which is not something to put in a response.
        return None, MISSING_FILE_REASON

    return link, None
