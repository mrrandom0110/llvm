"""Explicit, non-guessing symbol name resolution.

``resolve_symbol`` never silently picks one of several candidates and never
falls back to a different file than the one requested: a unique match is
``found``, zero matches is ``not_found``, and more than one match is
``ambiguous`` with the candidates listed for the caller to disambiguate.
"""

from __future__ import annotations

from .models import SymbolResolution
from .storage import IndexStorage


def resolve_symbol(
    storage: IndexStorage,
    name: str,
    *,
    relative_path: str | None = None,
) -> SymbolResolution:
    matches = storage.find_symbols_by_name(name, relative_path=relative_path)

    if len(matches) == 0:
        location = f" in {relative_path!r}" if relative_path is not None else ""
        return SymbolResolution(
            status="not_found",
            symbol=None,
            candidates=(),
            reason=f"No symbol named {name!r} found{location}",
        )

    if len(matches) == 1:
        return SymbolResolution(
            status="found",
            symbol=matches[0],
            candidates=(),
            reason=None,
        )

    return SymbolResolution(
        status="ambiguous",
        symbol=None,
        candidates=tuple(matches),
        reason=(
            f"Multiple symbols named {name!r} found; "
            "specify relative_path to disambiguate"
        ),
    )
