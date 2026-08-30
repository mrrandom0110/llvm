from __future__ import annotations

from pathlib import Path

import pytest

from netstack_academy.indexing.identity import resolve_symbol
from netstack_academy.indexing.models import SymbolInput
from netstack_academy.indexing.storage import IndexStorage


@pytest.fixture
def storage_with_duplicate_statics(tmp_path: Path):
    storage = IndexStorage.open(tmp_path / "index.sqlite3")
    storage.replace_symbols_and_edges(
        "commit-a",
        [
            SymbolInput(
                name="helper",
                kind="function",
                relative_path="net/ipv4/a.c",
                line=1,
                is_static=True,
            ),
            SymbolInput(
                name="helper",
                kind="function",
                relative_path="net/ipv6/b.c",
                line=20,
                is_static=True,
            ),
            SymbolInput(
                name="shared_util",
                kind="function",
                relative_path="net/util.c",
                line=1,
                is_static=False,
            ),
        ],
        [],
    )
    try:
        yield storage
    finally:
        storage.close()


def test_resolve_symbol_by_name_only_is_ambiguous_for_duplicate_statics(
    storage_with_duplicate_statics: IndexStorage,
) -> None:
    resolution = resolve_symbol(storage_with_duplicate_statics, "helper")

    assert resolution.status == "ambiguous"
    assert resolution.symbol is None
    assert len(resolution.candidates) == 2
    assert {c.relative_path for c in resolution.candidates} == {
        "net/ipv4/a.c",
        "net/ipv6/b.c",
    }


def test_resolve_symbol_with_relative_path_disambiguates_first_file(
    storage_with_duplicate_statics: IndexStorage,
) -> None:
    resolution = resolve_symbol(
        storage_with_duplicate_statics, "helper", relative_path="net/ipv4/a.c"
    )

    assert resolution.status == "found"
    assert resolution.symbol is not None
    assert resolution.symbol.relative_path == "net/ipv4/a.c"
    assert resolution.symbol.line == 1


def test_resolve_symbol_with_relative_path_disambiguates_second_file(
    storage_with_duplicate_statics: IndexStorage,
) -> None:
    resolution = resolve_symbol(
        storage_with_duplicate_statics, "helper", relative_path="net/ipv6/b.c"
    )

    assert resolution.status == "found"
    assert resolution.symbol is not None
    assert resolution.symbol.relative_path == "net/ipv6/b.c"
    assert resolution.symbol.line == 20


def test_resolve_symbol_unique_name_without_path_is_found(
    storage_with_duplicate_statics: IndexStorage,
) -> None:
    resolution = resolve_symbol(storage_with_duplicate_statics, "shared_util")

    assert resolution.status == "found"
    assert resolution.symbol is not None
    assert resolution.symbol.name == "shared_util"


def test_resolve_symbol_unknown_name_is_not_found(
    storage_with_duplicate_statics: IndexStorage,
) -> None:
    resolution = resolve_symbol(storage_with_duplicate_statics, "does_not_exist")

    assert resolution.status == "not_found"
    assert resolution.symbol is None
    assert resolution.candidates == ()
    assert resolution.reason is not None


def test_resolve_symbol_with_path_that_has_no_match_is_not_found_not_ambiguous(
    storage_with_duplicate_statics: IndexStorage,
) -> None:
    resolution = resolve_symbol(
        storage_with_duplicate_statics, "helper", relative_path="net/other/c.c"
    )

    assert resolution.status == "not_found"
    assert resolution.symbol is None


@pytest.mark.parametrize("relative_path", [None, "net/ipv4/a.c", "net/other/c.c"])
def test_resolve_symbol_never_returns_symbol_and_ambiguous_together(
    storage_with_duplicate_statics: IndexStorage, relative_path: str | None
) -> None:
    resolution = resolve_symbol(
        storage_with_duplicate_statics, "helper", relative_path=relative_path
    )

    if resolution.status == "ambiguous":
        assert resolution.symbol is None
        assert len(resolution.candidates) > 1
    elif resolution.status == "found":
        assert resolution.symbol is not None
        assert resolution.candidates == ()
    else:
        assert resolution.symbol is None


def test_resolve_symbol_reflects_latest_commit_after_reindex(
    tmp_path: Path,
) -> None:
    """Symbol identity is keyed by (relative path, location, commit); once the
    database is atomically replaced for a new HEAD, resolution must reflect
    the new commit's location and never resurrect the prior generation's row.
    """
    storage = IndexStorage.open(tmp_path / "index.sqlite3")
    try:
        storage.replace_symbols_and_edges(
            "commit-1",
            [
                SymbolInput(
                    name="tcp_input",
                    kind="function",
                    relative_path="net/ipv4/tcp_input.c",
                    line=10,
                )
            ],
            [],
        )
        storage.replace_symbols_and_edges(
            "commit-2",
            [
                SymbolInput(
                    name="tcp_input",
                    kind="function",
                    relative_path="net/ipv4/tcp_input.c",
                    line=15,
                )
            ],
            [],
        )

        current = resolve_symbol(storage, "tcp_input")
        assert current.status == "found"
        assert current.symbol is not None
        assert current.symbol.commit_hash == "commit-2"
        assert current.symbol.line == 15
    finally:
        storage.close()
