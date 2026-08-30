from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from netstack_academy.indexing.models import EdgeInput, SymbolInput
from netstack_academy.indexing.storage import IndexStorage


def _symbol(
    name: str,
    relative_path: str,
    line: int,
    *,
    kind: str = "function",
    is_static: bool = False,
    signature: str | None = None,
) -> SymbolInput:
    return SymbolInput(
        name=name,
        kind=kind,
        relative_path=relative_path,
        line=line,
        signature=signature,
        is_static=is_static,
    )


def test_open_creates_database_file(tmp_path: Path) -> None:
    db_path = tmp_path / "index.sqlite3"

    with IndexStorage.open(db_path) as storage:
        assert storage is not None

    assert db_path.exists()


def test_open_enables_wal_journal_mode(tmp_path: Path) -> None:
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        mode = storage.connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert mode.lower() == "wal"


def test_open_enables_foreign_keys(tmp_path: Path) -> None:
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        enabled = storage.connection.execute("PRAGMA foreign_keys").fetchone()[0]

    assert enabled == 1


def test_opening_same_database_path_twice_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "index.sqlite3"

    with IndexStorage.open(db_path) as first:
        first_version = first.schema_version

    with IndexStorage.open(db_path) as second:
        second_version = second.schema_version

    assert first_version == second_version


def test_fresh_databases_have_identical_schema(tmp_path: Path) -> None:
    with IndexStorage.open(tmp_path / "one.sqlite3") as one:
        schema_one = one.connection.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY name"
        ).fetchall()

    with IndexStorage.open(tmp_path / "two.sqlite3") as two:
        schema_two = two.connection.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY name"
        ).fetchall()

    assert schema_one == schema_two


def test_current_head_is_none_for_fresh_database(tmp_path: Path) -> None:
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        assert storage.current_head() is None


def test_close_closes_underlying_connection(tmp_path: Path) -> None:
    storage = IndexStorage.open(tmp_path / "index.sqlite3")
    storage.close()

    with pytest.raises(sqlite3.ProgrammingError):
        storage.connection.execute("SELECT 1")


def test_context_manager_closes_storage_on_exit(tmp_path: Path) -> None:
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        pass

    with pytest.raises(sqlite3.ProgrammingError):
        storage.connection.execute("SELECT 1")


def test_replace_symbols_and_edges_updates_current_head(tmp_path: Path) -> None:
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        storage.replace_symbols_and_edges(
            "deadbeef",
            [_symbol("tcp_input", "net/ipv4/tcp_input.c", 10)],
            [],
        )

        assert storage.current_head() == "deadbeef"


def test_replace_symbols_and_edges_returns_counts(tmp_path: Path) -> None:
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        result = storage.replace_symbols_and_edges(
            "deadbeef",
            [
                _symbol("helper", "net/ipv4/a.c", 1, is_static=True),
                _symbol("process", "net/ipv4/a.c", 6),
            ],
            [
                EdgeInput(
                    source_index=1,
                    target_index=0,
                    target_name="helper",
                    edge_type="call",
                    provenance="heuristic",
                )
            ],
        )

        assert result.symbol_count == 2
        assert result.edge_count == 1


def test_duplicate_static_function_names_produce_distinct_symbol_rows(
    tmp_path: Path,
) -> None:
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        storage.replace_symbols_and_edges(
            "deadbeef",
            [
                _symbol("helper", "net/ipv4/a.c", 1, is_static=True),
                _symbol("helper", "net/ipv6/b.c", 1, is_static=True),
            ],
            [],
        )

        matches = storage.find_symbols_by_name("helper")

    assert len(matches) == 2
    assert {symbol.relative_path for symbol in matches} == {
        "net/ipv4/a.c",
        "net/ipv6/b.c",
    }
    assert {symbol.id for symbol in matches} == {
        matches[0].id,
        matches[1].id,
    }
    assert matches[0].id != matches[1].id


def test_find_symbols_by_name_filters_by_relative_path(tmp_path: Path) -> None:
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        storage.replace_symbols_and_edges(
            "deadbeef",
            [
                _symbol("helper", "net/ipv4/a.c", 1, is_static=True),
                _symbol("helper", "net/ipv6/b.c", 1, is_static=True),
            ],
            [],
        )

        matches = storage.find_symbols_by_name("helper", relative_path="net/ipv4/a.c")

    assert len(matches) == 1
    assert matches[0].relative_path == "net/ipv4/a.c"


def test_search_symbols_uses_fts5_for_substring_matches(tmp_path: Path) -> None:
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        storage.replace_symbols_and_edges(
            "deadbeef",
            [
                _symbol(
                    "tcp_retransmit_skb",
                    "net/ipv4/tcp_output.c",
                    100,
                    signature="(struct sock *sk, struct sk_buff *skb)",
                ),
                _symbol("udp_sendmsg", "net/ipv4/udp.c", 200),
            ],
            [],
        )

        results = storage.search_symbols("retransmit")

    assert [symbol.name for symbol in results] == ["tcp_retransmit_skb"]


def test_outgoing_edges_returns_edges_originating_from_symbol(
    tmp_path: Path,
) -> None:
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        storage.replace_symbols_and_edges(
            "deadbeef",
            [
                _symbol("process", "net/ipv4/a.c", 6),
                _symbol("helper", "net/ipv4/a.c", 1, is_static=True),
            ],
            [
                EdgeInput(
                    source_index=0,
                    target_index=1,
                    target_name="helper",
                    edge_type="call",
                    provenance="heuristic",
                )
            ],
        )

        process_symbol = storage.find_symbols_by_name("process")[0]
        edges = storage.outgoing_edges(process_symbol.id)

    assert len(edges) == 1
    assert edges[0].target_name == "helper"
    assert edges[0].edge_type == "call"
    assert edges[0].provenance == "heuristic"


def test_incoming_edges_returns_edges_targeting_symbol(tmp_path: Path) -> None:
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        storage.replace_symbols_and_edges(
            "deadbeef",
            [
                _symbol("process", "net/ipv4/a.c", 6),
                _symbol("helper", "net/ipv4/a.c", 1, is_static=True),
            ],
            [
                EdgeInput(
                    source_index=0,
                    target_index=1,
                    target_name="helper",
                    edge_type="call",
                    provenance="heuristic",
                )
            ],
        )

        helper_symbol = storage.find_symbols_by_name("helper")[0]
        edges = storage.incoming_edges(helper_symbol.id)

    assert len(edges) == 1
    assert edges[0].source_symbol_id != helper_symbol.id


def test_edge_referencing_unknown_symbol_index_is_rejected(tmp_path: Path) -> None:
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        with pytest.raises(Exception):
            storage.replace_symbols_and_edges(
                "deadbeef",
                [_symbol("process", "net/ipv4/a.c", 6)],
                [
                    EdgeInput(
                        source_index=0,
                        target_index=99,
                        target_name="ghost",
                        edge_type="call",
                        provenance="heuristic",
                    )
                ],
            )


def test_failed_replace_leaves_previous_index_intact(tmp_path: Path) -> None:
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        storage.replace_symbols_and_edges(
            "first-head",
            [_symbol("tcp_input", "net/ipv4/tcp_input.c", 10)],
            [],
        )

        with pytest.raises(Exception):
            storage.replace_symbols_and_edges(
                "second-head",
                [_symbol("tcp_output", "net/ipv4/tcp_output.c", 20)],
                [
                    EdgeInput(
                        source_index=0,
                        target_index=99,
                        target_name="ghost",
                        edge_type="call",
                        provenance="heuristic",
                    )
                ],
            )

        assert storage.current_head() == "first-head"
        assert [s.name for s in storage.find_symbols_by_name("tcp_input")] == [
            "tcp_input"
        ]
        assert storage.find_symbols_by_name("tcp_output") == []


def _indexed_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    """All columns covered by at least one index (of any kind) on ``table``."""
    columns: set[str] = set()
    for index_row in connection.execute(f"PRAGMA index_list('{table}')").fetchall():
        index_name = index_row[1]
        for info_row in connection.execute(f"PRAGMA index_info('{index_name}')").fetchall():
            column_name = info_row[2]
            if column_name is not None:
                columns.add(column_name)
    return columns


def test_fresh_schema_indexes_symbol_name_column(tmp_path: Path) -> None:
    """``find_symbols_by_name``/symbol resolution filters on ``symbols.name``
    on every call; a fresh database must not force a full table scan for
    that lookup.
    """
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        indexed_columns = _indexed_columns(storage.connection, "symbols")

    assert "name" in indexed_columns


def test_fresh_schema_indexes_edge_source_and_target_columns(tmp_path: Path) -> None:
    """``outgoing_edges``/``incoming_edges`` filter on
    ``edges.source_symbol_id``/``edges.target_symbol_id`` respectively; a
    fresh database must not force a full table scan for either direction.
    """
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        indexed_columns = _indexed_columns(storage.connection, "edges")

    assert "source_symbol_id" in indexed_columns
    assert "target_symbol_id" in indexed_columns


def _symbol_names(storage: IndexStorage, query: str) -> list[str]:
    return [symbol.name for symbol in storage.search_symbols(query)]


def _seed_search_fixture(storage: IndexStorage) -> None:
    storage.replace_symbols_and_edges(
        "deadbeef",
        [
            _symbol(
                "tcp_retransmit_skb",
                "net/ipv4/tcp_output.c",
                100,
                signature="(struct sock *sk, struct sk_buff *skb)",
            ),
            _symbol("udp_sendmsg", "net/ipv4/udp.c", 200),
            _symbol("and_then_helper", "net/util.c", 300),
        ],
        [],
    )


@pytest.mark.parametrize(
    "query",
    [
        '"',
        '"tcp',
        'tcp"',
        ";",
        ";tcp",
        "tcp;",
        "*",
        "AND",
        "OR",
        "NOT",
        "tcp AND",
        "()",
        "(tcp",
        "tcp)",
        "tcp:input",
        "-tcp",
        "^tcp",
    ],
)
def test_search_symbols_never_raises_for_malformed_query_text(
    tmp_path: Path, query: str
) -> None:
    """FTS5 MATCH syntax treats quotes, semicolons, boolean keywords,
    wildcard/operator characters, and column-filter syntax specially; any
    of these appearing in ordinary user search text must never surface as
    an ``sqlite3.OperationalError`` all the way up to the caller. Worst
    case, the query is treated as literal text that matches nothing.
    """
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        _seed_search_fixture(storage)

        results = storage.search_symbols(query)

    assert isinstance(results, list)


def test_search_symbols_returns_empty_list_for_empty_query(tmp_path: Path) -> None:
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        _seed_search_fixture(storage)

        results = storage.search_symbols("")

    assert results == []


def test_search_symbols_treats_reserved_boolean_keyword_as_literal_text(
    tmp_path: Path,
) -> None:
    """``AND``/``OR``/``NOT`` are FTS5 boolean operators when unquoted, but
    a symbol search box has no boolean-query syntax exposed to the UI: a
    reserved keyword typed by a user is just the word they are searching
    for, and must be matched (or safely fail to match) as literal text.
    """
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        _seed_search_fixture(storage)

        results = _symbol_names(storage, "AND")

    assert results == ["and_then_helper"]


def test_search_symbols_treats_trailing_semicolon_as_literal_punctuation(
    tmp_path: Path,
) -> None:
    """A stray trailing semicolon (e.g. pasted from source code) must not
    turn an otherwise-valid search into an FTS5 syntax error, and must not
    prevent the intended symbol from being found.
    """
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        _seed_search_fixture(storage)

        results = _symbol_names(storage, "tcp_retransmit_skb;")

    assert results == ["tcp_retransmit_skb"]


def test_search_symbols_preserves_prefix_search_with_trailing_wildcard(
    tmp_path: Path,
) -> None:
    """The UI relies on an explicit trailing ``*`` for prefix/autocomplete
    search (e.g. typing ``retr*`` while a name is still being completed);
    hardening malformed input must not regress this normal, intentional
    wildcard usage.
    """
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        _seed_search_fixture(storage)

        results = _symbol_names(storage, "retr*")

    assert results == ["tcp_retransmit_skb"]


def test_search_symbols_preserves_exact_name_search(tmp_path: Path) -> None:
    """Hardening malformed input must not regress the ordinary case: an
    exact, well-formed symbol name must still be found.
    """
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        _seed_search_fixture(storage)

        results = _symbol_names(storage, "tcp_retransmit_skb")

    assert results == ["tcp_retransmit_skb"]
