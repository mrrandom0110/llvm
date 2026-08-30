"""Storage contract for semantic call/reference edges and their source sites.

A semantically derived edge is only useful for navigation if it records
*where* the call or reference happens: a call edge whose only location is
the caller's definition line cannot deep-link to the call site, and a
reference edge without a location carries no information at all beyond
"this symbol is used somewhere". So ``EdgeInput``/``Edge`` must carry the
call/reference site as a repository-relative path plus 1-based line and
column, and those three fields must round-trip through
``replace_symbols_and_edges``.

The site fields are optional: the regex fallback indexer's heuristic edges
do not (yet) supply one, and a semantic call with no ``fromRanges`` has no
site either, so an edge inserted without a site must read back with all
three fields ``None`` rather than failing to insert.

Because the three columns are new, an on-disk database written by the
version-1 schema must migrate forward in place, keeping its commit,
symbols, edges and FTS rows intact.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from netstack_academy.indexing.models import EdgeInput, SymbolInput
from netstack_academy.indexing.storage import SCHEMA_VERSION, IndexStorage

TCP_INPUT = "net/ipv4/tcp_input.c"
TCP_UTIL = "net/ipv4/tcp_util.c"

#: The version-1 schema, verbatim, so a v1 database can be created here
#: without depending on the current (migrated) production DDL.
_V1_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE schema_meta (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        version INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE commits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hash TEXT NOT NULL UNIQUE,
        indexed_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE current_head (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        commit_id INTEGER NOT NULL REFERENCES commits(id)
    )
    """,
    """
    CREATE TABLE symbols (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        commit_id INTEGER NOT NULL REFERENCES commits(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        kind TEXT NOT NULL,
        relative_path TEXT NOT NULL,
        line INTEGER NOT NULL,
        column INTEGER,
        signature TEXT,
        scope TEXT,
        is_static INTEGER NOT NULL DEFAULT 0,
        UNIQUE (relative_path, line, commit_id)
    )
    """,
    """
    CREATE TABLE edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        commit_id INTEGER NOT NULL REFERENCES commits(id) ON DELETE CASCADE,
        source_symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
        target_symbol_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
        target_name TEXT NOT NULL,
        edge_type TEXT NOT NULL CHECK (edge_type IN ('call', 'reference')),
        provenance TEXT NOT NULL CHECK (provenance IN ('heuristic', 'semantic'))
    )
    """,
    """
    CREATE VIRTUAL TABLE symbols_fts USING fts5(
        name,
        signature,
        symbol_id UNINDEXED
    )
    """,
)


def _symbol(name: str, relative_path: str, line: int) -> SymbolInput:
    return SymbolInput(
        name=name,
        kind="function",
        relative_path=relative_path,
        line=line,
        signature="(int x)",
    )


def _write_version_one_database(db_path: Path) -> None:
    """Create a populated database at schema version 1."""
    connection = sqlite3.connect(db_path, isolation_level=None)
    try:
        for statement in _V1_SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.execute("INSERT INTO schema_meta (id, version) VALUES (1, 1)")
        connection.execute(
            "INSERT INTO commits (id, hash, indexed_at) VALUES (1, ?, ?)",
            ("v1-head", "2024-01-01T00:00:00+00:00"),
        )
        connection.execute(
            """
            INSERT INTO symbols (
                id, commit_id, name, kind, relative_path, line,
                column, signature, scope, is_static
            ) VALUES (1, 1, 'tcp_process', 'function', ?, 1, NULL, '(int x)', NULL, 0)
            """,
            (TCP_INPUT,),
        )
        connection.execute(
            """
            INSERT INTO symbols (
                id, commit_id, name, kind, relative_path, line,
                column, signature, scope, is_static
            ) VALUES (2, 1, 'tcp_helper', 'function', ?, 1, NULL, '(int x)', NULL, 0)
            """,
            (TCP_UTIL,),
        )
        connection.execute(
            "INSERT INTO symbols_fts (name, signature, symbol_id) VALUES (?, ?, ?)",
            ("tcp_process", "(int x)", 1),
        )
        connection.execute(
            "INSERT INTO symbols_fts (name, signature, symbol_id) VALUES (?, ?, ?)",
            ("tcp_helper", "(int x)", 2),
        )
        connection.execute(
            """
            INSERT INTO edges (
                id, commit_id, source_symbol_id, target_symbol_id,
                target_name, edge_type, provenance
            ) VALUES (1, 1, 1, 2, 'tcp_helper', 'call', 'heuristic')
            """
        )
        connection.execute("INSERT INTO current_head (id, commit_id) VALUES (1, 1)")
    finally:
        connection.close()


def test_schema_version_accounts_for_edge_site_columns() -> None:
    """The site columns are a schema change, so the version must advance."""
    assert SCHEMA_VERSION >= 2


def test_semantic_call_edge_round_trips_its_call_site(tmp_path: Path) -> None:
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        storage.replace_symbols_and_edges(
            "deadbeef",
            [_symbol("tcp_process", TCP_INPUT, 1), _symbol("tcp_helper", TCP_UTIL, 1)],
            [
                EdgeInput(
                    source_index=0,
                    target_index=1,
                    target_name="tcp_helper",
                    edge_type="call",
                    provenance="semantic",
                    site_relative_path=TCP_INPUT,
                    site_line=3,
                    site_column=12,
                )
            ],
        )

        caller = storage.find_symbols_by_name("tcp_process")[0]
        edges = storage.outgoing_edges(caller.id)

    assert len(edges) == 1
    assert (
        edges[0].site_relative_path,
        edges[0].site_line,
        edges[0].site_column,
    ) == (TCP_INPUT, 3, 12)


def test_edge_without_a_site_reads_back_with_empty_site_fields(tmp_path: Path) -> None:
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        storage.replace_symbols_and_edges(
            "deadbeef",
            [_symbol("tcp_process", TCP_INPUT, 1), _symbol("tcp_helper", TCP_UTIL, 1)],
            [
                EdgeInput(
                    source_index=0,
                    target_index=1,
                    target_name="tcp_helper",
                    edge_type="call",
                    provenance="heuristic",
                )
            ],
        )

        caller = storage.find_symbols_by_name("tcp_process")[0]
        edges = storage.outgoing_edges(caller.id)

    assert edges[0].site_relative_path is None
    assert edges[0].site_line is None
    assert edges[0].site_column is None


def test_reference_edge_is_anchored_on_the_referenced_symbol(tmp_path: Path) -> None:
    """A reference edge records the referenced symbol plus the use site.

    There is no second symbol to point at (the use site is a position in a
    file, not necessarily inside another indexed definition), so the edge
    hangs off the referenced symbol with ``target_index=None`` and carries
    the location in its site columns.
    """
    with IndexStorage.open(tmp_path / "index.sqlite3") as storage:
        storage.replace_symbols_and_edges(
            "deadbeef",
            [_symbol("tcp_helper", TCP_UTIL, 1)],
            [
                EdgeInput(
                    source_index=0,
                    target_index=None,
                    target_name="tcp_helper",
                    edge_type="reference",
                    provenance="semantic",
                    site_relative_path=TCP_INPUT,
                    site_line=3,
                    site_column=12,
                )
            ],
        )

        helper = storage.find_symbols_by_name("tcp_helper")[0]
        edges = storage.references(helper.id)

    assert len(edges) == 1
    assert edges[0].edge_type == "reference"
    assert edges[0].provenance == "semantic"
    assert (
        edges[0].site_relative_path,
        edges[0].site_line,
        edges[0].site_column,
    ) == (TCP_INPUT, 3, 12)


def test_opening_a_version_one_database_migrates_it_without_data_loss(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "index.sqlite3"
    _write_version_one_database(db_path)

    with IndexStorage.open(db_path) as storage:
        assert storage.schema_version == SCHEMA_VERSION
        assert storage.current_head() == "v1-head"
        assert storage.symbol_count() == 2
        assert storage.edge_count() == 1

        caller = storage.find_symbols_by_name("tcp_process")[0]
        assert caller.relative_path == TCP_INPUT
        assert caller.commit_hash == "v1-head"
        assert [symbol.name for symbol in storage.search_symbols("tcp_helper")] == [
            "tcp_helper"
        ]

        migrated_edge = storage.outgoing_edges(caller.id)[0]
        assert migrated_edge.target_name == "tcp_helper"
        assert migrated_edge.provenance == "heuristic"
        assert migrated_edge.site_relative_path is None
        assert migrated_edge.site_line is None
        assert migrated_edge.site_column is None

        # The migrated database accepts the new site columns straight away.
        storage.replace_symbols_and_edges(
            "next-head",
            [_symbol("tcp_helper", TCP_UTIL, 1)],
            [
                EdgeInput(
                    source_index=0,
                    target_index=None,
                    target_name="tcp_helper",
                    edge_type="reference",
                    provenance="semantic",
                    site_relative_path=TCP_INPUT,
                    site_line=3,
                    site_column=12,
                )
            ],
        )

        helper = storage.find_symbols_by_name("tcp_helper")[0]
        assert storage.references(helper.id)[0].site_line == 3


def test_migrated_and_fresh_databases_agree_on_schema(tmp_path: Path) -> None:
    migrated_path = tmp_path / "migrated.sqlite3"
    _write_version_one_database(migrated_path)

    with IndexStorage.open(migrated_path) as migrated:
        migrated_columns = migrated.connection.execute(
            "SELECT name FROM pragma_table_info('edges') ORDER BY name"
        ).fetchall()

    with IndexStorage.open(tmp_path / "fresh.sqlite3") as fresh:
        fresh_columns = fresh.connection.execute(
            "SELECT name FROM pragma_table_info('edges') ORDER BY name"
        ).fetchall()

    assert migrated_columns == fresh_columns
    assert ("site_relative_path",) in fresh_columns
    assert ("site_line",) in fresh_columns
    assert ("site_column",) in fresh_columns
