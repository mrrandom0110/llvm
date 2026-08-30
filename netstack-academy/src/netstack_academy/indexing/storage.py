"""SQLite-backed storage for the commit-aware symbol index.

Schema overview
---------------

- ``schema_meta`` — single-row table tracking the applied schema version so
  migrations are idempotent.
- ``commits`` — records which HEADs have been indexed (for audit
  purposes). Only one commit's symbols/edges are retained at a time.
- ``current_head`` — single-row pointer to the ``commits`` row describing
  the generation currently on disk.
- ``symbols`` — one row per definition; uniqueness/identity is
  ``(relative_path, line, commit_id)`` so that duplicate ``static``
  functions defined in different files never collide.
- ``edges`` — call/reference edges between symbols in the same generation,
  each optionally carrying the call/reference *site* (``site_relative_path``,
  ``site_line``, ``site_column``) so a semantically derived edge can
  deep-link to the position where the call/use actually happens.
- ``symbols_fts`` — an FTS5 virtual table over symbol name/signature.

``replace_symbols_and_edges`` fully supersedes the previous generation
(rather than appending to it) inside a single transaction: the prior
generation is only deleted after the new generation has been completely
written, so any failure rolls back and leaves the previous generation and
``current_head()`` untouched.

Schema evolution
----------------

``_SCHEMA_STATEMENTS`` is the *current* schema and is written entirely with
``IF NOT EXISTS``, so it is safe to replay against any existing database.
Anything it cannot express -- adding a column to a table that already
exists -- lives in :func:`_migrate_edge_site_columns`, which is guarded by
the columns actually present rather than by the recorded version alone. A
version-1 database (whose ``edges`` table predates the site columns) is
therefore migrated in place, keeping its commit, symbols, edges and FTS
rows, and ends up with exactly the same schema as a freshly created one.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import Edge, EdgeInput, ReplaceIndexResult, Symbol, SymbolInput

SCHEMA_VERSION = 2

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS commits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hash TEXT NOT NULL UNIQUE,
        indexed_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS current_head (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        commit_id INTEGER NOT NULL REFERENCES commits(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS symbols (
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
    CREATE TABLE IF NOT EXISTS edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        commit_id INTEGER NOT NULL REFERENCES commits(id) ON DELETE CASCADE,
        source_symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
        target_symbol_id INTEGER REFERENCES symbols(id) ON DELETE CASCADE,
        target_name TEXT NOT NULL,
        edge_type TEXT NOT NULL CHECK (edge_type IN ('call', 'reference')),
        provenance TEXT NOT NULL CHECK (provenance IN ('heuristic', 'semantic')),
        site_relative_path TEXT,
        site_line INTEGER,
        site_column INTEGER
    )
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
        name,
        signature,
        symbol_id UNINDEXED
    )
    """,
    # Every lookup below is by symbol name or by one of the edge endpoints,
    # so those are the columns worth indexing; `(relative_path, line,
    # commit_id)` is already covered by the UNIQUE constraint on `symbols`.
    "CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)",
    "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_symbol_id)",
    "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_symbol_id)",
)

#: Columns added in schema version 2, applied to an ``edges`` table that
#: already exists (a version-1 database) since ``CREATE TABLE IF NOT
#: EXISTS`` cannot add them.
_EDGE_SITE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("site_relative_path", "ALTER TABLE edges ADD COLUMN site_relative_path TEXT"),
    ("site_line", "ALTER TABLE edges ADD COLUMN site_line INTEGER"),
    ("site_column", "ALTER TABLE edges ADD COLUMN site_column INTEGER"),
)

_SYMBOL_COLUMNS = (
    "symbols.id",
    "symbols.name",
    "symbols.kind",
    "symbols.relative_path",
    "symbols.line",
    "symbols.column",
    "symbols.signature",
    "symbols.scope",
    "symbols.is_static",
    "commits.hash",
)

_EDGE_COLUMNS = (
    "edges.id",
    "edges.source_symbol_id",
    "edges.target_symbol_id",
    "edges.target_name",
    "edges.edge_type",
    "edges.provenance",
    "commits.hash",
    "edges.site_relative_path",
    "edges.site_line",
    "edges.site_column",
)


def _row_to_symbol(row: tuple[Any, ...]) -> Symbol:
    (
        symbol_id,
        name,
        kind,
        relative_path,
        line,
        column,
        signature,
        scope,
        is_static,
        commit_hash,
    ) = row
    return Symbol(
        id=symbol_id,
        name=name,
        kind=kind,
        relative_path=relative_path,
        line=line,
        column=column,
        signature=signature,
        scope=scope,
        is_static=bool(is_static),
        commit_hash=commit_hash,
    )


def _row_to_edge(row: tuple[Any, ...]) -> Edge:
    (
        edge_id,
        source_symbol_id,
        target_symbol_id,
        target_name,
        edge_type,
        provenance,
        commit_hash,
        site_relative_path,
        site_line,
        site_column,
    ) = row
    return Edge(
        id=edge_id,
        source_symbol_id=source_symbol_id,
        target_symbol_id=target_symbol_id,
        target_name=target_name,
        edge_type=edge_type,
        provenance=provenance,
        commit_hash=commit_hash,
        site_relative_path=site_relative_path,
        site_line=site_line,
        site_column=site_column,
    )


def _migrate_edge_site_columns(connection: sqlite3.Connection) -> None:
    """Add any missing ``edges.site_*`` column, preserving existing rows.

    Guarded by the columns actually present rather than by the recorded
    schema version, so it is a no-op for a database just created from
    ``_SCHEMA_STATEMENTS`` (which already declares them) and completes a
    half-applied migration instead of failing on a duplicate column.
    """
    present = {
        row[0]
        for row in connection.execute("SELECT name FROM pragma_table_info('edges')")
    }
    for column, statement in _EDGE_SITE_COLUMNS:
        if column not in present:
            connection.execute(statement)


class IndexStorage:
    """SQLite-backed storage for symbols and edges of a single kernel repo."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @classmethod
    def open(cls, db_path: str | Path) -> "IndexStorage":
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(db_path, isolation_level=None)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")

        storage = cls(connection)
        storage._run_migrations()
        return storage

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT version FROM schema_meta WHERE id = 1"
        ).fetchone()
        return row[0] if row is not None else 0

    def _run_migrations(self) -> None:
        connection = self._connection
        connection.execute("BEGIN")
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    version INTEGER NOT NULL
                )
                """
            )
            row = connection.execute(
                "SELECT version FROM schema_meta WHERE id = 1"
            ).fetchone()
            current_version = row[0] if row is not None else 0

            if current_version < SCHEMA_VERSION:
                for statement in _SCHEMA_STATEMENTS:
                    connection.execute(statement)
                _migrate_edge_site_columns(connection)
                connection.execute(
                    """
                    INSERT INTO schema_meta (id, version) VALUES (1, ?)
                    ON CONFLICT(id) DO UPDATE SET version = excluded.version
                    """,
                    (SCHEMA_VERSION,),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def current_head(self) -> str | None:
        row = self._connection.execute(
            """
            SELECT commits.hash
            FROM current_head
            JOIN commits ON commits.id = current_head.commit_id
            WHERE current_head.id = 1
            """
        ).fetchone()
        return row[0] if row is not None else None

    def replace_symbols_and_edges(
        self,
        commit_hash: str,
        symbols: list[SymbolInput],
        edges: list[EdgeInput],
    ) -> ReplaceIndexResult:
        symbol_count = len(symbols)
        for edge in edges:
            if not (0 <= edge.source_index < symbol_count):
                raise IndexError(
                    f"EdgeInput.source_index {edge.source_index!r} is out of "
                    f"range for {symbol_count} symbols"
                )
            if edge.target_index is not None and not (
                0 <= edge.target_index < symbol_count
            ):
                raise IndexError(
                    f"EdgeInput.target_index {edge.target_index!r} is out of "
                    f"range for {symbol_count} symbols"
                )

        connection = self._connection
        connection.execute("BEGIN")
        try:
            connection.execute("DELETE FROM current_head")
            connection.execute("DELETE FROM edges")
            connection.execute("DELETE FROM symbols_fts")
            connection.execute("DELETE FROM symbols")
            connection.execute("DELETE FROM commits")

            indexed_at = datetime.now(timezone.utc).isoformat()
            cursor = connection.execute(
                "INSERT INTO commits (hash, indexed_at) VALUES (?, ?)",
                (commit_hash, indexed_at),
            )
            commit_id = cursor.lastrowid

            symbol_ids: list[int] = []
            for symbol in symbols:
                cursor = connection.execute(
                    """
                    INSERT INTO symbols (
                        commit_id, name, kind, relative_path, line,
                        column, signature, scope, is_static
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        commit_id,
                        symbol.name,
                        symbol.kind,
                        symbol.relative_path,
                        symbol.line,
                        symbol.column,
                        symbol.signature,
                        symbol.scope,
                        1 if symbol.is_static else 0,
                    ),
                )
                symbol_id = cursor.lastrowid
                symbol_ids.append(symbol_id)
                connection.execute(
                    "INSERT INTO symbols_fts (name, signature, symbol_id) VALUES (?, ?, ?)",
                    (symbol.name, symbol.signature, symbol_id),
                )

            for edge in edges:
                source_id = symbol_ids[edge.source_index]
                target_id = (
                    symbol_ids[edge.target_index]
                    if edge.target_index is not None
                    else None
                )
                connection.execute(
                    """
                    INSERT INTO edges (
                        commit_id, source_symbol_id, target_symbol_id,
                        target_name, edge_type, provenance,
                        site_relative_path, site_line, site_column
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        commit_id,
                        source_id,
                        target_id,
                        edge.target_name,
                        edge.edge_type,
                        edge.provenance,
                        edge.site_relative_path,
                        edge.site_line,
                        edge.site_column,
                    ),
                )

            connection.execute(
                "INSERT INTO current_head (id, commit_id) VALUES (1, ?)",
                (commit_id,),
            )

            connection.commit()
        except Exception:
            connection.rollback()
            raise

        return ReplaceIndexResult(
            commit_hash=commit_hash,
            symbol_count=len(symbols),
            edge_count=len(edges),
        )

    def find_symbols_by_name(
        self, name: str, *, relative_path: str | None = None
    ) -> list[Symbol]:
        query = (
            f"SELECT {', '.join(_SYMBOL_COLUMNS)} "
            "FROM symbols JOIN commits ON commits.id = symbols.commit_id "
            "WHERE symbols.name = ?"
        )
        params: list[Any] = [name]
        if relative_path is not None:
            query += " AND symbols.relative_path = ?"
            params.append(relative_path)
        query += " ORDER BY symbols.id"

        rows = self._connection.execute(query, params).fetchall()
        return [_row_to_symbol(row) for row in rows]

    def search_symbols(self, query: str, *, limit: int = 50) -> list[Symbol]:
        sql = (
            f"SELECT {', '.join(_SYMBOL_COLUMNS)} "
            "FROM symbols_fts "
            "JOIN symbols ON symbols.id = symbols_fts.symbol_id "
            "JOIN commits ON commits.id = symbols.commit_id "
            "WHERE symbols_fts MATCH ? "
            "ORDER BY rank "
            "LIMIT ?"
        )
        rows = self._connection.execute(sql, (query, limit)).fetchall()
        return [_row_to_symbol(row) for row in rows]

    def outgoing_edges(self, symbol_id: int) -> list[Edge]:
        sql = (
            f"SELECT {', '.join(_EDGE_COLUMNS)} "
            "FROM edges JOIN commits ON commits.id = edges.commit_id "
            "WHERE edges.source_symbol_id = ? "
            "ORDER BY edges.id"
        )
        rows = self._connection.execute(sql, (symbol_id,)).fetchall()
        return [_row_to_edge(row) for row in rows]

    def incoming_edges(self, symbol_id: int) -> list[Edge]:
        sql = (
            f"SELECT {', '.join(_EDGE_COLUMNS)} "
            "FROM edges JOIN commits ON commits.id = edges.commit_id "
            "WHERE edges.target_symbol_id = ? "
            "ORDER BY edges.id"
        )
        rows = self._connection.execute(sql, (symbol_id,)).fetchall()
        return [_row_to_edge(row) for row in rows]

    def references(self, symbol_id: int) -> list[Edge]:
        """All edges (incoming or outgoing) touching ``symbol_id``."""
        combined = self.outgoing_edges(symbol_id) + self.incoming_edges(symbol_id)
        combined.sort(key=lambda edge: edge.id)
        return combined

    def symbol_count(self) -> int:
        """Number of symbols in the currently indexed generation."""
        row = self._connection.execute("SELECT COUNT(*) FROM symbols").fetchone()
        return row[0] if row is not None else 0

    def edge_count(self) -> int:
        """Number of edges in the currently indexed generation."""
        row = self._connection.execute("SELECT COUNT(*) FROM edges").fetchone()
        return row[0] if row is not None else 0

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "IndexStorage":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
