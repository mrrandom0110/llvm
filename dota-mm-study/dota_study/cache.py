"""Дисковый кэш HTTP-ответов.

Суточный лимит API — главный дефицитный ресурс исследования, поэтому ни один
ответ не должен запрашиваться дважды. Кэш переживает перезапуск процесса и
позволяет продолжать выгрузку на следующие сутки с того же места.
"""

from __future__ import annotations

import json
import sqlite3
import time
import zlib
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS http_cache (
    key        TEXT PRIMARY KEY,
    status     INTEGER NOT NULL,
    fetched_at INTEGER NOT NULL,
    body       BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_http_cache_fetched ON http_cache (fetched_at);
"""


class HttpCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), timeout=60)
        self._conn.executescript(_SCHEMA)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()

    def get(self, key: str, max_age: int | None = None) -> Any | None:
        row = self._conn.execute(
            "SELECT status, fetched_at, body FROM http_cache WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        status, fetched_at, body = row
        if max_age is not None and time.time() - fetched_at > max_age:
            return None
        if status != 200:
            return None
        return json.loads(zlib.decompress(body).decode("utf-8"))

    def put(self, key: str, status: int, payload: Any) -> None:
        blob = zlib.compress(json.dumps(payload).encode("utf-8"), 6)
        self._conn.execute(
            "INSERT OR REPLACE INTO http_cache (key, status, fetched_at, body) VALUES (?,?,?,?)",
            (key, status, int(time.time()), blob),
        )
        self._conn.commit()

    def has(self, key: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM http_cache WHERE key = ? AND status = 200", (key,)
        ).fetchone()
        return row is not None

    def size(self) -> int:
        return self._conn.execute("SELECT count(*) FROM http_cache").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
