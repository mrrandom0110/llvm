"""Хранилище исследования на sqlite.

Одна база на весь проект: сырые выгрузки, промежуточные признаки и результаты
детекторов. Выбор в пользу sqlite, а не parquet, сделан потому, что выгрузка
возобновляемая и идёт много суток — нужны инкрементальные вставки и понятное
состояние прогресса.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import config

SCHEMA = """
PRAGMA journal_mode=WAL;

-- Кандидаты в выборку и статус их выгрузки.
CREATE TABLE IF NOT EXISTS players (
    account_id   INTEGER PRIMARY KEY,
    source       TEXT,              -- как аккаунт попал в выборку
    seed_rank    REAL,              -- avg_rank_tier матча, из которого он взят
    status       TEXT DEFAULT 'new',-- new | fetched | private | empty | failed
    n_matches    INTEGER,           -- всего матчей в истории
    n_ranked     INTEGER,           -- ranked-матчей в окне исследования
    first_match  INTEGER,
    last_match   INTEGER,
    fetched_at   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_players_status ON players (status);

-- Основная выгрузка: по одной строке на матч игрока.
CREATE TABLE IF NOT EXISTS player_matches (
    account_id    INTEGER NOT NULL,
    match_id      INTEGER NOT NULL,
    start_time    INTEGER,
    duration      INTEGER,
    player_slot   INTEGER,
    radiant_win   INTEGER,
    win           INTEGER,
    lobby_type    INTEGER,
    game_mode     INTEGER,
    hero_id       INTEGER,
    average_rank  REAL,
    party_size    INTEGER,
    leaver_status INTEGER,
    kills         INTEGER,
    deaths        INTEGER,
    assists       INTEGER,
    gold_per_min  INTEGER,
    xp_per_min    INTEGER,
    last_hits     INTEGER,
    hero_damage   INTEGER,
    net_worth     INTEGER,
    level         INTEGER,
    cluster       INTEGER,
    PRIMARY KEY (account_id, match_id)
);
CREATE INDEX IF NOT EXISTS idx_pm_account_time ON player_matches (account_id, start_time);
CREATE INDEX IF NOT EXISTS idx_pm_match ON player_matches (match_id);

-- Полные составы отдельных матчей (дорогая выгрузка: 1 вызов на матч).
CREATE TABLE IF NOT EXISTS match_meta (
    match_id      INTEGER PRIMARY KEY,
    start_time    INTEGER,
    duration      INTEGER,
    radiant_win   INTEGER,
    lobby_type    INTEGER,
    game_mode     INTEGER,
    avg_rank_tier REAL,
    n_known       INTEGER,   -- сколько из 10 игроков не анонимны
    focal_account INTEGER    -- игрок, ради которого матч выгружался
);

CREATE TABLE IF NOT EXISTS roster (
    match_id     INTEGER NOT NULL,
    account_id   INTEGER,
    player_slot  INTEGER NOT NULL,
    is_radiant   INTEGER,
    win          INTEGER,
    hero_id      INTEGER,
    rank_tier    INTEGER,
    kills        INTEGER,
    deaths       INTEGER,
    assists      INTEGER,
    gold_per_min INTEGER,
    xp_per_min   INTEGER,
    net_worth    INTEGER,
    leaver_status INTEGER,
    PRIMARY KEY (match_id, player_slot)
);
CREATE INDEX IF NOT EXISTS idx_roster_account ON roster (account_id);

-- Результаты детекторов смурфов и слабых игроков.
CREATE TABLE IF NOT EXISTS player_profile (
    account_id      INTEGER PRIMARY KEY,
    n_ranked        INTEGER,
    winrate         REAL,
    first_seen      INTEGER,
    last_seen       INTEGER,
    account_age_est INTEGER,   -- оценка даты регистрации по account_id
    rank_start      REAL,
    rank_end        REAL,
    rank_slope      REAL,      -- скорость подъёма, единиц ранга в месяц
    perf_z          REAL,      -- перформанс относительно своего брекета
    climb_z         REAL,
    smurf_score     REAL,
    weak_score      REAL,
    is_smurf        INTEGER,
    is_weak         INTEGER,
    region_switch   REAL,
    label           TEXT       -- resident | smurf | weak | unstable
);

-- Свободная таблица результатов, чтобы отчёт собирался из базы.
CREATE TABLE IF NOT EXISTS findings (
    test    TEXT NOT NULL,
    metric  TEXT NOT NULL,
    value   REAL,
    lo      REAL,
    hi      REAL,
    n       INTEGER,
    note    TEXT,
    PRIMARY KEY (test, metric)
);
"""

PM_COLUMNS: Sequence[str] = (
    "account_id",
    "match_id",
    "start_time",
    "duration",
    "player_slot",
    "radiant_win",
    "win",
    "lobby_type",
    "game_mode",
    "hero_id",
    "average_rank",
    "party_size",
    "leaver_status",
    "kills",
    "deaths",
    "assists",
    "gold_per_min",
    "xp_per_min",
    "last_hits",
    "hero_damage",
    "net_worth",
    "level",
    "cluster",
)

ROSTER_COLUMNS: Sequence[str] = (
    "match_id",
    "account_id",
    "player_slot",
    "is_radiant",
    "win",
    "hero_id",
    "rank_tier",
    "kills",
    "deaths",
    "assists",
    "gold_per_min",
    "xp_per_min",
    "net_worth",
    "leaver_status",
)


# Отдельная выборка «4600–5000»: не смешивается с основной, чтобы не
# сдвинуть цифры исследования подкрутки.
BRACKET_SCHEMA = """
CREATE TABLE IF NOT EXISTS bracket_watch (
    account_id   INTEGER PRIMARY KEY,
    source       TEXT,
    seed_rank    REAL,
    rank_tier    INTEGER,
    computed_mmr REAL,
    in_band      INTEGER,
    status       TEXT DEFAULT 'seen',
    n_matches    INTEGER,
    fetched_at   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_bracket_status ON bracket_watch (status, in_band);

CREATE TABLE IF NOT EXISTS bracket_matches (
    account_id    INTEGER NOT NULL,
    match_id      INTEGER NOT NULL,
    start_time    INTEGER,
    duration      INTEGER,
    player_slot   INTEGER,
    radiant_win   INTEGER,
    win           INTEGER,
    lobby_type    INTEGER,
    game_mode     INTEGER,
    hero_id       INTEGER,
    average_rank  REAL,
    party_size    INTEGER,
    leaver_status INTEGER,
    kills         INTEGER,
    deaths        INTEGER,
    assists       INTEGER,
    gold_per_min  INTEGER,
    xp_per_min    INTEGER,
    last_hits     INTEGER,
    hero_damage   INTEGER,
    net_worth     INTEGER,
    level         INTEGER,
    cluster       INTEGER,
    PRIMARY KEY (account_id, match_id)
);
CREATE INDEX IF NOT EXISTS idx_bm_account_time ON bracket_matches (account_id, start_time);
"""


# Снимок MMR уже выгруженных игроков: не смешивается с очередью players.
SNAPSHOT_SCHEMA = """
CREATE TABLE IF NOT EXISTS mmr_snapshot (
    account_id   INTEGER PRIMARY KEY,
    computed_mmr REAL,
    rank_tier    INTEGER,
    lobby_rank   REAL,
    fetched_at   INTEGER
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(str(path or config.STUDY_DB), timeout=120)
    conn.executescript(SCHEMA)
    conn.executescript(BRACKET_SCHEMA)
    conn.executescript(SNAPSHOT_SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


def _insert_many(
    conn: sqlite3.Connection, table: str, columns: Sequence[str], rows: Iterable[dict[str, Any]]
) -> int:
    placeholders = ",".join("?" * len(columns))
    sql = f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    payload = [tuple(row.get(col) for col in columns) for row in rows]
    if not payload:
        return 0
    conn.executemany(sql, payload)
    return len(payload)


def insert_player_matches(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    return _insert_many(conn, "player_matches", PM_COLUMNS, rows)


def insert_bracket_matches(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    return _insert_many(conn, "bracket_matches", PM_COLUMNS, rows)


def insert_roster(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> int:
    return _insert_many(conn, "roster", ROSTER_COLUMNS, rows)


def add_seeds(
    conn: sqlite3.Connection, seeds: Iterable[tuple[int, str, float | None]]
) -> int:
    """Регистрирует кандидатов, не затирая уже выгруженных."""
    rows = list(seeds)
    if not rows:
        return 0
    conn.executemany(
        "INSERT OR IGNORE INTO players (account_id, source, seed_rank) VALUES (?,?,?)",
        rows,
    )
    conn.commit()
    return conn.total_changes


def clear_findings(conn: sqlite3.Connection, *tests: str) -> None:
    """Удаляет прежние результаты теста перед новой записью.

    Без этого в таблице накапливаются величины из старых прогонов с другими
    настройками, и отчёт смешивал бы несопоставимые числа.
    """
    for test in tests:
        conn.execute("DELETE FROM findings WHERE test = ?", (test,))
    conn.commit()


def record_finding(
    conn: sqlite3.Connection,
    test: str,
    metric: str,
    value: float | None,
    lo: float | None = None,
    hi: float | None = None,
    n: int | None = None,
    note: str = "",
) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO findings (test, metric, value, lo, hi, n, note) VALUES (?,?,?,?,?,?,?)",
        (test, metric, value, lo, hi, n, note),
    )
    conn.commit()


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    out = {}
    for table in ("players", "player_matches", "match_meta", "roster", "player_profile"):
        out[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    out["players_fetched"] = conn.execute(
        "SELECT count(*) FROM players WHERE status = 'fetched'"
    ).fetchone()[0]
    return out
