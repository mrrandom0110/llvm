"""Выгрузка историй игроков.

Это самая ценная часть бюджета API: один вызов `/players/{id}/matches`
возвращает до нескольких тысяч матчей одного игрока. Именно поэтому основной
объём исследования строится на историях, а не на выгрузке отдельных матчей.

История берётся целиком, без фильтра по датам на стороне сервера: самый ранний
матч нужен для оценки возраста аккаунта в детекторе смурфов.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any, Sequence

from . import db
from .api import OpenDotaClient, OpenDotaError, QuotaExhausted

log = logging.getLogger(__name__)

# net_worth в проекции истории недоступен, поэтому не запрашивается.
HISTORY_FIELDS: Sequence[str] = (
    "match_id",
    "player_slot",
    "radiant_win",
    "duration",
    "game_mode",
    "lobby_type",
    "hero_id",
    "start_time",
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
    "level",
    "cluster",
)


def _rows_for_player(account_id: int, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in history:
        slot = entry.get("player_slot")
        radiant_win = entry.get("radiant_win")
        if slot is None or radiant_win is None:
            continue
        is_radiant = slot < 128
        row = {
            "account_id": account_id,
            "win": int(bool(radiant_win) == is_radiant),
            "radiant_win": int(bool(radiant_win)),
        }
        for field in HISTORY_FIELDS:
            if field == "radiant_win":
                continue
            row[field] = entry.get(field)
        rows.append(row)
    return rows


def crawl_players(
    client: OpenDotaClient,
    conn: sqlite3.Connection,
    limit_players: int,
    history_limit: int = 5000,
    reserve: int = 0,
) -> dict[str, int]:
    """Выгружает истории игроков со статусом `new`.

    Возобновляемая операция: статус каждого игрока фиксируется в базе сразу,
    поэтому прерывание по квоте или по времени не теряет прогресс.
    """
    pending = conn.execute(
        "SELECT account_id FROM players WHERE status = 'new' ORDER BY random() LIMIT ?",
        (limit_players,),
    ).fetchall()

    stats = {"fetched": 0, "private": 0, "empty": 0, "rows": 0, "ranked_rows": 0}
    started = time.time()

    for idx, record in enumerate(pending, 1):
        account_id = record["account_id"]
        if client.budget_left() <= reserve:
            log.warning("остановка: бюджет опустился до резерва %d", reserve)
            break
        try:
            history = client.player_matches(
                account_id, limit=history_limit, project=HISTORY_FIELDS
            )
        except QuotaExhausted:
            log.warning("квота исчерпана на игроке %d", account_id)
            break
        except OpenDotaError as exc:
            log.warning("игрок %s пропущен: %s", account_id, exc)
            conn.execute(
                "UPDATE players SET status='failed' WHERE account_id=?", (account_id,)
            )
            conn.commit()
            continue

        if history is None:
            conn.execute(
                "UPDATE players SET status='private', fetched_at=? WHERE account_id=?",
                (int(time.time()), account_id),
            )
            stats["private"] += 1
            conn.commit()
            continue

        rows = _rows_for_player(account_id, history)
        if not rows:
            conn.execute(
                "UPDATE players SET status='empty', fetched_at=? WHERE account_id=?",
                (int(time.time()), account_id),
            )
            stats["empty"] += 1
            conn.commit()
            continue

        db.insert_player_matches(conn, rows)
        times = [r["start_time"] for r in rows if r["start_time"]]
        ranked = sum(1 for r in rows if r["lobby_type"] == 7)
        conn.execute(
            """UPDATE players
               SET status='fetched', n_matches=?, n_ranked=?, first_match=?,
                   last_match=?, fetched_at=?
               WHERE account_id=?""",
            (
                len(rows),
                ranked,
                min(times) if times else None,
                max(times) if times else None,
                int(time.time()),
                account_id,
            ),
        )
        conn.commit()
        stats["fetched"] += 1
        stats["rows"] += len(rows)
        stats["ranked_rows"] += ranked

        if idx % 50 == 0:
            rate = idx / max(time.time() - started, 1e-9)
            log.info(
                "%d/%d игроков | %s матчей (%s ranked) | приватных %d | бюджет %d | %.1f игрок/с",
                idx,
                len(pending),
                f"{stats['rows']:,}",
                f"{stats['ranked_rows']:,}",
                stats["private"],
                client.budget_left(),
                rate,
            )

    return stats
