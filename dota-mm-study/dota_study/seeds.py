"""Формирование выборки игроков.

Рамка выборки — случайные публичные ranked-матчи, стратифицированные по
брекетам. Это даёт выборку игроков, взвешенную по активности: чем чаще человек
играет, тем вероятнее он попадёт в выборку. Именно такая выборка отвечает на
вопрос «как устроен типичный матч», а не «как устроен типичный аккаунт».

Побочная выгода: один вызов `/matches/{id}` приносит в среднем около восьми
account_id и одновременно полный состав матча, который нужен тесту C. Поэтому
сбор семян и выгрузка составов делят между собой одни и те же вызовы API.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any, Iterable

from . import db
from .api import OpenDotaClient, OpenDotaError, QuotaExhausted
from .config import LOBBY_RANKED

log = logging.getLogger(__name__)

_SAMPLE_SQL = """
select match_id, avg_rank_tier
from (
  select match_id, avg_rank_tier,
         row_number() over (
           partition by floor(avg_rank_tier / 10)
           order by random()
         ) as rn
  from public_matches
  where start_time >= {lo} and start_time < {hi}
    and lobby_type = {lobby}
    and duration > 900
    and avg_rank_tier is not null
) t
where rn <= {per_bracket}
"""


def sample_match_ids(
    client: OpenDotaClient,
    per_bracket: int = 40,
    days_back: int = 14,
    windows: int = 6,
) -> list[tuple[int, float]]:
    """Стратифицированная случайная выборка match_id по брекетам.

    Берём несколько суточных окон вместо одного длинного: у SQL-эксплорера
    жёсткий таймаут чтения, а разные окна заодно снижают зависимость выборки от
    одного конкретного дня недели.
    """
    now = int(time.time())
    step = max(days_back // max(windows, 1), 1)
    out: list[tuple[int, float]] = []
    seen: set[int] = set()

    for idx in range(windows):
        hi = now - idx * step * 86400
        lo = hi - 86400
        sql = _SAMPLE_SQL.format(
            lo=lo, hi=hi, lobby=LOBBY_RANKED, per_bracket=per_bracket
        )
        try:
            rows = client.explorer(sql, max_age=30 * 86400)
        except QuotaExhausted:
            break
        except OpenDotaError as exc:
            log.warning("окно выборки %d пропущено: %s", idx, exc)
            continue
        for row in rows:
            match_id = int(row["match_id"])
            if match_id in seen:
                continue
            seen.add(match_id)
            out.append((match_id, float(row["avg_rank_tier"])))
        log.info("окно %d: накоплено %d матчей", idx, len(out))

    return out


def _roster_rows(match: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    radiant_win = match.get("radiant_win")
    for player in match.get("players") or []:
        slot = player.get("player_slot")
        if slot is None:
            continue
        is_radiant = slot < 128
        win = None
        if radiant_win is not None:
            win = int(bool(radiant_win) == is_radiant)
        rows.append(
            {
                "match_id": match["match_id"],
                "account_id": player.get("account_id"),
                "player_slot": slot,
                "is_radiant": int(is_radiant),
                "win": win,
                "hero_id": player.get("hero_id"),
                "rank_tier": player.get("rank_tier"),
                "kills": player.get("kills"),
                "deaths": player.get("deaths"),
                "assists": player.get("assists"),
                "gold_per_min": player.get("gold_per_min"),
                "xp_per_min": player.get("xp_per_min"),
                "net_worth": player.get("net_worth"),
                "leaver_status": player.get("leaver_status"),
            }
        )
    return rows


def ingest_match(
    conn: sqlite3.Connection,
    match: dict[str, Any],
    focal_account: int | None = None,
) -> list[int]:
    """Сохраняет состав матча и возвращает найденные account_id."""
    rows = _roster_rows(match)
    known = [r["account_id"] for r in rows if r["account_id"]]
    conn.execute(
        """INSERT OR REPLACE INTO match_meta
           (match_id, start_time, duration, radiant_win, lobby_type, game_mode,
            avg_rank_tier, n_known, focal_account)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            match["match_id"],
            match.get("start_time"),
            match.get("duration"),
            int(bool(match.get("radiant_win"))) if match.get("radiant_win") is not None else None,
            match.get("lobby_type"),
            match.get("game_mode"),
            match.get("avg_rank_tier"),
            len(known),
            focal_account,
        ),
    )
    db.insert_roster(conn, rows)
    return known


def harvest_from_matches(
    client: OpenDotaClient,
    conn: sqlite3.Connection,
    sampled: Iterable[tuple[int, float]],
    budget: int,
) -> int:
    """Выгружает составы матчей и регистрирует найденных игроков как кандидатов."""
    spent = 0
    new_seeds = 0
    for match_id, avg_rank in sampled:
        if spent >= budget:
            break
        already = conn.execute(
            "SELECT 1 FROM match_meta WHERE match_id = ?", (match_id,)
        ).fetchone()
        if already:
            continue
        try:
            match = client.match(match_id)
        except QuotaExhausted:
            log.warning("квота исчерпана, собрано %d кандидатов", new_seeds)
            break
        except OpenDotaError as exc:
            log.warning("матч %s пропущен: %s", match_id, exc)
            continue
        spent += 1
        if not match or not match.get("players"):
            continue
        accounts = ingest_match(conn, match)
        added = db.add_seeds(
            conn, [(acc, "public_sample", avg_rank) for acc in accounts]
        )
        new_seeds += len(accounts)
        conn.commit()
        if spent % 25 == 0:
            log.info(
                "выгружено %d матчей, кандидатов накоплено %d, бюджет %d",
                spent,
                conn.execute("SELECT count(*) FROM players").fetchone()[0],
                client.budget_left(),
            )
    return new_seeds


def expand_via_peers(
    client: OpenDotaClient,
    conn: sqlite3.Connection,
    limit_players: int = 30,
    min_games_together: int = 20,
) -> int:
    """Расширение выборки через сокомандников.

    Один вызов даёт десятки account_id, то есть на порядок дешевле выгрузки
    составов. Взамен выборка смещается в сторону тех, кто играет в пати, поэтому
    источник помечается отдельно и используется только как дополнение.
    """
    rows = conn.execute(
        """SELECT account_id FROM players
           WHERE status = 'fetched' ORDER BY random() LIMIT ?""",
        (limit_players,),
    ).fetchall()
    added = 0
    for row in rows:
        try:
            peers = client.player_peers(row["account_id"])
        except QuotaExhausted:
            break
        except OpenDotaError:
            continue
        if not peers:
            continue
        candidates = [
            (int(p["account_id"]), "peers", None)
            for p in peers
            if p.get("account_id") and (p.get("with_games") or 0) >= min_games_together
        ]
        db.add_seeds(conn, candidates)
        added += len(candidates)
    return added
