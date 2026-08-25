"""Этап 3: массовая выгрузка историй игроков."""

from __future__ import annotations

import argparse
import logging

from dota_study import crawl, db
from dota_study.api import OpenDotaClient
from dota_study.config import STUDY_WINDOW_START

log = logging.getLogger("crawl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=2000)
    parser.add_argument("--history-limit", type=int, default=5000)
    parser.add_argument(
        "--reserve",
        type=int,
        default=700,
        help="сколько вызовов оставить следующим этапам",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    client = OpenDotaClient()
    conn = db.connect()

    stats = crawl.crawl_players(
        client,
        conn,
        limit_players=args.players,
        history_limit=args.history_limit,
        reserve=args.reserve,
    )

    log.info(
        "выгружено игроков: %d (приватных %d, пустых %d)",
        stats["fetched"],
        stats["private"],
        stats["empty"],
    )

    totals = conn.execute(
        f"""SELECT count(*) AS all_rows,
                   count(*) FILTER (WHERE lobby_type = 7) AS ranked,
                   count(*) FILTER (WHERE lobby_type = 7
                                      AND start_time >= {STUDY_WINDOW_START}
                                      AND average_rank IS NOT NULL) AS in_window,
                   count(DISTINCT account_id) AS players
            FROM player_matches"""
    ).fetchone()
    log.info(
        "в базе: %s строк, %s ranked, %s в окне исследования, игроков %s",
        f"{totals['all_rows']:,}",
        f"{totals['ranked']:,}",
        f"{totals['in_window']:,}",
        f"{totals['players']:,}",
    )
    log.info("вызовов API: %d, бюджет %d", client.stats["requests"], client.budget_left())
    client.close()


if __name__ == "__main__":
    main()
