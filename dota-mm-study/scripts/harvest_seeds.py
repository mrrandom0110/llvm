"""Этап 2: формирование выборки игроков.

Случайные ranked-матчи, стратифицированные по брекетам, разбираются на
участников. Те же вызовы сохраняют составы, которые понадобятся тесту C.
"""

from __future__ import annotations

import argparse
import logging

from dota_study import db, seeds
from dota_study.api import OpenDotaClient

log = logging.getLogger("seeds")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-bracket", type=int, default=40)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--budget", type=int, default=220, help="вызовов на выгрузку матчей")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    client = OpenDotaClient()
    conn = db.connect()

    sampled = seeds.sample_match_ids(
        client, per_bracket=args.per_bracket, windows=args.windows
    )
    log.info("выбрано %d матчей-кандидатов", len(sampled))

    seeds.harvest_from_matches(client, conn, sampled, budget=args.budget)

    stats = db.counts(conn)
    by_source = conn.execute(
        "SELECT source, count(*) FROM players GROUP BY source"
    ).fetchall()
    log.info("итого кандидатов: %d", stats["players"])
    for source, count in by_source:
        log.info("  источник %-14s %d", source, count)
    log.info(
        "составов сохранено: %d матчей, %d строк; вызовов API: %d; бюджет %d",
        stats["match_meta"],
        stats["roster"],
        client.stats["requests"],
        client.budget_left(),
    )
    client.close()


if __name__ == "__main__":
    main()
