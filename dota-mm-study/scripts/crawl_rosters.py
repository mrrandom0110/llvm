"""Этап 6: выгрузка составов для теста C.

Стратегия расходования бюджета. Матчи, уже выгруженные на этапе формирования
выборки, дают наблюдения бесплатно: в каждом из них около восьми участников, и
для части из них история уже известна, а значит известна и их серия. Платный
добор идёт адресно — выбираются матчи с длинными сериями, поскольку именно
крайние ячейки определяют мощность теста.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from dota_study import db, features, seeds
from dota_study.api import OpenDotaClient, OpenDotaError, QuotaExhausted

log = logging.getLogger("rosters")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--min-streak", type=int, default=2)
    parser.add_argument("--recent-months", type=int, default=18)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    client = OpenDotaClient()
    conn = db.connect()

    full = features.build_features(conn)
    sample = features.analysis_sample(full)

    cutoff = int(sample["start_time"].max()) - args.recent_months * 30 * 86400
    recent = sample[sample["start_time"] >= cutoff].copy()
    log.info("матчей в свежем окне: %s", f"{len(recent):,}")

    have = set(
        row[0] for row in conn.execute("SELECT match_id FROM match_meta").fetchall()
    )
    free = recent[recent["match_id"].isin(have)]
    log.info(
        "наблюдений без затрат: %s матчей уже выгружено, из них с известной серией %s",
        f"{len(have):,}",
        f"{len(free):,}",
    )

    # Адресный добор: крайние серии информативнее, поэтому берём их с перевесом.
    pool = recent[recent["prev_streak"].abs() >= args.min_streak]
    pool = pool[~pool["match_id"].isin(have)]
    if pool.empty:
        log.warning("нет кандидатов для добора")
        return

    targets = _stratified_pick(pool, args.budget, args.min_streak)
    log.info("к выгрузке %d матчей, распределение серий: %s",
             len(targets), dict(targets["prev_streak"].clip(-4, 4).value_counts().sort_index()))

    fetched = 0
    for row in targets.itertuples(index=False):
        if fetched >= args.budget or client.budget_left() <= 5:
            break
        try:
            match = client.match(int(row.match_id))
        except QuotaExhausted:
            log.warning("квота исчерпана")
            break
        except OpenDotaError as exc:
            log.warning("матч %s пропущен: %s", row.match_id, exc)
            continue
        fetched += 1
        if not match or not match.get("players"):
            continue
        seeds.ingest_match(conn, match, focal_account=int(row.account_id))
        conn.commit()
        if fetched % 50 == 0:
            log.info("выгружено %d/%d, бюджет %d", fetched, len(targets), client.budget_left())

    stats = db.counts(conn)
    log.info(
        "составов в базе: %s матчей, %s строк; вызовов API %d",
        f"{stats['match_meta']:,}",
        f"{stats['roster']:,}",
        client.stats["requests"],
    )
    client.close()


def _stratified_pick(
    pool: pd.DataFrame, budget: int, min_streak: int, per_player: int = 6
) -> pd.DataFrame:
    """Отбор блоками по игрокам, а не по отдельным матчам.

    Ключевое требование теста C — сравнивать матчи одного и того же человека
    между собой. Если брать по одному матчу у каждого игрока, фиксированные
    эффекты оценить невозможно и весь эффект серии смешается с различиями между
    людьми. Поэтому у каждого отобранного игрока берётся несколько матчей с
    разными сериями: и после побед, и после поражений.
    """
    pool = pool.copy()
    pool["cell"] = pool["prev_streak"].clip(-4, 4)
    pool["side"] = np.sign(pool["prev_streak"])

    # Годятся только те, у кого есть матчи и после побед, и после поражений:
    # иначе внутри игрока не будет разброса объясняющей переменной.
    sides = pool.groupby("account_id")["side"].nunique()
    eligible = sides[sides >= 2].index
    pool = pool[pool["account_id"].isin(eligible)]
    if pool.empty:
        return pool

    n_players = max(budget // per_player, 1)
    rng = np.random.default_rng(11)
    candidates = pool["account_id"].drop_duplicates()
    chosen = candidates.sample(min(n_players, len(candidates)), random_state=7)

    picks = []
    for account_id in chosen:
        subset = pool[pool["account_id"] == account_id]
        # Внутри игрока балансируем победные и проигрышные серии.
        half = max(per_player // 2, 1)
        for side in (1, -1):
            side_subset = subset[subset["side"] == side]
            if side_subset.empty:
                continue
            picks.append(
                side_subset.sample(
                    min(half, len(side_subset)), random_state=int(rng.integers(1e6))
                )
            )
    if not picks:
        return pool.head(0)
    out = pd.concat(picks).drop_duplicates("match_id")
    return out.sample(frac=1.0, random_state=5).head(budget)


if __name__ == "__main__":
    main()
