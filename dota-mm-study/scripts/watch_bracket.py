"""Набор и разбор игроков полосы 4600–5000 MMR.

Этапы: матчи Divine → профили (MMR/медаль) → истории → слабые/сильные
и сравнение активности. Всё лежит в отдельных таблицах, основная выборка
исследования не меняется.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import pandas as pd

from dota_study import db, seeds
from dota_study.api import OpenDotaClient, OpenDotaError, QuotaExhausted, RateLimited
from dota_study.config import DATA_DIR
from dota_study.crawl import HISTORY_FIELDS, _rows_for_player
from dota_study.plotting import ACCENT, MUTED, NEUTRAL, save
from dota_study.stats import bracket

log = logging.getLogger("bracket")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harvest-matches", type=int, default=70)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--crawl", type=int, default=160)
    parser.add_argument("--history-limit", type=int, default=800)
    parser.add_argument("--skip-harvest", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-crawl", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    client = OpenDotaClient()
    conn = db.connect()

    if not args.skip_harvest:
        _harvest(client, conn, args)
    _import_existing_divine(conn)
    if not args.skip_probe:
        _probe(client, conn)
    _recompute_in_band(conn)
    if not args.skip_crawl:
        _crawl(client, conn, args)
    results = _analyse(conn)
    (DATA_DIR / "bracket_activity.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=float)
    )
    _plot(results)
    _log_plain(results)
    client.close()
    log.info("готово")


def _harvest(client: OpenDotaClient, conn, args) -> None:
    sampled = seeds.sample_divine_match_ids(
        client, per_window=max(args.harvest_matches // args.windows, 8), windows=args.windows
    )
    log.info("матчей Divine в выборке: %d", len(sampled))
    spent = 0
    for match_id, avg_rank in sampled:
        if spent >= args.harvest_matches:
            break
        try:
            match = client.match(match_id)
        except QuotaExhausted:
            log.warning("квота на составах исчерпана")
            break
        except RateLimited:
            log.warning("429 на составе, пауза 45 с")
            time.sleep(45)
            continue
        except OpenDotaError as exc:
            log.warning("матч %s: %s", match_id, exc)
            continue
        spent += 1
        if not match:
            continue
        for player in match.get("players") or []:
            acc = player.get("account_id")
            if not acc:
                continue
            tier = player.get("rank_tier")
            in_band = int(bracket.in_mmr_band(None, tier)) if tier is not None else None
            conn.execute(
                """INSERT OR IGNORE INTO bracket_watch
                   (account_id, source, seed_rank, rank_tier, in_band, status)
                   VALUES (?,?,?,?,?, 'seen')""",
                (int(acc), "divine_match", avg_rank, tier, in_band),
            )
        conn.commit()
        if spent % 20 == 0:
            n = conn.execute("SELECT count(*) FROM bracket_watch").fetchone()[0]
            log.info("составов %d, аккаунтов %d, бюджет %d", spent, n, client.budget_left())
    log.info(
        "в наблюдении %d аккаунтов",
        conn.execute("SELECT count(*) FROM bracket_watch").fetchone()[0],
    )


def _probe(client: OpenDotaClient, conn) -> None:
    rows = conn.execute(
        """SELECT account_id, rank_tier FROM bracket_watch
           WHERE computed_mmr IS NULL AND status IN ('seen', 'fetched')"""
    ).fetchall()
    log.info("профили без MMR: %d", len(rows))
    for i, row in enumerate(rows, 1):
        acc = row["account_id"]
        try:
            info = client.player(acc)
        except QuotaExhausted:
            break
        except RateLimited:
            log.warning("429 на профиле, пауза 45 с")
            time.sleep(45)
            continue
        except OpenDotaError as exc:
            log.warning("профиль %s: %s", acc, exc)
            continue
        if not info:
            conn.execute(
                "UPDATE bracket_watch SET status='private', fetched_at=? WHERE account_id=?",
                (int(time.time()), acc),
            )
            conn.commit()
            continue
        mmr, tier = bracket.extract_mmr(info)
        if tier is None:
            tier = row["rank_tier"]
        in_band = int(bracket.in_mmr_band(mmr, tier))
        conn.execute(
            """UPDATE bracket_watch
               SET computed_mmr=?, rank_tier=?, in_band=?, fetched_at=?
               WHERE account_id=?""",
            (mmr, tier, in_band, int(time.time()), acc),
        )
        conn.commit()
        if i % 40 == 0:
            log.info("проверено профилей %d / %d", i, len(rows))
    inside = conn.execute(
        "SELECT count(*) FROM bracket_watch WHERE in_band = 1"
    ).fetchone()[0]
    log.info("в полосе 4600–5000 / Divine 4–9: %d", inside)


def _crawl(client: OpenDotaClient, conn, args) -> None:
    pending = conn.execute(
        """SELECT account_id FROM bracket_watch
           WHERE in_band = 1 AND status IN ('seen')
           ORDER BY random() LIMIT ?""",
        (args.crawl,),
    ).fetchall()
    log.info("выгрузка историй: %d игроков", len(pending))
    ok = 0
    for row in pending:
        acc = row["account_id"]
        try:
            history = client.player_matches(
                acc, limit=args.history_limit, project=list(HISTORY_FIELDS), significant=0
            )
        except QuotaExhausted:
            break
        except RateLimited:
            log.warning("429, пауза 45 с")
            time.sleep(45)
            continue
        except OpenDotaError as exc:
            log.warning("история %s: %s", acc, exc)
            continue
        if history is None:
            conn.execute(
                "UPDATE bracket_watch SET status='private' WHERE account_id=?", (acc,)
            )
            conn.commit()
            continue
        rows = _rows_for_player(acc, history)
        if not rows:
            conn.execute(
                "UPDATE bracket_watch SET status='empty' WHERE account_id=?", (acc,)
            )
            conn.commit()
            continue
        db.insert_bracket_matches(conn, rows)
        conn.execute(
            "UPDATE bracket_watch SET status='fetched', n_matches=?, fetched_at=? WHERE account_id=?",
            (len(rows), int(time.time()), acc),
        )
        conn.commit()
        ok += 1
        if ok % 20 == 0:
            log.info("историй сохранено %d, бюджет %d", ok, client.budget_left())
    log.info("новых историй: %d", ok)


def _recompute_in_band(conn) -> None:
    """Пересчитать полосу по уже известным MMR и медалям, без новых запросов."""
    rows = conn.execute(
        "SELECT account_id, computed_mmr, rank_tier FROM bracket_watch"
    ).fetchall()
    for row in rows:
        flag = int(bracket.in_mmr_band(row["computed_mmr"], row["rank_tier"]))
        conn.execute(
            "UPDATE bracket_watch SET in_band=? WHERE account_id=?",
            (flag, row["account_id"]),
        )
    conn.commit()
    inside = conn.execute(
        "SELECT count(*) FROM bracket_watch WHERE in_band = 1"
    ).fetchone()[0]
    log.info("после пересчёта в полосе: %d", inside)


def _import_existing_divine(conn) -> None:
    """Игроки основной выборки, которые уже сидят в этой полосе по рангу лобби."""
    existing = pd.read_sql_query(
        """SELECT account_id, avg(average_rank) AS r
           FROM player_matches
           WHERE lobby_type = 7 AND average_rank IS NOT NULL
           GROUP BY account_id
           HAVING r >= 70 AND r < 80""",
        conn,
    )
    if existing.empty:
        return
    copied = 0
    for acc in existing["account_id"]:
        has = conn.execute(
            "SELECT 1 FROM bracket_matches WHERE account_id = ? LIMIT 1", (int(acc),)
        ).fetchone()
        if has:
            continue
        rows = conn.execute(
            "SELECT * FROM player_matches WHERE account_id = ?", (int(acc),)
        ).fetchall()
        db.insert_bracket_matches(conn, [dict(r) for r in rows])
        conn.execute(
            """INSERT OR IGNORE INTO bracket_watch
               (account_id, source, seed_rank, status, n_matches)
               VALUES (?,?,?, 'fetched', ?)""",
            (
                int(acc),
                "main_sample",
                float(existing.loc[existing["account_id"] == acc, "r"].iloc[0]),
                len(rows),
            ),
        )
        copied += 1
    conn.commit()
    if copied:
        log.info("добавлено из основной выборки: %d", copied)


def _analyse(conn) -> dict:
    db.clear_findings(conn, "G_bracket")
    watch = pd.read_sql_query("SELECT * FROM bracket_watch WHERE in_band = 1", conn)
    matches = pd.read_sql_query("SELECT * FROM bracket_matches", conn)
    if not watch.empty and not matches.empty:
        matches = matches[matches["account_id"].isin(set(watch["account_id"]))]
    log.info("матчей в разборе %s, игроков в полосе %s", f"{len(matches):,}", f"{len(watch):,}")
    if matches.empty:
        return {"n_matches": 0, "n_in_band": int(len(watch))}

    skill = bracket.player_skill(matches)
    if skill.empty:
        return {"n_matches": int(len(matches)), "n_enough": 0}
    activity = bracket.player_activity(matches, skill)
    summary = bracket.group_summary(activity)
    hours = bracket.hour_hist(matches, skill)
    counts = skill["group"].value_counts().to_dict()
    log.info("группы: %s", counts)

    if "слабый" in summary and "сильный" in summary:
        diff = summary["разница_сильный_минус_слабый"]
        db.record_finding(
            conn, "G_bracket", "per_week_diff", diff.get("per_week"),
            n=summary["сильный"]["n"] + summary["слабый"]["n"],
            note="каток в неделю: сильный минус слабый",
        )
        db.record_finding(
            conn, "G_bracket", "night_diff", diff.get("night_msk"),
            n=summary["сильный"]["n"] + summary["слабый"]["n"],
            note="доля ночных каток (МСК 0–5): сильный минус слабый",
        )

    statuses = {
        row["status"]: row["n"]
        for row in conn.execute(
            "SELECT status, count(*) AS n FROM bracket_watch GROUP BY status"
        )
    }
    return {
        "n_matches": int(len(matches)),
        "n_watch": int(conn.execute("SELECT count(*) FROM bracket_watch").fetchone()[0]),
        "n_in_band": int(len(watch)),
        "n_enough": int(len(skill)),
        "n_private": int(statuses.get("private", 0)),
        "statuses": {str(k): int(v) for k, v in statuses.items()},
        "counts": {str(k): int(v) for k, v in counts.items()},
        "summary": summary,
        "hours": hours,
        "mmr_lo": bracket.MMR_LO,
        "mmr_hi": bracket.MMR_HI,
        "recent_days": bracket.RECENT_DAYS,
    }


def _log_plain(results: dict) -> None:
    summary = results.get("summary") or {}
    for name in ("слабый", "сильный"):
        info = summary.get(name)
        if not info:
            continue
        log.info(
            "%s n=%d | %.1f каток/нед | вечер %.1f каток | ночь МСК %.0f%% | прайм %.0f%% | винрейт %.3f",
            name,
            info["n"],
            info.get("per_week", float("nan")),
            info.get("session_len", float("nan")),
            100 * info.get("night_msk", float("nan")),
            100 * info.get("prime_msk", float("nan")),
            info.get("winrate", float("nan")),
        )


def _plot(results: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    hours = results.get("hours") or {}
    summary = results.get("summary") or {}
    if not hours and not summary:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    ax = axes[0]
    colors = {"слабый": MUTED, "сильный": ACCENT, "середина": NEUTRAL}
    for name, series in hours.items():
        if name == "середина":
            continue
        ax.plot(series["hour"], 100 * np.array(series["share"]), "o-",
                color=colors.get(name, NEUTRAL), label=name, ms=3)
    ax.set_xlabel("час, Москва")
    ax.set_ylabel("доля каток, %")
    ax.set_title("Когда заходят в игру")
    ax.legend()

    ax = axes[1]
    labels = ["каток в неделю", "длина вечера", "ночь 0–5 МСК, %", "прайм 18–22, %"]
    keys = ["per_week", "session_len", "night_msk", "prime_msk"]
    weak = summary.get("слабый") or {}
    strong = summary.get("сильный") or {}
    wvals = [
        weak.get("per_week", 0),
        weak.get("session_len", 0),
        100 * weak.get("night_msk", 0),
        100 * weak.get("prime_msk", 0),
    ]
    svals = [
        strong.get("per_week", 0),
        strong.get("session_len", 0),
        100 * strong.get("night_msk", 0),
        100 * strong.get("prime_msk", 0),
    ]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, wvals, 0.36, color=MUTED, label="слабый в этом пуле")
    ax.bar(x + 0.18, svals, 0.36, color=ACCENT, label="сильный в этом пуле")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Как часто и какими вечерами")
    ax.legend(fontsize=8)
    save(fig, "bracket_activity.png")


if __name__ == "__main__":
    main()
