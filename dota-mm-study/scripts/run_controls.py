"""Этап 1: позитивные контроли на публичных матчах.

Проверяем, что конвейер воспроизводит известные факты о Dota 2. Пока это не
сделано, результаты основных тестов интерпретировать нельзя.
"""

from __future__ import annotations

import argparse
import json
import logging

from dota_study import db
from dota_study.api import OpenDotaClient
from dota_study.config import DATA_DIR
from dota_study.controls import bracket_label, collect_controls, wilson_interval
from dota_study.plotting import ACCENT, MUTED, NEUTRAL, save

log = logging.getLogger("controls")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--hero-windows", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    client = OpenDotaClient()
    conn = db.connect()

    res = collect_controls(client, days=args.days, hero_windows=args.hero_windows)
    if res.total.n == 0:
        raise SystemExit("не удалось собрать ни одного дня данных")

    lo, hi = wilson_interval(res.total.radiant_wins, res.total.n)
    log.info(
        "ИТОГО %s ranked-матчей за %d суток; Radiant %.4f [%.4f, %.4f]",
        f"{res.total.n:,}",
        res.days,
        res.total.winrate,
        lo,
        hi,
    )
    db.record_finding(
        conn,
        "E_controls",
        "radiant_winrate",
        res.total.winrate,
        lo,
        hi,
        res.total.n,
        f"публичные ranked-матчи за {res.days} суток",
    )

    # Перевес Radiant по брекетам: если он есть везде, это не артефакт выборки.
    brackets = sorted(b for b in res.by_bracket if b >= 0)
    for bracket in brackets:
        tally = res.by_bracket[bracket]
        blo, bhi = wilson_interval(tally.radiant_wins, tally.n)
        log.info(
            "  %-10s n=%9s Radiant %.4f [%.4f, %.4f]",
            bracket_label(bracket),
            f"{tally.n:,}",
            tally.winrate,
            blo,
            bhi,
        )
        db.record_finding(
            conn,
            "E_controls",
            f"radiant_winrate_{bracket_label(bracket)}",
            tally.winrate,
            blo,
            bhi,
            tally.n,
        )

    for bucket in ("short", "medium", "long"):
        tally = res.by_duration.get(bucket)
        if tally and tally.n:
            dlo, dhi = wilson_interval(tally.radiant_wins, tally.n)
            db.record_finding(
                conn,
                "E_controls",
                f"radiant_winrate_duration_{bucket}",
                tally.winrate,
                dlo,
                dhi,
                tally.n,
            )
            log.info(
                "  длительность %-6s n=%9s Radiant %.4f",
                bucket,
                f"{tally.n:,}",
                tally.winrate,
            )

    # Разброс винрейтов героев: пайплайн должен видеть, что герои не равны.
    heroes = {h["id"]: h["localized_name"] for h in client.heroes()}
    hero_rows = [
        (heroes.get(hid, str(hid)), n, wins, wins / n)
        for hid, (n, wins) in res.hero_stats.items()
        if n >= 2000
    ]
    hero_rows.sort(key=lambda r: r[3])
    if hero_rows:
        spread = hero_rows[-1][3] - hero_rows[0][3]
        log.info(
            "герои: %d шт., разброс винрейта %.1f п.п. (%s %.3f ... %s %.3f)",
            len(hero_rows),
            spread * 100,
            hero_rows[0][0],
            hero_rows[0][3],
            hero_rows[-1][0],
            hero_rows[-1][3],
        )
        db.record_finding(
            conn,
            "E_controls",
            "hero_winrate_spread",
            spread,
            n=sum(r[1] for r in hero_rows),
            note=f"{hero_rows[0][0]} {hero_rows[0][3]:.3f} .. {hero_rows[-1][0]} {hero_rows[-1][3]:.3f}",
        )

    _plot(res, hero_rows, brackets)

    payload = {
        "days": res.days,
        "total_matches": res.total.n,
        "radiant_winrate": res.total.winrate,
        "radiant_ci": [lo, hi],
        "by_bracket": {
            bracket_label(b): {
                "n": res.by_bracket[b].n,
                "winrate": res.by_bracket[b].winrate,
            }
            for b in brackets
        },
        "bracket_share": {
            bracket_label(b): res.by_bracket[b].n / res.total.n for b in brackets
        },
        "hero_winrates": {name: {"n": n, "winrate": wr} for name, n, _, wr in hero_rows},
    }
    (DATA_DIR / "controls.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    log.info("готово; вызовов API израсходовано: %d", client.stats["requests"])
    client.close()


def _plot(res, hero_rows, brackets) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    ax = axes[0]
    labels = [bracket_label(b) for b in brackets]
    values = [res.by_bracket[b].winrate for b in brackets]
    errs = [
        [
            res.by_bracket[b].winrate - wilson_interval(res.by_bracket[b].radiant_wins, res.by_bracket[b].n)[0]
            for b in brackets
        ],
        [
            wilson_interval(res.by_bracket[b].radiant_wins, res.by_bracket[b].n)[1] - res.by_bracket[b].winrate
            for b in brackets
        ],
    ]
    ax.errorbar(range(len(brackets)), values, yerr=errs, fmt="o", color=ACCENT, capsize=3)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_xticks(range(len(brackets)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("винрейт Radiant")
    ax.set_title(f"Перевес Radiant по брекетам\n{res.total.n:,} матчей, 99% ДИ")

    ax = axes[1]
    shares = [res.by_bracket[b].n / res.total.n for b in brackets]
    ax.bar(range(len(brackets)), shares, color=NEUTRAL, alpha=0.8)
    ax.set_xticks(range(len(brackets)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("доля матчей")
    ax.set_title("Население брекетов")

    ax = axes[2]
    if hero_rows:
        wrs = np.array([r[3] for r in hero_rows])
        ax.hist(wrs, bins=24, color=NEUTRAL, alpha=0.8)
        ax.axvline(0.5, color=MUTED, ls="--", lw=1)
        ax.set_xlabel("винрейт героя")
        ax.set_ylabel("героев")
        ax.set_title(f"Винрейты героев\nразброс {100 * (wrs.max() - wrs.min()):.1f} п.п.")

    save(fig, "controls.png")


if __name__ == "__main__":
    main()
