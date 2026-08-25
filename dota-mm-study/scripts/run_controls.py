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
    db.clear_findings(conn, "E_controls")

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

    party = _party_control(conn)
    if party and not party.get("insufficient"):
        log.info(
            "пати против соло (годы %s, покрытие поля %.0f%%): %.4f против %.4f, "
            "разница %+.4f (99%% ДИ %+.4f, %+.4f)",
            party["years"],
            100 * party["coverage"],
            party["party_winrate"],
            party["solo_winrate"],
            party["difference"],
            party["lo"],
            party["hi"],
        )
        db.record_finding(
            conn,
            "E_controls",
            "party_advantage",
            party["difference"],
            party["lo"],
            party["hi"],
            party["n"],
            f"пати {party['party_winrate']:.4f} против соло {party['solo_winrate']:.4f}; "
            f"поле party_size заполнено у {100 * party['coverage']:.0f}% матчей",
        )
        payload_party = party
    else:
        payload_party = None
        log.warning("контроль по пати не рассчитан: поле party_size заполнено слишком редко")

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
        "party": payload_party,
    }
    (DATA_DIR / "controls.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    log.info("готово; вызовов API израсходовано: %d", client.stats["requests"])
    client.close()


def _party_control(conn) -> dict[str, float] | None:
    """Преимущество игры в пати над соло — ещё один заведомо существующий эффект.

    Считается по уже выгруженным историям, без обращений к API. Сравнение
    внутриигровое: у одних и тех же людей берутся их соло-матчи и их же матчи в
    пати, поэтому различия между игроками не подменяют эффект.
    """
    import numpy as np
    import pandas as pd

    from dota_study import features

    full = features.analysis_sample(features.build_features(conn))
    coverage = float(full["party_size"].notna().mean())

    # Поле заполняется неравномерно во времени, и в годы с низким покрытием
    # попадают не случайные матчи. Сравнивать можно только там, где данные есть
    # у большинства матчей, иначе эффект пати смешается с эффектом периода.
    year = pd.to_datetime(full["start_time"], unit="s").dt.year
    by_year = full.groupby(year)["party_size"].apply(lambda s: s.notna().mean())
    good_years = by_year[by_year >= 0.5].index
    df = full[year.isin(good_years)].dropna(subset=["party_size"])
    if len(df) < 10_000:
        return {"coverage": coverage, "insufficient": True}
    df = df.assign(in_party=(df["party_size"] > 1).astype(float))

    # Оба режима должны встречаться у игрока, иначе сравнивать не с чем.
    both = df.groupby("account_id")["in_party"].nunique()
    df = df[df["account_id"].isin(both[both == 2].index)]
    if len(df) < 10_000:
        return {"coverage": coverage, "insufficient": True}

    from dota_study.stats.streaks import fixed_effects_lpm

    fe = fixed_effects_lpm(
        df["win"].to_numpy(dtype=float),
        np.column_stack([df["in_party"].to_numpy(dtype=float)]),
        df["account_id"].to_numpy(),
        ["in_party"],
    )
    lo, hi = fe.ci("in_party")
    grouped = df.groupby("in_party")["win"].mean()
    return {
        "party_winrate": float(grouped.get(1.0, np.nan)),
        "solo_winrate": float(grouped.get(0.0, np.nan)),
        "difference": float(fe.coef[0]),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(len(df)),
        "coverage": coverage,
        "years": sorted(int(y) for y in good_years),
        "insufficient": False,
    }


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
