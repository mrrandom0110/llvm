"""Поиск зависимостей: что вообще предсказывает победу в выборке.

У каждого признака две цифры. Сырая — «как выглядит, если не знать игрока».
Внутриигровая — тот же человек в двух состояниях. Расхождение между ними —
это отбор, а не механизм.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from dota_study import db, features
from dota_study.config import DATA_DIR
from dota_study.plotting import ACCENT, MUTED, NEUTRAL, save
from dota_study.stats import scan
from dota_study.stats.streaks import fixed_effects_lpm

log = logging.getLogger("scan")

LABELS = {
    "is_radiant": "сторона Radiant",
    "in_party": "игра в пати",
    "same_hero": "тот же герой, что в прошлом матче",
    "comfort_hero": "герой из последних пяти матчей",
    "is_weekend": "выходные",
    "first_of_session": "первый матч вечера",
    "late_session": "шестой матч вечера и дальше",
    "short_pause": "пауза ≤ 1 ч внутри вечера",
    "long_break": "перерыв ≥ 24 ч",
    "after_wins": "серия ≥ 2 побед",
    "after_losses": "серия ≥ 2 поражений",
    "lobby_above": "лобби выше своего обычного ранга",
    "lobby_below": "лобби ниже своего обычного ранга",
    "off_hours": "необычный час (сдвиг ≥ 6 ч от медианы)",
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    conn = db.connect()
    db.clear_findings(conn, "F_scan")

    sample = features.analysis_sample(features.build_features(conn))
    df = scan.enrich(sample)
    log.info("выборка %s матчей", f"{len(df):,}")

    binaries = {}
    for col, title in LABELS.items():
        subset = df
        if col == "in_party":
            year = pd.to_datetime(df["start_time"], unit="s", utc=True).dt.year
            coverage = df.groupby(year)["party_size"].apply(lambda s: s.notna().mean())
            good = coverage[coverage >= 0.5].index
            subset = df[year.isin(good)]
            log.info("пати: годы с покрытием ≥50%%: %s", list(map(int, good)))
        result = scan.binary_effect(subset, col)
        if result is None:
            log.info("  %-40s недостаточно данных", title)
            continue
        binaries[col] = result
        log.info(
            "  %-40s сырой %+6.2f п.п. | внутри %+6.2f [%+.2f, %+.2f] | n=%s",
            title,
            100 * result["raw_diff"],
            100 * result["within"],
            100 * result["lo"],
            100 * result["hi"],
            f"{result['n']:,}",
        )
        db.record_finding(
            conn,
            "F_scan",
            col,
            result["within"],
            result["lo"],
            result["hi"],
            result["n"],
            f"{title}; сырой разрыв {result['raw_diff']:+.4f}",
        )

    df["session_bin"] = scan.session_bins(df).astype(str)
    sessions = scan.bins_within(df, "session_bin", "1-й матч")
    if sessions:
        log.info("позиция в сессии (внутри, от первого матча):")
        for level, wr, n, w in zip(
            sessions["levels"], sessions["raw"], sessions["n"], sessions["within"]
        ):
            log.info("  %-12s сырой %.4f  внутри %+.4f  n=%s", level, wr, w, f"{n:,}")

    continue_ = scan.session_continue(df)
    if continue_:
        log.info(
            "ещё один матч в тот же вечер: после победы %.3f, после поражения %.3f (разница %+.3f)",
            continue_["after_win"],
            continue_["after_loss"],
            continue_["diff"],
        )

    # Линейный наклон позиции в сессии — компактная сводка «усталости».
    fe_pos = fixed_effects_lpm(
        df["win"].to_numpy(dtype=float),
        np.column_stack([df["session_pos"].to_numpy(dtype=float)]),
        df["account_id"].to_numpy(),
        ["session_pos"],
    )
    pos = {
        "within": float(fe_pos.coef[0]),
        "se": float(fe_pos.se[0]),
        "lo": float(fe_pos.ci("session_pos")[0]),
        "hi": float(fe_pos.ci("session_pos")[1]),
        "n": int(fe_pos.n_obs),
    }
    log.info(
        "наклон по номеру матча в вечере: %+.4f за матч [%+.4f, %+.4f]",
        pos["within"],
        pos["lo"],
        pos["hi"],
    )

    results = {
        "binaries": binaries,
        "session": sessions,
        "continue_session": continue_,
        "session_pos_slope": pos,
        "labels": LABELS,
    }
    (DATA_DIR / "dependencies.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=float)
    )
    _plot(results)
    log.info("готово")


def _plot(results: dict) -> None:
    import matplotlib.pyplot as plt

    items = []
    for key, info in results["binaries"].items():
        items.append((results["labels"][key], info["within"], info["lo"], info["hi"]))
    items.sort(key=lambda row: row[1])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), gridspec_kw={"width_ratios": [1.3, 1]})

    ax = axes[0]
    names = [x[0] for x in items]
    vals = np.array([x[1] for x in items]) * 100
    lo = np.array([x[2] for x in items]) * 100
    hi = np.array([x[3] for x in items]) * 100
    y = np.arange(len(names))
    ax.barh(y, vals, color=ACCENT, alpha=0.85)
    ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="none", ecolor=NEUTRAL, capsize=3)
    ax.axvline(0.0, color=MUTED, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("эффект внутри игрока, п.п.")
    ax.set_title("Что меняет шанс победы\nу одного и того же человека")

    ax = axes[1]
    sess = results.get("session") or {}
    if sess.get("levels"):
        ax.plot(sess["levels"], 100 * np.array(sess["within"]), "o-", color=ACCENT)
        ax.axhline(0.0, color=MUTED, lw=1)
        ax.set_ylabel("к первому матчу вечера, п.п.")
        ax.set_title("Позиция в игровом вечере")
        ax.tick_params(axis="x", rotation=20)

    save(fig, "dependencies.png")


if __name__ == "__main__":
    main()
