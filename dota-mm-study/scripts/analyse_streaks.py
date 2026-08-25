"""Отдельный разбор: «выиграл N подряд — каков шанс выиграть следующий матч».

У вопроса два разных ответа, и весь смысл этого этапа в том, чтобы их
разделить.

* Предсказательный: увидев человека на серии из N побед, чего ждать от его
  следующего матча. Здесь работает отбор — длинные серии чаще бывают у сильных.
* Причинный: меняет ли сама серия шансы **одного и того же** человека.

Разделение делается перестановочным тестом: порядок исходов внутри истории
игрока перемешивается, его собственный винрейт и длина истории сохраняются.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from dota_study import db, features
from dota_study.config import DATA_DIR
from dota_study.plotting import ACCENT, MUTED, NEUTRAL, save
from dota_study.stats import hotstreak

log = logging.getLogger("streaks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-streak", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=400)
    parser.add_argument("--min-games", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    conn = db.connect()
    db.clear_findings(conn, "B_hotstreak")

    sample = features.analysis_sample(features.build_features(conn))
    base_rate = float(sample["win"].mean())
    log.info(
        "выборка: %s матчей, %s игроков, общий винрейт %.4f",
        f"{len(sample):,}",
        f"{sample['account_id'].nunique():,}",
        base_rate,
    )

    log.info("=" * 78)
    log.info("Нулевой закон 1: перемешивание всей истории игрока")
    full = hotstreak.permutation_null(
        sample,
        max_streak=args.max_streak,
        n_permutations=args.permutations,
        min_games=args.min_games,
        label="случайный порядок при том же винрейте игрока",
    )
    _log_table(full, base_rate)

    log.info("=" * 78)
    log.info("Нулевой закон 2: перемешивание только внутри игровой сессии")
    within = hotstreak.permutation_null(
        sample,
        max_streak=args.max_streak,
        n_permutations=args.permutations,
        min_games=args.min_games,
        within="session_id",
        label="случайный порядок внутри вечера",
    )
    _log_excess(within)

    results = {
        "base_rate": base_rate,
        "n_matches": int(len(sample)),
        "n_players": int(sample["account_id"].nunique()),
        "full_shuffle": full.as_frame().to_dict("list"),
        "within_session": within.as_frame().to_dict("list"),
        "permutations": args.permutations,
    }

    log.info("=" * 78)
    log.info("Однороден ли эффект: разбивка по когортам и брекетам")
    results["heterogeneity"] = _heterogeneity(conn, sample, args)

    log.info("=" * 78)
    log.info("Внутри одного вечера или через паузу")
    results["session_split"] = _session_split(sample, args)

    _record(conn, full, base_rate)
    _record_session_split(conn, results["session_split"])
    (DATA_DIR / "streak_question.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=float)
    )
    _plot(full, within, base_rate, results)
    log.info("готово")


def _log_table(answer: hotstreak.StreakAnswer, base_rate: float) -> None:
    frame = hotstreak.conditional_table(answer, min_n=300)
    log.info(
        "%6s %10s %10s %10s %10s %10s",
        "серия",
        "матчей",
        "факт",
        "отбор",
        "нул.закон",
        "сверх нуля",
    )
    for row in frame.itertuples(index=False):
        log.info(
            "%+6d %10s %10.4f %+10.4f %10.4f %+10.4f",
            row.streak,
            f"{int(row.n):,}",
            row.observed,
            row.null_mean - base_rate,
            row.null_mean,
            row.excess,
        )


def _log_excess(answer: hotstreak.StreakAnswer) -> None:
    frame = hotstreak.conditional_table(answer, min_n=300)
    for row in frame.itertuples(index=False):
        if row.streak < 1:
            continue
        significant = "" if row.null_lo <= row.observed <= row.null_hi else "  <- вне интервала"
        log.info(
            "  серия %+d: факт %.4f, нулевой закон %.4f [%.4f, %.4f]%s",
            row.streak,
            row.observed,
            row.null_mean,
            row.null_lo,
            row.null_hi,
            significant,
        )


def _heterogeneity(conn, sample: pd.DataFrame, args) -> dict:
    """Одинаков ли эффект серии у разных групп игроков."""
    labels = pd.read_sql_query("SELECT account_id, label FROM player_profile", conn)
    out: dict[str, dict] = {}
    if not labels.empty:
        merged = sample.merge(labels, on="account_id", how="left")
        for label in ("resident", "smurf", "weak"):
            subset = merged[merged["label"] == label]
            if subset["account_id"].nunique() < 20:
                continue
            answer = hotstreak.permutation_null(
                subset,
                max_streak=4,
                n_permutations=max(args.permutations // 2, 100),
                min_games=args.min_games,
            )
            excess = _mean_excess(answer)
            out[label] = excess
            log.info(
                "  %-9s игроков %4d: превышение над нулевым законом при серии 2-4: %+.4f",
                label,
                subset["account_id"].nunique(),
                excess["mean_excess_wins"],
            )

    sample = sample.assign(bracket=(sample["average_rank"] // 10).astype("Int64"))
    for bracket, subset in sample.groupby("bracket"):
        if subset["account_id"].nunique() < 40:
            continue
        answer = hotstreak.permutation_null(
            subset,
            max_streak=4,
            n_permutations=max(args.permutations // 2, 100),
            min_games=args.min_games,
        )
        excess = _mean_excess(answer)
        out[f"bracket_{int(bracket)}"] = excess
        log.info(
            "  брекет %d, игроков %4d: превышение %+.4f",
            int(bracket),
            subset["account_id"].nunique(),
            excess["mean_excess_wins"],
        )
    return out


def _mean_excess(answer: hotstreak.StreakAnswer) -> dict:
    frame = answer.as_frame()
    wins = frame[(frame["streak"] >= 2) & (frame["n"] >= 200)]
    losses = frame[(frame["streak"] <= -2) & (frame["n"] >= 200)]
    return {
        "mean_excess_wins": float(wins["excess"].mean()) if len(wins) else float("nan"),
        "mean_excess_losses": float(losses["excess"].mean()) if len(losses) else float("nan"),
        "n": int(frame["n"].sum()),
    }


def _session_split(sample: pd.DataFrame, args) -> dict:
    """Серия, набранная за один вечер, против серии, разорванной паузой.

    Если зависимость исхода от серии — это про форму человека в конкретный
    вечер, она должна исчезать, когда серия прервана сном.
    """
    df = sample.sort_values(["account_id", "start_time"], kind="stable").copy()
    df["same_session"] = (
        df.groupby("account_id")["session_id"].diff().fillna(1) == 0
    )
    out = {}
    for flag, title in ((True, "серия внутри одного вечера"), (False, "первый матч вечера")):
        subset = df[df["same_session"] == flag]
        if len(subset) < 50_000:
            continue
        grouped = subset.groupby(subset["prev_streak"].clip(-4, 4))["win"].agg(["mean", "size"])
        grouped = grouped[grouped["size"] >= 500]
        wins = grouped.loc[grouped.index >= 2, "mean"]
        losses = grouped.loc[grouped.index <= -2, "mean"]
        out[title] = {
            "n": int(grouped["size"].sum()),
            "after_wins": float(wins.mean()) if len(wins) else float("nan"),
            "after_losses": float(losses.mean()) if len(losses) else float("nan"),
        }
        log.info(
            "  %-28s n=%9s | после побед %.4f | после поражений %.4f",
            title,
            f"{int(grouped['size'].sum()):,}",
            out[title]["after_wins"],
            out[title]["after_losses"],
        )
    return out


def _record(conn, answer: hotstreak.StreakAnswer, base_rate: float) -> None:
    frame = hotstreak.conditional_table(answer, min_n=300)
    for row in frame.itertuples(index=False):
        if row.streak < 1:
            continue
        db.record_finding(
            conn,
            "B_hotstreak",
            f"p_win_after_{int(row.streak)}_wins",
            float(row.observed),
            n=int(row.n),
            note=(
                f"отбор даёт {row.null_mean - base_rate:+.4f}, "
                f"сверх случайного порядка {row.excess:+.4f}"
            ),
        )


def _record_session_split(conn, split: dict) -> None:
    """Насколько перерыв между вечерами гасит эффект серии.

    Ключевая проверка механизма: алгоритм, следящий за серией, не заметил бы,
    что человек сходил спать. Форма человека — заметила бы.
    """
    for title, info in split.items():
        spread = info["after_wins"] - info["after_losses"]
        db.record_finding(
            conn,
            "B_hotstreak",
            f"spread_{'same_session' if 'вечера' in title and 'внутри' in title else 'after_break'}",
            float(spread),
            n=int(info["n"]),
            note=f"{title}: после побед {info['after_wins']:.4f}, после поражений {info['after_losses']:.4f}",
        )


def _plot(full, within, base_rate: float, results: dict) -> None:
    import matplotlib.pyplot as plt

    frame = hotstreak.conditional_table(full, min_n=300)
    wframe = hotstreak.conditional_table(within, min_n=300)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

    ax = axes[0]
    ax.fill_between(
        frame["streak"], frame["null_lo"], frame["null_hi"], color=MUTED, alpha=0.3,
        label="случайный порядок, 99%",
    )
    ax.plot(frame["streak"], frame["null_mean"], color=MUTED, lw=1.2)
    ax.plot(frame["streak"], frame["observed"], "o-", color=ACCENT, lw=1.4, ms=4,
            label="наблюдение")
    ax.axhline(base_rate, color=NEUTRAL, ls=":", lw=1, label="средний винрейт выборки")
    ax.set_xlabel("серия перед матчем")
    ax.set_ylabel("доля побед в следующем матче")
    ax.set_title("Шанс победить после серии\nи то же при случайном порядке")
    ax.legend(fontsize=7)

    ax = axes[1]
    selection = frame["null_mean"] - base_rate
    genuine = frame["excess"]
    ax.bar(frame["streak"] - 0.2, 100 * selection, width=0.4, color=NEUTRAL,
           label="отбор: у кого бывают такие серии")
    ax.bar(frame["streak"] + 0.2, 100 * genuine, width=0.4, color=ACCENT,
           label="сверх случайного порядка")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("серия перед матчем")
    ax.set_ylabel("вклад, п.п.")
    ax.set_title("Из чего складывается эффект серии")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.plot(wframe["streak"], 100 * wframe["excess"], "o-", color=ACCENT, ms=4,
            label="сверх перемешивания внутри вечера")
    ax.plot(frame["streak"], 100 * frame["excess"], "s--", color=NEUTRAL, ms=4,
            label="сверх перемешивания всей истории")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("серия перед матчем")
    ax.set_ylabel("превышение, п.п.")
    ax.set_title("Это просто удачный вечер?")
    ax.legend(fontsize=7)

    save(fig, "streak_question.png")


if __name__ == "__main__":
    main()
