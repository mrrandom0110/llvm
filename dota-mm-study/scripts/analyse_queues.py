"""Проверка легенды о win queue / lose queue.

Если поиск идёт в разных пулах по знаку серии, игроки одного матча должны
чаще иметь одинаковый знак серии, чем при подборе только по рейтингу.
"""

from __future__ import annotations

import argparse
import json
import logging

from dota_study import db, features
from dota_study.config import DATA_DIR
from dota_study.plotting import ACCENT, MUTED, NEUTRAL, save
from dota_study.stats import queues

log = logging.getLogger("queues")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=300)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    conn = db.connect()
    db.clear_findings(conn, "E_queues")

    sample = features.analysis_sample(features.build_features(conn))
    rows = queues.shared_match_rows(sample)
    log.info(
        "матчей с ≥2 игроками выборки: %s, строк %s",
        f"{rows['match_id'].nunique():,}",
        f"{len(rows):,}",
    )

    answer = queues.permutation_null(rows, n_permutations=args.permutations)
    _, pairs = queues.prepare(rows)
    ally = pairs[pairs["same_team"]]
    enemy = pairs[~pairs["same_team"]]
    ally_duos = ally.groupby(["account_id_a", "account_id_b"]).size()
    extra = {
        "n_ally_matches": int(ally["match_id"].nunique()),
        "n_enemy_matches": int(enemy["match_id"].nunique()),
        "n_ally_duos": int(len(ally_duos)),
        "mean_ally_coplays": float(ally_duos.mean()) if len(ally_duos) else 0.0,
    }
    _log(answer)
    log.info(
        "союзники: %s матчей, %s уникальных пар, в среднем %.1f совместных матчей",
        f"{extra['n_ally_matches']:,}",
        f"{extra['n_ally_duos']:,}",
        extra["mean_ally_coplays"],
    )
    log.info("соперники: %s уникальных матчей", f"{extra['n_enemy_matches']:,}")
    _record(conn, answer)

    results = {
        "n_matches": answer.n_matches,
        "n_pairs": answer.n_pairs,
        "n_ally": answer.n_ally,
        "n_enemy": answer.n_enemy,
        **extra,
        "permutations": answer.n_permutations,
        "observed": answer.observed,
        "null_mean": answer.null_mean,
        "null_lo": answer.null_lo,
        "null_hi": answer.null_hi,
        "excess": {k: answer.excess(k) for k in answer.observed},
        "by_streak": answer.by_streak,
    }
    (DATA_DIR / "queues.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=float)
    )
    _plot(answer)
    log.info("готово")


def _log(answer: queues.QueueAnswer) -> None:
    log.info(
        "пар %s (союзники %s, соперники %s), перестановок %d",
        f"{answer.n_pairs:,}",
        f"{answer.n_ally:,}",
        f"{answer.n_enemy:,}",
        answer.n_permutations,
    )
    for key, title in (
        ("same_sign", "все пары"),
        ("same_sign_ally", "союзники"),
        ("same_sign_enemy", "соперники"),
        ("corr_ally", "корреляция с союзниками"),
        ("corr_enemy", "корреляция с соперниками"),
    ):
        log.info(
            "  %-28s факт %.4f | нуль %.4f [%.4f, %.4f] | сверх %+.4f",
            title,
            answer.observed[key],
            answer.null_mean[key],
            answer.null_lo[key],
            answer.null_hi[key],
            answer.excess(key),
        )


def _record(conn, answer: queues.QueueAnswer) -> None:
    for key, note in (
        ("same_sign_ally", "доля пар союзников с тем же знаком серии"),
        ("same_sign_enemy", "доля пар соперников с тем же знаком серии"),
        ("corr_enemy", "корреляция серии с серией соперника; ноль — один пул"),
    ):
        db.record_finding(
            conn,
            "E_queues",
            key,
            float(answer.observed[key]),
            float(answer.null_lo[key]),
            float(answer.null_hi[key]),
            n=answer.n_ally if "ally" in key else answer.n_enemy,
            note=f"{note}; сверх нуля {answer.excess(key):+.4f}",
        )


def _plot(answer: queues.QueueAnswer) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    labels = ["союзники", "соперники"]
    keys = ["same_sign_ally", "same_sign_enemy"]
    obs = [answer.observed[k] for k in keys]
    null = [answer.null_mean[k] for k in keys]
    lo = [answer.null_lo[k] for k in keys]
    hi = [answer.null_hi[k] for k in keys]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, obs, width=0.36, color=ACCENT, label="наблюдение")
    ax.bar(x + 0.18, null, width=0.36, color=NEUTRAL, alpha=0.7, label="только рейтинг")
    ax.errorbar(x + 0.18, null, yerr=[np.array(null) - lo, hi - np.array(null)],
                fmt="none", ecolor=MUTED, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("доля пар с тем же знаком серии")
    ax.set_title("Одинаковый знак серии в одном матче")
    ax.legend(fontsize=8)
    ymin = min(min(obs), min(lo)) - 0.03
    ymax = max(max(obs), max(hi)) + 0.03
    ax.set_ylim(ymin, ymax)

    ax = axes[1]
    curve = answer.by_streak
    ax.plot(curve["streak"], curve["ally"], "o-", color=ACCENT, label="средняя серия союзника")
    ax.plot(curve["streak"], curve["enemy"], "s--", color=NEUTRAL, label="средняя серия соперника")
    ax.axhline(0.0, color=MUTED, lw=1)
    ax.axvline(0.0, color=MUTED, lw=0.8)
    ax.set_xlabel("ваша серия перед матчем")
    ax.set_ylabel("серия партнёра")
    ax.set_title("С кем вас ставят при вашей серии")
    ax.legend(fontsize=8)

    save(fig, "queues.png")


if __name__ == "__main__":
    main()
