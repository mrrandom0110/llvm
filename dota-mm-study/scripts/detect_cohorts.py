"""Этап 4: выделение смурфов, слабых игроков и «жителей рейтинга»."""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from dota_study import db, features, smurf
from dota_study.plotting import ACCENT, MUTED, NEUTRAL, save

log = logging.getLogger("cohorts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-matches", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    conn = db.connect()
    db.clear_findings(conn, "D_cohorts")

    log.info("построение признаков...")
    full = features.build_features(conn)
    sample = features.analysis_sample(full)
    log.info("матчей в окне исследования: %s", f"{len(sample):,}")

    history_meta = pd.read_sql_query(
        """SELECT account_id,
                  first_match AS first_match_all,
                  last_match  AS last_match_all,
                  n_matches   AS n_matches_all
           FROM players WHERE status = 'fetched'""",
        conn,
    )

    profiles = smurf.build_profiles(sample, history_meta=history_meta)
    profiles = profiles[profiles["n_ranked"] >= args.min_matches].reset_index(drop=True)
    log.info("игроков с историей от %d матчей: %d", args.min_matches, len(profiles))

    scored = smurf.score_cohorts(profiles)
    validation = smurf.validate(scored)
    log.info(
        "валидация: заведомых смурфов %s, заведомых жителей %s, AUC %.3f (без возраста %.3f)",
        validation.get("n_smurf"),
        validation.get("n_resident"),
        validation.get("auc", float("nan")),
        validation.get("auc_no_age", float("nan")),
    )

    counts = scored["label"].value_counts().to_dict()
    log.info("когорты: %s", counts)

    for label in ("resident", "smurf", "weak"):
        subset = scored[scored["label"] == label]
        if subset.empty:
            continue
        log.info(
            "  %-9s n=%4d винрейт %.4f | подъём ранга %+.2f/мес | перформанс %+.2f | стаж %.0f мес",
            label,
            len(subset),
            subset["winrate"].mean(),
            subset["rank_slope"].mean(),
            subset["perf_mean"].mean(),
            subset["span_months"].mean(),
        )

    _persist(conn, scored)
    db.record_finding(
        conn,
        "D_cohorts",
        "smurf_auc",
        validation.get("auc"),
        n=validation.get("n"),
        note=(
            "AUC разделения заведомых смурфов и жителей рейтинга; "
            f"без возрастной компоненты {validation.get('auc_no_age', float('nan')):.3f}"
        ),
    )
    db.record_finding(
        conn, "D_cohorts", "smurf_share", counts.get("smurf", 0) / max(len(scored), 1),
        n=len(scored)
    )
    db.record_finding(
        conn, "D_cohorts", "weak_share", counts.get("weak", 0) / max(len(scored), 1),
        n=len(scored)
    )

    _plot(scored)
    log.info("готово")


def _persist(conn, scored: pd.DataFrame) -> None:
    rows = []
    for row in scored.itertuples(index=False):
        rows.append(
            (
                int(row.account_id),
                int(row.n_ranked),
                float(row.winrate),
                int(row.first_match) if pd.notna(row.first_match) else None,
                int(row.last_match) if pd.notna(row.last_match) else None,
                int(row.registration_est) if pd.notna(row.registration_est) else None,
                _f(row.rank_start),
                _f(row.rank_end),
                _f(row.rank_slope),
                _f(row.perf_z),
                _f(row.climb_z),
                _f(row.smurf_score),
                _f(row.weak_score),
                int(row.is_smurf),
                int(row.is_weak),
                _f(row.n_clusters),
                str(row.label),
            )
        )
    conn.executemany(
        """INSERT OR REPLACE INTO player_profile
           (account_id, n_ranked, winrate, first_seen, last_seen, account_age_est,
            rank_start, rank_end, rank_slope, perf_z, climb_z, smurf_score,
            weak_score, is_smurf, is_weak, region_switch, label)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()


def _f(value) -> float | None:
    return float(value) if pd.notna(value) else None


def _plot(scored: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    ax = axes[0]
    for label, color in (("resident", NEUTRAL), ("smurf", ACCENT), ("weak", MUTED)):
        subset = scored[scored["label"] == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["n_ranked"],
            subset["winrate"],
            s=8,
            alpha=0.5,
            color=color,
            label=f"{label} ({len(subset)})",
        )
    ax.set_xscale("log")
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("ranked-матчей в окне")
    ax.set_ylabel("винрейт")
    ax.set_title("Винрейт и объём истории")
    ax.legend(fontsize=7)

    ax = axes[1]
    valid = scored.dropna(subset=["rank_slope"])
    ax.scatter(
        valid["rank_slope"], valid["perf_mean"], s=8, alpha=0.4,
        c=np.where(valid["label"] == "smurf", ACCENT, NEUTRAL),
    )
    ax.set_xlabel("подъём ранга, единиц в месяц")
    ax.set_ylabel("перформанс относительно брекета")
    ax.set_title("Признаки смурфа")

    ax = axes[2]
    ages = scored["account_age_months"].dropna()
    ax.hist(ages, bins=40, color=NEUTRAL, alpha=0.85)
    ax.set_xlabel("оценка возраста аккаунта, месяцев")
    ax.set_ylabel("игроков")
    ax.set_title("Возраст аккаунтов по номеру")

    save(fig, "cohorts.png")


if __name__ == "__main__":
    main()
