"""Этап 5а: построение нулевой модели честного матчмейкинга.

Здесь решается главная методологическая трудность исследования. Честная
рейтинговая система — это не одна точка, а семейство: её предсказания зависят от
того, насколько подвижен навык игроков, каков шаг обновления рейтинга и как
широко окно подбора. Поэтому нулевая гипотеза формируется двумя способами.

1. Диапазон. Прогоняется сетка параметров и фиксируется, какие значения
   статистик честная система вообще способна выдать.
2. Точечное предсказание. Единственный свободный параметр — нестационарность
   навыка — подбирается так, чтобы совпал наблюдаемый разброс винрейтов. После
   этого наклон эффекта серии становится предсказанием модели, а не подгонкой.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from dota_study import db, features
from dota_study.config import DATA_DIR
from dota_study.plotting import ACCENT, MUTED, NEUTRAL, save
from dota_study.sim import calibrate as cal
from dota_study.sim.fair_mm import SimConfig
from dota_study.stats import dispersion

log = logging.getLogger("simulation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-games", type=int, default=200)
    parser.add_argument("--players", type=int, default=9000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    conn = db.connect()
    sample = features.analysis_sample(features.build_features(conn))
    log.info(
        "выборка: %s матчей, %s игроков",
        f"{len(sample):,}",
        f"{sample['account_id'].nunique():,}",
    )

    db.clear_findings(conn, "null_model")
    targets = cal.observed_targets(sample, min_games=args.min_games)
    log.info(
        "наблюдаемые ориентиры: медиана игр %.0f, разброс log(n) %.2f, подвижность ранга %.3f",
        targets.median_games,
        targets.games_log_sd,
        targets.rank_volatility_ratio,
    )

    tallies = sample.groupby("account_id")["win"].agg(["sum", "size"])
    tallies = tallies[tallies["size"] >= args.min_games]
    disp_real = dispersion.analyse(
        tallies["sum"].to_numpy(), tallies["size"].to_numpy(), n_boot=1000
    )
    log.info(
        "наблюдаемая дисперсия винрейтов: phi=%.3f [%.3f, %.3f] по %d игрокам",
        disp_real.phi,
        disp_real.phi_lo,
        disp_real.phi_hi,
        disp_real.n_players,
    )

    base = replace(
        SimConfig(n_players=args.players),
        n_rounds=int(max(targets.median_games * 2.5, 200)),
        activity_sd=float(np.clip(targets.games_log_sd, 0.1, 2.0)),
    )
    base = cal.match_activity(base, targets.median_games, min_games=args.min_games)
    log.info("база симуляции: %d раундов, участие %.2f, активность %.2f",
             base.n_rounds, base.participation, base.activity_sd)

    log.info("=" * 70)
    log.info("Калибровка нестационарности навыка по наблюдаемому разбросу винрейтов")
    fitted, grid = cal.calibrate_to_dispersion(
        disp_real.phi, base, min_games=args.min_games
    )
    log.info("\n%s", grid.round(5).to_string(index=False))
    log.info("выбран дрейф навыка %.4f", fitted.skill_drift)

    # Предсказание наклона откалиброванной моделью и его неопределённость по
    # соседним по разбросу конфигурациям.
    disp_fit, slope_fit, _ = cal.dispersion_of(fitted, min_games=args.min_games, n_boot=400)
    log.info(
        "откалиброванная честная модель: phi=%.3f (цель %.3f), наклон серии %+.5f ± %.5f",
        disp_fit.phi,
        disp_real.phi,
        slope_fit.coef[0],
        slope_fit.se[0],
    )

    log.info("=" * 70)
    log.info("Диапазон достижимого честной системой")
    sens = cal.sensitivity_grid(fitted, min_games=args.min_games)
    log.info("\n%s", sens.round(5).to_string(index=False))

    payload = {
        "targets": {
            "median_games": targets.median_games,
            "games_log_sd": targets.games_log_sd,
            "rank_volatility_ratio": targets.rank_volatility_ratio,
        },
        "observed_phi": disp_real.phi,
        "observed_phi_ci": [disp_real.phi_lo, disp_real.phi_hi],
        "base_config": {k: v for k, v in base.__dict__.items()},
        "fitted_config": {k: v for k, v in fitted.__dict__.items()},
        "calibration_grid": grid.to_dict("list"),
        "fitted_phi": disp_fit.phi,
        "fitted_slope": float(slope_fit.coef[0]),
        "fitted_slope_se": float(slope_fit.se[0]),
        "sensitivity": sens.to_dict("list"),
        "fair_phi_range": [float(sens["phi"].min()), float(sens["phi"].max())],
        "fair_slope_range": [float(sens["slope"].min()), float(sens["slope"].max())],
    }
    (DATA_DIR / "simulation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=float)
    )

    db.record_finding(
        conn, "null_model", "fitted_slope", float(slope_fit.coef[0]),
        float(slope_fit.coef[0] - 2.576 * slope_fit.se[0]),
        float(slope_fit.coef[0] + 2.576 * slope_fit.se[0]),
        note=f"предсказание честной модели, откалиброванной на phi={disp_real.phi:.2f}",
    )
    db.record_finding(
        conn, "null_model", "fair_phi_min", float(sens["phi"].min()),
        note="минимум разброса винрейтов, достижимый честной системой",
    )
    db.record_finding(
        conn, "null_model", "fair_slope_min", float(sens["slope"].min()),
        note="самый отрицательный наклон серии, достижимый честной системой",
    )

    _plot(grid, sens, disp_real.phi)
    log.info("готово")


def _plot(grid: pd.DataFrame, sens: pd.DataFrame, phi_real: float) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    ax = axes[0]
    ax.plot(grid["skill_drift"], grid["phi"], "o-", color=NEUTRAL, lw=1.4)
    ax.axhline(phi_real, color=ACCENT, ls="--", lw=1.4, label=f"наблюдение {phi_real:.2f}")
    ax.axhline(1.0, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("нестационарность навыка в модели")
    ax.set_ylabel("разброс винрейтов phi")
    ax.set_title("Калибровка честной модели")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.scatter(sens["phi"], sens["slope"], s=26, color=NEUTRAL)
    for _, row in sens.iterrows():
        ax.annotate(row["вариант"], (row["phi"], row["slope"]), fontsize=5.5,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0.0, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("разброс винрейтов phi")
    ax.set_ylabel("наклон эффекта серии")
    ax.set_title("Что вообще способна выдать честная система")

    save(fig, "null_model.png")


if __name__ == "__main__":
    main()
