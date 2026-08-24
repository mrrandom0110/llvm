"""Этап 5: тесты A и B против нулевой модели честного матчмейкинга."""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from dota_study import db, features
from dota_study.config import DATA_DIR
from dota_study.plotting import ACCENT, MUTED, NEUTRAL, save
from dota_study.sim.fair_mm import SimConfig, simulate, to_frame
from dota_study.stats import dispersion, streaks

log = logging.getLogger("tests_ab")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-games", type=int, default=200)
    parser.add_argument("--max-streak", type=int, default=4)
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    conn = db.connect()

    full = features.build_features(conn)
    sample = features.analysis_sample(full)
    log.info(
        "выборка: %s матчей, %s игроков",
        f"{len(sample):,}",
        f"{sample['account_id'].nunique():,}",
    )

    labels = pd.read_sql_query("SELECT account_id, label FROM player_profile", conn)
    if not labels.empty:
        sample = sample.merge(labels, on="account_id", how="left")
    else:
        sample["label"] = "resident"

    results: dict[str, object] = {}

    # ---- Тест A ----------------------------------------------------------
    log.info("=" * 70)
    log.info("ТЕСТ A: дисперсия карьерных винрейтов")
    tallies = sample.groupby("account_id")["win"].agg(["sum", "size"])
    tallies = tallies[tallies["size"] >= args.min_games]
    log.info("игроков с историей от %d матчей: %d", args.min_games, len(tallies))

    disp_real = dispersion.analyse(
        tallies["sum"].to_numpy(), tallies["size"].to_numpy(), n_boot=args.boot
    )
    _log_dispersion("реальные данные", disp_real)
    results["dispersion_real"] = _disp_dict(disp_real)

    # Без смурфов: проверяем, не они ли создают весь избыточный разброс.
    smurfs = set(labels.loc[labels["label"] == "smurf", "account_id"]) if not labels.empty else set()
    clean = sample[~sample["account_id"].isin(smurfs)]
    clean_tallies = clean.groupby("account_id")["win"].agg(["sum", "size"])
    clean_tallies = clean_tallies[clean_tallies["size"] >= args.min_games]
    disp_clean = None
    if len(clean_tallies) > 30:
        disp_clean = dispersion.analyse(
            clean_tallies["sum"].to_numpy(), clean_tallies["size"].to_numpy(), n_boot=args.boot
        )
        _log_dispersion("без смурфов", disp_clean)
        results["dispersion_no_smurf"] = _disp_dict(disp_clean)

    # Та же статистика в заведомо честной системе.
    sim = simulate(SimConfig(n_players=6000, n_rounds=int(max(tallies["size"].mean(), 200))))
    sim_n = np.bincount(sim.player_id)
    sim_w = np.bincount(sim.player_id, weights=sim.win)
    keep = sim_n >= args.min_games
    disp_sim = dispersion.analyse(sim_w[keep], sim_n[keep], n_boot=min(args.boot, 800))
    _log_dispersion("симуляция честного MM", disp_sim)
    results["dispersion_sim"] = _disp_dict(disp_sim)

    db.record_finding(
        conn, "A_dispersion", "phi_real", disp_real.phi, disp_real.phi_lo, disp_real.phi_hi,
        disp_real.n_players, disp_real.verdict(),
    )
    db.record_finding(
        conn, "A_dispersion", "phi_sim", disp_sim.phi, disp_sim.phi_lo, disp_sim.phi_hi,
        disp_sim.n_players, "нулевая модель",
    )
    db.record_finding(
        conn, "A_dispersion", "true_winrate_sd", disp_real.true_sd, n=disp_real.n_players,
        note="разброс истинных винрейтов после снятия биномиального шума",
    )

    # ---- Тест B ----------------------------------------------------------
    log.info("=" * 70)
    log.info("ТЕСТ B: влияние серии на исход")

    curve = streaks.winrate_by_streak(sample, max_streak=5)
    for s, wr, lo, hi, n in zip(curve.streak, curve.winrate, curve.lo, curve.hi, curve.n):
        log.info("  серия %+d: винрейт %.4f [%.4f, %.4f] n=%s", s, wr, lo, hi, f"{n:,}")

    fe_real = streaks.streak_slope(sample, max_streak=args.max_streak, controls=True)
    coefs = fe_real.as_dict()
    log.info(
        "наклон (реальные, с контролями): %+.5f ± %.5f (p=%.2e)",
        *coefs["streak_linear"],
    )

    fe_real_nc = streaks.streak_slope(sample, max_streak=args.max_streak, controls=False)
    log.info(
        "наклон (реальные, без контролей): %+.5f ± %.5f",
        fe_real_nc.coef[0],
        fe_real_nc.se[0],
    )

    sim_df = features.add_streaks(to_frame(sim))
    fe_sim = streaks.streak_slope(sim_df, max_streak=args.max_streak, controls=False)
    log.info("наклон (честная симуляция): %+.5f ± %.5f", fe_sim.coef[0], fe_sim.se[0])

    comparison = streaks.StreakComparison(real=fe_real_nc, simulated=fe_sim)
    log.info(
        "разница реальность минус честная модель: %+.5f ± %.5f, z=%.2f, p=%.4f",
        comparison.diff,
        comparison.se_diff,
        comparison.z,
        comparison.p_value,
    )

    # Индикаторы серий: асимметрия побед и поражений — признак гипотезы H_engage.
    y, X, groups, names = streaks.streak_design(sample, max_streak=args.max_streak)
    fe_ind = streaks.fixed_effects_lpm(y, X, groups, names)
    ind = fe_ind.as_dict()
    for name in names:
        if name.startswith("streak_"):
            log.info("  %-12s %+.5f ± %.5f (p=%.3f)", name, *ind[name])

    asym = _asymmetry(fe_ind, args.max_streak)
    log.info(
        "асимметрия побед и поражений: %+.5f ± %.5f (p=%.3f)",
        asym["value"],
        asym["se"],
        asym["p"],
    )

    runs = streaks.runs_test(sample)
    log.info(
        "runs-тест: игроков %s, средний z %+.3f, объединённый z %+.2f (p=%.3g)",
        f"{runs.get('n_players', 0):,}",
        runs.get("mean_z", float("nan")),
        runs.get("combined_z", float("nan")),
        runs.get("p_value", float("nan")),
    )
    runs_sim = streaks.runs_test(sim_df)
    log.info(
        "runs-тест (честная симуляция): средний z %+.3f",
        runs_sim.get("mean_z", float("nan")),
    )

    for test, metric, res in (
        ("B_streaks", "slope_real", fe_real_nc.coef[0]),
        ("B_streaks", "slope_real_controlled", fe_real.coef[0]),
        ("B_streaks", "slope_sim", fe_sim.coef[0]),
        ("B_streaks", "slope_diff", comparison.diff),
    ):
        db.record_finding(conn, test, metric, float(res), n=len(sample))
    db.record_finding(
        conn, "B_streaks", "slope_diff_p", comparison.p_value, n=len(sample),
        note="реальность против честной модели",
    )
    db.record_finding(
        conn, "B_streaks", "asymmetry", asym["value"], n=len(sample), note=f"p={asym['p']:.4f}"
    )
    db.record_finding(
        conn, "B_streaks", "runs_mean_z", runs.get("mean_z"), n=runs.get("n_players"),
        note=f"честная модель {runs_sim.get('mean_z', float('nan')):+.3f}",
    )

    results["streaks"] = {
        "curve": {
            "streak": curve.streak.tolist(),
            "winrate": curve.winrate.tolist(),
            "lo": curve.lo.tolist(),
            "hi": curve.hi.tolist(),
            "n": curve.n.tolist(),
        },
        "slope_real": float(fe_real_nc.coef[0]),
        "slope_real_se": float(fe_real_nc.se[0]),
        "slope_real_controlled": float(fe_real.coef[0]),
        "slope_real_controlled_se": float(fe_real.se[0]),
        "slope_sim": float(fe_sim.coef[0]),
        "slope_sim_se": float(fe_sim.se[0]),
        "diff": comparison.diff,
        "diff_se": comparison.se_diff,
        "diff_p": comparison.p_value,
        "indicators": {k: v for k, v in ind.items() if k.startswith("streak_")},
        "asymmetry": asym,
        "runs": runs,
        "runs_sim": runs_sim,
        "n_obs": int(len(sample)),
        "n_players": int(sample["account_id"].nunique()),
    }

    (DATA_DIR / "tests_ab.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=float)
    )
    _plot(tallies, disp_real, disp_sim, curve, fe_ind, names, sim_df)
    log.info("готово")


def _asymmetry(fe: streaks.FEResult, max_streak: int) -> dict[str, float]:
    """Сумма коэффициентов симметричных серий: при симметрии эффекта равна нулю."""
    from scipy import stats as sps

    idx_pos = [fe.names.index(f"streak_{k:+d}") for k in range(1, max_streak + 1)]
    idx_neg = [fe.names.index(f"streak_{-k:+d}") for k in range(1, max_streak + 1)]
    value = float(fe.coef[idx_pos].sum() + fe.coef[idx_neg].sum())
    se = float(np.sqrt((fe.se[idx_pos] ** 2).sum() + (fe.se[idx_neg] ** 2).sum()))
    z = value / se if se > 0 else np.nan
    return {"value": value, "se": se, "p": float(2 * sps.norm.sf(abs(z)))}


def _disp_dict(res: dispersion.DispersionResult) -> dict[str, object]:
    return {
        "n_players": res.n_players,
        "n_matches": res.n_matches,
        "mean_winrate": res.mean_winrate,
        "phi": res.phi,
        "phi_ci": [res.phi_lo, res.phi_hi],
        "observed_sd": res.observed_sd,
        "binomial_sd": res.binomial_sd,
        "true_sd": res.true_sd,
        "p_underdispersion": res.p_underdispersion,
        "lrt_p": res.lrt_p,
        "verdict": res.verdict(),
    }


def _log_dispersion(title: str, res: dispersion.DispersionResult) -> None:
    log.info(
        "  %-24s phi=%.3f [%.3f, %.3f] | наблюдаемый SD %.4f против биномиального %.4f "
        "| истинный разброс %.4f | %s",
        title,
        res.phi,
        res.phi_lo,
        res.phi_hi,
        res.observed_sd,
        res.binomial_sd,
        res.true_sd,
        res.verdict(),
    )


def _plot(tallies, disp_real, disp_sim, curve, fe_ind, names, sim_df) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    ax = axes[0]
    rates = (tallies["sum"] / tallies["size"]).to_numpy()
    ax.hist(rates, bins=40, density=True, color=NEUTRAL, alpha=0.8, label="наблюдаемое")
    grid = np.linspace(rates.min(), rates.max(), 200)
    binom_pdf = np.exp(-0.5 * ((grid - disp_real.mean_winrate) / disp_real.binomial_sd) ** 2)
    binom_pdf /= binom_pdf.sum() * (grid[1] - grid[0])
    ax.plot(grid, binom_pdf, color=ACCENT, lw=1.6, label="чистая случайность")
    ax.axvline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("карьерный винрейт")
    ax.set_title(
        f"Тест A: разброс винрейтов\nphi={disp_real.phi:.2f} против {disp_sim.phi:.2f} в честной модели"
    )
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.errorbar(
        curve.streak,
        curve.winrate,
        yerr=[curve.winrate - curve.lo, curve.hi - curve.winrate],
        fmt="o-",
        color=ACCENT,
        capsize=3,
        lw=1,
    )
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("серия перед матчем")
    ax.set_ylabel("доля побед")
    ax.set_title("Тест B: исход после серии\n(сырые доли, 99% ДИ)")

    ax = axes[2]
    keys = [n for n in names if n.startswith("streak_")]
    idx = [names.index(k) for k in keys]
    xs = [int(k.split("_")[1]) for k in keys]
    coef = fe_ind.coef[idx]
    se = fe_ind.se[idx]
    ax.errorbar(xs, coef, yerr=2.576 * se, fmt="o", color=ACCENT, capsize=3)
    ax.axhline(0.0, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("серия перед матчем")
    ax.set_ylabel("сдвиг вероятности победы")
    ax.set_title("Тест B: с фиксированными эффектами\nи контролем на движение рейтинга")

    save(fig, "tests_ab.png")


if __name__ == "__main__":
    main()
