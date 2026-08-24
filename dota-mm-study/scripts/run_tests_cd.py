"""Этап 7: тесты C и D — состав команды и разложение эффекта серии."""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from dota_study import db, features
from dota_study.config import DATA_DIR
from dota_study.plotting import ACCENT, MUTED, NEUTRAL, save
from dota_study.stats import roster as roster_stats

log = logging.getLogger("tests_cd")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-streak", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    conn = db.connect()

    full = features.build_features(conn)
    sample = features.analysis_sample(full)
    roster = pd.read_sql_query("SELECT * FROM roster", conn)
    log.info(
        "составов: %s строк по %s матчам",
        f"{len(roster):,}",
        f"{roster['match_id'].nunique():,}",
    )

    # Независимая оценка силы игрока по его собственной истории. Она точнее
    # ранга, но известна только для выгруженной части выборки.
    strength = (
        sample.groupby("account_id")
        .agg(skill=("average_rank", "mean"))["skill"]
    )
    log.info("независимая оценка силы есть для %s игроков", f"{len(strength):,}")

    focal = sample[sample["match_id"].isin(set(roster["match_id"]))][
        ["match_id", "account_id", "prev_streak", "win"]
    ]
    log.info("фокальных наблюдений-кандидатов: %s", f"{len(focal):,}")

    obs = roster_stats.build_roster_observations(
        roster, focal, skill_column="rank_tier", external_skill=strength
    )
    log.info("наблюдений с достаточным составом: %s", f"{len(obs):,}")

    # Проверка на устойчивость: та же оценка, но силой служит только ранг из
    # данных матча. Смешивание двух разных измерителей могло бы создать мнимую
    # асимметрию, если доступность независимой оценки различается у союзников и
    # соперников, поэтому основной вывод обязан выдерживать однородную меру.
    obs_rank_only = roster_stats.build_roster_observations(
        roster, focal, skill_column="rank_tier", external_skill=None
    )

    results: dict[str, object] = {}

    # ---- Тест C ----------------------------------------------------------
    log.info("=" * 70)
    log.info("ТЕСТ C: перекос состава в зависимости от серии")
    asym = roster_stats.test_asymmetry(obs, max_streak=args.max_streak)
    if asym.get("n", 0) == 0:
        log.warning("недостаточно данных для теста C")
    else:
        log.info(
            "наблюдений %s по %s игрокам, средняя доля анонимов в матче %.2f",
            f"{asym['n']:,}",
            f"{asym['n_players']:,}",
            asym["mean_anon_share"],
        )
        log.info(
            "средний перекос союзники минус соперники: %+.4f ± %.4f",
            asym["mean_delta"],
            asym["se_delta"],
        )
        for streak, row in asym["by_streak"].iterrows():
            log.info(
                "  серия %+d: перекос %+.4f ± %.4f (n=%d)",
                streak,
                row["mean"],
                row["se"],
                int(row["size"]),
            )
        slope = asym.get("slope")
        if slope is not None:
            lo, hi = slope.ci("streak")
            log.info(
                "наклон перекоса по серии: %+.5f ± %.5f, 99%% ДИ [%+.5f, %+.5f]",
                slope.coef[0],
                slope.se[0],
                lo,
                hi,
            )
            # Перевод в вероятность победы, чтобы границу можно было сравнить
            # с эффектом серии из теста B.
            value, value_se = roster_stats.delta_value_in_winrate(obs)
            channel = roster_stats.roster_channel_in_winrate(
                float(slope.coef[0]), float(slope.se[0]), value, value_se
            )
            if channel:
                log.info(
                    "  единица перекоса стоит %+.4f вероятности победы (SE %.4f)",
                    value,
                    value_se,
                )
                log.info(
                    "  вклад канала состава в эффект серии: %+.5f, 99%% ДИ [%+.5f, %+.5f]",
                    channel["effect"],
                    channel["lo"],
                    channel["hi"],
                )
                db.record_finding(
                    conn, "C_roster", "channel_in_winrate", channel["effect"],
                    channel["lo"], channel["hi"], asym["n"],
                    "вклад подбора состава в эффект серии, в вероятности победы",
                )
                results["roster_channel_winrate"] = channel
                results["delta_value"] = {"coef": value, "se": value_se}
            db.record_finding(
                conn, "C_roster", "delta_slope", float(slope.coef[0]), lo, hi,
                asym["n"], "перекос состава как функция серии; ноль — честный подбор",
            )
        db.record_finding(
            conn, "C_roster", "mean_delta", asym["mean_delta"],
            asym["mean_delta"] - 2.576 * asym["se_delta"],
            asym["mean_delta"] + 2.576 * asym["se_delta"],
            asym["n"],
        )

        asym_rank = roster_stats.test_asymmetry(obs_rank_only, max_streak=args.max_streak)
        if asym_rank.get("n", 0) > 0 and asym_rank.get("slope") is not None:
            log.info(
                "устойчивость (только ранг из матча): наклон %+.5f ± %.5f по %s наблюдениям",
                asym_rank["slope"].coef[0],
                asym_rank["slope"].se[0],
                f"{asym_rank['n']:,}",
            )
            db.record_finding(
                conn, "C_roster", "delta_slope_rank_only",
                float(asym_rank["slope"].coef[0]),
                *asym_rank["slope"].ci("streak"),
                n=asym_rank["n"],
                note="проверка на однородной мере силы",
            )
            results["test_c_rank_only"] = {
                "n": asym_rank["n"],
                "slope": float(asym_rank["slope"].coef[0]),
                "slope_se": float(asym_rank["slope"].se[0]),
                "mean_delta": asym_rank["mean_delta"],
            }
        results["test_c"] = {
            "n": asym["n"],
            "n_players": asym["n_players"],
            "mean_delta": asym["mean_delta"],
            "se_delta": asym["se_delta"],
            "anon_share": asym["mean_anon_share"],
            "by_streak": asym["by_streak"].reset_index().to_dict("list"),
            "slope": float(slope.coef[0]) if slope is not None else None,
            "slope_se": float(slope.se[0]) if slope is not None else None,
        }

    # ---- Тест D ----------------------------------------------------------
    log.info("=" * 70)
    log.info("ТЕСТ D: разложение эффекта серии на каналы")
    decomposition = roster_stats.decompose_streak(sample, obs, max_streak=4)
    channel_names = {
        "own_performance": "собственный перформанс",
        "ally_skill": "сила союзников",
        "enemy_skill": "сила соперников",
        "delta": "перекос состава",
        "skill_spread": "разброс силы в матче",
    }
    results["test_d"] = {}
    for key, title in channel_names.items():
        fe = decomposition.get(key)
        if fe is None:
            continue
        lo, hi = fe.ci("streak")
        log.info(
            "  %-22s %+.5f ± %.5f, 99%% ДИ [%+.5f, %+.5f], n=%s",
            title,
            fe.coef[0],
            fe.se[0],
            lo,
            hi,
            f"{fe.n_obs:,}",
        )
        db.record_finding(conn, "D_decomposition", key, float(fe.coef[0]), lo, hi, fe.n_obs, title)
        results["test_d"][key] = {
            "coef": float(fe.coef[0]),
            "se": float(fe.se[0]),
            "ci": [lo, hi],
            "n": fe.n_obs,
        }

    (DATA_DIR / "tests_cd.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=float)
    )
    if asym.get("n", 0) > 0:
        _plot(asym, decomposition, channel_names)
    log.info("готово")


def _plot(asym, decomposition, channel_names) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ax = axes[0]
    table = asym["by_streak"]
    ax.errorbar(
        table.index,
        table["mean"],
        yerr=2.576 * table["se"],
        fmt="o",
        color=ACCENT,
        capsize=3,
    )
    ax.axhline(0.0, color=MUTED, ls="--", lw=1)
    ax.set_xlabel("серия перед матчем")
    ax.set_ylabel("сила союзников минус соперников")
    ax.set_title(
        f"Тест C: перекос состава\n{asym['n']:,} наблюдений, 99% ДИ"
    )

    ax = axes[1]
    keys = [k for k in channel_names if k in decomposition]
    values = [decomposition[k].coef[0] for k in keys]
    errs = [2.576 * decomposition[k].se[0] for k in keys]
    positions = np.arange(len(keys))
    ax.barh(positions, values, xerr=errs, color=NEUTRAL, alpha=0.85, capsize=3)
    ax.axvline(0.0, color=MUTED, ls="--", lw=1)
    ax.set_yticks(positions)
    ax.set_yticklabels([channel_names[k] for k in keys])
    ax.set_xlabel("изменение за шаг серии")
    ax.set_title("Тест D: каналы эффекта серии")

    save(fig, "tests_cd.png")


if __name__ == "__main__":
    main()
