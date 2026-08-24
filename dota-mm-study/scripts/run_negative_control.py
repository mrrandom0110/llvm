"""Этап 5б: негативный контроль — увидели бы мы подкрутку, если бы она была.

Позитивные контроли доказывают, что конвейер видит известные эффекты в данных.
Здесь проверяется обратное и не менее важное: что **сами тесты чувствительны к
подкрутке**. В симулятор внедряется вмешательство известной силы, и измеряется,
как на него реагируют статистики. Без этой проверки вывод «эффекта не найдено»
не имел бы силы: он мог бы означать, что тесты слепы.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace

import pandas as pd

from dota_study import db
from dota_study.config import DATA_DIR
from dota_study.plotting import ACCENT, MUTED, NEUTRAL, save
from dota_study.sim.rigged_mm import RigConfig, detection_curve

log = logging.getLogger("negative_control")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=6000)
    parser.add_argument("--max-rounds", type=int, default=1500)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    conn = db.connect()
    db.clear_findings(conn, "negative_control")

    sim_path = DATA_DIR / "simulation.json"
    if not sim_path.exists():
        raise SystemExit("сначала выполните scripts.run_simulation")
    fitted = json.loads(sim_path.read_text())["fitted_config"]

    base = RigConfig(
        **{k: v for k, v in fitted.items() if k in RigConfig.__dataclass_fields__}
    )
    base = replace(
        base, n_players=args.players, n_rounds=min(fitted["n_rounds"], args.max_rounds)
    )
    log.info(
        "нулевая база: %d игроков, %d раундов, дрейф навыка %.3f",
        base.n_players,
        base.n_rounds,
        base.skill_drift,
    )

    log.info("=" * 70)
    log.info("Механизм прямого смещения исхода против серии")
    outcome = detection_curve(base, mechanism="outcome")
    log.info("\n%s", outcome.round(5).to_string(index=False))

    baseline = float(outcome.loc[outcome["сила_пп"] == 0, "slope"].iloc[0])
    outcome["сдвиг наклона"] = outcome["slope"] - baseline
    log.info(
        "внедрённая подкрутка восстанавливается тестом почти один к одному: "
        "при внедрении 1.0 п.п. наклон сдвигается на %.2f п.п.",
        100 * float(outcome.loc[outcome["сила_пп"] == 1.0, "сдвиг наклона"].iloc[0]),
    )

    log.info("=" * 70)
    log.info("Механизм перекоса состава внутри матча")
    roster = detection_curve(
        base, strengths_pp=(0.0, 0.25, 0.5, 1.0), mechanism="roster"
    )
    log.info("\n%s", roster.round(5).to_string(index=False))
    log.info(
        "эффект слабый даже при перекосе в каждом матче: подобранные в один матч "
        "игроки почти равны по рейтингу, поэтому перестановка составов физически "
        "не способна сильно сдвинуть исход"
    )

    for _, row in outcome.iterrows():
        db.record_finding(
            conn,
            "negative_control",
            f"outcome_rig_{row['сила_пп']:.2f}pp",
            float(row["slope"]),
            n=int(row["n_players"]),
            note="наклон, который дал бы тест при подкрутке такой силы",
        )

    payload = {
        "outcome": outcome.to_dict("list"),
        "roster": roster.to_dict("list"),
        "baseline_slope": baseline,
    }
    (DATA_DIR / "negative_control.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=float)
    )

    _plot(outcome, roster)
    log.info("готово")


def _plot(outcome: pd.DataFrame, roster: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    ab = json.loads((DATA_DIR / "tests_ab.json").read_text()) if (DATA_DIR / "tests_ab.json").exists() else {}
    observed = ab.get("streaks", {}).get("slope_real")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    ax = axes[0]
    ax.errorbar(
        outcome["сила_пп"],
        100 * outcome["slope"],
        yerr=100 * 2.576 * outcome["slope_se"],
        fmt="o-",
        color=NEUTRAL,
        capsize=3,
        label="подкрученная система",
    )
    if observed is not None:
        ax.axhline(
            100 * observed, color=ACCENT, ls="--", lw=1.5,
            label=f"наблюдение {100 * observed:+.2f} п.п.",
        )
    ax.axhline(0, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("внедрённая подкрутка, п.п. за шаг серии")
    ax.set_ylabel("измеренный наклон, п.п.")
    ax.set_title("Тест восстанавливает внедрённый эффект")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(outcome["сила_пп"], outcome["runs_mean_z"], "o-", color=NEUTRAL,
            label="подкрученная система")
    ax.axhline(0, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("внедрённая подкрутка, п.п. за шаг серии")
    ax.set_ylabel("средний z теста серий")
    ax.set_title("Подкрутка укорачивает серии\n(положительный z)")
    ax.legend(fontsize=7)

    save(fig, "negative_control.png")


if __name__ == "__main__":
    main()
