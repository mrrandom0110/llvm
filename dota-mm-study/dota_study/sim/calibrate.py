"""Калибровка нулевой модели и анализ её чувствительности.

Сравнивать реальные данные с симулятором честно можно только тогда, когда
симулятор настроен по величинам, **не участвующим в самих тестах**. Иначе
рассуждение станет круговым: подогнав модель под наблюдаемый разброс винрейтов,
мы гарантированно получим совпадение и ничего не проверим.

Поэтому калибровочные цели выбраны так, чтобы не пересекаться с проверяемыми
статистиками:

* сколько матчей приходится на игрока — задаёт биномиальный шум винрейта;
* насколько подвижен рейтинг игрока относительно разброса рейтингов в
  популяции — задаёт нестационарность навыка и величину шага Elo.

Разброс винрейтов и автокорреляция исходов остаются предсказанием модели.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .fair_mm import SimConfig, SimResult, simulate


@dataclass
class CalibrationTargets:
    median_games: float
    games_log_sd: float
    rank_volatility_ratio: float


def rank_volatility_ratio(
    df: pd.DataFrame,
    rank_column: str = "average_rank",
    min_games: int = 200,
    window: int = 50,
) -> float:
    """Подвижность рейтинга игрока в единицах разброса рейтингов в популяции.

    Величина безразмерна, поэтому сопоставима между реальной шкалой рангов и
    внутренней шкалой симулятора.
    """
    usable = df.dropna(subset=[rank_column])
    counts = usable.groupby("account_id")[rank_column].transform("size")
    usable = usable[counts >= min_games]
    if usable.empty:
        return float("nan")

    def _volatility(series: pd.Series) -> float:
        rolled = series.rolling(window, min_periods=window // 2).mean().dropna()
        return rolled.std(ddof=0) if len(rolled) > 2 else np.nan

    within = usable.groupby("account_id")[rank_column].apply(_volatility)
    between = usable.groupby("account_id")[rank_column].mean().std(ddof=0)
    if not np.isfinite(between) or between == 0:
        return float("nan")
    return float(np.nanmedian(within) / between)


def observed_targets(sample: pd.DataFrame, min_games: int = 200) -> CalibrationTargets:
    counts = sample.groupby("account_id").size()
    eligible = counts[counts >= min_games]
    return CalibrationTargets(
        median_games=float(eligible.median()) if len(eligible) else float("nan"),
        games_log_sd=float(np.log(counts.clip(lower=1)).std(ddof=0)),
        rank_volatility_ratio=rank_volatility_ratio(sample, min_games=min_games),
    )


def simulated_targets(result: SimResult, min_games: int = 200) -> CalibrationTargets:
    counts = np.bincount(result.player_id, minlength=result.config.n_players)
    eligible = counts[counts >= min_games]
    frame = pd.DataFrame(
        {
            "account_id": result.player_id,
            "average_rank": result.match_rating.astype(float),
        }
    )
    return CalibrationTargets(
        median_games=float(np.median(eligible)) if len(eligible) else float("nan"),
        games_log_sd=float(np.log(np.clip(counts, 1, None)).std(ddof=0)),
        rank_volatility_ratio=rank_volatility_ratio(frame, min_games=min_games),
    )


def match_activity(
    cfg: SimConfig, target_median_games: float, min_games: int = 200, iterations: int = 3
) -> SimConfig:
    """Подгоняет число раундов под наблюдаемую длину истории игрока.

    Неоднородная активность имеет смысл только при неполном участии: если в
    каждом раунде играют все, длина истории у всех одинакова и равна числу
    раундов. Поэтому доля участников меньше единицы, а число раундов
    подбирается итеративно.
    """
    if not np.isfinite(target_median_games):
        return cfg
    for _ in range(iterations):
        observed = simulated_targets(simulate(cfg), min_games=min_games).median_games
        if not np.isfinite(observed) or observed <= 0:
            break
        ratio = target_median_games / observed
        if 0.95 <= ratio <= 1.05:
            break
        cfg = replace(cfg, n_rounds=int(np.clip(cfg.n_rounds * ratio, 200, 20000)))
    return cfg


def calibrate(
    targets: CalibrationTargets,
    base: SimConfig | None = None,
    drift_grid: tuple[float, ...] = (0.0, 0.016, 0.032, 0.048, 0.064, 0.096, 0.128),
    min_games: int = 200,
) -> tuple[SimConfig, pd.DataFrame]:
    """Подбирает дрейф навыка так, чтобы совпала подвижность рейтинга.

    Число раундов и разброс активности выставляются по наблюдаемой длине
    историй, после чего единственным свободным параметром остаётся дрейф навыка.
    """
    cfg = base or SimConfig()
    if np.isfinite(targets.games_log_sd):
        cfg = replace(cfg, activity_sd=float(np.clip(targets.games_log_sd, 0.1, 2.0)))
    cfg = replace(cfg, n_rounds=int(max(targets.median_games * 2.5, 200)))
    cfg = match_activity(cfg, targets.median_games, min_games=min_games)

    rows = []
    best_cfg, best_gap = cfg, np.inf
    for drift in drift_grid:
        trial = replace(cfg, skill_drift=drift)
        result = simulate(trial)
        sim_t = simulated_targets(result, min_games=min_games)
        gap = abs(sim_t.rank_volatility_ratio - targets.rank_volatility_ratio)
        rows.append(
            {
                "skill_drift": drift,
                "median_games": sim_t.median_games,
                "rank_volatility_ratio": sim_t.rank_volatility_ratio,
                "gap": gap,
            }
        )
        if np.isfinite(gap) and gap < best_gap:
            best_gap, best_cfg = gap, trial

    return best_cfg, pd.DataFrame(rows)


def dispersion_of(cfg: SimConfig, min_games: int = 200, n_boot: int = 200):
    """Коэффициент дисперсии и наклон серии для заданной конфигурации."""
    from ..features import add_streaks
    from ..stats import dispersion, streaks
    from .fair_mm import to_frame

    result = simulate(cfg)
    counts = np.bincount(result.player_id, minlength=cfg.n_players)
    wins = np.bincount(result.player_id, weights=result.win, minlength=cfg.n_players)
    keep = counts >= min_games
    disp = dispersion.analyse(wins[keep], counts[keep], n_boot=n_boot)
    frame = add_streaks(to_frame(result))
    slope = streaks.streak_slope(frame, controls=False)
    return disp, slope, result


def calibrate_to_dispersion(
    phi_target: float,
    base: SimConfig,
    drift_grid: tuple[float, ...] = (0.0, 0.004, 0.008, 0.012, 0.016, 0.024, 0.032, 0.048),
    min_games: int = 200,
) -> tuple[SimConfig, pd.DataFrame]:
    """Настраивает нестационарность навыка так, чтобы совпал разброс винрейтов.

    Такая калибровка решает проблему неидентифицируемости. Разброс винрейтов в
    честной модели монотонно растёт с нестационарностью навыка, поэтому по нему
    можно однозначно подобрать единственный свободный параметр. После этого
    **наклон серии перестаёт быть подгоняемой величиной и становится
    предсказанием модели**, которое можно честно сравнить с наблюдением.

    Цена такого решения: тест A больше не является независимой проверкой, он
    работает как односторонняя фальсификация (недодисперсия) и как якорь
    калибровки для теста B.
    """
    rows = []
    best_cfg, best_gap = base, np.inf
    for drift in drift_grid:
        trial = replace(base, skill_drift=drift)
        disp, slope, _ = dispersion_of(trial, min_games=min_games)
        gap = abs(disp.phi - phi_target)
        rows.append(
            {
                "skill_drift": drift,
                "phi": disp.phi,
                "true_sd": disp.true_sd,
                "slope": float(slope.coef[0]),
                "slope_se": float(slope.se[0]),
                "gap": gap,
            }
        )
        if gap < best_gap:
            best_gap, best_cfg = gap, trial
    return best_cfg, pd.DataFrame(rows)


def sensitivity_grid(base: SimConfig, min_games: int = 200) -> pd.DataFrame:
    """Диапазон статистик, достижимый честной системой при разумных параметрах.

    Смысл: вместо одной точки нулевой модели получаем область. Реальные данные
    признаются несовместимыми с честным подбором только если выходят за неё.
    """
    from ..stats import dispersion, streaks
    from .fair_mm import to_frame

    variations: list[tuple[str, SimConfig]] = [("базовая", base)]
    for drift in (0.0, base.skill_drift * 2, base.skill_drift * 4):
        variations.append((f"дрейф навыка {drift:.3f}", replace(base, skill_drift=drift)))
    for k in (base.k_factor / 2, base.k_factor * 2):
        variations.append((f"шаг Elo {k:.3f}", replace(base, k_factor=k)))
    for noise in (base.mm_noise / 2, base.mm_noise * 3):
        variations.append((f"окно подбора {noise:.3f}", replace(base, mm_noise=noise)))
    for share in (0.02, 0.05):
        variations.append(
            (f"смурфов {share:.0%}", replace(base, smurf_fraction=share))
        )
    variations.append(("случайные команды", replace(base, team_assignment="random")))

    rows = []
    for label, cfg in variations:
        result = simulate(cfg)
        counts = np.bincount(result.player_id, minlength=cfg.n_players)
        wins = np.bincount(result.player_id, weights=result.win, minlength=cfg.n_players)
        keep = counts >= min_games
        if keep.sum() < 30:
            continue
        disp = dispersion.analyse(wins[keep], counts[keep], n_boot=200)
        frame = to_frame(result)
        from ..features import add_streaks

        frame = add_streaks(frame)
        slope = streaks.streak_slope(frame, controls=False)
        rows.append(
            {
                "вариант": label,
                "phi": disp.phi,
                "phi_lo": disp.phi_lo,
                "phi_hi": disp.phi_hi,
                "true_sd": disp.true_sd,
                "slope": float(slope.coef[0]),
                "slope_se": float(slope.se[0]),
                "n_players": int(keep.sum()),
                "volatility": simulated_targets(result, min_games).rank_volatility_ratio,
            }
        )
    return pd.DataFrame(rows)
