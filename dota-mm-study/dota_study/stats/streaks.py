"""Тест B: влияние предшествующей серии на исход матча.

Ключевая тонкость: отрицательная зависимость исхода от серии сама по себе
**не** является признаком подкрутки. В честной рейтинговой системе она возникает
неизбежно — после победы рейтинг вырос, значит следующий соперник сильнее.
Поэтому наблюдаемая величина сравнивается с той же величиной в симуляторе
честного матчмейкинга, а не с нулём.

Оценивание идёт линейной вероятностной моделью с фиксированными эффектами
игрока. При вероятностях около 0.5 она практически совпадает с логистической,
но допускает поглощение тысяч индивидуальных эффектов на миллионах наблюдений,
что для логита с фиктивными переменными вычислительно недоступно.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StreakCurve:
    streak: np.ndarray
    winrate: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    n: np.ndarray


@dataclass
class FEResult:
    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    n_obs: int
    n_groups: int

    def as_dict(self) -> dict[str, tuple[float, float, float]]:
        out = {}
        for i, name in enumerate(self.names):
            z = self.coef[i] / self.se[i] if self.se[i] > 0 else np.nan
            out[name] = (float(self.coef[i]), float(self.se[i]), float(2 * stats.norm.sf(abs(z))))
        return out

    def ci(self, name: str, conf: float = 0.99) -> tuple[float, float]:
        i = self.names.index(name)
        z = stats.norm.ppf(1 - (1 - conf) / 2)
        return (self.coef[i] - z * self.se[i], self.coef[i] + z * self.se[i])


def winrate_by_streak(
    df: pd.DataFrame, max_streak: int = 5, conf: float = 0.99
) -> StreakCurve:
    """Наблюдаемая доля побед в зависимости от предшествующей серии."""
    clipped = df["prev_streak"].clip(-max_streak, max_streak)
    grouped = df.groupby(clipped)["win"].agg(["mean", "sum", "size"])
    grouped = grouped.sort_index()
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    n = grouped["size"].to_numpy(dtype=float)
    wins = grouped["sum"].to_numpy(dtype=float)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return StreakCurve(
        streak=grouped.index.to_numpy(),
        winrate=p,
        lo=center - half,
        hi=center + half,
        n=n.astype(int),
    )


def _within_transform(values: np.ndarray, group_codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Вычитание внутригруппового среднего — поглощение фиксированных эффектов."""
    counts = np.bincount(group_codes, minlength=n_groups).astype(float)
    counts[counts == 0] = 1.0
    if values.ndim == 1:
        sums = np.bincount(group_codes, weights=values, minlength=n_groups)
        return values - (sums / counts)[group_codes]
    out = np.empty_like(values, dtype=float)
    for j in range(values.shape[1]):
        sums = np.bincount(group_codes, weights=values[:, j], minlength=n_groups)
        out[:, j] = values[:, j] - (sums / counts)[group_codes]
    return out


def fixed_effects_lpm(
    y: np.ndarray,
    X: np.ndarray,
    groups: np.ndarray,
    names: list[str],
) -> FEResult:
    """МНК с поглощёнными фиксированными эффектами и кластеризацией по группам.

    Кластеризация обязательна: наблюдения одного игрока зависимы, и без неё
    стандартные ошибки были бы занижены в разы, а любой шум выглядел бы значимым.
    """
    codes, uniques = pd.factorize(groups, sort=False)
    n_groups = len(uniques)

    y_w = _within_transform(np.asarray(y, dtype=float), codes, n_groups)
    X_w = _within_transform(np.asarray(X, dtype=float), codes, n_groups)

    xtx = X_w.T @ X_w
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (X_w.T @ y_w)
    resid = y_w - X_w @ beta

    # Кластерная «сэндвич»-оценка ковариации.
    meat = np.zeros_like(xtx)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(np.diff(sorted_codes)) + 1
    for chunk in np.split(order, boundaries):
        xg = X_w[chunk]
        ug = resid[chunk]
        score = xg.T @ ug
        meat += np.outer(score, score)

    n_obs, k = X_w.shape
    dof_scale = (n_groups / max(n_groups - 1, 1)) * (
        (n_obs - 1) / max(n_obs - k - n_groups, 1)
    )
    cov = xtx_inv @ meat @ xtx_inv * dof_scale
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return FEResult(names=names, coef=beta, se=se, n_obs=n_obs, n_groups=n_groups)


def streak_design(
    df: pd.DataFrame,
    max_streak: int = 4,
    controls: bool = True,
    reference: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Строит матрицу плана: индикаторы серий плюс контроли.

    Индикаторы, а не линейный член, позволяют увидеть асимметрию между сериями
    побед и поражений, которую предсказывает гипотеза H_engage.

    Базовой категорией берётся серия из одного поражения — одна из самых
    населённых ячеек. Использовать в этой роли нулевую серию нельзя: она
    встречается только в самом первом матче игрока, и оценки относительно почти
    пустой ячейки имели бы бессмысленно широкие интервалы.
    """
    df = df[df["prev_streak"] != 0]
    clipped = df["prev_streak"].clip(-max_streak, max_streak).to_numpy()
    columns: list[np.ndarray] = []
    names: list[str] = []
    for k in range(-max_streak, max_streak + 1):
        if k == 0 or k == reference:
            continue
        columns.append((clipped == k).astype(float))
        names.append(f"streak_{k:+d}")

    if controls:
        extra = {
            "rank_delta": df.get("rank_delta"),
            "party_size": df.get("party_size"),
            "session_pos": df.get("session_pos"),
        }
        for name, series in extra.items():
            if series is None:
                continue
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            if np.all(np.isnan(values)):
                continue
            values = np.nan_to_num(values, nan=float(np.nanmean(values)))
            columns.append(values)
            names.append(name)

    X = np.column_stack(columns)
    y = df["win"].to_numpy(dtype=float)
    groups = df["account_id"].to_numpy()
    return y, X, groups, names


def streak_slope(
    df: pd.DataFrame, max_streak: int = 4, controls: bool = True
) -> FEResult:
    """Линейный наклон: насколько каждая дополнительная победа в серии меняет шанс."""
    clipped = df["prev_streak"].clip(-max_streak, max_streak).to_numpy(dtype=float)
    columns = [clipped]
    names = ["streak_linear"]
    if controls:
        for name in ("rank_delta", "party_size", "session_pos"):
            if name not in df:
                continue
            values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
            if np.all(np.isnan(values)):
                continue
            columns.append(np.nan_to_num(values, nan=float(np.nanmean(values))))
            names.append(name)
    X = np.column_stack(columns)
    return fixed_effects_lpm(
        df["win"].to_numpy(dtype=float), X, df["account_id"].to_numpy(), names
    )


def asymmetric_slopes(
    df: pd.DataFrame, max_streak: int = 4, controls: bool = True
) -> tuple[FEResult, dict[str, float]]:
    """Отдельные наклоны для серий побед и серий поражений.

    Гипотеза H_engage предсказывает именно асимметрию: система «наказывает» за
    победную серию и «выручает» после серии поражений, а значит два наклона
    должны различаться. Симметричный эффект такой формы не имеет.
    """
    from scipy import stats as sps

    data = df[df["prev_streak"] != 0]
    clipped = data["prev_streak"].clip(-max_streak, max_streak).to_numpy(dtype=float)
    columns = [np.maximum(clipped, 0.0), np.minimum(clipped, 0.0)]
    names = ["slope_after_wins", "slope_after_losses"]

    if controls:
        for name in ("rank_delta", "party_size", "session_pos"):
            if name not in data:
                continue
            values = pd.to_numeric(data[name], errors="coerce").to_numpy(dtype=float)
            if np.all(np.isnan(values)):
                continue
            columns.append(np.nan_to_num(values, nan=float(np.nanmean(values))))
            names.append(name)

    fe = fixed_effects_lpm(
        data["win"].to_numpy(dtype=float),
        np.column_stack(columns),
        data["account_id"].to_numpy(),
        names,
    )
    i_pos = fe.names.index("slope_after_wins")
    i_neg = fe.names.index("slope_after_losses")
    diff = float(fe.coef[i_pos] - fe.coef[i_neg])
    se = float(np.sqrt(fe.se[i_pos] ** 2 + fe.se[i_neg] ** 2))
    z = diff / se if se > 0 else np.nan
    return fe, {
        "difference": diff,
        "se": se,
        "z": float(z),
        "p": float(2 * sps.norm.sf(abs(z))),
    }


def runs_test(df: pd.DataFrame, min_games: int = 100) -> dict[str, float]:
    """Тест Уолда-Вольфовица на случайность последовательности исходов.

    Проверяет не уровень винрейта, а его структуру: подкрутка должна порождать
    больше чередований (меньше длинных серий), чем случайная последовательность
    с той же долей побед.
    """
    z_values: list[float] = []
    for _, group in df.groupby("account_id", sort=False):
        wins = group["win"].to_numpy()
        n = len(wins)
        if n < min_games:
            continue
        n1 = int(wins.sum())
        n2 = n - n1
        if n1 == 0 or n2 == 0:
            continue
        runs = 1 + int(np.sum(wins[1:] != wins[:-1]))
        expected = 2 * n1 * n2 / n + 1
        variance = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n * n * (n - 1))
        if variance <= 0:
            continue
        z_values.append((runs - expected) / np.sqrt(variance))

    if not z_values:
        return {"n_players": 0}
    arr = np.array(z_values)
    # Стауффер: под нулевой гипотезой среднее z равно нулю, а сумма z,
    # делённая на корень из числа игроков, снова стандартна нормальна.
    combined = float(arr.sum() / np.sqrt(len(arr)))
    return {
        "n_players": len(arr),
        "mean_z": float(arr.mean()),
        "combined_z": combined,
        "p_value": float(2 * stats.norm.sf(abs(combined))),
    }


@dataclass
class StreakComparison:
    real: FEResult
    simulated: FEResult
    diff: float = field(init=False)
    se_diff: float = field(init=False)
    z: float = field(init=False)
    p_value: float = field(init=False)

    def __post_init__(self) -> None:
        i_real = self.real.names.index("streak_linear")
        i_sim = self.simulated.names.index("streak_linear")
        self.diff = float(self.real.coef[i_real] - self.simulated.coef[i_sim])
        self.se_diff = float(
            np.sqrt(self.real.se[i_real] ** 2 + self.simulated.se[i_sim] ** 2)
        )
        self.z = self.diff / self.se_diff if self.se_diff > 0 else np.nan
        self.p_value = float(2 * stats.norm.sf(abs(self.z)))
