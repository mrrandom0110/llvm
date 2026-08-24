"""Тесты C и D: состав команды и разложение эффекта серии.

Тест C — самый прямой из возможных. Если система «наказывает» за победную
серию, наказание должно быть чем-то физически выражено, а единственный рычаг у
матчмейкера — состав матча. Поэтому измеряется разница средней силы союзников и
соперников как функция серии, предшествующей матчу. При честном подборе это
математический ноль при любой серии.

Тест D разделяет два механизма, дающих одинаковую отрицательную автокорреляцию:

* тильт — игрок после серии поражений играет хуже сам;
* подкрутка — игроку выдают худших союзников.

Различить их можно, потому что первый механизм виден в собственном перформансе
игрока, а второй — в силе его команды.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .streaks import fixed_effects_lpm


@dataclass
class RosterObservation:
    match_id: int
    account_id: int
    prev_streak: int
    ally_skill: float
    enemy_skill: float
    delta: float
    n_ally_known: int
    n_enemy_known: int
    anon_share: float


def build_roster_observations(
    roster: pd.DataFrame,
    focal: pd.DataFrame,
    skill_column: str = "rank_tier",
    external_skill: pd.Series | None = None,
    min_known_per_side: int = 2,
) -> pd.DataFrame:
    """Собирает наблюдения «сила союзников против силы соперников».

    `focal` содержит матчи игроков, для которых известна предшествующая серия.
    `external_skill` — необязательная оценка силы по независимой истории игрока;
    она точнее ранга, но доступна только для выгруженной части выборки.
    """
    roster = roster.copy()
    if external_skill is not None:
        roster["skill"] = roster["account_id"].map(external_skill)
        fallback = roster[skill_column].astype(float)
        # Ранг используется там, где собственной истории нет: иначе выборка
        # схлопнулась бы до горстки матчей.
        roster["skill"] = roster["skill"].fillna(_rescale(fallback, roster["skill"]))
    else:
        roster["skill"] = roster[skill_column].astype(float)

    by_match = {mid: grp for mid, grp in roster.groupby("match_id", sort=False)}
    records: list[dict[str, float]] = []

    for row in focal.itertuples(index=False):
        group = by_match.get(row.match_id)
        if group is None:
            continue
        me = group[group["account_id"] == row.account_id]
        if me.empty:
            continue
        is_radiant = bool(me.iloc[0]["is_radiant"])
        same = group[(group["is_radiant"] == is_radiant) & (group["account_id"] != row.account_id)]
        other = group[group["is_radiant"] != is_radiant]

        ally = same["skill"].dropna()
        enemy = other["skill"].dropna()
        if len(ally) < min_known_per_side or len(enemy) < min_known_per_side:
            continue

        all_skill = group["skill"].dropna()
        records.append(
            {
                "match_id": row.match_id,
                "account_id": row.account_id,
                "prev_streak": row.prev_streak,
                "win": row.win,
                "ally_skill": float(ally.mean()),
                "enemy_skill": float(enemy.mean()),
                "delta": float(ally.mean() - enemy.mean()),
                # Разброс силы внутри матча: второй рычаг, которым матчмейкер
                # мог бы вмешиваться, — не перекос между командами, а само
                # расширение окна подбора для игрока на серии.
                "skill_spread": float(all_skill.std(ddof=0)) if len(all_skill) > 2 else float("nan"),
                "n_ally_known": int(len(ally)),
                "n_enemy_known": int(len(enemy)),
                "anon_share": float(group["account_id"].isna().mean()),
            }
        )

    return pd.DataFrame.from_records(records)


def _rescale(source: pd.Series, target: pd.Series) -> pd.Series:
    """Приводит ранг к шкале внешней оценки силы, чтобы их можно было смешивать."""
    valid = target.dropna()
    if valid.empty or source.dropna().empty:
        return source
    src_mean, src_std = source.mean(), source.std(ddof=0)
    if not np.isfinite(src_std) or src_std == 0:
        return pd.Series(np.full(len(source), valid.mean()), index=source.index)
    return (source - src_mean) / src_std * valid.std(ddof=0) + valid.mean()


def test_asymmetry(obs: pd.DataFrame, max_streak: int = 3) -> dict[str, object]:
    """Зависит ли перекос состава от предшествующей серии."""
    if obs.empty:
        return {"n": 0}
    data = obs.copy()
    data["streak_c"] = data["prev_streak"].clip(-max_streak, max_streak)

    grouped = data.groupby("streak_c")["delta"].agg(["mean", "std", "size"])
    grouped["se"] = grouped["std"] / np.sqrt(grouped["size"])

    # Наклон с фиксированными эффектами игрока: сравниваем матчи одного и того
    # же человека между собой, чтобы различия игроков не подменяли эффект серии.
    usable = data[data["account_id"].map(data["account_id"].value_counts()) >= 2]
    slope = None
    if len(usable) > 50:
        X = np.column_stack([usable["streak_c"].to_numpy(dtype=float)])
        slope = fixed_effects_lpm(
            usable["delta"].to_numpy(dtype=float),
            X,
            usable["account_id"].to_numpy(),
            ["streak"],
        )

    overall_mean = float(data["delta"].mean())
    overall_se = float(data["delta"].std(ddof=1) / np.sqrt(len(data)))
    return {
        "n": int(len(data)),
        "n_players": int(data["account_id"].nunique()),
        "mean_delta": overall_mean,
        "se_delta": overall_se,
        "by_streak": grouped,
        "slope": slope,
        "mean_anon_share": float(data["anon_share"].mean()),
    }


def delta_value_in_winrate(obs: pd.DataFrame) -> tuple[float, float]:
    """Во сколько побед обходится единица перекоса состава.

    Наклон перекоса измеряется в единицах ранга, а эффект серии — в вероятности
    победы, и напрямую их не сравнить. Переводной коэффициент оценивается прямо
    по данным: насколько меняется исход матча при изменении перекоса на единицу.
    """
    if obs.empty or "win" not in obs:
        return float("nan"), float("nan")
    usable = obs.dropna(subset=["delta", "win"])
    counts = usable["account_id"].map(usable["account_id"].value_counts())
    usable = usable[counts >= 2]
    if len(usable) < 50:
        return float("nan"), float("nan")
    fe = fixed_effects_lpm(
        usable["win"].to_numpy(dtype=float),
        np.column_stack([usable["delta"].to_numpy(dtype=float)]),
        usable["account_id"].to_numpy(),
        ["delta"],
    )
    return float(fe.coef[0]), float(fe.se[0])


def roster_channel_in_winrate(
    delta_slope: float, delta_slope_se: float, value: float, value_se: float
) -> dict[str, float]:
    """Вклад канала состава в эффект серии, выраженный в вероятности победы.

    Произведение двух оценок; неопределённость считается дельта-методом.
    """
    if not all(np.isfinite([delta_slope, delta_slope_se, value, value_se])):
        return {}
    effect = delta_slope * value
    se = np.sqrt((value * delta_slope_se) ** 2 + (delta_slope * value_se) ** 2)
    return {
        "effect": float(effect),
        "se": float(se),
        "lo": float(effect - 2.576 * se),
        "hi": float(effect + 2.576 * se),
    }


def decompose_streak(
    sample: pd.DataFrame, obs: pd.DataFrame, max_streak: int = 4
) -> dict[str, object]:
    """Разложение эффекта серии на каналы.

    Собственный перформанс считается по всей выборке, поэтому оценивается точно.
    Каналы состава ограничены матчами с выгруженными составами.
    """
    result: dict[str, object] = {}

    own = sample.dropna(subset=["perf_index"])
    if len(own) > 1000:
        streak = own["prev_streak"].clip(-max_streak, max_streak).to_numpy(dtype=float)
        result["own_performance"] = fixed_effects_lpm(
            own["perf_index"].to_numpy(dtype=float),
            np.column_stack([streak]),
            own["account_id"].to_numpy(),
            ["streak"],
        )

    if not obs.empty:
        streak = obs["prev_streak"].clip(-max_streak, max_streak).to_numpy(dtype=float)
        counts = obs["account_id"].map(obs["account_id"].value_counts())
        usable = counts >= 2
        if usable.sum() > 50:
            for channel in ("ally_skill", "enemy_skill", "delta", "skill_spread"):
                if channel not in obs:
                    continue
                values = obs.loc[usable, channel]
                keep = values.notna().to_numpy()
                if keep.sum() < 50:
                    continue
                result[channel] = fixed_effects_lpm(
                    values.to_numpy(dtype=float)[keep],
                    np.column_stack([streak[usable.to_numpy()][keep]]),
                    obs.loc[usable, "account_id"].to_numpy()[keep],
                    ["streak"],
                )
    return result
