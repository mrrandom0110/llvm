"""Проверка легенды о раздельных очередях поиска (win queue / lose queue).

Народная формулировка: после победной серии вас кладут в одну очередь поиска,
после проигрышной — в другую. У этого утверждения два проверяемых варианта.

* Одна очередь на весь матч. Все десять игроков взяты из одного пула, значит
  и союзники, и соперники чаще имеют тот же знак серии, чем при подборе
  только по рейтингу.
* Очередь на команду. Вашу пятёрку набрали из одной очереди, вражескую — из
  другой. Тогда союзники похожи на вас по серии, а соперники — наоборот.

Честный подбор по рейтингу тоже даёт слабую похожесть серий: сильные игроки
чаще бывают на победной серии, а в один матч попадают близкие по рейтингу.
Поэтому наблюдение сравнивается не с 50%, а с перестановочным нулём: серии
перемешиваются внутри брекета и недели, а кто с кем попал в матч не меняется.
Остаётся ровно тот избыток похожести, который мог бы дать отдельный пул.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def add_side(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_radiant"] = (df["player_slot"].fillna(0).astype(int) < 128).astype(int)
    return df


def shared_match_rows(sample: pd.DataFrame, min_players: int = 2) -> pd.DataFrame:
    """Строки выборки из матчей, где есть хотя бы двое игроков с историей."""
    counts = sample.groupby("match_id").size()
    keep = counts[counts >= min_players].index
    out = add_side(sample.loc[sample["match_id"].isin(keep)].copy())
    out = out[out["prev_streak"] != 0]
    return out


def directed_pairs(rows: pd.DataFrame) -> pd.DataFrame:
    """Каждая упорядоченная пара игроков выборки из одного матча."""
    cols = [
        "match_id",
        "account_id",
        "prev_streak",
        "is_radiant",
        "average_rank",
        "party_size",
        "start_time",
    ]
    missing = [c for c in cols if c not in rows.columns]
    if missing:
        raise KeyError(f"в выборке нет колонок {missing}")
    slim = rows[cols]
    left = slim.rename(columns={c: f"{c}_a" for c in cols if c != "match_id"})
    right = slim.rename(columns={c: f"{c}_b" for c in cols if c != "match_id"})
    pairs = left.merge(right, on="match_id")
    pairs = pairs[pairs["account_id_a"] != pairs["account_id_b"]]
    pairs["same_team"] = pairs["is_radiant_a"] == pairs["is_radiant_b"]
    pairs["same_sign"] = np.sign(pairs["prev_streak_a"]) == np.sign(pairs["prev_streak_b"])
    return pairs.reset_index(drop=True)


@dataclass
class QueueAnswer:
    n_matches: int
    n_pairs: int
    n_ally: int
    n_enemy: int
    observed: dict[str, float]
    null_mean: dict[str, float]
    null_lo: dict[str, float]
    null_hi: dict[str, float]
    n_permutations: int
    by_streak: dict[str, list]

    def excess(self, key: str) -> float:
        return self.observed[key] - self.null_mean[key]


def attach_row_index(rows: pd.DataFrame) -> pd.DataFrame:
    """Номер строки нужен, чтобы подставлять перемешанные серии в уже собранные пары."""
    out = rows.reset_index(drop=True).copy()
    out["_row"] = np.arange(len(out))
    return out


def _cells(rows: pd.DataFrame) -> dict[tuple, np.ndarray]:
    week = (rows["start_time"].to_numpy() // (7 * 86400)).astype(np.int64)
    bracket = (rows["average_rank"].to_numpy() // 10).astype(np.int64)
    cells: dict[tuple, list[int]] = {}
    for i, key in enumerate(zip(bracket.tolist(), week.tolist())):
        cells.setdefault(key, []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in cells.items()}


def _summarize(pairs: pd.DataFrame) -> dict[str, float]:
    ally = pairs[pairs["same_team"]]
    enemy = pairs[~pairs["same_team"]]
    return {
        "same_sign": float(pairs["same_sign"].mean()),
        "same_sign_ally": float(ally["same_sign"].mean()) if len(ally) else float("nan"),
        "same_sign_enemy": float(enemy["same_sign"].mean()) if len(enemy) else float("nan"),
        "corr": _corr(pairs["prev_streak_a"], pairs["prev_streak_b"]),
        "corr_ally": _corr(ally["prev_streak_a"], ally["prev_streak_b"]),
        "corr_enemy": _corr(enemy["prev_streak_a"], enemy["prev_streak_b"]),
    }


def _corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 20:
        return float("nan")
    left = a.clip(-5, 5).to_numpy(dtype=float)
    right = b.clip(-5, 5).to_numpy(dtype=float)
    if left.std() == 0 or right.std() == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _partner_curve(pairs: pd.DataFrame, max_streak: int = 4) -> dict[str, list]:
    """Средняя серия партнёра как функция собственной серии."""
    data = pairs.copy()
    data["own"] = data["prev_streak_a"].clip(-max_streak, max_streak)
    out: dict[str, list] = {"streak": [], "ally": [], "enemy": [], "n_ally": [], "n_enemy": []}
    for streak, group in data.groupby("own"):
        ally = group[group["same_team"]]["prev_streak_b"]
        enemy = group[~group["same_team"]]["prev_streak_b"]
        out["streak"].append(int(streak))
        out["ally"].append(float(ally.mean()) if len(ally) else float("nan"))
        out["enemy"].append(float(enemy.mean()) if len(enemy) else float("nan"))
        out["n_ally"].append(int(len(ally)))
        out["n_enemy"].append(int(len(enemy)))
    return out


def prepare(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Готовит строки и пары с индексом для быстрых перестановок."""
    if "is_radiant" not in rows.columns:
        rows = add_side(rows)
    indexed = attach_row_index(rows)
    left = indexed.rename(columns={c: f"{c}_a" for c in indexed.columns if c != "match_id"})
    right = indexed.rename(columns={c: f"{c}_b" for c in indexed.columns if c != "match_id"})
    pairs = left.merge(right, on="match_id")
    pairs = pairs[pairs["account_id_a"] != pairs["account_id_b"]].copy()
    pairs["same_team"] = pairs["is_radiant_a"] == pairs["is_radiant_b"]
    pairs["same_sign"] = np.sign(pairs["prev_streak_a"]) == np.sign(pairs["prev_streak_b"])
    return indexed, pairs.reset_index(drop=True)


def permutation_null(
    rows: pd.DataFrame,
    n_permutations: int = 300,
    seed: int = 20260824,
) -> QueueAnswer:
    """Наблюдаемая похожесть серий против подбора только по рейтингу."""
    indexed, pairs = prepare(rows)
    if pairs.empty:
        raise ValueError("нет пар игроков в одних матчах")
    observed = _summarize(pairs)
    by_streak = _partner_curve(pairs)
    cells = _cells(indexed)
    rng = np.random.default_rng(seed)
    keys = list(observed)
    null = {key: np.empty(n_permutations) for key in keys}
    streaks = indexed["prev_streak"].to_numpy().copy()

    for i in range(n_permutations):
        shuffled = streaks.copy()
        for members in cells.values():
            if len(members) < 2:
                continue
            shuffled[members] = rng.permutation(shuffled[members])
        trial = pairs.copy()
        trial["prev_streak_a"] = shuffled[pairs["_row_a"].to_numpy()]
        trial["prev_streak_b"] = shuffled[pairs["_row_b"].to_numpy()]
        trial["same_sign"] = np.sign(trial["prev_streak_a"]) == np.sign(
            trial["prev_streak_b"]
        )
        stats = _summarize(trial)
        for key in keys:
            null[key][i] = stats[key]

    return QueueAnswer(
        n_matches=int(indexed["match_id"].nunique()),
        n_pairs=int(len(pairs)),
        n_ally=int(pairs["same_team"].sum()),
        n_enemy=int((~pairs["same_team"]).sum()),
        observed=observed,
        null_mean={k: float(np.nanmean(v)) for k, v in null.items()},
        null_lo={k: float(np.nanquantile(v, 0.005)) for k, v in null.items()},
        null_hi={k: float(np.nanquantile(v, 0.995)) for k, v in null.items()},
        n_permutations=n_permutations,
        by_streak=by_streak,
    )
