"""Ответ на вопрос «выиграл N подряд — каков шанс на следующую победу».

У вопроса два разных ответа, и путать их нельзя.

1. **Предсказательный.** Если мы видим человека на серии из N побед, какова
   вероятность, что он выиграет следующий матч? Здесь работает и отбор: длинные
   серии чаще случаются у сильных игроков, поэтому увидеть серию из десяти
   побед — само по себе свидетельство, что перед нами хороший игрок.

2. **Причинный.** Для одного и того же человека — меняет ли сам факт серии его
   шансы в следующем матче? Здесь отбор нужно устранить.

Второй ответ получается перестановочным тестом: последовательность исходов
каждого игрока перемешивается, при этом сохраняется его собственный винрейт и
длина истории. Такой нулевой закон автоматически учитывает две вещи, которые
обычно портят подобные подсчёты:

* отбор по силе игрока — у каждого игрока винрейт остаётся своим;
* смещение конечных последовательностей, описанное Миллером и Санджурхо: даже у
  честной монетки доля решек после серии решек в конечной последовательности
  систематически ниже половины, и без поправки это выглядело бы как «подкрутка».
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def run_lengths(outcomes: np.ndarray) -> np.ndarray:
    """Длина серии одинаковых исходов, заканчивающейся в каждой позиции.

    Работает по последней оси, поэтому годится и для одной последовательности,
    и для пачки перемешанных копий сразу.
    """
    arr = np.asarray(outcomes)
    n = arr.shape[-1]
    change = np.empty(arr.shape, dtype=bool)
    change[..., 0] = True
    change[..., 1:] = arr[..., 1:] != arr[..., :-1]
    idx = np.broadcast_to(np.arange(n), arr.shape)
    start = np.maximum.accumulate(np.where(change, idx, 0), axis=-1)
    return idx - start + 1


def signed_prev_streak(outcomes: np.ndarray) -> np.ndarray:
    """Знаковая длина серии, предшествующей каждой позиции."""
    arr = np.asarray(outcomes)
    lengths = run_lengths(arr)
    out = np.zeros(arr.shape, dtype=np.int32)
    prev_len = lengths[..., :-1]
    prev_win = arr[..., :-1]
    out[..., 1:] = np.where(prev_win == 1, prev_len, -prev_len)
    return out


def tally(streaks: np.ndarray, outcomes: np.ndarray, max_streak: int) -> tuple[np.ndarray, np.ndarray]:
    """Суммирует победы и матчи по ячейкам серии.

    Возвращает два массива длиной 2*max_streak+1, индексируемых сдвигом на
    max_streak: позиция i соответствует серии i - max_streak.
    """
    size = 2 * max_streak + 1
    clipped = np.clip(streaks, -max_streak, max_streak) + max_streak
    flat = clipped.ravel()
    wins = np.bincount(flat, weights=outcomes.ravel(), minlength=size)[:size]
    total = np.bincount(flat, minlength=size)[:size]
    return wins, total


@dataclass
class StreakAnswer:
    streak: np.ndarray
    observed_wins: np.ndarray
    observed_n: np.ndarray
    null_mean: np.ndarray
    null_lo: np.ndarray
    null_hi: np.ndarray
    n_permutations: int
    label: str

    @property
    def observed_rate(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.observed_wins / self.observed_n

    @property
    def excess(self) -> np.ndarray:
        return self.observed_rate - self.null_mean

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "streak": self.streak,
                "n": self.observed_n.astype(int),
                "observed": self.observed_rate,
                "null_mean": self.null_mean,
                "null_lo": self.null_lo,
                "null_hi": self.null_hi,
                "excess": self.excess,
            }
        )


def permutation_null(
    df: pd.DataFrame,
    max_streak: int = 10,
    n_permutations: int = 400,
    within: str | None = None,
    seed: int = 20260824,
    min_games: int = 50,
    label: str = "перемешивание истории игрока",
) -> StreakAnswer:
    """Наблюдаемая кривая и её нулевой закон при случайном порядке исходов.

    `within` задаёт блоки, внутри которых разрешено перемешивание. Если указать
    колонку сессии, порядок будет перемешиваться только внутри игровых вечеров,
    а различия между вечерами сохранятся. Сравнение двух таких нулевых законов
    показывает, объясняется ли зависимость от серии просто тем, что бывают
    удачные и неудачные вечера.
    """
    rng = np.random.default_rng(seed)
    size = 2 * max_streak + 1
    obs_wins = np.zeros(size)
    obs_n = np.zeros(size)
    null_wins = np.zeros((n_permutations, size))
    null_n = np.zeros((n_permutations, size))

    ordered = df.sort_values(["account_id", "start_time"], kind="stable")
    for _, group in ordered.groupby("account_id", sort=False):
        outcomes = group["win"].to_numpy(dtype=np.int8)
        if len(outcomes) < min_games:
            continue

        streaks = signed_prev_streak(outcomes)
        w, t = tally(streaks, outcomes.astype(float), max_streak)
        obs_wins += w
        obs_n += t

        permuted = _permute(outcomes, n_permutations, rng, group, within)
        p_streaks = signed_prev_streak(permuted)
        clipped = np.clip(p_streaks, -max_streak, max_streak) + max_streak
        # Одна свёртка на всю пачку вместо цикла по перестановкам: сдвигаем
        # индекс ячейки на номер перестановки и считаем всё разом.
        offset = (np.arange(n_permutations)[:, None] * size + clipped).ravel()
        counts = np.bincount(offset, minlength=n_permutations * size)
        wins_counts = np.bincount(
            offset, weights=permuted.ravel().astype(float), minlength=n_permutations * size
        )
        null_n += counts.reshape(n_permutations, size)
        null_wins += wins_counts.reshape(n_permutations, size)

    with np.errstate(invalid="ignore", divide="ignore"):
        null_rates = null_wins / null_n
    return StreakAnswer(
        streak=np.arange(-max_streak, max_streak + 1),
        observed_wins=obs_wins,
        observed_n=obs_n,
        null_mean=np.nanmean(null_rates, axis=0),
        null_lo=np.nanquantile(null_rates, 0.005, axis=0),
        null_hi=np.nanquantile(null_rates, 0.995, axis=0),
        n_permutations=n_permutations,
        label=label,
    )


def _permute(
    outcomes: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
    group: pd.DataFrame,
    within: str | None,
) -> np.ndarray:
    """Пачка перемешанных копий последовательности исходов."""
    n = len(outcomes)
    if within is None or within not in group:
        keys = rng.random((n_permutations, n))
        order = np.argsort(keys, axis=1)
        return outcomes[order]

    # Перемешивание внутри блоков: сортируем по паре (блок, случайный ключ),
    # тогда элементы не покидают свой блок.
    blocks = pd.factorize(group[within])[0].astype(np.int64)
    keys = rng.random((n_permutations, n))
    order = np.lexsort((keys, np.broadcast_to(blocks, (n_permutations, n))), axis=1)
    return outcomes[order]


def conditional_table(answer: StreakAnswer, min_n: int = 300) -> pd.DataFrame:
    frame = answer.as_frame()
    return frame[frame["n"] >= min_n].reset_index(drop=True)
