"""Симулятор честного матчмейкинга.

Это нулевая модель всего исследования. Система по построению **не содержит
никакой подкрутки**: подбор смотрит исключительно на рейтинг, исход разыгрывается
только силой команд. Тем не менее в такой системе винрейты сходятся к 50%, а у
исходов появляется отрицательная автокорреляция — просто потому, что после
победы рейтинг вырос и соперник стал сильнее.

Именно поэтому наблюдаемые в реальных данных значения нужно сравнивать не с
нулём, а с распределением тех же статистик здесь.

Реализация векторизована: за один раунд формируются и разыгрываются все матчи
пула сразу, что позволяет за секунды получать миллионы матчей.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimConfig:
    n_players: int = 20_000
    n_rounds: int = 600
    skill_sd: float = 1.0
    # Насколько разница сил команд превращается в вероятность победы.
    # Подобрано так, чтобы разброс винрейтов был реалистичным.
    beta: float = 0.9
    # Шаг обновления рейтинга. В Dota 2 это около 25-30 MMR за игру.
    k_factor: float = 0.030
    # Ширина окна подбора: шум, добавляемый к рейтингу при сортировке очереди.
    mm_noise: float = 0.05
    # Дрейф истинного навыка за раунд: обучение и деградация игроков.
    skill_drift: float = 0.004
    # Доля игроков, стартующих с рейтингом сильно ниже своей истинной силы.
    smurf_fraction: float = 0.0
    smurf_gap: float = 2.0
    # Постоянное преимущество стороны Radiant, чтобы симуляция была сопоставима
    # с реальными данными, где оно составляет около трёх процентных пунктов.
    side_bias: float = 0.12
    # Доля игроков, участвующих в раунде. Строго меньше единицы, иначе разброс
    # активности не проявляется: при полном участии у всех одинаковая история.
    participation: float = 0.35
    # Разброс игровой активности. Без него у всех была бы одинаковая длина
    # истории, тогда как в реальности она различается на порядки, а от неё
    # зависит и точность рейтинга, и биномиальный шум винрейта.
    activity_sd: float = 0.9
    seed: int = 20260824
    # Насколько подбор балансирует команды: 'snake' раскладывает отсортированных
    # по рейтингу игроков так, чтобы суммы были близки, 'random' — случайно.
    team_assignment: str = "snake"


@dataclass
class SimResult:
    player_id: np.ndarray
    win: np.ndarray
    round_id: np.ndarray
    match_rating: np.ndarray
    rating: np.ndarray
    true_skill: np.ndarray
    config: SimConfig = field(repr=False, default_factory=SimConfig)

    @property
    def n_matches(self) -> int:
        return len(self.player_id) // 10


def _assign_teams(order: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Раскладка отсортированных по рейтингу игроков на две команды.

    `order` имеет форму (n_matches, 10). Возвращает булеву маску Radiant.
    """
    n_matches = order.shape[0]
    if mode == "random":
        mask = np.zeros((n_matches, 10), dtype=bool)
        for i in range(n_matches):
            mask[i, rng.choice(10, 5, replace=False)] = True
        return mask
    # «Змейка»: позиции 0,3,4,7,8 против 1,2,5,6,9 дают почти равные суммы
    # рейтингов, что имитирует балансировку команд матчмейкером.
    pattern = np.zeros(10, dtype=bool)
    pattern[[0, 3, 4, 7, 8]] = True
    return np.broadcast_to(pattern, (n_matches, 10)).copy()


def simulate(config: SimConfig | None = None) -> SimResult:
    cfg = config or SimConfig()
    rng = np.random.default_rng(cfg.seed)

    true_skill = rng.normal(0.0, cfg.skill_sd, cfg.n_players)
    rating = true_skill + rng.normal(0.0, 0.35, cfg.n_players)

    if cfg.smurf_fraction > 0:
        n_smurf = int(cfg.n_players * cfg.smurf_fraction)
        smurfs = rng.choice(cfg.n_players, n_smurf, replace=False)
        rating[smurfs] -= cfg.smurf_gap

    per_round = int(cfg.n_players * cfg.participation) // 10 * 10
    n_matches_round = per_round // 10
    total = cfg.n_rounds * per_round

    # Логарифм веса активности: кто-то играет каждый вечер, кто-то раз в месяц.
    log_activity = rng.normal(0.0, cfg.activity_sd, cfg.n_players)

    out_player = np.empty(total, dtype=np.int32)
    out_win = np.empty(total, dtype=np.int8)
    out_round = np.empty(total, dtype=np.int32)
    out_match_rating = np.empty(total, dtype=np.float32)
    cursor = 0

    for rnd in range(cfg.n_rounds):
        # Взвешенный отбор без возвращения через трюк Гумбеля: прибавляем к
        # логарифму веса гумбелев шум и берём верхние k. Это точная выборка и
        # работает за один проход, в отличие от последовательного отбора.
        keys = log_activity + rng.gumbel(0.0, 1.0, cfg.n_players)
        pool = np.argpartition(-keys, per_round - 1)[:per_round]

        # Подбор смотрит только на рейтинг: сортируем очередь по зашумлённому
        # рейтингу и нарезаем подряд идущими десятками.
        queue_key = rating[pool] + rng.normal(0.0, cfg.mm_noise, per_round)
        pool = pool[np.argsort(queue_key, kind="stable")]
        blocks = pool.reshape(n_matches_round, 10)

        radiant_mask = _assign_teams(blocks, cfg.team_assignment, rng)
        skills = true_skill[blocks]
        radiant_skill = np.where(radiant_mask, skills, 0.0).sum(axis=1)
        dire_skill = np.where(~radiant_mask, skills, 0.0).sum(axis=1)

        margin = cfg.beta * (radiant_skill - dire_skill) + cfg.side_bias
        p_radiant = 1.0 / (1.0 + np.exp(-margin))
        radiant_win = rng.random(n_matches_round) < p_radiant

        player_win = np.where(
            radiant_mask, radiant_win[:, None], ~radiant_win[:, None]
        )

        # Обновление рейтинга по Elo: ожидание считается по рейтингам команд,
        # то есть системе доступен только рейтинг, но не истинная сила.
        radiant_rating = np.where(radiant_mask, rating[blocks], 0.0).sum(axis=1)
        dire_rating = np.where(~radiant_mask, rating[blocks], 0.0).sum(axis=1)
        expected_radiant = 1.0 / (1.0 + np.exp(-cfg.beta * (radiant_rating - dire_rating)))
        delta_radiant = cfg.k_factor * (radiant_win.astype(float) - expected_radiant)
        delta = np.where(radiant_mask, delta_radiant[:, None], -delta_radiant[:, None])
        np.add.at(rating, blocks.ravel(), delta.ravel())

        # Средний рейтинг матча — прямой аналог поля average_rank в реальных
        # данных, поэтому на нём можно калибровать симулятор.
        match_rating = (radiant_rating + dire_rating) / 10.0

        flat = blocks.ravel()
        out_player[cursor : cursor + per_round] = flat
        out_win[cursor : cursor + per_round] = player_win.ravel().astype(np.int8)
        out_round[cursor : cursor + per_round] = rnd
        out_match_rating[cursor : cursor + per_round] = np.repeat(match_rating, 10)
        cursor += per_round

        if cfg.skill_drift:
            true_skill += rng.normal(0.0, cfg.skill_drift, cfg.n_players)

    return SimResult(
        player_id=out_player[:cursor],
        win=out_win[:cursor],
        round_id=out_round[:cursor],
        match_rating=out_match_rating[:cursor],
        rating=rating,
        true_skill=true_skill,
        config=cfg,
    )


def to_frame(result: SimResult):
    """Приводит результат симуляции к тому же виду, что и реальные данные."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "account_id": result.player_id.astype(np.int64),
            "win": result.win.astype(np.int64),
            "start_time": result.round_id.astype(np.int64) * 3600,
            "average_rank": result.match_rating.astype(np.float64),
        }
    )
    return df.sort_values(["account_id", "start_time"], kind="stable").reset_index(drop=True)
