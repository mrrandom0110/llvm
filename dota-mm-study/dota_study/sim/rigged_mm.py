"""Симулятор системы с намеренно внедрённой подкруткой.

Это негативный контроль исследования, и по значимости он равен позитивному.
Позитивные контроли доказывают, что конвейер видит известные эффекты в данных.
Здесь доказывается другое: что **сами тесты способны обнаружить подкрутку**,
если бы она существовала. Без такой проверки вывод «эффекта не найдено» ничего
не стоил бы — он мог бы означать, что тесты слепы.

Внедряются оба механизма, которыми подкрутка вообще может быть реализована:

* через состав — игроку с победной серией достаются более слабые союзники;
* через исход — вероятность победы напрямую смещается против серии.

Прогоняя тесты на этих данных при разной силе вмешательства, получаем порог
обнаружения: какого размера подкрутку исследование заметило бы наверняка.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fair_mm import SimConfig, SimResult, _assign_teams


@dataclass
class RigConfig(SimConfig):
    # Доля матчей, в которых состав перекашивается против игрока с самой
    # длинной победной серией.
    rig_roster_prob: float = 0.0
    # Прямое смещение вероятности победы против серии, в единицах логита на шаг.
    rig_outcome: float = 0.0
    # Серия, начиная с которой вмешательство включается.
    rig_threshold: int = 2
    mechanism: str = "roster"  # roster | outcome | both


def _rigged_mask(
    blocks: np.ndarray, streaks: np.ndarray, apply: np.ndarray
) -> np.ndarray:
    """Состав, перекошенный против игроков с победной серией.

    Блоки отсортированы по рейтингу, поэтому номер столбца — это ранг силы
    внутри матча. Игроку с самой длинной победной серией выдаются четыре самых
    слабых союзника, а все сильные уходят к соперникам.
    """
    n_matches = blocks.shape[0]
    mask = _assign_teams(blocks, "snake", np.random.default_rng(0))
    if not apply.any():
        return mask

    target = np.argmax(streaks, axis=1)
    key = np.tile(np.arange(10), (n_matches, 1)).astype(float)
    key[np.arange(n_matches), target] = -1.0
    weakest_five = np.argsort(key, axis=1)[:, :5]

    rigged = np.zeros((n_matches, 10), dtype=bool)
    np.put_along_axis(rigged, weakest_five, True, axis=1)
    return np.where(apply[:, None], rigged, mask)


def simulate_rigged(config: RigConfig) -> SimResult:
    cfg = config
    rng = np.random.default_rng(cfg.seed)

    true_skill = rng.normal(0.0, cfg.skill_sd, cfg.n_players)
    rating = true_skill + rng.normal(0.0, 0.35, cfg.n_players)
    streak = np.zeros(cfg.n_players, dtype=np.int32)
    log_activity = rng.normal(0.0, cfg.activity_sd, cfg.n_players)

    per_round = int(cfg.n_players * cfg.participation) // 10 * 10
    n_matches_round = per_round // 10
    total = cfg.n_rounds * per_round

    out_player = np.empty(total, dtype=np.int32)
    out_win = np.empty(total, dtype=np.int8)
    out_round = np.empty(total, dtype=np.int32)
    out_match_rating = np.empty(total, dtype=np.float32)
    cursor = 0

    for rnd in range(cfg.n_rounds):
        keys = log_activity + rng.gumbel(0.0, 1.0, cfg.n_players)
        pool = np.argpartition(-keys, per_round - 1)[:per_round]
        queue_key = rating[pool] + rng.normal(0.0, cfg.mm_noise, per_round)
        pool = pool[np.argsort(queue_key, kind="stable")]
        blocks = pool.reshape(n_matches_round, 10)
        block_streaks = streak[blocks]

        if cfg.mechanism in ("roster", "both") and cfg.rig_roster_prob > 0:
            eligible = block_streaks.max(axis=1) >= cfg.rig_threshold
            apply = eligible & (rng.random(n_matches_round) < cfg.rig_roster_prob)
            radiant_mask = _rigged_mask(blocks, block_streaks, apply)
        else:
            radiant_mask = _assign_teams(blocks, cfg.team_assignment, rng)

        skills = true_skill[blocks]
        radiant_skill = np.where(radiant_mask, skills, 0.0).sum(axis=1)
        dire_skill = np.where(~radiant_mask, skills, 0.0).sum(axis=1)
        margin = cfg.beta * (radiant_skill - dire_skill) + cfg.side_bias

        if cfg.mechanism in ("outcome", "both") and cfg.rig_outcome != 0:
            # Прямое смещение исхода против команды, чьи игроки на подъёме.
            capped = np.clip(block_streaks, -5, 5)
            radiant_streak = np.where(radiant_mask, capped, 0).sum(axis=1)
            dire_streak = np.where(~radiant_mask, capped, 0).sum(axis=1)
            margin -= cfg.rig_outcome * (radiant_streak - dire_streak)

        p_radiant = 1.0 / (1.0 + np.exp(-margin))
        radiant_win = rng.random(n_matches_round) < p_radiant
        player_win = np.where(radiant_mask, radiant_win[:, None], ~radiant_win[:, None])

        radiant_rating = np.where(radiant_mask, rating[blocks], 0.0).sum(axis=1)
        dire_rating = np.where(~radiant_mask, rating[blocks], 0.0).sum(axis=1)
        expected = 1.0 / (1.0 + np.exp(-cfg.beta * (radiant_rating - dire_rating)))
        delta_radiant = cfg.k_factor * (radiant_win.astype(float) - expected)
        delta = np.where(radiant_mask, delta_radiant[:, None], -delta_radiant[:, None])
        np.add.at(rating, blocks.ravel(), delta.ravel())

        flat = blocks.ravel()
        wins_flat = player_win.ravel()
        out_player[cursor : cursor + per_round] = flat
        out_win[cursor : cursor + per_round] = wins_flat.astype(np.int8)
        out_round[cursor : cursor + per_round] = rnd
        out_match_rating[cursor : cursor + per_round] = np.repeat(
            (radiant_rating + dire_rating) / 10.0, 10
        )
        cursor += per_round

        # Обновление серий: подкрутка должна опираться на актуальное состояние.
        won = wins_flat
        current = streak[flat]
        streak[flat] = np.where(
            won == 1, np.where(current > 0, current + 1, 1),
            np.where(current < 0, current - 1, -1),
        )

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


def pp_to_logit(pp_per_step: float) -> float:
    """Перевод силы подкрутки из процентных пунктов в логит-единицы.

    Вблизи вероятности 0.5 производная логистической функции равна 0.25,
    поэтому сдвиг логита на x меняет вероятность примерно на x/4.
    """
    return 4.0 * pp_per_step / 100.0


def detection_curve(
    base: RigConfig,
    strengths_pp: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0),
    mechanism: str = "outcome",
    min_games: int = 200,
):
    """Порог обнаружения: как тесты реагируют на подкрутку заданной силы.

    Сила задаётся в процентных пунктах вероятности победы за шаг серии — в тех
    же единицах, в которых измеряется наблюдаемый эффект, чтобы результаты можно
    было сопоставлять напрямую.
    """
    import pandas as pd
    from dataclasses import replace

    from ..features import add_streaks
    from ..stats import dispersion, streaks as streak_stats
    from .fair_mm import to_frame

    rows = []
    for strength in strengths_pp:
        if mechanism == "roster":
            # Для механизма состава сила трактуется как доля матчей, в которых
            # состав перекашивается против игрока на серии.
            cfg = replace(
                base, mechanism="roster", rig_roster_prob=min(strength, 1.0), rig_outcome=0.0
            )
        else:
            cfg = replace(
                base,
                mechanism="outcome",
                rig_outcome=pp_to_logit(strength),
                rig_roster_prob=0.0,
            )
        result = simulate_rigged(cfg)
        counts = np.bincount(result.player_id, minlength=cfg.n_players)
        wins = np.bincount(result.player_id, weights=result.win, minlength=cfg.n_players)
        keep = counts >= min_games
        disp = dispersion.analyse(wins[keep], counts[keep], n_boot=200)
        frame = add_streaks(to_frame(result))
        slope = streak_stats.streak_slope(frame, controls=False)
        runs = streak_stats.runs_test(frame, min_games=min_games)
        rows.append(
            {
                "механизм": mechanism,
                "сила_пп": strength,
                "phi": disp.phi,
                "phi_lo": disp.phi_lo,
                "phi_hi": disp.phi_hi,
                "slope": float(slope.coef[0]),
                "slope_se": float(slope.se[0]),
                "runs_mean_z": runs.get("mean_z", float("nan")),
                "n_players": int(keep.sum()),
            }
        )
    return pd.DataFrame(rows)
