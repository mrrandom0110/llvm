"""Проверка оценщиков на данных с заранее известным ответом.

Смысл этих тестов не в покрытии кода, а в доверии к числам: каждая статистика
проверяется на синтетических данных, для которых правильный ответ выводится
аналитически. Если оценщик смещён, весь вывод исследования обесценивается.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dota_study import features
from dota_study.stats import bracket, dispersion, hotstreak, queues, roster, streaks, theories


def test_streaks_are_computed_before_the_match() -> None:
    """Серия описывает прошлое, а не текущий матч."""
    df = pd.DataFrame(
        {
            "account_id": [1, 1, 1, 1, 1, 2, 2],
            "start_time": [1, 2, 3, 4, 5, 1, 2],
            "win": [1, 1, 0, 0, 1, 0, 1],
        }
    )
    out = features.add_streaks(df)
    assert out["prev_streak"].tolist() == [0, 1, 2, -1, -2, 0, -1]


def test_signed_prev_streak_is_the_history_before_each_match() -> None:
    """Знаковая серия смотрит только назад и умеет работать пачками."""
    outcomes = np.array([1, 1, 0, 0, 1])
    assert hotstreak.signed_prev_streak(outcomes).tolist() == [0, 1, 2, -1, -2]

    batch = np.array([[1, 1, 0], [0, 0, 1]])
    assert hotstreak.signed_prev_streak(batch).tolist() == [[0, 1, 2], [0, -1, -2]]


def test_permutation_null_excess_vanishes_for_a_fair_coin() -> None:
    """При независимых бросках превышение над перемешиванием около нуля."""
    rng = np.random.default_rng(11)
    n_players, n_games = 80, 180
    account = np.repeat(np.arange(n_players), n_games)
    start_time = np.tile(np.arange(n_games), n_players)
    df = pd.DataFrame(
        {
            "account_id": account,
            "start_time": start_time,
            "win": rng.integers(0, 2, n_players * n_games),
        }
    )
    answer = hotstreak.permutation_null(
        df, max_streak=4, n_permutations=80, min_games=50, seed=3
    )
    mid = answer.as_frame()
    mid = mid[(mid["streak"] >= 1) & (mid["streak"] <= 4)]
    assert abs(float(mid["excess"].mean())) < 0.015


def test_dispersion_detects_pure_binomial() -> None:
    """У однородной популяции коэффициент дисперсии равен единице."""
    rng = np.random.default_rng(1)
    games = np.full(4000, 400)
    wins = rng.binomial(games, 0.5)
    result = dispersion.analyse(wins, games, n_boot=200)
    assert result.phi == pytest.approx(1.0, abs=0.1)
    assert result.phi_lo < 1.0 < result.phi_hi
    assert result.true_sd < 0.005


def test_dispersion_recovers_known_heterogeneity() -> None:
    """При заданном разбросе истинных винрейтов оценщик его восстанавливает."""
    rng = np.random.default_rng(2)
    true_sd = 0.04
    p = np.clip(rng.normal(0.5, true_sd, 4000), 0.05, 0.95)
    games = np.full(4000, 500)
    wins = rng.binomial(games, p)
    result = dispersion.analyse(wins, games, n_boot=200)
    assert result.true_sd == pytest.approx(true_sd, abs=0.006)
    assert result.phi > 1.5
    assert "жёсткая подкрутка отвергается" in result.verdict()


def test_dispersion_detects_forced_fifty_percent() -> None:
    """Принудительное сведение к 50% проявляется как недодисперсия."""
    rng = np.random.default_rng(3)
    n_players, n_games = 1500, 400
    wins = np.empty(n_players, dtype=int)
    for i in range(n_players):
        # Каждое поражение повышает шанс победы и наоборот — ровно то, что
        # приписывают «подкрутке».
        balance, won = 0, 0
        for _ in range(n_games):
            p = 0.5 - 0.05 * np.clip(balance, -5, 5)
            outcome = rng.random() < p
            won += outcome
            balance += 1 if outcome else -1
        wins[i] = won
    result = dispersion.analyse(wins, np.full(n_players, n_games), n_boot=200)
    assert result.phi < 1.0
    assert result.phi_hi < 1.0
    assert result.verdict().startswith("недодисперсия")


def test_fixed_effects_recovers_known_slope() -> None:
    """Регрессия с поглощением эффектов групп восстанавливает заданный наклон."""
    rng = np.random.default_rng(4)
    n_players, per_player = 300, 200
    account = np.repeat(np.arange(n_players), per_player)
    # Большие индивидуальные различия, которые обязаны быть поглощены.
    alpha = rng.normal(0.5, 0.15, n_players)[account]
    x = rng.normal(0.0, 1.0, n_players * per_player)
    true_slope = 0.02
    p = np.clip(alpha + true_slope * x, 0.02, 0.98)
    y = (rng.random(len(p)) < p).astype(float)

    result = streaks.fixed_effects_lpm(y, np.column_stack([x]), account, ["x"])
    assert result.coef[0] == pytest.approx(true_slope, abs=0.004)
    lo, hi = result.ci("x")
    assert lo < true_slope < hi


def test_fixed_effects_absorbs_group_differences() -> None:
    """Различия между игроками не должны просачиваться в оценку наклона.

    Здесь объясняющая переменная скоррелирована с уровнем игрока, но истинного
    эффекта нет. Без фиксированных эффектов оценка была бы сильно смещена.
    """
    rng = np.random.default_rng(5)
    n_players, per_player = 400, 150
    account = np.repeat(np.arange(n_players), per_player)
    level = rng.normal(0.5, 0.1, n_players)
    x = level[account] + rng.normal(0, 0.3, n_players * per_player)
    y = (rng.random(len(x)) < np.clip(level[account], 0.02, 0.98)).astype(float)

    naive = np.polyfit(x, y, 1)[0]
    result = streaks.fixed_effects_lpm(y, np.column_stack([x]), account, ["x"])
    assert abs(naive) > 0.05, "иначе тест не проверяет то, ради чего написан"
    assert result.coef[0] == pytest.approx(0.0, abs=0.01)


def test_clustered_errors_exceed_independent_ones() -> None:
    """Кластеризация обязана расширять интервал при зависимости внутри игрока.

    Зависимость создаётся на уровне сессий: она не поглощается фиксированным
    эффектом игрока и одновременно присутствует в объясняющей переменной и в
    ошибке. Именно в такой ситуации обычные стандартные ошибки занижены, и
    любой шум выглядел бы значимым.
    """
    rng = np.random.default_rng(6)
    n_players, n_sessions, per_session = 200, 20, 15
    account = np.repeat(np.arange(n_players), n_sessions * per_session)
    session_shock = np.repeat(
        rng.normal(0, 1, n_players * n_sessions), per_session
    )
    x = session_shock + rng.normal(0, 0.5, len(account))
    y = 0.3 * session_shock + rng.normal(0, 0.1, len(account))

    clustered = streaks.fixed_effects_lpm(y, np.column_stack([x]), account, ["x"])

    # Обычная гомоскедастичная ошибка той же регрессии для сравнения.
    codes = pd.factorize(account)[0]
    x_w = streaks._within_transform(x, codes, n_players)
    y_w = streaks._within_transform(y, codes, n_players)
    beta = (x_w @ y_w) / (x_w @ x_w)
    resid = y_w - beta * x_w
    sigma2 = resid @ resid / (len(y_w) - n_players - 1)
    plain_se = np.sqrt(sigma2 / (x_w @ x_w))

    assert clustered.se[0] > 1.5 * plain_se


def test_runs_test_flags_alternating_sequences() -> None:
    """Слишком частое чередование исходов даёт положительный z."""
    alternating = pd.DataFrame(
        {"account_id": np.repeat([1, 2], 400), "win": np.tile([0, 1], 400)}
    )
    result = streaks.runs_test(alternating, min_games=100)
    assert result["mean_z"] > 5

    rng = np.random.default_rng(7)
    random_seq = pd.DataFrame(
        {
            "account_id": np.repeat(np.arange(200), 300),
            "win": rng.integers(0, 2, 200 * 300),
        }
    )
    result = streaks.runs_test(random_seq, min_games=100)
    assert abs(result["mean_z"]) < 0.2


def test_asymmetric_slopes_separate_directions() -> None:
    """Разные наклоны для побед и поражений оцениваются раздельно."""
    rng = np.random.default_rng(8)
    rows = []
    for player in range(150):
        for streak in list(range(-4, 0)) + list(range(1, 5)):
            for _ in range(60):
                # Эффект есть только после побед.
                p = 0.5 + (0.01 * streak if streak > 0 else 0.0)
                rows.append(
                    {
                        "account_id": player,
                        "prev_streak": streak,
                        "win": int(rng.random() < p),
                    }
                )
    df = pd.DataFrame(rows)
    _, asym = streaks.asymmetric_slopes(df, controls=False)
    assert asym["difference"] > 0.005
    assert asym["p"] < 0.01


def test_roster_observations_split_teams_correctly() -> None:
    """Союзники и соперники определяются по стороне фокального игрока."""
    roster_df = pd.DataFrame(
        {
            "match_id": [1] * 10,
            "account_id": list(range(10)),
            "player_slot": list(range(5)) + list(range(128, 133)),
            "is_radiant": [1] * 5 + [0] * 5,
            "rank_tier": [50, 52, 54, 56, 58, 60, 62, 64, 66, 68],
        }
    )
    focal = pd.DataFrame(
        {"match_id": [1], "account_id": [0], "prev_streak": [3], "win": [1]}
    )
    obs = roster.build_roster_observations(roster_df, focal)
    assert len(obs) == 1
    row = obs.iloc[0]
    assert row["ally_skill"] == pytest.approx(55.0)   # 52,54,56,58
    assert row["enemy_skill"] == pytest.approx(64.0)  # 60..68
    assert row["delta"] == pytest.approx(-9.0)


def _queue_matches(n_matches: int, assign, seed: int = 1) -> pd.DataFrame:
    """Синтетические матчи: по два игрока выборки на сторону."""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = 1_700_000_000
    for mid in range(n_matches):
        streaks = assign(rng)
        for i, streak in enumerate(streaks):
            rows.append(
                {
                    "match_id": mid,
                    "account_id": mid * 10 + i,
                    "player_slot": i if i < 2 else 128 + (i - 2),
                    "prev_streak": streak,
                    "average_rank": 40,
                    "party_size": 1,
                    "start_time": t0 + mid * 60,
                }
            )
    return pd.DataFrame(rows)


def test_queue_detector_is_quiet_when_streaks_are_independent() -> None:
    """При независимом знаке серии избыток похожести около нуля."""

    def assign(rng):
        return rng.choice([-2, -1, 1, 2], size=4)

    rows = _queue_matches(400, assign)
    answer = queues.permutation_null(rows, n_permutations=80, seed=3)
    assert abs(answer.excess("same_sign_enemy")) < 0.03
    assert abs(answer.excess("same_sign_ally")) < 0.03


def test_queue_detector_finds_a_shared_search_pool() -> None:
    """Если весь матч из одной очереди, соперники тоже имеют тот же знак."""

    def assign(rng):
        sign = rng.choice([-1, 1])
        return [sign * int(rng.integers(1, 4)) for _ in range(4)]

    rows = _queue_matches(300, assign)
    answer = queues.permutation_null(rows, n_permutations=80, seed=4)
    assert answer.observed["same_sign_enemy"] > 0.9
    assert answer.excess("same_sign_enemy") > 0.3


def test_queue_detector_finds_opposing_team_queues() -> None:
    """Если команды из разных очередей, союзники совпадают, соперники — нет."""

    def assign(rng):
        return [2, 3, -2, -1]

    rows = _queue_matches(200, assign)
    answer = queues.permutation_null(rows, n_permutations=60, seed=5)
    assert answer.observed["same_sign_ally"] == pytest.approx(1.0)
    assert answer.observed["same_sign_enemy"] == pytest.approx(0.0)
    assert answer.excess("same_sign_ally") > 0.2
    assert answer.excess("same_sign_enemy") < -0.2


def test_binary_scan_recovers_a_known_side_effect() -> None:
    """Если Radiant выигрывает чаще у того же игрока, сканер это видит."""
    from dota_study.stats import scan

    rng = np.random.default_rng(2)
    rows = []
    for player in range(80):
        for t in range(80):
            radiant = int(rng.random() < 0.5)
            p = 0.47 + 0.08 * radiant
            rows.append(
                {
                    "account_id": player,
                    "win": int(rng.random() < p),
                    "is_radiant": radiant,
                }
            )
    result = scan.binary_effect(pd.DataFrame(rows), "is_radiant", min_n=1000)
    assert result is not None
    assert result["within"] == pytest.approx(0.08, abs=0.03)


def test_mmr_band_accepts_mmr_or_divine_medal() -> None:
    assert bracket.in_mmr_band(4800, 50) is True
    assert bracket.in_mmr_band(3000, 75) is True
    assert bracket.in_mmr_band(3000, 60) is False
    assert bracket.in_mmr_band(None, 76) is True
    assert bracket.in_mmr_band(None, 60) is False


def test_extract_mmr_reads_all_opendota_shapes() -> None:
    assert bracket.extract_mmr({"computed_mmr": 4720.4, "rank_tier": 75}) == (4720.4, 75)
    assert bracket.extract_mmr({"mmr_estimate": {"estimate": 4810}, "rank_tier": 74}) == (
        4810.0,
        74,
    )
    assert bracket.extract_mmr({"solo_competitive_rank": 4900, "rank_tier": 75})[0] == 4900.0
    assert bracket.extract_mmr(None) == (None, None)
    assert bracket.extract_mmr({}) == (None, None)


def test_bracket_skill_splits_obvious_gaps() -> None:
    rows = []
    t0 = 1_700_000_000
    for acc, wr, gpm in ((1, 0.62, 620), (2, 0.38, 380)):
        rng = np.random.default_rng(acc)
        for i in range(40):
            rows.append(
                {
                    "account_id": acc,
                    "match_id": acc * 100 + i,
                    "start_time": t0 + i * 3600,
                    "duration": 2000,
                    "player_slot": 0,
                    "win": int(rng.random() < wr),
                    "lobby_type": 7,
                    "leaver_status": 0,
                    "kills": 8 if wr > 0.5 else 3,
                    "deaths": 4 if wr > 0.5 else 9,
                    "assists": 10,
                    "gold_per_min": gpm,
                    "xp_per_min": gpm + 50,
                }
            )
    skill = bracket.player_skill(pd.DataFrame(rows))
    by_id = skill.set_index("account_id")["group"]
    assert by_id.loc[1] == "сильный"
    assert by_id.loc[2] == "слабый"


def test_party_raises_next_lobby_rank() -> None:
    """В пати лобби выше своего номера — оценщик это ловит."""
    rows = []
    for acc in (1, 2, 3, 4):
        for i in range(40):
            party = i % 2 == 0
            rows.append(
                {
                    "account_id": acc,
                    "match_id": acc * 100 + i,
                    "start_time": 1_700_000_000 + i * 3600,
                    "average_rank": 72.0 + (3.0 if party else 0.0),
                    "rank_delta": 3.0 if party else 0.0,
                    "rank_baseline": 72.0,
                    "party_size": 2 if party else 1,
                    "win": 1,
                    "year": 2024,
                }
            )
    df = pd.DataFrame(rows)
    out = theories.party_lobby_effect(df)
    assert out["within"] == pytest.approx(3.0, abs=0.2)
    assert out["lo"] > 1.0


def test_weakest_three_detects_stacked_side() -> None:
    """Трое слабых на одной стороне дают счёт выше случайной рассадки."""
    # Radiant: 50,51,52,80,81  Dire: 70,71,72,73,74 — трое слабых все на свете.
    ranks = np.array([50, 51, 52, 80, 81, 70, 71, 72, 73, 74], dtype=float)
    radiant = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    observed = theories.weakest_three_together(ranks, radiant)
    assert observed == 1.0
    rng = np.random.default_rng(0)
    nulls = [theories.weakest_three_together(ranks, rng.permutation(radiant)) for _ in range(80)]
    assert observed >= np.quantile(nulls, 0.8)


def test_smurf_pair_excess_finds_isolated_pool() -> None:
    """Смурфы, которые играют только друг с другом, дают положительный избыток."""
    rows = []
    # Два матча: смурфы вместе, жители вместе.
    for match_id, labels, ranks in (
        (1, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [55] * 10),
        (2, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [55] * 10),
        (3, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [55] * 10),
        (4, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [55] * 10),
    ):
        for i, (lab, rank) in enumerate(zip(labels, ranks)):
            rows.append(
                {
                    "match_id": match_id,
                    "account_id": match_id * 10 + i,
                    "is_smurf": lab,
                    "avg_rank_tier": rank,
                    "start_time": 1_700_000_000,
                    "player_slot": i if i < 5 else 128 + i - 5,
                }
            )
    out = theories.smurf_pool_excess(pd.DataFrame(rows), n_perm=40, rng=np.random.default_rng(2))
    assert out["excess"] > 0.1


def test_next_lobby_follows_performance_after_loss() -> None:
    """Красивый слив поднимает следующее лобби — оценщик видит сдвиг."""
    rows = []
    t0 = 1_700_000_000
    for acc in range(8):
        for i in range(30):
            loss = i % 2 == 0
            high = (i % 4 == 0)
            rows.append(
                {
                    "account_id": acc,
                    "match_id": acc * 100 + i,
                    "start_time": t0 + i * 3600,
                    "win": 0 if loss else 1,
                    "perf_index": 1.5 if high else -1.5,
                    "rank_delta": 0.0,
                    "career_pos": i,
                }
            )
    df = pd.DataFrame(rows)
    pieces = []
    for _, grp in df.groupby("account_id"):
        g = grp.sort_values("start_time").copy()
        # Следующее лобби жёстче именно после красивого поражения.
        g["next_rank_delta"] = np.where((g["win"] == 0) & (g["perf_index"] > 0), 4.0, 0.0)
        pieces.append(g)
    out = theories.next_lobby_after_perf(pd.concat(pieces, ignore_index=True), after_win=False)
    assert out["diff"] > 1.0


def test_away_cluster_penalty_is_detected() -> None:
    rows = []
    for acc in (1, 2, 3):
        for i in range(30):
            away = i >= 20
            rows.append(
                {
                    "account_id": acc,
                    "cluster": 2 if away else 1,
                    "win": 0 if away else 1,
                    "rank_delta": 0.0,
                    "start_time": 1_700_000_000 + i,
                }
            )
    out = theories.away_cluster_effect(pd.DataFrame(rows))
    assert out["win_within"] < -0.3


def test_calibration_mobility_sees_later_rank_moves() -> None:
    rows = []
    t0 = 1_700_000_000
    for acc in range(5):
        for i in range(80):
            rank = 50 + (20 if i >= 30 else 0)
            rows.append(
                {
                    "account_id": acc,
                    "start_time": t0 + i * 86400,
                    "average_rank": rank,
                }
            )
    out = theories.calibration_mobility(pd.DataFrame(rows), early=30)
    assert out["share_changed_bracket"] == pytest.approx(1.0)
    assert out["median_abs_move"] >= 3


def test_patch_shift_detects_rank_jump() -> None:
    patch = 1_681_948_800  # 2023-04-20
    rows = []
    for i in range(100):
        t = patch - 10 * 86400 + i * 86400
        rows.append(
            {
                "start_time": t,
                "average_rank": 40.0 if t < patch else 55.0,
                "match_id": i,
                "win": 1,
                "player_slot": 0,
            }
        )
    out = theories.patch_shift(pd.DataFrame(rows), patch_ts=patch, window_days=20)
    assert out["rank_after"] - out["rank_before"] > 10


def test_wilson_interval_covers_true_rate() -> None:
    from dota_study.controls import wilson_interval

    lo, hi = wilson_interval(530, 1000, conf=0.99)
    assert lo < 0.53 < hi
    assert hi - lo < 0.09
    # Крайний случай, на котором наивный нормальный интервал разваливается.
    lo, hi = wilson_interval(0, 50)
    assert lo >= 0.0 and hi > 0.0
