"""Построение признаков из историй игроков.

Центральный признак — серия, предшествующая матчу. Она считается по полной
хронологической последовательности ranked-матчей игрока, а не по отфильтрованной
выборке: иначе выброшенный матч разрывал бы серию и создавал ложные нули.

Сессия определяется по паузе между матчами. Это нужно тесту на гипотезу H_engage:
если вмешательство привязано к удержанию игрока, эффект должен зависеть от того,
как долго человек уже играет подряд.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from .config import MIN_DURATION_SEC, STUDY_WINDOW_START

SESSION_GAP_SEC = 3 * 3600
ABANDON_STATUS = 2


def load_player_matches(
    conn: sqlite3.Connection,
    ranked_only: bool = True,
    min_duration: int = MIN_DURATION_SEC,
) -> pd.DataFrame:
    where = [f"duration >= {min_duration}"]
    if ranked_only:
        where.append("lobby_type = 7")
    sql = f"SELECT * FROM player_matches WHERE {' AND '.join(where)}"
    df = pd.read_sql_query(sql, conn)
    return df.sort_values(["account_id", "start_time"], kind="stable").reset_index(drop=True)


def add_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """Знаковая длина серии, предшествующей каждому матчу.

    +k означает k побед подряд перед этим матчем, -k означает k поражений.
    Первому матчу игрока присваивается 0.
    """
    df = df.copy()
    out = np.zeros(len(df), dtype=np.int32)
    prev_account = None
    streak = 0
    accounts = df["account_id"].to_numpy()
    wins = df["win"].to_numpy()
    for i in range(len(df)):
        if accounts[i] != prev_account:
            streak = 0
            prev_account = accounts[i]
        out[i] = streak
        if wins[i] == 1:
            streak = streak + 1 if streak > 0 else 1
        else:
            streak = streak - 1 if streak < 0 else -1
    df["prev_streak"] = out
    return df


def add_sessions(df: pd.DataFrame, gap: int = SESSION_GAP_SEC) -> pd.DataFrame:
    """Номер сессии, позиция матча в сессии и её итоговая длина."""
    df = df.copy()
    delta = df.groupby("account_id", sort=False)["start_time"].diff()
    new_session = (delta.isna()) | (delta > gap)
    df["session_id"] = new_session.groupby(df["account_id"], sort=False).cumsum()
    grouped = df.groupby(["account_id", "session_id"], sort=False)
    df["session_pos"] = grouped.cumcount()
    df["session_len"] = grouped["match_id"].transform("size")
    df["hours_since_prev"] = (delta / 3600.0).fillna(np.nan)
    return df


def add_rank_dynamics(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Динамика брекета матча как прокси движения рейтинга.

    Прямого доступа к MMR в API нет, поэтому используется `average_rank` матча.
    Скользящее среднее по предыдущим матчам даёт базовый уровень игрока, а
    отклонение от него — то самое движение рейтинга, которым честная система
    объясняет зависимость исхода от серии.
    """
    df = df.copy()
    grouped = df.groupby("account_id", sort=False)["average_rank"]
    baseline = grouped.transform(
        lambda s: s.shift(1).rolling(window, min_periods=5).mean()
    )
    df["rank_baseline"] = baseline
    df["rank_delta"] = df["average_rank"] - baseline
    df["rank_prev_delta"] = grouped.transform(lambda s: s.diff())
    return df


def add_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Перформанс игрока относительно его брекета и героя.

    Нормировка обязательна: GPM растёт и с рангом, и с длительностью матча, и
    зависит от героя, поэтому сырое значение несопоставимо между игроками.
    """
    df = df.copy()
    df["gpm_norm"] = df["gold_per_min"] / df["duration"].clip(lower=1) * 1800
    df["kda"] = (df["kills"] + df["assists"]) / df["deaths"].clip(lower=1)

    df["bracket"] = (df["average_rank"] // 10).astype("Int64")
    for metric in ("gold_per_min", "xp_per_min", "last_hits", "hero_damage", "kda"):
        key = ["bracket", "hero_id"]
        grouped = df.groupby(key, sort=False)[metric]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        df[f"z_{metric}"] = (df[metric] - mean) / std

    perf_cols = [f"z_{m}" for m in ("gold_per_min", "xp_per_min", "last_hits", "hero_damage", "kda")]
    df["perf_index"] = df[perf_cols].mean(axis=1)
    return df


def build_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """Полный конвейер признаков по ranked-истории."""
    df = load_player_matches(conn)
    df = add_streaks(df)
    df = add_sessions(df)
    df = add_rank_dynamics(df)
    df = add_performance(df)
    df["in_window"] = (
        (df["start_time"] >= STUDY_WINDOW_START) & df["average_rank"].notna()
    )
    df["abandoned"] = (df["leaver_status"].fillna(0) >= ABANDON_STATUS).astype(int)
    return df


def analysis_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Выборка для основных тестов согласно критериям предрегистрации."""
    mask = df["in_window"] & (df["abandoned"] == 0)
    return df.loc[mask].copy()
