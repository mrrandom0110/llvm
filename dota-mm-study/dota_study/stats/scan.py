"""Поиск зависимостей исхода от наблюдаемых признаков.

Сырой винрейт смешивает отбор (в пати чаще заходят сильные) и настоящую
связь. Поэтому у каждого признака две цифры: доля побед как есть и наклон
внутри игрока — тот же человек в двух состояниях.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .streaks import fixed_effects_lpm


def enrich(sample: pd.DataFrame) -> pd.DataFrame:
    df = sample.sort_values(["account_id", "start_time"], kind="stable").copy()
    df["is_radiant"] = (df["player_slot"].fillna(0).astype(int) < 128).astype(int)
    ts = pd.to_datetime(df["start_time"], unit="s", utc=True)
    df["hour_utc"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["year"] = ts.dt.year
    prev_hero = df.groupby("account_id", sort=False)["hero_id"].shift(1)
    df["same_hero"] = (df["hero_id"] == prev_hero).astype("Int64")
    df["in_party"] = np.where(
        df["party_size"].notna(), (df["party_size"] > 1).astype(float), np.nan
    )
    df["late_session"] = (df["session_pos"] >= 5).astype(int)
    df["first_of_session"] = (df["session_pos"] == 0).astype(int)
    hours = df["hours_since_prev"]
    df["long_break"] = np.where(hours.notna(), (hours >= 24).astype(float), np.nan)
    df["short_pause"] = np.where(
        hours.notna() & (df["session_pos"] > 0),
        (hours <= 1).astype(float),
        np.nan,
    )
    df["after_wins"] = (df["prev_streak"] >= 2).astype(int)
    df["after_losses"] = (df["prev_streak"] <= -2).astype(int)
    df["lobby_above"] = np.where(
        df["rank_delta"].notna(), (df["rank_delta"] > 2).astype(float), np.nan
    )
    df["lobby_below"] = np.where(
        df["rank_delta"].notna(), (df["rank_delta"] < -2).astype(float), np.nan
    )
    median_hour = df.groupby("account_id", sort=False)["hour_utc"].transform("median")
    circ = np.minimum(((df["hour_utc"] - median_hour) % 24), ((median_hour - df["hour_utc"]) % 24))
    df["off_hours"] = (circ >= 6).astype(int)
    # Герой, который уже встречался в последних пяти матчах этого игрока.
    recent = (
        df.groupby("account_id", sort=False)["hero_id"]
        .shift(1)
        .to_frame("h1")
        .assign(
            h2=df.groupby("account_id", sort=False)["hero_id"].shift(2),
            h3=df.groupby("account_id", sort=False)["hero_id"].shift(3),
            h4=df.groupby("account_id", sort=False)["hero_id"].shift(4),
            h5=df.groupby("account_id", sort=False)["hero_id"].shift(5),
        )
    )
    df["comfort_hero"] = (
        (df["hero_id"].to_numpy()[:, None] == recent.to_numpy())
        .any(axis=1)
        .astype(int)
    )
    return df


def binary_effect(df: pd.DataFrame, column: str, min_n: int = 20_000) -> dict | None:
    """Сырой разрыв и внутриигровой эффект бинарного признака."""
    data = df.dropna(subset=[column, "win", "account_id"]).copy()
    data[column] = data[column].astype(float)
    if data[column].nunique() < 2 or len(data) < min_n:
        return None
    counts = data[column].value_counts()
    if counts.min() < min_n / 10:
        return None
    raw = data.groupby(column)["win"].agg(["mean", "size"])
    fe = fixed_effects_lpm(
        data["win"].to_numpy(dtype=float),
        np.column_stack([data[column].to_numpy(dtype=float)]),
        data["account_id"].to_numpy(),
        [column],
    )
    lo, hi = fe.ci(column)
    ones = float(raw.loc[1.0, "mean"]) if 1.0 in raw.index else float("nan")
    zeros = float(raw.loc[0.0, "mean"]) if 0.0 in raw.index else float("nan")
    return {
        "n": int(len(data)),
        "n_on": int(counts.get(1.0, 0)),
        "raw_on": ones,
        "raw_off": zeros,
        "raw_diff": ones - zeros,
        "within": float(fe.coef[0]),
        "se": float(fe.se[0]),
        "lo": float(lo),
        "hi": float(hi),
    }


def bins_within(df: pd.DataFrame, column: str, reference: str, min_n: int = 8_000) -> dict:
    """Винрейт по корзинам и внутриигровые отклонения от опорной корзины."""
    data = df.dropna(subset=[column, "win"]).copy()
    data[column] = data[column].astype(str)
    raw = data.groupby(column, observed=True)["win"].agg(["mean", "size"])
    raw = raw[raw["size"] >= min_n]
    if reference not in raw.index or len(raw) < 2:
        return {}
    names = [level for level in raw.index if level != reference]
    X = np.column_stack([(data[column] == level).astype(float) for level in names])
    fe = fixed_effects_lpm(
        data["win"].to_numpy(dtype=float),
        X,
        data["account_id"].to_numpy(),
        names,
    )
    within = {reference: 0.0}
    lo_map = {reference: 0.0}
    hi_map = {reference: 0.0}
    for name in names:
        within[name] = float(fe.coef[fe.names.index(name)])
        a, b = fe.ci(name)
        lo_map[name] = float(a)
        hi_map[name] = float(b)
    ordered = list(raw.index)
    return {
        "levels": ordered,
        "raw": [float(raw.loc[x, "mean"]) for x in ordered],
        "n": [int(raw.loc[x, "size"]) for x in ordered],
        "within": [within[x] for x in ordered],
        "lo": [lo_map[x] for x in ordered],
        "hi": [hi_map[x] for x in ordered],
        "reference": reference,
    }


def session_continue(df: pd.DataFrame) -> dict:
    """Играет ли человек ещё раз в тот же вечер после победы и после поражения."""
    data = df.sort_values(["account_id", "start_time"], kind="stable").copy()
    nxt = data.groupby("account_id", sort=False)["session_id"].shift(-1)
    data["continues"] = (nxt == data["session_id"]).astype(float)
    last_seen = data.groupby("account_id", sort=False)["start_time"].transform("max")
    body = data[data["start_time"] < last_seen]
    grouped = body.groupby("win")["continues"].agg(["mean", "size"])
    if 0 not in grouped.index or 1 not in grouped.index:
        return {}
    return {
        "after_loss": float(grouped.loc[0, "mean"]),
        "after_win": float(grouped.loc[1, "mean"]),
        "n_loss": int(grouped.loc[0, "size"]),
        "n_win": int(grouped.loc[1, "size"]),
        "diff": float(grouped.loc[1, "mean"] - grouped.loc[0, "mean"]),
    }


def session_bins(df: pd.DataFrame) -> pd.Series:
    pos = df["session_pos"]
    return pd.cut(
        pos,
        bins=[-0.1, 0, 1, 2, 4, 20],
        labels=["1-й матч", "2-й", "3-й", "4–5-й", "6+"],
    )


def break_bins(df: pd.DataFrame) -> pd.Series:
    hours = df["hours_since_prev"]
    out = pd.Series(index=df.index, dtype="object")
    out[df["session_pos"] == 0] = "первый / после паузы"
    mid = df["session_pos"] > 0
    out[mid & (hours <= 1)] = "≤1 ч"
    out[mid & (hours > 1) & (hours <= 3)] = "1–3 ч"
    return out
