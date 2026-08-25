"""Слабые и сильные в одной полосе рейтинга, плюс когда они играют.

Полоса 4600–5000 (или Divine, если MMR не посчитан) — это один «номер».
Сила внутри неё разная: один плюсует и доминирует на героях, другой минусует
и умирает чаще. Дальше сравнивается не медаль, а режим жизни: сколько каток
в неделю, в какие часы, длинные ли вечера.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dota_study.features import ABANDON_STATUS, add_sessions
from dota_study.config import LOBBY_RANKED, MIN_DURATION_SEC

MMR_LO = 4600
MMR_HI = 5000
# Запасной фильтр, если OpenDota не посчитала MMR: Divine 4–9.
TIER_LO = 74
TIER_HI = 79
RECENT_DAYS = 60
MIN_RECENT = 25


def extract_mmr(info: dict | None) -> tuple[float | None, int | None]:
    """MMR и медаль из ответа `/players/{id}`.

    OpenDota то отдаёт `computed_mmr`, то только `mmr_estimate`, то ничего
    кроме `rank_tier`. Берём первое, что есть.
    """
    if not info:
        return None, None
    tier = info.get("rank_tier")
    mmr = info.get("computed_mmr")
    if mmr is None:
        mmr = info.get("solo_competitive_rank") or info.get("competitive_rank")
    if mmr is None:
        mmr = (info.get("mmr_estimate") or {}).get("estimate")
    try:
        mmr_f = float(mmr) if mmr is not None else None
        if mmr_f is not None and not np.isfinite(mmr_f):
            mmr_f = None
    except (TypeError, ValueError):
        mmr_f = None
    try:
        tier_i = int(tier) if tier is not None else None
    except (TypeError, ValueError):
        tier_i = None
    return mmr_f, tier_i


def in_mmr_band(mmr: float | None, rank_tier: int | None) -> bool:
    """Полоса «этот номер»: MMR 4600–5000 или медаль Divine 4–5.

    OpenDota часто рисует Divine 4–5 как 4200–4500, поэтому медаль не
    выкидываем, если оценка MMR чуть ниже нижней границы. Иначе в пуле
    остаются единицы.
    """
    if mmr is not None and np.isfinite(mmr) and MMR_LO <= float(mmr) <= MMR_HI:
        return True
    if rank_tier is None:
        return False
    return TIER_LO <= int(rank_tier) <= TIER_HI


def clean_ranked(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["abandoned"] = out["leaver_status"].fillna(0) >= ABANDON_STATUS
    mask = (
        (out["lobby_type"] == LOBBY_RANKED)
        & (out["duration"].fillna(0) >= MIN_DURATION_SEC)
        & (~out["abandoned"])
        & out["start_time"].notna()
    )
    return out.loc[mask].sort_values(["account_id", "start_time"], kind="stable")


def player_skill(df: pd.DataFrame) -> pd.DataFrame:
    """Оценка силы внутри полосы по недавним ranked-матчам."""
    ranked = add_sessions(clean_ranked(df))
    if ranked.empty:
        return pd.DataFrame()
    horizon = int(ranked["start_time"].max()) - RECENT_DAYS * 86400
    recent = ranked[ranked["start_time"] >= horizon].copy()
    kda = (recent["kills"] + recent["assists"]) / recent["deaths"].clip(lower=1)
    recent["kda"] = kda
    g = recent.groupby("account_id").agg(
        n=("win", "size"),
        winrate=("win", "mean"),
        gpm=("gold_per_min", "mean"),
        xpm=("xp_per_min", "mean"),
        kda=("kda", "mean"),
        deaths=("deaths", "mean"),
        abandon=("abandoned", "mean"),
        last=("start_time", "max"),
        first=("start_time", "min"),
    )
    g = g[g["n"] >= MIN_RECENT].copy()
    if g.empty:
        return g
    g["skill"] = (
        _z(g["winrate"]) + _z(g["gpm"]) + _z(g["xpm"]) + _z(g["kda"]) - _z(g["deaths"])
    ) / 5.0
    q_lo, q_hi = g["skill"].quantile([0.25, 0.75])
    g["group"] = "середина"
    g.loc[g["skill"] <= q_lo, "group"] = "слабый"
    g.loc[g["skill"] >= q_hi, "group"] = "сильный"
    return g.reset_index()


def player_activity(df: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Как часто и в какие часы играет каждый, у кого уже есть группа."""
    ranked = add_sessions(clean_ranked(df))
    if ranked.empty or profiles.empty:
        return pd.DataFrame()
    horizon = int(ranked["start_time"].max()) - RECENT_DAYS * 86400
    recent = ranked[ranked["start_time"] >= horizon].copy()
    ts = pd.to_datetime(recent["start_time"], unit="s", utc=True)
    recent["hour_utc"] = ts.dt.hour
    recent["hour_msk"] = (recent["hour_utc"] + 3) % 24
    recent["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    recent["night_msk"] = recent["hour_msk"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    recent["prime_msk"] = recent["hour_msk"].isin([18, 19, 20, 21, 22]).astype(int)

    weeks = RECENT_DAYS / 7.0
    rows = []
    for account, grp in recent.groupby("account_id"):
        sessions = grp.groupby("session_id").size()
        gaps = grp["start_time"].diff() / 3600.0
        rows.append(
            {
                "account_id": account,
                "games": int(len(grp)),
                "per_week": float(len(grp) / weeks),
                "session_len": float(sessions.mean()),
                "weekend": float(grp["is_weekend"].mean()),
                "night_msk": float(grp["night_msk"].mean()),
                "prime_msk": float(grp["prime_msk"].mean()),
                "mean_hour_msk": float(grp["hour_msk"].mean()),
                "gap_hours": float(gaps.dropna().median()) if gaps.notna().any() else float("nan"),
            }
        )
    act = pd.DataFrame(rows)
    return act.merge(profiles, on="account_id", how="inner")


def group_summary(activity: pd.DataFrame) -> dict:
    """Средние по слабым и сильным плюс разница."""
    out: dict[str, dict] = {}
    metrics = [
        "per_week",
        "session_len",
        "weekend",
        "night_msk",
        "prime_msk",
        "mean_hour_msk",
        "winrate",
        "gpm",
        "kda",
    ]
    for group in ("слабый", "сильный", "середина"):
        sub = activity[activity["group"] == group]
        if sub.empty:
            continue
        info = {"n": int(len(sub))}
        for col in metrics:
            if col not in sub:
                continue
            info[col] = float(sub[col].mean())
        out[group] = info
    if "слабый" in out and "сильный" in out:
        out["разница_сильный_минус_слабый"] = {
            col: out["сильный"][col] - out["слабый"][col]
            for col in metrics
            if col in out["сильный"] and col in out["слабый"]
        }
    return out


def hour_hist(df: pd.DataFrame, profiles: pd.DataFrame) -> dict:
    ranked = clean_ranked(df)
    if ranked.empty:
        return {}
    horizon = int(ranked["start_time"].max()) - RECENT_DAYS * 86400
    recent = ranked[ranked["start_time"] >= horizon].merge(
        profiles[["account_id", "group"]], on="account_id"
    )
    ts = pd.to_datetime(recent["start_time"], unit="s", utc=True)
    recent["hour_msk"] = (ts.dt.hour + 3) % 24
    out = {}
    for group, sub in recent.groupby("group"):
        counts = sub["hour_msk"].value_counts(normalize=True).sort_index()
        out[str(group)] = {
            "hour": [int(h) for h in range(24)],
            "share": [float(counts.get(h, 0.0)) for h in range(24)],
            "n": int(len(sub)),
        }
    return out


def _z(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std
