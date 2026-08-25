"""Оценщики остальных теорий матчмейкинга.

Каждая функция отвечает на один заранее зафиксированный вопрос из THEORIES.md.
Нуль — тот же человек в другом состоянии или перестановка внутри ячейки.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dota_study.stats.scan import binary_effect
from dota_study.stats.streaks import fixed_effects_lpm

PATCH_733 = 1_681_948_800  # 2023-04-20 00:00 UTC


def _years_with_party_coverage(df: pd.DataFrame, floor: float = 0.5) -> pd.Series:
    ts = pd.to_datetime(df["start_time"], unit="s", utc=True)
    year = ts.dt.year
    coverage = df.groupby(year)["party_size"].apply(lambda s: s.notna().mean())
    return year.isin(coverage[coverage >= floor].index)


def party_lobby_effect(df: pd.DataFrame) -> dict:
    """Насколько лобби в пати выше обычного номера того же человека."""
    data = df.copy()
    if "in_party" not in data:
        data["in_party"] = np.where(
            data["party_size"].notna(), (data["party_size"] > 1).astype(float), np.nan
        )
    if "year" in data.columns:
        # тесты могут передать уже узкий кадр
        pass
    elif "start_time" in data.columns:
        keep = _years_with_party_coverage(data)
        data = data.loc[keep]
    data = data.dropna(subset=["in_party", "rank_delta", "account_id"])
    fe = fixed_effects_lpm(
        data["rank_delta"].to_numpy(dtype=float),
        np.column_stack([data["in_party"].to_numpy(dtype=float)]),
        data["account_id"].to_numpy(),
        ["in_party"],
    )
    lo, hi = fe.ci("in_party")
    return {
        "n": int(len(data)),
        "within": float(fe.coef[0]),
        "se": float(fe.se[0]),
        "lo": float(lo),
        "hi": float(hi),
    }


def weakest_three_together(ranks: np.ndarray, radiant: np.ndarray) -> float:
    """1, если трое самых слабых (меньший rank_tier) на одной стороне."""
    order = np.argsort(ranks)
    weak = radiant[order[:3]]
    return float(weak.min() == weak.max())


def skill_stacking(roster: pd.DataFrame, n_perm: int = 199, rng=None) -> dict:
    """Доля матчей, где трое слабых на одной стороне, против перестановок."""
    rng = rng or np.random.default_rng(7)
    observed = []
    null_draw = []
    used = 0
    for match_id, grp in roster.groupby("match_id"):
        ranks = pd.to_numeric(grp["rank_tier"], errors="coerce")
        ok = ranks.notna()
        if ok.sum() < 8:
            continue
        g = grp.loc[ok]
        ranks_v = ranks.loc[ok].to_numpy(dtype=float)
        if "is_radiant" in g:
            side = g["is_radiant"].to_numpy(dtype=int)
        else:
            side = (g["player_slot"].to_numpy(dtype=int) < 128).astype(int)
        if side.min() == side.max():
            continue
        observed.append(weakest_three_together(ranks_v, side))
        used += 1
        for _ in range(max(n_perm // 10, 5)):
            null_draw.append(weakest_three_together(ranks_v, rng.permutation(side)))
    if not observed:
        return {"n_matches": 0}
    obs = float(np.mean(observed))
    null = np.asarray(null_draw, dtype=float)
    return {
        "n_matches": used,
        "observed": obs,
        "null_mean": float(null.mean()),
        "null_lo": float(np.quantile(null, 0.005)),
        "null_hi": float(np.quantile(null, 0.995)),
        "excess": obs - float(null.mean()),
    }


def smurf_mirror(roster: pd.DataFrame, n_perm: int = 199, rng=None) -> dict:
    """P(смурф у врагов | смурф у нас) против рассадки тех же людей."""
    rng = rng or np.random.default_rng(11)
    if "is_smurf" not in roster.columns:
        return {"n_matches": 0}
    obs_vals = []
    null_vals = []
    used = 0
    for _, grp in roster.groupby("match_id"):
        flag = grp["is_smurf"].fillna(0).to_numpy(dtype=int)
        if flag.sum() == 0:
            continue
        if "is_radiant" in grp:
            side = grp["is_radiant"].to_numpy(dtype=int)
        else:
            side = (grp["player_slot"].to_numpy(dtype=int) < 128).astype(int)
        obs_vals.append(_mirror_rate(flag, side))
        used += 1
        for _ in range(max(n_perm // 15, 4)):
            null_vals.append(_mirror_rate(flag, rng.permutation(side)))
    if not obs_vals:
        return {"n_matches": 0}
    obs = float(np.mean(obs_vals))
    null = np.asarray(null_vals, dtype=float)
    return {
        "n_matches": used,
        "observed": obs,
        "null_mean": float(null.mean()),
        "excess": obs - float(null.mean()),
        "null_lo": float(np.quantile(null, 0.005)),
        "null_hi": float(np.quantile(null, 0.995)),
    }


def _mirror_rate(flag: np.ndarray, side: np.ndarray) -> float:
    rates = []
    for s in (0, 1):
        ours = flag[side == s]
        theirs = flag[side != s]
        if ours.size == 0 or theirs.size == 0:
            continue
        if ours.max() == 0:
            continue
        rates.append(float(theirs.mean()) if ours.sum() else np.nan)
    rates = [r for r in rates if r == r]
    return float(np.mean(rates)) if rates else 0.0


def smurf_pool_excess(roster: pd.DataFrame, n_perm: int = 80, rng=None) -> dict:
    """Избыток пар смурф–смурф относительно перестановки меток в брекете×неделе."""
    rng = rng or np.random.default_rng(3)
    data = roster.copy()
    if "is_smurf" not in data:
        return {"n_pairs": 0}
    if "avg_rank_tier" not in data and "rank_tier" in data:
        data["avg_rank_tier"] = data.groupby("match_id")["rank_tier"].transform("mean")
    ts = pd.to_datetime(data["start_time"], unit="s", utc=True)
    data["week"] = ts.dt.isocalendar().week.astype(int)
    data["year"] = ts.dt.year
    data["bracket"] = (pd.to_numeric(data["avg_rank_tier"], errors="coerce") // 10).astype("Int64")
    data["is_smurf"] = data["is_smurf"].fillna(0).astype(int)

    def pair_rate(frame: pd.DataFrame) -> float:
        rates = []
        for _, grp in frame.groupby("match_id"):
            flags = grp["is_smurf"].to_numpy()
            if len(flags) < 2:
                continue
            # доля партнёров-смурфов у смурфа (союзники и враги)
            if flags.sum() == 0:
                continue
            # все пары
            same = 0
            total = 0
            for i in range(len(flags)):
                if flags[i] != 1:
                    continue
                for j in range(len(flags)):
                    if i == j:
                        continue
                    total += 1
                    same += int(flags[j] == 1)
            if total:
                rates.append(same / total)
        return float(np.mean(rates)) if rates else 0.0

    observed = pair_rate(data)
    nulls = []
    for _ in range(n_perm):
        shuffled = data.copy()
        for _, idx in data.groupby(["bracket", "year", "week"]).groups.items():
            vals = shuffled.loc[idx, "is_smurf"].to_numpy()
            shuffled.loc[idx, "is_smurf"] = rng.permutation(vals)
        nulls.append(pair_rate(shuffled))
    null = np.asarray(nulls, dtype=float)
    return {
        "n_pairs": int(data["is_smurf"].sum()),
        "n_matches": int(data["match_id"].nunique()),
        "observed": observed,
        "null_mean": float(null.mean()) if len(null) else float("nan"),
        "null_lo": float(np.quantile(null, 0.005)) if len(null) else float("nan"),
        "null_hi": float(np.quantile(null, 0.995)) if len(null) else float("nan"),
        "excess": observed - float(null.mean()) if len(null) else float("nan"),
    }


def next_lobby_after_perf(
    df: pd.DataFrame, after_win: bool = False, early: int | None = None
) -> dict:
    """Следующий rank_delta после сильного vs слабого поражения (или победы)."""
    data = df.sort_values(["account_id", "start_time"], kind="stable").copy()
    if "next_rank_delta" not in data:
        data["next_rank_delta"] = data.groupby("account_id", sort=False)["rank_delta"].shift(-1)
    if "career_pos" not in data:
        data["career_pos"] = data.groupby("account_id", sort=False).cumcount()
    want_win = 1 if after_win else 0
    body = data[(data["win"] == want_win) & data["perf_index"].notna() & data["next_rank_delta"].notna()]
    if early is not None:
        body = body[body["career_pos"] < early]
    else:
        body = body[body["career_pos"] >= 20]
    if body.empty:
        return {"n": 0, "diff": float("nan")}
    q_lo, q_hi = body["perf_index"].quantile([0.25, 0.75])
    low = body[body["perf_index"] <= q_lo]["next_rank_delta"]
    high = body[body["perf_index"] >= q_hi]["next_rank_delta"]
    if low.empty or high.empty:
        return {"n": int(len(body)), "diff": float("nan")}
    diff = float(high.mean() - low.mean())
    # грубый ДИ по двум выборкам
    se = np.sqrt(high.var(ddof=1) / len(high) + low.var(ddof=1) / len(low))
    z = 2.576
    return {
        "n": int(len(body)),
        "n_high": int(len(high)),
        "n_low": int(len(low)),
        "high": float(high.mean()),
        "low": float(low.mean()),
        "diff": diff,
        "se": float(se) if se == se else float("nan"),
        "lo": diff - z * float(se) if se == se else float("nan"),
        "hi": diff + z * float(se) if se == se else float("nan"),
        "after_win": after_win,
    }


def lobby_spread_slices(matches: pd.DataFrame, rank_col: str = "average_rank") -> dict:
    """Разброс ранга внутри матча ночью, в выходные и на высоком рейтинге.

    В истории игрока `average_rank` — среднее лобби, поэтому ст. откл. по нему
    ноль. Для настоящего разброса нужна колонка на человека (`rank_tier` состава).
    """
    if rank_col not in matches.columns:
        return {}
    rows = matches.dropna(subset=["match_id", rank_col, "start_time"])
    if rows.empty:
        return {}
    spread = rows.groupby("match_id").agg(
        std=(rank_col, "std"),
        mean=(rank_col, "mean"),
        n=(rank_col, "size"),
        start_time=("start_time", "min"),
    )
    spread = spread[spread["n"] >= 2].dropna(subset=["std"])
    ts = pd.to_datetime(spread["start_time"], unit="s", utc=True)
    spread["hour"] = ts.dt.hour
    spread["weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    spread["night"] = spread["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)
    spread["immortal"] = (spread["mean"] >= 80).astype(int)
    out = {"n_matches": int(len(spread))}
    for name, col in (("night", "night"), ("weekend", "weekend"), ("immortal", "immortal")):
        a = spread.loc[spread[col] == 1, "std"]
        b = spread.loc[spread[col] == 0, "std"]
        # Immortal в выборке составов почти нет — срез из трёх матчей не публикуем.
        if a.empty or b.empty or len(a) < 20 or len(b) < 20:
            continue
        diff = float(a.mean() - b.mean())
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        z = 2.576
        out[name] = {
            "on": float(a.mean()),
            "off": float(b.mean()),
            "diff": diff,
            "lo": diff - z * float(se),
            "hi": diff + z * float(se),
            "n_on": int(len(a)),
            "n_off": int(len(b)),
        }
    return out


def away_cluster_effect(df: pd.DataFrame) -> dict:
    """Победа и лобби у того же человека не на своём сервере."""
    data = df.dropna(subset=["cluster", "account_id"]).copy()
    home = data.groupby("account_id")["cluster"].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
    data = data.join(home.rename("home_cluster"), on="account_id")
    data["away_cluster"] = (data["cluster"] != data["home_cluster"]).astype(float)
    result: dict = {"n": int(len(data)), "share_away": float(data["away_cluster"].mean())}
    if data["away_cluster"].nunique() < 2:
        result["win_within"] = float("nan")
        return result
    if "win" in data:
        win_eff = binary_effect(data, "away_cluster", min_n=20)
        if win_eff:
            result["win_within"] = win_eff["within"]
            result["win_lo"] = win_eff["lo"]
            result["win_hi"] = win_eff["hi"]
            result["n"] = win_eff["n"]
        else:
            # маленькие синтетические выборки: обычный FE
            fe = fixed_effects_lpm(
                data["win"].to_numpy(dtype=float),
                np.column_stack([data["away_cluster"].to_numpy(dtype=float)]),
                data["account_id"].to_numpy(),
                ["away"],
            )
            lo, hi = fe.ci("away")
            result["win_within"] = float(fe.coef[0])
            result["win_lo"] = float(lo)
            result["win_hi"] = float(hi)
    if "rank_delta" in data and data["rank_delta"].notna().any():
        body = data.dropna(subset=["rank_delta"])
        if body["away_cluster"].nunique() > 1 and body["account_id"].nunique() > 1:
            fe = fixed_effects_lpm(
                body["rank_delta"].to_numpy(dtype=float),
                np.column_stack([body["away_cluster"].to_numpy(dtype=float)]),
                body["account_id"].to_numpy(),
                ["away"],
            )
            lo, hi = fe.ci("away")
            result["lobby_within"] = float(fe.coef[0])
            result["lobby_lo"] = float(lo)
            result["lobby_hi"] = float(hi)
    return result


def calibration_mobility(df: pd.DataFrame, early: int = 30) -> dict:
    """Двигается ли номер после первых ranked-матчей."""
    data = df.dropna(subset=["account_id", "average_rank", "start_time"]).copy()
    data = data.sort_values(["account_id", "start_time"], kind="stable")
    data["pos"] = data.groupby("account_id", sort=False).cumcount()
    later = data[data["pos"] >= early]
    if later.empty:
        return {"n_players": 0}
    early_rank = (
        data[data["pos"] < early].groupby("account_id")["average_rank"].mean()
    )
    later_mean = later.groupby("account_id")["average_rank"].mean()
    both = pd.DataFrame({"early": early_rank, "later": later_mean}).dropna()
    if both.empty:
        return {"n_players": 0}
    move = both["later"] - both["early"]
    changed = (both["later"] // 10).astype(int) != (both["early"] // 10).astype(int)
    horizon = data.groupby("account_id")["start_time"].agg(["min", "max"])
    span_days = (horizon["max"] - horizon["min"]) / 86400.0
    return {
        "n_players": int(len(both)),
        "share_changed_bracket": float(changed.mean()),
        "median_abs_move": float(move.abs().median()),
        "mean_move": float(move.mean()),
        "median_span_days": float(span_days.reindex(both.index).median()),
    }


def returning_swing(df: pd.DataFrame, gap_days: float = 30.0) -> dict:
    """Скачок лобби сразу после длинной паузы."""
    data = df.sort_values(["account_id", "start_time"], kind="stable").copy()
    if "hours_since_prev" not in data:
        delta = data.groupby("account_id", sort=False)["start_time"].diff()
        data["hours_since_prev"] = delta / 3600.0
    body = data.dropna(subset=["rank_delta", "hours_since_prev"])
    after = body.loc[body["hours_since_prev"] >= gap_days * 24, "rank_delta"].abs()
    normal = body.loc[body["hours_since_prev"] < 24, "rank_delta"].abs()
    if after.empty or normal.empty:
        return {"n_after": int(after.size), "n_normal": int(normal.size)}
    diff = float(after.mean() - normal.mean())
    se = np.sqrt(after.var(ddof=1) / len(after) + normal.var(ddof=1) / len(normal))
    z = 2.576
    return {
        "n_after": int(len(after)),
        "n_normal": int(len(normal)),
        "after": float(after.mean()),
        "normal": float(normal.mean()),
        "diff": diff,
        "lo": diff - z * float(se),
        "hi": diff + z * float(se),
    }


def patch_shift(df: pd.DataFrame, patch_ts: int = PATCH_733, window_days: int = 30) -> dict:
    """Сдвиг среднего average_rank и разброса лобби вокруг патча."""
    lo = patch_ts - window_days * 86400
    hi = patch_ts + window_days * 86400
    window = df[(df["start_time"] >= lo) & (df["start_time"] < hi)].copy()
    if window.empty:
        return {"n": 0}
    before = window[window["start_time"] < patch_ts]
    after = window[window["start_time"] >= patch_ts]
    out = {
        "n_before": int(len(before)),
        "n_after": int(len(after)),
        "rank_before": float(before["average_rank"].mean()) if len(before) else float("nan"),
        "rank_after": float(after["average_rank"].mean()) if len(after) else float("nan"),
    }
    if "match_id" in window:
        def _spread(part):
            g = part.dropna(subset=["average_rank"]).groupby("match_id")["average_rank"].std()
            g = g[part.groupby("match_id").size() >= 2]
            return float(g.mean()) if len(g) else float("nan")

        out["spread_before"] = _spread(before)
        out["spread_after"] = _spread(after)
    if "win" in window and "player_slot" in window:
        rad_b = before[before["player_slot"] < 128]["win"].mean() if len(before) else float("nan")
        rad_a = after[after["player_slot"] < 128]["win"].mean() if len(after) else float("nan")
        out["radiant_before"] = float(rad_b) if rad_b == rad_b else float("nan")
        out["radiant_after"] = float(rad_a) if rad_a == rad_a else float("nan")
    return out


def medal_lobby_gap(df: pd.DataFrame, recent: int = 20) -> dict:
    """Расхождение длинного базового номера и среднего лобби последних матчей."""
    data = df.dropna(subset=["average_rank"]).copy()
    data = data.sort_values(["account_id", "start_time"], kind="stable")
    rows = []
    for acc, grp in data.groupby("account_id"):
        if len(grp) < recent + 10:
            continue
        base = float(grp["average_rank"].iloc[:-recent].mean())
        last = float(grp["average_rank"].iloc[-recent:].mean())
        rows.append({"account_id": acc, "gap": last - base})
    if not rows:
        return {"n": 0}
    gaps = pd.Series([r["gap"] for r in rows])
    return {
        "n": int(len(gaps)),
        "median_gap": float(gaps.median()),
        "mean_abs_gap": float(gaps.abs().mean()),
        "share_abs_ge_3": float((gaps.abs() >= 3).mean()),
    }


def off_role_effect(df: pd.DataFrame, support_heroes: set[int]) -> dict:
    """Чужая роль: победа и лобби у того же человека."""
    data = df.dropna(subset=["hero_id", "account_id"]).copy()
    data["is_support"] = data["hero_id"].isin(support_heroes).astype(int)
    home = data.groupby("account_id")["is_support"].mean()
    main_support = home >= 0.5
    data = data.join(main_support.rename("main_support"), on="account_id")
    data["off_role"] = (data["is_support"] != data["main_support"].astype(int)).astype(float)
    out: dict = {"n": int(len(data)), "share_off": float(data["off_role"].mean())}
    if data["off_role"].nunique() < 2:
        return out
    if "win" in data:
        fe = fixed_effects_lpm(
            data["win"].to_numpy(dtype=float),
            np.column_stack([data["off_role"].to_numpy(dtype=float)]),
            data["account_id"].to_numpy(),
            ["off"],
        )
        lo, hi = fe.ci("off")
        out["win_within"] = float(fe.coef[0])
        out["win_lo"] = float(lo)
        out["win_hi"] = float(hi)
    body = data.dropna(subset=["rank_delta"]) if "rank_delta" in data else None
    if body is not None and len(body) and body["off_role"].nunique() > 1:
        fe = fixed_effects_lpm(
            body["rank_delta"].to_numpy(dtype=float),
            np.column_stack([body["off_role"].to_numpy(dtype=float)]),
            body["account_id"].to_numpy(),
            ["off"],
        )
        lo, hi = fe.ci("off")
        out["lobby_within"] = float(fe.coef[0])
        out["lobby_lo"] = float(lo)
        out["lobby_hi"] = float(hi)
    return out


def mmr_explains_lobby(snapshot: pd.DataFrame) -> dict:
    """Насколько снимок MMR/медали предсказывает среднее лобби последних матчей."""
    data = snapshot.dropna(subset=["lobby_rank"]).copy()
    if len(data) < 8:
        return {"n": int(len(data))}
    out = {"n": int(len(data))}
    if "computed_mmr" in data:
        mmr_rows = data.dropna(subset=["computed_mmr"])
        if len(mmr_rows) >= 8:
            mmr = mmr_rows["computed_mmr"].to_numpy(dtype=float)
            y_mmr = mmr_rows["lobby_rank"].to_numpy(dtype=float)
            out["corr"] = float(np.corrcoef(mmr, y_mmr)[0, 1])
            x = (mmr - mmr.mean()) / (mmr.std() or 1)
            coef = np.polyfit(x, y_mmr, 1)
            resid = y_mmr - np.polyval(coef, x)
            out["slope"] = float(coef[0])
            out["resid_std"] = float(resid.std())
            out["n"] = int(len(mmr_rows))
    if "rank_tier" in data:
        medal_rows = data.dropna(subset=["rank_tier"])
        if len(medal_rows) >= 8:
            out["corr_medal"] = float(
                np.corrcoef(
                    medal_rows["rank_tier"].to_numpy(dtype=float),
                    medal_rows["lobby_rank"].to_numpy(dtype=float),
                )[0, 1]
            )
    return out
