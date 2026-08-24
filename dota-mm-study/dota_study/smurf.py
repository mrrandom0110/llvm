"""Детекторы смурфов, бустед-аккаунтов и слабых игроков.

Зачем это нужно основному вопросу исследования. Смурф — главный конфаундер:
он даёт настоящие 70% побед, раздувает хвосты распределения винрейтов и создаёт
у соседей по матчу то самое ощущение «несбалансированных команд», которое и
породило легенду о подкрутке. Пока смурфы не выделены, невозможно сказать,
объясняется ли сверхдисперсия реальными различиями в силе или дефектом выборки.

Размеченных данных не существует, поэтому используется скоринг по признакам с
последующей полу-размеченной валидацией: за заведомых смурфов принимаются
молодые аккаунты, добравшиеся до Divine и выше, за заведомых «жителей рейтинга» —
аккаунты с многолетней историей и стабильным брекетом. Ни та, ни другая группа
не используется при построении score, только для проверки его разделяющей силы.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MONTH = 30 * 86400
DIVINE_TIER = 70


def estimate_registration_time(df: pd.DataFrame) -> pd.Series:
    """Оценка даты регистрации аккаунта по его номеру.

    Steam выдаёт account_id почти монотонно по времени, поэтому номер аккаунта
    сам по себе датирует регистрацию. Калибровка строится по нашим же данным:
    для каждого аккаунта известен его первый матч, и минимальное время первого
    матча среди всех аккаунтов с номером не больше данного даёт нижнюю границу
    даты регистрации. Кумулятивный минимум делает оценку монотонной по номеру.
    """
    ordered = df.sort_values("account_id")
    running_min = ordered["first_match"].cummin()
    return running_min.reindex(df.index)


def _safe_z(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def _rank_slope(group: pd.DataFrame) -> float:
    """Скорость подъёма по рангам в единицах ранга за месяц."""
    valid = group.dropna(subset=["average_rank", "start_time"])
    if len(valid) < 30:
        return np.nan
    x = valid["start_time"].to_numpy(dtype=float) / MONTH
    y = valid["average_rank"].to_numpy(dtype=float)
    if x.max() - x.min() < 0.5:
        return np.nan
    slope = np.polyfit(x - x.mean(), y, 1)[0]
    return float(slope)


def build_profiles(matches: pd.DataFrame, first_n: int = 50) -> pd.DataFrame:
    """Сводка по каждому игроку: динамика ранга, перформанс, ранняя доминантность."""
    df = matches.sort_values(["account_id", "start_time"], kind="stable")

    agg = df.groupby("account_id").agg(
        n_ranked=("match_id", "size"),
        wins=("win", "sum"),
        first_match=("start_time", "min"),
        last_match=("start_time", "max"),
        rank_start=("average_rank", lambda s: s.head(20).mean()),
        rank_end=("average_rank", lambda s: s.tail(20).mean()),
        rank_max=("average_rank", "max"),
        perf_mean=("perf_index", "mean"),
        abandon_rate=("abandoned", "mean"),
        deaths_mean=("deaths", "mean"),
        party_rate=("party_size", lambda s: (s.fillna(1) > 1).mean()),
        n_clusters=("cluster", "nunique"),
    )
    agg["winrate"] = agg["wins"] / agg["n_ranked"]
    agg["span_months"] = (agg["last_match"] - agg["first_match"]) / MONTH

    slopes = df.groupby("account_id")[["start_time", "average_rank"]].apply(_rank_slope)
    agg["rank_slope"] = slopes

    # Перформанс в первых матчах: смурф силён сразу, а не после освоения.
    early = df.groupby("account_id").head(first_n)
    agg["perf_early"] = early.groupby("account_id")["perf_index"].mean()
    agg["winrate_early"] = early.groupby("account_id")["win"].mean()

    agg = agg.reset_index()
    agg["registration_est"] = estimate_registration_time(agg).to_numpy()
    agg["account_age_months"] = (agg["last_match"] - agg["registration_est"]) / MONTH
    return agg


@dataclass
class CohortThresholds:
    smurf_quantile: float = 0.95
    weak_quantile: float = 0.95


def score_cohorts(profiles: pd.DataFrame, thresholds: CohortThresholds | None = None) -> pd.DataFrame:
    """Складывает признаки в два индекса: смурфа и слабого игрока."""
    th = thresholds or CohortThresholds()
    out = profiles.copy()

    # Смурф: быстро поднимается, играет выше своего брекета, аккаунт молодой,
    # и всё это при коротком стаже.
    climb = _safe_z(out["rank_slope"].fillna(out["rank_slope"].median()))
    perf = _safe_z(out["perf_early"].fillna(0.0))
    youth = -_safe_z(np.log1p(out["account_age_months"].clip(lower=0).fillna(0)))
    early_win = _safe_z(out["winrate_early"].fillna(0.5))
    inexperience = -_safe_z(np.log1p(out["n_ranked"]))

    out["climb_z"] = climb
    out["perf_z"] = perf
    out["smurf_score"] = (
        0.30 * climb + 0.30 * perf + 0.15 * youth + 0.15 * early_win + 0.10 * inexperience
    )

    # Слабый игрок: устойчиво ниже уровня своего брекета, часто бросает матчи,
    # много умирает, рейтинг сползает вниз.
    out["weak_score"] = (
        0.40 * (-_safe_z(out["perf_mean"].fillna(0.0)))
        + 0.25 * _safe_z(out["abandon_rate"].fillna(0.0))
        + 0.20 * _safe_z(out["deaths_mean"].fillna(out["deaths_mean"].median()))
        + 0.15 * (-climb)
    )

    out["is_smurf"] = (out["smurf_score"] >= out["smurf_score"].quantile(th.smurf_quantile)).astype(int)
    out["is_weak"] = (out["weak_score"] >= out["weak_score"].quantile(th.weak_quantile)).astype(int)

    out["label"] = "resident"
    out.loc[out["is_weak"] == 1, "label"] = "weak"
    out.loc[out["is_smurf"] == 1, "label"] = "smurf"
    return out


def semi_supervised_labels(profiles: pd.DataFrame) -> pd.DataFrame:
    """Заведомые случаи для проверки разделяющей силы score.

    Эти метки не участвуют в построении score, только в его валидации.
    """
    out = profiles.copy()
    likely_smurf = (
        (out["account_age_months"] < 12)
        & (out["rank_max"] >= DIVINE_TIER)
        & (out["n_ranked"] >= 50)
    )
    likely_resident = (
        (out["span_months"] >= 24)
        & (out["rank_slope"].abs() < 0.3)
        & (out["n_ranked"] >= 200)
    )
    out["truth"] = np.where(likely_smurf, "smurf", np.where(likely_resident, "resident", ""))
    return out


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Площадь под ROC через статистику Манна-Уитни."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def validate(profiles: pd.DataFrame) -> dict[str, float]:
    labelled = semi_supervised_labels(profiles)
    subset = labelled[labelled["truth"] != ""]
    if subset.empty:
        return {"n": 0}
    y = (subset["truth"] == "smurf").astype(int).to_numpy()
    return {
        "n": int(len(subset)),
        "n_smurf": int(y.sum()),
        "n_resident": int((1 - y).sum()),
        "auc": auc(subset["smurf_score"].to_numpy(), y),
    }
