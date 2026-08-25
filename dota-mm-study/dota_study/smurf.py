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


def calibrate_account_age(account_ids: pd.Series, first_match: pd.Series) -> pd.Series:
    """Датировка регистрации аккаунта по его номеру.

    Steam выдаёт account_id почти монотонно по времени регистрации, поэтому сам
    номер датирует аккаунт. Калибровочная кривая строится по нашим же данным:
    внутри окон по номеру берётся медиана даты первого матча, после чего кривая
    делается монотонной кумулятивным максимумом.

    Оценка нужна там, где наблюдаемый первый матч вводит в заблуждение: у
    вернувшегося после долгого перерыва игрока история может начинаться недавно
    при старом аккаунте, и без поправки он выглядел бы смурфом.
    """
    frame = pd.DataFrame({"account_id": account_ids, "first_match": first_match}).dropna()
    if len(frame) < 20:
        return pd.Series(np.nan, index=account_ids.index)

    frame = frame.sort_values("account_id")
    n_bins = max(min(len(frame) // 25, 60), 4)
    frame["bin"] = pd.qcut(frame["account_id"], n_bins, duplicates="drop")
    curve = frame.groupby("bin", observed=True).agg(
        account_id=("account_id", "median"), first_match=("first_match", "median")
    )
    curve["first_match"] = curve["first_match"].cummax()
    curve = curve.dropna().sort_values("account_id")
    if len(curve) < 2:
        return pd.Series(np.nan, index=account_ids.index)

    estimated = np.interp(
        account_ids.to_numpy(dtype=float),
        curve["account_id"].to_numpy(dtype=float),
        curve["first_match"].to_numpy(dtype=float),
    )
    return pd.Series(estimated, index=account_ids.index)


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


def build_profiles(
    matches: pd.DataFrame,
    history_meta: pd.DataFrame | None = None,
    first_n: int = 50,
) -> pd.DataFrame:
    """Сводка по каждому игроку: динамика ранга, перформанс, ранняя доминантность.

    `history_meta` содержит границы полной истории аккаунта, включая матчи до
    окна исследования. Без неё возраст аккаунта систематически занижался бы:
    первый матч в окне — это не первый матч в жизни аккаунта.
    """
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

    if history_meta is not None and not history_meta.empty:
        agg = agg.merge(history_meta, on="account_id", how="left", suffixes=("", "_all"))
    if "first_match_all" not in agg:
        agg["first_match_all"] = agg["first_match"]
    agg["first_match_all"] = agg["first_match_all"].fillna(agg["first_match"])

    agg["registration_est"] = calibrate_account_age(
        agg["account_id"], agg["first_match_all"]
    )
    # Возраст аккаунта: берём более раннюю из двух оценок, потому что обе дают
    # верхнюю границу даты регистрации, а ошибиться в сторону «аккаунт старый»
    # безопаснее, чем записать ветерана в смурфы.
    origin = agg[["first_match_all", "registration_est"]].min(axis=1)
    agg["registration_est"] = origin
    agg["account_age_months"] = (agg["last_match"] - origin) / MONTH
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
    # Вариант без возрастной компоненты: нужен для честной валидации, поскольку
    # проверочная метка сама опирается на возраст аккаунта.
    out["smurf_score_no_age"] = 0.40 * climb + 0.40 * perf + 0.20 * early_win

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
    # Метка опирается на длину наблюдаемой карьеры и достигнутый брекет —
    # величины, которые не входят в сам score. Аккаунт, за десять месяцев
    # добравшийся до Divine, почти наверняка принадлежит опытному игроку.
    likely_smurf = (
        (out["span_months"] < 12)
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
    """Проверка разделяющей силы score на заведомых случаях.

    Это проверка согласованности, а не независимая валидация: настоящей разметки
    смурфов не существует, и проверочная метка частично опирается на те же
    наблюдаемые величины. Поэтому отдельно приводится AUC варианта score без
    возрастной компоненты — метка от него не зависит напрямую.
    """
    labelled = semi_supervised_labels(profiles)
    subset = labelled[labelled["truth"] != ""]
    if subset.empty:
        return {"n": 0}
    y = (subset["truth"] == "smurf").astype(int).to_numpy()
    out = {
        "n": int(len(subset)),
        "n_smurf": int(y.sum()),
        "n_resident": int((1 - y).sum()),
        "auc": auc(subset["smurf_score"].to_numpy(), y),
    }
    if "smurf_score_no_age" in subset:
        out["auc_no_age"] = auc(subset["smurf_score_no_age"].to_numpy(), y)
    return out
