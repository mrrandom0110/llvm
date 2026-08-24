"""Тест A: дисперсия карьерных винрейтов.

Логика теста. Если бы система жёстко тянула каждого к 50%, наблюдаемый разброс
винрейтов оказался бы **меньше** биномиального шума: система гасила бы
отклонения, которые при честной случайности обязаны накапливаться. Такая
недодисперсия при честной игре практически недостижима, поэтому это самый
сильный фальсифицирующий признак во всём исследовании.

Обратная ситуация — сверхдисперсия — означает, что у игроков есть устойчивые
различия в силе, которые система не гасит.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special, stats


@dataclass
class DispersionResult:
    n_players: int
    n_matches: int
    mean_winrate: float
    phi: float
    phi_lo: float
    phi_hi: float
    p_underdispersion: float
    p_overdispersion: float
    observed_sd: float
    binomial_sd: float
    true_sd: float
    alpha: float
    beta: float
    lrt_stat: float
    lrt_p: float

    def verdict(self) -> str:
        if self.phi_hi < 1.0:
            return "недодисперсия: свидетельство в пользу жёсткой подкрутки"
        if self.phi_lo > 1.0:
            return "сверхдисперсия: жёсткая подкрутка отвергается"
        return "дисперсия неотличима от биномиальной"


def pearson_dispersion(wins: np.ndarray, games: np.ndarray) -> tuple[float, float, float]:
    """Коэффициент дисперсии Пирсона и хвостовые вероятности.

    Под нулевой гипотезой (у всех игроков одна и та же вероятность победы)
    статистика Пирсона распределена как хи-квадрат с m-1 степенями свободы.
    """
    p = wins.sum() / games.sum()
    expected = games * p
    variance = games * p * (1 - p)
    chi2 = float(np.sum((wins - expected) ** 2 / variance))
    dof = len(wins) - 1
    phi = chi2 / dof
    p_under = float(stats.chi2.cdf(chi2, dof))
    p_over = float(stats.chi2.sf(chi2, dof))
    return phi, p_under, p_over


def _betabinom_nll(params: np.ndarray, wins: np.ndarray, games: np.ndarray) -> float:
    log_alpha, log_beta = params
    alpha, beta = np.exp(log_alpha), np.exp(log_beta)
    if not np.isfinite(alpha) or not np.isfinite(beta):
        return 1e18
    ll = (
        special.betaln(wins + alpha, games - wins + beta)
        - special.betaln(alpha, beta)
    )
    return -float(np.sum(ll))


def fit_beta_binomial(wins: np.ndarray, games: np.ndarray) -> tuple[float, float, float]:
    """Подгонка бета-биномиальной модели и LRT против чистой биномиальной.

    Бета-биномиальная модель описывает популяцию, в которой у каждого игрока
    своя истинная вероятность победы. Биномиальная — вырожденный случай, когда
    все вероятности одинаковы.
    """
    p0 = wins.sum() / games.sum()
    # Стартуем с умеренной концентрации, логарифмируем ради положительности.
    start = np.log([p0 * 50, (1 - p0) * 50])
    fit = optimize.minimize(
        _betabinom_nll, start, args=(wins, games), method="Nelder-Mead",
        options={"maxiter": 4000, "fatol": 1e-8, "xatol": 1e-8},
    )
    alpha, beta = np.exp(fit.x)

    ll_bb = -fit.fun
    ll_bin = float(np.sum(stats.binom.logpmf(wins, games, p0)))
    # Биномиальная модель вложена в бета-биномиальную, но лежит на границе
    # пространства параметров, поэтому нулевое распределение статистики — смесь
    # хи-квадрат, и p-значение делится пополам.
    lrt = 2 * (ll_bb - ll_bin)
    lrt_p = 0.5 * float(stats.chi2.sf(max(lrt, 0.0), 1))
    return alpha, beta, lrt, lrt_p


def bootstrap_phi(
    wins: np.ndarray, games: np.ndarray, n_boot: int = 2000, seed: int = 7
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = len(wins)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, m, m)
        draws[i] = pearson_dispersion(wins[idx], games[idx])[0]
    return float(np.quantile(draws, 0.005)), float(np.quantile(draws, 0.995))


def analyse(
    wins: np.ndarray, games: np.ndarray, n_boot: int = 2000
) -> DispersionResult:
    wins = np.asarray(wins, dtype=float)
    games = np.asarray(games, dtype=float)
    phi, p_under, p_over = pearson_dispersion(wins, games)
    lo, hi = bootstrap_phi(wins, games, n_boot=n_boot)
    alpha, beta, lrt, lrt_p = fit_beta_binomial(wins, games)

    rates = wins / games
    observed_sd = float(rates.std(ddof=1))
    p = wins.sum() / games.sum()
    # Ожидаемый чисто биномиальный разброс при тех же размерах истории.
    binomial_sd = float(np.sqrt(np.mean(p * (1 - p) / games)))
    true_var = max(observed_sd**2 - binomial_sd**2, 0.0)

    return DispersionResult(
        n_players=len(wins),
        n_matches=int(games.sum()),
        mean_winrate=float(p),
        phi=phi,
        phi_lo=lo,
        phi_hi=hi,
        p_underdispersion=p_under,
        p_overdispersion=p_over,
        observed_sd=observed_sd,
        binomial_sd=binomial_sd,
        true_sd=float(np.sqrt(true_var)),
        alpha=float(alpha),
        beta=float(beta),
        lrt_stat=float(lrt),
        lrt_p=float(lrt_p),
    )
