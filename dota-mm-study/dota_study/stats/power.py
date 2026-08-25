"""Расчёт мощности до сбора данных.

Смысл упражнения: заранее понять, какой размер эффекта мы вообще способны
отличить от нуля при доступном объёме, и зафиксировать целевой объём выборки.
Без этого нулевой результат невозможно интерпретировать — непонятно, эффекта
нет или не хватило данных.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats


def _z(alpha: float, two_sided: bool = True) -> float:
    return stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)


@dataclass(frozen=True)
class ProportionPower:
    delta: float
    alpha: float
    power: float
    n_per_group: int

    def __str__(self) -> str:
        return (
            f"эффект {self.delta * 100:.2f} п.п. при alpha={self.alpha}, "
            f"мощности {self.power:.0%}: нужно {self.n_per_group:,} наблюдений в группе"
        )


def n_for_proportion_diff(
    delta: float, p: float = 0.5, alpha: float = 0.01, power: float = 0.8
) -> ProportionPower:
    """Размер каждой из двух групп для сравнения долей."""
    z_a = _z(alpha)
    z_b = stats.norm.ppf(power)
    n = (z_a + z_b) ** 2 * 2 * p * (1 - p) / delta**2
    return ProportionPower(delta, alpha, power, int(math.ceil(n)))


def mde_for_proportion(n_per_group: int, p: float = 0.5, alpha: float = 0.01, power: float = 0.8) -> float:
    """Минимально детектируемый эффект при заданном объёме."""
    z_a = _z(alpha)
    z_b = stats.norm.ppf(power)
    return (z_a + z_b) * math.sqrt(2 * p * (1 - p) / n_per_group)


def dispersion_se(n_players: int) -> float:
    """SE коэффициента дисперсии phi при биномиальной нулевой гипотезе.

    Статистика Пирсона X^2 распределена как chi2 с m-1 степенями свободы,
    поэтому phi = X^2/(m-1) имеет дисперсию 2/(m-1).
    """
    return math.sqrt(2.0 / max(n_players - 1, 1))


def mde_for_dispersion(n_players: int, alpha: float = 0.01, power: float = 0.8) -> float:
    """Минимально детектируемое отклонение phi от единицы."""
    return (_z(alpha) + stats.norm.ppf(power)) * dispersion_se(n_players)


def matches_needed_for_streak_cell(
    n_per_cell: int, streak_len: int, base_rate: float = 0.5
) -> int:
    """Сколько всего player-matches нужно, чтобы набрать ячейку серии длины k.

    Доля игр, которым предшествует серия ровно из k одинаковых исходов, при
    независимых исходах составляет base_rate**k * (1 - base_rate).
    """
    share = base_rate**streak_len * (1 - base_rate)
    return int(math.ceil(n_per_cell / share))


def roster_delta_mde(
    n_matches: int, skill_sd: float = 0.05, alpha: float = 0.01, power: float = 0.8
) -> float:
    """MDE для разницы средней силы 4 союзников и 5 соперников.

    Дисперсия разницы средних: sd^2/4 + sd^2/5 на матч.
    """
    per_match_sd = skill_sd * math.sqrt(1 / 4 + 1 / 5)
    se = per_match_sd / math.sqrt(max(n_matches, 1))
    return (_z(alpha) + stats.norm.ppf(power)) * se


def report() -> str:
    lines: list[str] = []
    add = lines.append

    add("## Тест B — эффект серии на вероятность победы")
    add("")
    add("| Размер эффекта | Наблюдений в ячейке | Нужно player-matches при серии длины 3 |")
    add("| --- | --- | --- |")
    for delta in (0.005, 0.01, 0.02, 0.03):
        pw = n_for_proportion_diff(delta)
        total = matches_needed_for_streak_cell(pw.n_per_group, 3)
        add(f"| {delta * 100:.1f} п.п. | {pw.n_per_group:,} | {total:,} |")
    add("")
    for n in (500_000, 1_000_000, 2_000_000):
        cell = int(n * 0.5**3 * 0.5)
        add(f"- При {n:,} player-matches ячейка серии из 3 содержит ~{cell:,} игр, "
            f"MDE = {mde_for_proportion(cell) * 100:.2f} п.п.")

    add("")
    add("## Тест A — дисперсия карьерных винрейтов")
    add("")
    add("| Игроков в когорте | SE(phi) | MDE отклонения phi от 1 |")
    add("| --- | --- | --- |")
    for m in (500, 1000, 2000, 3000, 5000):
        add(f"| {m:,} | {dispersion_se(m):.4f} | {mde_for_dispersion(m):.3f} |")

    add("")
    add("## Тест C — асимметрия силы союзников и соперников")
    add("")
    add("| Матчей с составами | MDE разницы средних (в долях винрейта) |")
    add("| --- | --- |")
    for n in (500, 1000, 2000, 5000):
        add(f"| {n:,} | {roster_delta_mde(n) * 100:.2f} п.п. |")

    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
