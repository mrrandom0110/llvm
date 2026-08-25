"""Позитивные контроли на публичных матчах.

Смысл этапа: до основных тестов убедиться, что конвейер вообще способен
обнаружить эффект известного размера. Если пайплайн не воспроизводит перевес
Radiant, который в Dota 2 существует заведомо, то нулевой результат основных
тестов ничего не значил бы — он мог бы объясняться поломкой обработки.

Данные берутся SQL-эксплорером OpenDota. У эксплорера жёсткий таймаут чтения:
окно в сутки проходит, месячное уже нет, поэтому агрегаты собираются посуточно,
по одному вызову API на день.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from scipy import stats

from .api import OpenDotaClient, OpenDotaError, QuotaExhausted
from .config import LOBBY_RANKED, MIN_DURATION_SEC, RANK_NAMES

log = logging.getLogger(__name__)

DAY = 86400


def wilson_interval(wins: int, n: int, conf: float = 0.99) -> tuple[float, float]:
    """Доверительный интервал Уилсона: устойчив при долях близких к 0 и 1."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


@dataclass
class BracketTally:
    n: int = 0
    radiant_wins: int = 0

    @property
    def winrate(self) -> float:
        return self.radiant_wins / self.n if self.n else float("nan")


@dataclass
class ControlResults:
    days: int = 0
    total: BracketTally = field(default_factory=BracketTally)
    by_bracket: dict[int, BracketTally] = field(default_factory=dict)
    by_duration: dict[str, BracketTally] = field(default_factory=dict)
    daily: list[tuple[int, int, int]] = field(default_factory=list)
    hero_stats: dict[int, tuple[int, int]] = field(default_factory=dict)


_BRACKET_SQL = """
select
  floor(coalesce(avg_rank_tier, -10) / 10)::int as bracket,
  case
    when duration < 1800 then 'short'
    when duration < 2700 then 'medium'
    else 'long'
  end as len_bucket,
  count(*) as n,
  sum(case when radiant_win then 1 else 0 end) as radiant_wins
from public_matches
where start_time >= {lo} and start_time < {hi}
  and lobby_type = {lobby} and duration > {min_dur}
group by 1, 2
"""

_HERO_SQL = """
select hero_id, count(*) as n, sum(win) as wins from (
  select unnest(radiant_team) as hero_id,
         case when radiant_win then 1 else 0 end as win
  from public_matches
  where start_time >= {lo} and start_time < {hi}
    and lobby_type = {lobby} and duration > {min_dur}
  union all
  select unnest(dire_team) as hero_id,
         case when radiant_win then 0 else 1 end as win
  from public_matches
  where start_time >= {lo} and start_time < {hi}
    and lobby_type = {lobby} and duration > {min_dur}
) t
where hero_id > 0
group by 1
"""


def collect_controls(
    client: OpenDotaClient,
    days: int = 10,
    end_time: int | None = None,
    hero_windows: int = 4,
    hero_window_hours: int = 24,
) -> ControlResults:
    """Собирает агрегаты по публичным ranked-матчам за последние `days` суток."""
    end = end_time or int(time.time())
    out = ControlResults()

    for offset in range(days):
        hi = end - offset * DAY
        lo = hi - DAY
        sql = _BRACKET_SQL.format(
            lo=lo, hi=hi, lobby=LOBBY_RANKED, min_dur=MIN_DURATION_SEC
        )
        try:
            rows = client.explorer(sql, max_age=30 * DAY)
        except QuotaExhausted:
            log.warning("квота исчерпана после %d суток", offset)
            break
        except OpenDotaError as exc:
            log.warning("день -%d пропущен: %s", offset, exc)
            continue

        day_n = day_w = 0
        for row in rows:
            bracket = int(row["bracket"])
            n = int(row["n"])
            wins = int(row["radiant_wins"])
            tally = out.by_bracket.setdefault(bracket, BracketTally())
            tally.n += n
            tally.radiant_wins += wins
            bucket = out.by_duration.setdefault(row["len_bucket"], BracketTally())
            bucket.n += n
            bucket.radiant_wins += wins
            day_n += n
            day_w += wins

        if day_n:
            out.total.n += day_n
            out.total.radiant_wins += day_w
            out.daily.append((lo, day_n, day_w))
            out.days += 1
            log.info(
                "сутки -%d: %s матчей, Radiant %.4f", offset, f"{day_n:,}", day_w / day_n
            )

    # Винрейты героев требуют разворачивания массивов составов, это в десять раз
    # тяжелее, поэтому берём несколько коротких окон вместо полных суток.
    for idx in range(hero_windows):
        hi = end - idx * DAY
        lo = hi - hero_window_hours * 3600
        sql = _HERO_SQL.format(lo=lo, hi=hi, lobby=LOBBY_RANKED, min_dur=MIN_DURATION_SEC)
        try:
            rows = client.explorer(sql, max_age=30 * DAY)
        except QuotaExhausted:
            break
        except OpenDotaError as exc:
            log.warning("окно героев %d пропущено: %s", idx, exc)
            continue
        for row in rows:
            hero_id = int(row["hero_id"])
            n, wins = out.hero_stats.get(hero_id, (0, 0))
            out.hero_stats[hero_id] = (n + int(row["n"]), wins + int(row["wins"]))

    return out


def bracket_label(bracket: int) -> str:
    if bracket < 0:
        return "Без ранга"
    return RANK_NAMES.get(bracket, f"tier {bracket}")
