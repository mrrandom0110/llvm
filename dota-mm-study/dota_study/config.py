"""Общие константы и пути исследования."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DOTA_STUDY_DATA", PROJECT_ROOT / "data"))
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

CACHE_DB = DATA_DIR / "http_cache.sqlite"
STUDY_DB = DATA_DIR / "study.sqlite"

API_BASE = "https://api.opendota.com/api"
API_KEY = os.environ.get("OPENDOTA_API_KEY") or None

# Free tier: 60 запросов в минуту, порядка 2000-3000 в сутки. Держим запас,
# чтобы не упереться в 429 на границе окна.
RATE_LIMIT_PER_MINUTE = int(os.environ.get("DOTA_STUDY_RPM", "50"))
DAILY_BUDGET = int(os.environ.get("DOTA_STUDY_DAILY_BUDGET", "2900"))

# Ranked matchmaking, только он влияет на MMR.
LOBBY_RANKED = 7

# All Pick / Ranked All Pick / Captains Mode / Turbo и прочее.
# Для основной выборки берём режимы, где итог матча осмыслен как ranked-игра.
VALID_GAME_MODES = {1, 2, 3, 4, 5, 12, 16, 22}

# Матчи короче этого почти всегда развал/абандон на старте.
MIN_DURATION_SEC = 600

# average_rank заполняется OpenDota только у относительно свежих матчей:
# у аккаунтов с историей с 2014 покрытие ~1%, у активных игроков 2023+ ~100%.
# Ниже этой отметки контроль на рейтинг невозможен.
STUDY_WINDOW_START = 1672531200  # 2023-01-01 UTC

# Ранги OpenDota: rank_tier = medal*10 + star. average_rank в истории матчей
# использует ту же шкалу.
RANK_NAMES = {
    1: "Herald",
    2: "Guardian",
    3: "Crusader",
    4: "Archon",
    5: "Legend",
    6: "Ancient",
    7: "Divine",
    8: "Immortal",
}


def rank_name(avg_rank: float | None) -> str:
    """Человекочитаемое имя медали по значению average_rank."""
    if avg_rank is None:
        return "Unknown"
    return RANK_NAMES.get(int(avg_rank) // 10, "Unknown")


def ensure_dirs() -> None:
    for path in (DATA_DIR, REPORTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
