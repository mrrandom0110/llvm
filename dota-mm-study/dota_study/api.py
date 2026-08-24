"""Клиент OpenDota API с ограничением скорости, повторами и кэшем.

Устройство рассчитано на free tier (60 запросов в минуту, ~2-3 тысячи в сутки),
поэтому клиент:

* никогда не повторяет уже выполненный запрос (дисковый кэш);
* сам выдерживает паузы, чтобы не получать 429;
* считает израсходованный суточный бюджет в персистентном журнале, так что
  выгрузку можно останавливать и продолжать;
* поднимает :class:`QuotaExhausted`, когда бюджет исчерпан, вместо того чтобы
  молча портить выборку частичными данными.
"""

from __future__ import annotations

import logging
import sqlite3
import time
import urllib.parse
from collections import deque
from typing import Any, Iterable

import requests

from . import config
from .cache import HttpCache

log = logging.getLogger(__name__)

_QUOTA_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_calls (
    ts INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_api_calls_ts ON api_calls (ts);
"""


class QuotaExhausted(RuntimeError):
    """Суточный бюджет вызовов исчерпан."""


class OpenDotaError(RuntimeError):
    pass


class OpenDotaClient:
    def __init__(
        self,
        cache_path=None,
        rpm: int = config.RATE_LIMIT_PER_MINUTE,
        daily_budget: int = config.DAILY_BUDGET,
        api_key: str | None = config.API_KEY,
        offline: bool = False,
    ):
        config.ensure_dirs()
        self.cache = HttpCache(cache_path or config.CACHE_DB)
        self.rpm = rpm
        self.daily_budget = daily_budget
        self.api_key = api_key
        self.offline = offline
        self._recent: deque[float] = deque()
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "dota-mm-study/1.0 (research; open data)"
        self._quota = sqlite3.connect(str(config.CACHE_DB), timeout=60)
        self._quota.executescript(_QUOTA_SCHEMA)
        self._quota.commit()
        self.remaining_day: int | None = None
        self.stats = {"cache_hits": 0, "requests": 0, "errors": 0}

    # -- бюджет ---------------------------------------------------------

    def calls_last_24h(self) -> int:
        cutoff = int(time.time()) - 86400
        return self._quota.execute(
            "SELECT count(*) FROM api_calls WHERE ts > ?", (cutoff,)
        ).fetchone()[0]

    def budget_left(self) -> int:
        """Оценка снизу: минимум из нашего счётчика и заголовка сервера."""
        local = self.daily_budget - self.calls_last_24h()
        if self.remaining_day is not None:
            return max(0, min(local, self.remaining_day))
        return max(0, local)

    def _record_call(self) -> None:
        self._quota.execute("INSERT INTO api_calls (ts) VALUES (?)", (int(time.time()),))
        self._quota.commit()

    # -- ограничение скорости -------------------------------------------

    def _throttle(self) -> None:
        now = time.monotonic()
        while self._recent and now - self._recent[0] > 60:
            self._recent.popleft()
        if len(self._recent) >= self.rpm:
            sleep_for = 60 - (now - self._recent[0]) + 0.25
            if sleep_for > 0:
                log.debug("rate limit: пауза %.1f с", sleep_for)
                time.sleep(sleep_for)
        self._recent.append(time.monotonic())

    # -- запросы ---------------------------------------------------------

    @staticmethod
    def _cache_key(path: str, params: dict[str, Any] | None) -> str:
        if not params:
            return path
        flat: list[tuple[str, str]] = []
        for key, value in params.items():
            if isinstance(value, (list, tuple)):
                flat.extend((key, str(v)) for v in value)
            else:
                flat.append((key, str(value)))
        return path + "?" + urllib.parse.urlencode(sorted(flat))

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        max_age: int | None = None,
        retries: int = 4,
    ) -> Any:
        key = self._cache_key(path, params)
        cached = self.cache.get(key, max_age=max_age)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached
        if self.offline:
            raise QuotaExhausted(f"offline-режим, а {key} нет в кэше")
        if self.budget_left() <= 0:
            raise QuotaExhausted(
                f"исчерпан суточный бюджет ({self.daily_budget} вызовов); "
                "выгрузку можно продолжить позже, кэш сохранён"
            )

        request_params = dict(params or {})
        if self.api_key:
            request_params["api_key"] = self.api_key

        delay = 4.0
        for attempt in range(retries + 1):
            self._throttle()
            try:
                response = self._session.get(
                    f"{config.API_BASE}{path}", params=request_params, timeout=120
                )
            except requests.RequestException as exc:
                self.stats["errors"] += 1
                if attempt == retries:
                    raise OpenDotaError(f"сеть недоступна для {key}: {exc}") from exc
                time.sleep(delay)
                delay *= 2
                continue

            self._record_call()
            self.stats["requests"] += 1
            header = response.headers.get("x-rate-limit-remaining-day")
            if header is not None:
                try:
                    self.remaining_day = int(header)
                except ValueError:
                    pass

            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise OpenDotaError(f"невалидный JSON от {key}") from exc
                self.cache.put(key, 200, payload)
                return payload

            if response.status_code == 429:
                self.stats["errors"] += 1
                if self.remaining_day == 0:
                    raise QuotaExhausted("сервер сообщил, что суточный лимит исчерпан")
                time.sleep(max(delay, 15.0))
                delay *= 2
                continue

            if response.status_code in (500, 502, 503, 504):
                self.stats["errors"] += 1
                if attempt == retries:
                    raise OpenDotaError(f"{response.status_code} на {key}")
                time.sleep(delay)
                delay *= 2
                continue

            if response.status_code in (403, 404):
                # Приватный профиль или несуществующий матч: состояние
                # постоянное, запоминаем, чтобы не тратить на него квоту снова.
                self.cache.put(key, response.status_code, None)
                return None

            # Прочие 4xx (в первую очередь 400 от эксплорера при таймауте SQL)
            # могут быть временными, поэтому кэшировать их нельзя.
            self.stats["errors"] += 1
            detail = response.text[:200].replace("\n", " ")
            raise OpenDotaError(f"{response.status_code} на {key}: {detail}")

        raise OpenDotaError(f"не удалось получить {key} за {retries + 1} попыток")

    # -- конкретные эндпоинты --------------------------------------------

    def explorer(self, sql: str, max_age: int | None = None) -> list[dict[str, Any]]:
        """SQL по публичной базе OpenDota. Один вызов может вернуть тысячи строк."""
        payload = self.get("/explorer", {"sql": " ".join(sql.split())}, max_age=max_age)
        if payload is None:
            raise OpenDotaError("explorer вернул пустой ответ")
        if payload.get("err"):
            raise OpenDotaError(f"ошибка SQL: {payload['err']}")
        return payload.get("rows") or []

    def player(self, account_id: int) -> dict[str, Any] | None:
        return self.get(f"/players/{account_id}")

    def player_matches(
        self,
        account_id: int,
        limit: int = 2000,
        project: Iterable[str] | None = None,
        **filters: Any,
    ) -> list[dict[str, Any]] | None:
        params: dict[str, Any] = {"limit": limit, **filters}
        if project:
            params["project"] = list(project)
        return self.get(f"/players/{account_id}/matches", params)

    def player_peers(self, account_id: int) -> list[dict[str, Any]] | None:
        return self.get(f"/players/{account_id}/peers")

    def match(self, match_id: int) -> dict[str, Any] | None:
        return self.get(f"/matches/{match_id}")

    def public_matches(self, less_than_match_id: int | None = None, **filters: Any):
        params = dict(filters)
        if less_than_match_id is not None:
            params["less_than_match_id"] = less_than_match_id
        return self.get("/publicMatches", params)

    def heroes(self) -> list[dict[str, Any]]:
        return self.get("/heroes", max_age=30 * 86400) or []

    def close(self) -> None:
        self.cache.close()
        self._quota.close()
        self._session.close()
