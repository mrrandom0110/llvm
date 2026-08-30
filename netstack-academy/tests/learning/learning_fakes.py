"""Test doubles for the learning-side tests: an injectable clock and a
minimal symbol-index stand-in.

Both are deliberately tiny and behavioural. ``FakeClock`` exists because
every scheduling assertion in this directory has to be exact -- a real
``datetime.now()`` would make "due tomorrow" untestable. ``FakeSymbolIndex``
records the arguments it was called with so the search tests can prove the
service forwards a query (and its limit) rather than reimplementing symbol
search itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any


class FakeClock:
    """A monotonic, manually advanced UTC clock."""

    def __init__(self, start: datetime | None = None) -> None:
        self.now = start if start is not None else datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **timedelta_kwargs: float) -> datetime:
        self.now = self.now + timedelta(**timedelta_kwargs)
        return self.now


@dataclass(frozen=True)
class FakeSymbol:
    """The subset of a persisted symbol the learning UI actually shows."""

    id: int
    name: str
    kind: str
    relative_path: str
    line: int


class FakeSymbolIndex:
    """Stands in for :class:`netstack_academy.indexing.service.IndexService`.

    The learning service must depend only on a narrow ``search_symbols``
    contract, so this double implements exactly that and nothing else.
    """

    def __init__(self, results: list[FakeSymbol] | None = None) -> None:
        self.results = results if results is not None else []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def search_symbols(self, query: str, **kwargs: Any) -> list[FakeSymbol]:
        self.calls.append((query, dict(kwargs)))
        return list(self.results)


class ExplodingSymbolIndex:
    """A symbol index that must never be consulted."""

    def search_symbols(self, query: str, **kwargs: Any) -> list[FakeSymbol]:
        raise AssertionError(f"symbol index must not be queried (query={query!r})")
