from __future__ import annotations

from pathlib import Path

import pytest

from netstack_academy.learning.store import LearningStore

from learning_fakes import FakeClock


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store(tmp_path: Path, clock: FakeClock):
    """A real SQLite-backed store on disk with an injected clock."""
    with LearningStore.open(tmp_path / "learning.sqlite3", clock=clock) as opened:
        yield opened
