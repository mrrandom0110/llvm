from __future__ import annotations

from pathlib import Path

import pytest

from content_builder import (
    draft_lesson_frontmatter,
    lesson_frontmatter,
    write_lesson,
    write_module,
)


@pytest.fixture
def content_root(tmp_path: Path) -> Path:
    """A small but complete two-module curriculum on disk.

    Module ``rx-path`` (order 1) holds one published lesson; module
    ``tx-path`` (order 2) holds one published and one draft lesson. The
    directory names deliberately do not match the intended display order,
    so any test asserting ordering is asserting the loader's own sort and
    not the filesystem's.
    """
    rx = write_module(
        tmp_path,
        directory="20-rx",
        module_id="module-rx",
        slug="rx-path",
        title="Receive path",
        order=1,
    )
    write_lesson(rx, frontmatter=lesson_frontmatter())

    tx = write_module(
        tmp_path,
        directory="10-tx",
        module_id="module-tx",
        slug="tx-path",
        title="Transmit path",
        order=2,
    )
    write_lesson(
        tx,
        frontmatter=lesson_frontmatter(
            lesson_id="lesson-qdisc-dequeue",
            slug="qdisc-dequeue",
            title="Qdisc dequeue",
            order=10,
        ),
    )
    write_lesson(tx, frontmatter=draft_lesson_frontmatter())

    return tmp_path
