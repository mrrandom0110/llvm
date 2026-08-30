"""Helpers that write real curriculum content trees to a temporary directory.

The curriculum loader reads Markdown files with YAML frontmatter from a
content root laid out as::

    <content_root>/<module-directory>/module.md
    <content_root>/<module-directory>/<lesson-file>.md

Every helper here writes real files with real YAML so the loader under test
is exercised against the same bytes production content would have; nothing
is mocked. ``overrides`` maps a frontmatter key to a replacement value, and
mapping a key to :data:`OMIT` deletes it, which is how tests state "this
required field is missing" without rebuilding a whole document by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

#: Sentinel meaning "remove this frontmatter key entirely".
OMIT = object()

SCHEMA_VERSION = 1

MODULE_FILENAME = "module.md"

DEFAULT_LESSON_BODY = """\
## Where this happens

The driver hands the `sk_buff` to the stack from softirq context.

```c
static int napi_poll(struct napi_struct *n, struct list_head *repoll)
{
        return 0;
}
```

| Field | Meaning |
| --- | --- |
| `weight` | budget for one poll |

See [the NAPI docs](https://docs.kernel.org/networking/napi.html).
"""


def _frontmatter_document(frontmatter: dict[str, Any], body: str) -> str:
    return "---\n" + yaml.safe_dump(frontmatter, sort_keys=False) + "---\n\n" + body


def _apply_overrides(
    base: dict[str, Any], overrides: dict[str, Any] | None
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (overrides or {}).items():
        if value is OMIT:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def write_module(
    content_root: Path,
    *,
    directory: str,
    module_id: str | None = None,
    slug: str | None = None,
    title: str = "Receive path",
    order: int = 1,
    summary: str = "How a frame becomes an sk_buff.",
    overrides: dict[str, Any] | None = None,
    body: str = "Module overview.\n",
) -> Path:
    """Write ``<content_root>/<directory>/module.md`` and return the directory."""
    module_directory = content_root / directory
    module_directory.mkdir(parents=True, exist_ok=True)

    frontmatter: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": module_id if module_id is not None else f"module-{directory}",
        "slug": slug if slug is not None else directory,
        "title": title,
        "order": order,
        "summary": summary,
    }
    document = _frontmatter_document(_apply_overrides(frontmatter, overrides), body)
    (module_directory / MODULE_FILENAME).write_text(document, encoding="utf-8")
    return module_directory


def lesson_frontmatter(
    *,
    lesson_id: str = "lesson-napi-poll",
    slug: str = "napi-poll",
    title: str = "The NAPI poll loop",
    order: int = 10,
    status: str = "published",
) -> dict[str, Any]:
    """A complete, valid ``published`` lesson frontmatter mapping."""
    return {
        "schema_version": SCHEMA_VERSION,
        "id": lesson_id,
        "slug": slug,
        "title": title,
        "order": order,
        "status": status,
        "summary": "How the NAPI poll loop drains a device queue.",
        "objectives": [
            "Explain when napi_poll runs",
            "Name the budget that bounds one poll",
        ],
        "prerequisites": [],
        "packet_stage": "rx-softirq",
        "execution_context": "softirq",
        "ownership": "The NAPI instance is owned by the device driver.",
        "locking": "NAPI_STATE_SCHED bit serializes pollers.",
        "rcu": "rcu_read_lock() is held across the receive handler.",
        "structures": [
            {"name": "struct napi_struct", "fields": ["poll", "weight", "state"]},
        ],
        "config_caveats": ["CONFIG_RPS moves work to a remote CPU."],
        "version_caveats": ["Budget accounting changed in v5.15."],
        "tracepoints": ["napi:napi_poll"],
        "source_symbols": [
            {"name": "napi_poll", "path": "net/core/dev.c"},
            {"name": "netif_receive_skb"},
        ],
        "lab": {
            "commands": ["cat /proc/net/softnet_stat"],
            "expected_observations": ["The second column stays at zero."],
            "cleanup": ["true"],
        },
        "quiz": [
            {
                "id": "q-context",
                "prompt": "In which context does napi_poll run?",
                "options": [
                    {"id": "a", "text": "Hard IRQ"},
                    {"id": "b", "text": "Softirq"},
                ],
                "answer": "b",
                "explanation": "NAPI polling is deferred to NET_RX_SOFTIRQ.",
            },
        ],
        "mastery_gate": {"min_quiz_score": 0.8, "required_review_level": 2},
    }


def draft_lesson_frontmatter(
    *,
    lesson_id: str = "lesson-draft",
    slug: str = "draft-lesson",
    title: str = "Draft: GRO coalescing",
    order: int = 20,
) -> dict[str, Any]:
    """The minimum frontmatter a ``draft`` lesson may carry."""
    return {
        "schema_version": SCHEMA_VERSION,
        "id": lesson_id,
        "slug": slug,
        "title": title,
        "order": order,
        "status": "draft",
    }


def write_lesson(
    module_directory: Path,
    *,
    filename: str | None = None,
    frontmatter: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
    body: str = DEFAULT_LESSON_BODY,
) -> Path:
    """Write one lesson Markdown file into ``module_directory``."""
    base = frontmatter if frontmatter is not None else lesson_frontmatter()
    merged = _apply_overrides(base, overrides)
    name = filename if filename is not None else f"{merged.get('slug', 'lesson')}.md"

    path = module_directory / name
    path.write_text(_frontmatter_document(merged, body), encoding="utf-8")
    return path


def write_raw_lesson(module_directory: Path, *, filename: str, text: str) -> Path:
    """Write arbitrary bytes as a lesson file (malformed frontmatter cases)."""
    path = module_directory / filename
    path.write_text(text, encoding="utf-8")
    return path
