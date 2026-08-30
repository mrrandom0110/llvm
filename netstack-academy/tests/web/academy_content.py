"""Writes the tiny real curriculum the web tests are served from.

Two modules, four lessons, real Markdown with real YAML frontmatter, loaded
by the real :func:`~netstack_academy.curriculum.loader.load_curriculum`.
Nothing here is a stub: a page test that renders a hand-built model object
proves the template works against a shape the loader may never produce.

Three details are deliberate rather than decorative:

- **Directory names contradict the declared order.** ``20-rx`` declares
  ``order: 1`` and ``10-tx`` declares ``order: 2``, so a dashboard that
  lists modules in filesystem order gets the answer wrong.
- **The body carries hostile Markdown.** ``BODY_XSS_MARKER`` appears inside
  a ``<script>``, a ``javascript:`` link and an ``onerror`` attribute. It
  must not survive to the page, and the ordinary content around it must.
- **The lab carries a command that would be destructive if executed.**
  ``LAB_SENTINEL_PATH`` names a file the server must never create: the app
  *displays* lab commands, it does not run them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1

#: Text that only ever appears in a quiz question's ``explanation``. A page
#: or payload containing it before an attempt is recorded has leaked the
#: answer key.
QUIZ_EXPLANATION_MARKER = "answer-key-only-explanation"

#: The trusted answer to ``q-context``, so a test can submit a wrong answer
#: without hard-coding the option letter in three places.
CORRECT_OPTION_ID = "b"
WRONG_OPTION_ID = "a"
QUESTION_ID = "q-context"

#: Payload embedded in the lesson body through every hostile construct the
#: sanitizer is expected to defeat.
BODY_XSS_MARKER = "bodyXssPayload"

#: A lab command names this path. The file must not exist after any request.
LAB_SENTINEL_PATH = "/tmp/netstack-academy-lab-must-not-run"

#: Unique prose used to prove the lesson body is server-rendered (i.e. that
#: reading a lesson never depends on JavaScript).
BODY_PROSE_MARKER = "until the budget is spent"

NAPI_BODY = f"""\
## Where this happens

The driver hands the `sk_buff` to the stack from softirq context, and
`napi_poll` keeps calling the device's poll method {BODY_PROSE_MARKER}.

```c
#include <linux/skbuff.h>

static int napi_poll(struct napi_struct *n, struct list_head *repoll)
{{
        return n->poll(n, n->weight);
}}
```

| Field | Meaning |
| --- | --- |
| `weight` | budget for one poll |

See [the NAPI docs](https://docs.kernel.org/networking/napi.html).

<script>{BODY_XSS_MARKER}()</script>

[Read more](javascript:{BODY_XSS_MARKER}())

<img src="x" onerror="{BODY_XSS_MARKER}()">
"""

GRO_BODY = """\
## Coalescing

GRO merges segments before they reach the socket, which is why a capture
taken above the driver shows fewer, larger packets than the wire carried.
"""

QDISC_BODY = """\
## Dequeue

A qdisc hands one `sk_buff` at a time to the driver's transmit routine.
"""


def _document(frontmatter: dict[str, Any], body: str) -> str:
    return "---\n" + yaml.safe_dump(frontmatter, sort_keys=False) + "---\n\n" + body


def _module_frontmatter(
    *, module_id: str, slug: str, title: str, order: int, summary: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": module_id,
        "slug": slug,
        "title": title,
        "order": order,
        "summary": summary,
    }


def _napi_frontmatter() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": "lesson-napi-poll",
        "slug": "napi-poll",
        "title": "The NAPI poll loop",
        "order": 10,
        "status": "published",
        "summary": "How the NAPI poll loop drains a device queue.",
        "objectives": [
            "Explain when napi_poll runs",
            "Name the budget that bounds one poll",
        ],
        "prerequisites": [],
        "packet_stage": "rx-softirq",
        "execution_context": "softirq, with hard IRQs enabled",
        "ownership": "The NAPI instance is owned by the device driver.",
        "locking": "The NAPI_STATE_SCHED bit serializes pollers.",
        "rcu": "rcu_read_lock() is held across the receive handler.",
        "structures": [
            {"name": "struct napi_struct", "fields": ["poll", "weight", "state"]},
            {"name": "struct softnet_data", "fields": ["poll_list"]},
        ],
        "config_caveats": [
            "CONFIG_RPS moves work to a remote CPU.",
            "CONFIG_NET_RX_BUSY_POLL lets a socket poll the device itself.",
        ],
        "version_caveats": ["Budget accounting changed in v5.15."],
        "tracepoints": ["napi:napi_poll", "net:netif_receive_skb"],
        "source_symbols": [
            {"name": "napi_poll", "path": "net/core/dev.c"},
            {"name": "netif_receive_skb"},
        ],
        "lab": {
            "commands": [
                "cat /proc/net/softnet_stat",
                f"touch {LAB_SENTINEL_PATH}",
            ],
            "expected_observations": [
                "The second column stays at zero while the interface is idle.",
            ],
            "cleanup": [f"rm -f {LAB_SENTINEL_PATH}"],
        },
        "quiz": [
            {
                "id": QUESTION_ID,
                "prompt": "In which context does napi_poll run?",
                "options": [
                    {"id": WRONG_OPTION_ID, "text": "Hard IRQ"},
                    {"id": CORRECT_OPTION_ID, "text": "Softirq"},
                ],
                "answer": CORRECT_OPTION_ID,
                "explanation": (
                    "NAPI polling is deferred to NET_RX_SOFTIRQ "
                    f"({QUIZ_EXPLANATION_MARKER})."
                ),
            },
        ],
        "mastery_gate": {"min_quiz_score": 0.8, "required_review_level": 2},
    }


def _gro_frontmatter() -> dict[str, Any]:
    frontmatter = _napi_frontmatter()
    frontmatter.update(
        {
            "id": "lesson-gro",
            "slug": "gro-coalescing",
            "title": "GRO coalescing",
            "order": 20,
            "summary": "Merging segments before they reach the socket.",
            "objectives": ["Describe when GRO gives up on a flow"],
            # The lock state the lesson page has to render: this lesson is
            # unreachable until the NAPI lesson is completed.
            "prerequisites": ["lesson-napi-poll"],
            "source_symbols": [{"name": "netif_receive_skb"}],
            "tracepoints": ["net:netif_receive_skb"],
            "structures": [{"name": "struct napi_struct", "fields": ["gro_hash"]}],
        }
    )
    return frontmatter


def _qdisc_frontmatter() -> dict[str, Any]:
    frontmatter = _napi_frontmatter()
    frontmatter.update(
        {
            "id": "lesson-qdisc",
            "slug": "qdisc-dequeue",
            "title": "Qdisc dequeue",
            "order": 10,
            "summary": "How packets leave a queueing discipline.",
            "objectives": ["Name the lock a qdisc dequeue holds"],
            "prerequisites": [],
            "packet_stage": "tx-softirq",
            "source_symbols": [{"name": "helper", "path": "net/ipv4/a.c"}],
            "tracepoints": ["qdisc:qdisc_dequeue"],
            "structures": [{"name": "struct Qdisc", "fields": ["enqueue"]}],
        }
    )
    return frontmatter


def _draft_frontmatter() -> dict[str, Any]:
    """The minimum a ``draft`` lesson may carry: identity and nothing else."""
    return {
        "schema_version": SCHEMA_VERSION,
        "id": "lesson-draft",
        "slug": "bql-draft",
        "title": "Draft: byte queue limits",
        "order": 20,
        "status": "draft",
    }


def write_academy_content(content_root: Path) -> Path:
    """Write the two-module curriculum and return ``content_root``."""
    content_root.mkdir(parents=True, exist_ok=True)

    # Directory names deliberately disagree with the declared order.
    rx = content_root / "20-rx"
    tx = content_root / "10-tx"
    rx.mkdir(parents=True, exist_ok=True)
    tx.mkdir(parents=True, exist_ok=True)

    (rx / "module.md").write_text(
        _document(
            _module_frontmatter(
                module_id="module-rx",
                slug="rx-path",
                title="Receive path",
                order=1,
                summary="How a frame becomes an sk_buff.",
            ),
            "Everything from the device queue to the socket.\n",
        ),
        encoding="utf-8",
    )
    (rx / "napi-poll.md").write_text(
        _document(_napi_frontmatter(), NAPI_BODY), encoding="utf-8"
    )
    (rx / "gro-coalescing.md").write_text(
        _document(_gro_frontmatter(), GRO_BODY), encoding="utf-8"
    )

    (tx / "module.md").write_text(
        _document(
            _module_frontmatter(
                module_id="module-tx",
                slug="tx-path",
                title="Transmit path",
                order=2,
                summary="How an sk_buff reaches the wire.",
            ),
            "From the socket down to the driver.\n",
        ),
        encoding="utf-8",
    )
    (tx / "qdisc-dequeue.md").write_text(
        _document(_qdisc_frontmatter(), QDISC_BODY), encoding="utf-8"
    )
    (tx / "bql-draft.md").write_text(
        _document(_draft_frontmatter(), "Notes to self.\n"), encoding="utf-8"
    )

    # A directory with no module.md: assets live next to content and must
    # not be mistaken for a module.
    (content_root / "assets").mkdir(parents=True, exist_ok=True)
    (content_root / "assets" / "notes.md").write_text("scratch\n", encoding="utf-8")

    return content_root


def write_invalid_content(content_root: Path) -> Path:
    """Write a content root whose lesson fails published-tier validation."""
    content_root.mkdir(parents=True, exist_ok=True)
    module_directory = content_root / "broken"
    module_directory.mkdir(parents=True, exist_ok=True)

    (module_directory / "module.md").write_text(
        _document(
            _module_frontmatter(
                module_id="module-broken",
                slug="broken",
                title="Broken module",
                order=1,
                summary="Deliberately invalid content.",
            ),
            "Overview.\n",
        ),
        encoding="utf-8",
    )

    frontmatter = _napi_frontmatter()
    # Published, but with no objectives and no quiz: exactly the state an
    # author reaches halfway through writing a lesson.
    frontmatter["objectives"] = []
    frontmatter["quiz"] = []
    (module_directory / "half-written.md").write_text(
        _document(frontmatter, "Half a lesson.\n"), encoding="utf-8"
    )

    return content_root
