"""A tiny real kernel-ish repository and a real symbol index over it.

The web tests need a symbol index with every shape the symbol pages have to
cope with, so this builds one by writing real C files, committing them with
real ``git``, and persisting real rows through
:class:`~netstack_academy.indexing.storage.IndexStorage` -- the same code
path the production indexer commits through.

The fixture deliberately contains four awkward cases:

- ``helper`` is defined ``static`` twice, in different files, so a lookup by
  name alone is *ambiguous* rather than answerable.
- ``gone_symbol`` lives in a file that is not on disk, which is what an
  index built before a ``git pull`` looks like. No deep link can be built
  for it, and the page still has to render.
- ``escaped_symbol``'s stored path escapes the repository. Nothing
  production writes such a row, but the deep-link builder is the last thing
  standing between a corrupt index and a link that points outside the
  kernel tree, so the page must refuse it.
- The edges cover both provenances: a ``semantic`` call with a call site,
  a ``heuristic`` call with none, and a ``semantic`` reference.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from netstack_academy.indexing.models import EdgeInput, SymbolInput
from netstack_academy.indexing.storage import IndexStorage

DEV_C = "net/core/dev.c"
IPV4_C = "net/ipv4/a.c"
IPV6_C = "net/ipv6/b.c"
REMOVED_C = "net/core/removed.c"
ESCAPING_PATH = "../../etc/passwd"

NAPI_POLL_LINE = 12
NETIF_RECEIVE_SKB_LINE = 18
IPV4_HELPER_LINE = 3
IPV6_HELPER_LINE = 3
GONE_SYMBOL_LINE = 7
ESCAPED_SYMBOL_LINE = 1

#: Where the semantic provider saw ``napi_poll`` call ``netif_receive_skb``.
CALL_SITE_LINE = 15
CALL_SITE_COLUMN = 16

#: Where the semantic provider saw ``napi_poll`` referenced.
REFERENCE_SITE_LINE = 10
REFERENCE_SITE_COLUMN = 20

DEV_C_SOURCE = """\
#include <linux/netdevice.h>

static void trace_napi(struct napi_struct *n)
{
        (void)n;
}

/*
 * The poll loop.
 */

int napi_poll(struct napi_struct *n, int budget)
{
        trace_napi(n);
        return netif_receive_skb(NULL);
}

int netif_receive_skb(struct sk_buff *skb)
{
        return helper(0);
}
"""

IPV4_C_SOURCE = """\
#include <linux/kernel.h>

static int helper(int x)
{
        return x + 1;
}

int process(int x)
{
        return helper(x) + napi_poll(NULL, 0);
}
"""

IPV6_C_SOURCE = """\
#include <linux/kernel.h>

static int helper(int x)
{
        return x - 1;
}
"""


@dataclass(frozen=True, slots=True)
class IndexedGeneration:
    """What was persisted for one commit."""

    head: str
    symbol_count: int
    edge_count: int


def _run_git(*args: str, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def write_kernel_repo(repo: Path) -> str:
    """Write the C sources, commit them, and return the resulting ``HEAD``."""
    (repo / "net" / "core").mkdir(parents=True, exist_ok=True)
    (repo / "net" / "ipv4").mkdir(parents=True, exist_ok=True)
    (repo / "net" / "ipv6").mkdir(parents=True, exist_ok=True)

    (repo / DEV_C).write_text(DEV_C_SOURCE, encoding="utf-8")
    (repo / IPV4_C).write_text(IPV4_C_SOURCE, encoding="utf-8")
    (repo / IPV6_C).write_text(IPV6_C_SOURCE, encoding="utf-8")

    _run_git("init", cwd=repo)
    _run_git("config", "user.email", "test@example.com", cwd=repo)
    _run_git("config", "user.name", "Test User", cwd=repo)
    _run_git("add", ".", cwd=repo)
    _run_git("commit", "-m", "Add network fixture sources", cwd=repo)
    return _run_git("rev-parse", "HEAD", cwd=repo)


def index_symbols() -> list[SymbolInput]:
    """The persisted generation's symbols, in insertion order."""
    return [
        SymbolInput(
            name="napi_poll",
            kind="function",
            relative_path=DEV_C,
            line=NAPI_POLL_LINE,
            column=None,
            signature="int napi_poll(struct napi_struct *n, int budget)",
            scope=None,
            is_static=False,
        ),
        SymbolInput(
            name="netif_receive_skb",
            kind="function",
            relative_path=DEV_C,
            line=NETIF_RECEIVE_SKB_LINE,
            column=None,
            signature="int netif_receive_skb(struct sk_buff *skb)",
            scope=None,
            is_static=False,
        ),
        SymbolInput(
            name="helper",
            kind="function",
            relative_path=IPV4_C,
            line=IPV4_HELPER_LINE,
            column=None,
            signature="static int helper(int x)",
            scope=None,
            is_static=True,
        ),
        SymbolInput(
            name="helper",
            kind="function",
            relative_path=IPV6_C,
            line=IPV6_HELPER_LINE,
            column=None,
            signature="static int helper(int x)",
            scope=None,
            is_static=True,
        ),
        SymbolInput(
            name="gone_symbol",
            kind="function",
            relative_path=REMOVED_C,
            line=GONE_SYMBOL_LINE,
            column=None,
            signature="int gone_symbol(void)",
            scope=None,
            is_static=False,
        ),
        SymbolInput(
            name="escaped_symbol",
            kind="function",
            relative_path=ESCAPING_PATH,
            line=ESCAPED_SYMBOL_LINE,
            column=None,
            signature="int escaped_symbol(void)",
            scope=None,
            is_static=False,
        ),
    ]


def index_edges() -> list[EdgeInput]:
    """Call/reference edges over :func:`index_symbols`' batch indices."""
    return [
        # napi_poll -> netif_receive_skb, seen by the semantic provider,
        # with the position of the call itself.
        EdgeInput(
            source_index=0,
            target_index=1,
            target_name="netif_receive_skb",
            edge_type="call",
            provenance="semantic",
            site_relative_path=DEV_C,
            site_line=CALL_SITE_LINE,
            site_column=CALL_SITE_COLUMN,
        ),
        # netif_receive_skb -> napi_poll, guessed by the regex fallback: no
        # site, and a provenance the page has to label as such.
        EdgeInput(
            source_index=1,
            target_index=0,
            target_name="napi_poll",
            edge_type="call",
            provenance="heuristic",
        ),
        # netif_receive_skb -> helper, resolved to the ipv4 definition.
        EdgeInput(
            source_index=1,
            target_index=2,
            target_name="helper",
            edge_type="call",
            provenance="heuristic",
        ),
        # A use of napi_poll, anchored on napi_poll itself.
        EdgeInput(
            source_index=0,
            target_index=None,
            target_name="napi_poll",
            edge_type="reference",
            provenance="semantic",
            site_relative_path=IPV4_C,
            site_line=REFERENCE_SITE_LINE,
            site_column=REFERENCE_SITE_COLUMN,
        ),
    ]


def index_kernel_repo(storage: IndexStorage, *, head: str) -> IndexedGeneration:
    """Persist the fixture generation at ``head`` into ``storage``."""
    result = storage.replace_symbols_and_edges(head, index_symbols(), index_edges())
    return IndexedGeneration(
        head=result.commit_hash,
        symbol_count=result.symbol_count,
        edge_count=result.edge_count,
    )
