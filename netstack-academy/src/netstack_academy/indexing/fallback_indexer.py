"""Lightweight, regex-based C source fallback indexer.

This is deliberately *not* a real C parser or AST: it is a bounded,
best-effort heuristic scanner used when a semantic provider (clangd) is
unavailable and/or to supplement ctags (which has no call graph). It is
restricted to a configured set of ``roots`` and extracts:

- function *definitions* (``kind`` immaterial beyond the ctags/storage
  contract; the fallback indexer only ever finds functions), recording
  whether each is C ``static`` (file-scoped) for identity purposes, and
- direct named call expressions (``name(...)``) inside each function body.

Call resolution never guesses across static boundaries: a call to a
``static`` function only resolves to a definition in the *caller's own
file*. A call to a non-``static`` function resolves only when the name is
globally unique among the definitions found in this run; otherwise the
edge is retained with ``target_relative_path=None`` (unresolved) rather
than picking one of several candidates. All edges produced here carry
``provenance="heuristic"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from .paths import normalize_relative_path

_CONTROL_KEYWORDS = frozenset(
    {
        "if",
        "for",
        "while",
        "switch",
        "return",
        "sizeof",
        "else",
        "do",
        "defined",
    }
)

# A function *definition* signature: an optional `static` qualifier, at
# least two whitespace/pointer-separated identifiers (return type, then
# name -- this is what naturally excludes single-identifier constructs like
# `if (x)` or a bare call `helper(x)`), a parenthesized parameter list with
# no `;`/`{`/`}` inside it, and either nothing else on the line or an
# opening brace, up to end of line. A trailing `;` (a prototype) never
# matches, since it isn't accounted for after the closing paren.
#
# The return-type group's repetitions each require a *mandatory* trailing
# separator (whitespace, or a pointer `*` with optional surrounding
# whitespace). This is the boundary that distinguishes the return type from
# the function name: the name token sits directly against `(` with no
# separator, so it can never be consumed by this group, no matter how the
# engine backtracks. (An earlier version made the trailing separator
# optional, which let the group's own backtracking swallow all but the last
# character of the name -- e.g. `helper` collapsed to `r` -- since regex
# backtracking gives back the *minimum* needed for the rest of the pattern
# to match, and the name group only requires one character.)
_DEFINITION_RE = re.compile(
    r"^(?P<static>static\s+)?"
    r"(?:[A-Za-z_]\w*\s*\*\s*|[A-Za-z_]\w*\s+)+"
    r"(?P<name>[A-Za-z_]\w*)\s*"
    r"\(\s*(?P<params>[^;{}()]*)\s*\)\s*"
    r"(?P<brace>\{)?\s*$"
)

_CALL_RE = re.compile(r"\b(?P<name>[A-Za-z_]\w*)\s*\(")


@dataclass(frozen=True, slots=True)
class FallbackSymbol:
    """A function definition discovered by the regex fallback scanner."""

    name: str
    kind: str
    relative_path: str
    line: int
    signature: str | None
    is_static: bool


@dataclass(frozen=True, slots=True)
class FallbackEdge:
    """A heuristic call edge discovered by the regex fallback scanner."""

    source_name: str
    source_relative_path: str
    source_line: int
    target_name: str
    target_relative_path: str | None
    line: int
    provenance: str = "heuristic"


@dataclass(frozen=True, slots=True)
class FallbackIndexResult:
    """The outcome of a fallback indexing run."""

    symbols: list[FallbackSymbol] = field(default_factory=list)
    edges: list[FallbackEdge] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _RawCall:
    source_name: str
    source_relative_path: str
    source_line: int
    target_name: str
    call_line: int


def _iter_source_files(kernel_repo: Path, roots: Iterable[str]) -> Iterator[Path]:
    seen: set[Path] = set()
    for root in roots:
        root_path = kernel_repo / root
        if not root_path.exists():
            continue
        if root_path.is_file():
            candidates: Iterable[Path] = (root_path,) if root_path.suffix == ".c" else ()
        else:
            candidates = sorted(root_path.rglob("*.c"))
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def _scan_file(
    absolute_path: Path, relative_path: str
) -> tuple[list[FallbackSymbol], list[_RawCall], list[str]]:
    symbols: list[FallbackSymbol] = []
    raw_calls: list[_RawCall] = []
    diagnostics: list[str] = []

    try:
        text = absolute_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        diagnostics.append(f"{relative_path}: could not read file ({exc})")
        return symbols, raw_calls, diagnostics

    lines = text.splitlines()
    line_count = len(lines)
    index = 0

    while index < line_count:
        match = _DEFINITION_RE.match(lines[index].strip())
        if match is None:
            index += 1
            continue

        name = match.group("name")
        if name in _CONTROL_KEYWORDS:
            index += 1
            continue

        signature_line_no = index + 1

        if match.group("brace") is not None:
            depth = 1
            body_start = index + 1
        else:
            lookahead = index + 1
            while lookahead < line_count and lines[lookahead].strip() == "":
                lookahead += 1
            if lookahead >= line_count or lines[lookahead].strip() != "{":
                # Not actually a definition (e.g. a multi-line prototype).
                index += 1
                continue
            depth = 1
            body_start = lookahead + 1

        is_static = match.group("static") is not None
        params = match.group("params") or ""
        symbols.append(
            FallbackSymbol(
                name=name,
                kind="function",
                relative_path=relative_path,
                line=signature_line_no,
                signature=f"({params})" if params else "()",
                is_static=is_static,
            )
        )

        cursor = body_start
        while cursor < line_count and depth > 0:
            body_line = lines[cursor]
            depth += body_line.count("{") - body_line.count("}")
            if depth > 0:
                for call_match in _CALL_RE.finditer(body_line):
                    target_name = call_match.group("name")
                    if target_name in _CONTROL_KEYWORDS:
                        continue
                    raw_calls.append(
                        _RawCall(
                            source_name=name,
                            source_relative_path=relative_path,
                            source_line=signature_line_no,
                            target_name=target_name,
                            call_line=cursor + 1,
                        )
                    )
            cursor += 1

        index = cursor

    return symbols, raw_calls, diagnostics


def _resolve_edges(
    symbols: list[FallbackSymbol], raw_calls: list[_RawCall]
) -> list[FallbackEdge]:
    by_name: dict[str, list[FallbackSymbol]] = {}
    for symbol in symbols:
        by_name.setdefault(symbol.name, []).append(symbol)

    edges: list[FallbackEdge] = []
    for call in raw_calls:
        candidates = by_name.get(call.target_name, [])

        same_file_static = [
            candidate
            for candidate in candidates
            if candidate.is_static and candidate.relative_path == call.source_relative_path
        ]

        target_relative_path: str | None
        if same_file_static:
            target_relative_path = call.source_relative_path
        else:
            non_static_candidates = [
                candidate for candidate in candidates if not candidate.is_static
            ]
            target_relative_path = (
                non_static_candidates[0].relative_path
                if len(non_static_candidates) == 1
                else None
            )

        edges.append(
            FallbackEdge(
                source_name=call.source_name,
                source_relative_path=call.source_relative_path,
                source_line=call.source_line,
                target_name=call.target_name,
                target_relative_path=target_relative_path,
                line=call.call_line,
                provenance="heuristic",
            )
        )

    return edges


def index_fallback(
    kernel_repo: Path, *, roots: Iterable[str] | None = None
) -> FallbackIndexResult:
    """Scan C sources under ``roots`` for function definitions and calls."""
    selected_roots = list(roots) if roots is not None else ["."]

    all_symbols: list[FallbackSymbol] = []
    all_raw_calls: list[_RawCall] = []
    all_diagnostics: list[str] = []

    for absolute_path in _iter_source_files(kernel_repo, selected_roots):
        relative_path = normalize_relative_path(absolute_path, kernel_repo=kernel_repo)
        symbols, raw_calls, diagnostics = _scan_file(absolute_path, relative_path)
        all_symbols.extend(symbols)
        all_raw_calls.extend(raw_calls)
        all_diagnostics.extend(diagnostics)

    edges = _resolve_edges(all_symbols, all_raw_calls)

    return FallbackIndexResult(
        symbols=all_symbols, edges=edges, diagnostics=all_diagnostics
    )
