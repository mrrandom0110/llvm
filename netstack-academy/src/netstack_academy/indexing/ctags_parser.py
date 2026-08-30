"""Tolerant parser for Universal Ctags ``--output-format=json`` output.

Universal Ctags emits one JSON object per line (JSON Lines). This parser is
deliberately forgiving: ctags output is an external, versioned tool
boundary, so a single malformed or unexpected line must never abort the
whole run. Any line that is not valid JSON, is not a ``{"_type": "tag"}``
record, or is missing a required field (``name``/``path``/``line``) is
skipped and recorded as a human-readable diagnostic string instead of
raising.

``is_static`` is derived from Universal Ctags' ``file`` boolean field, which
marks symbols whose visibility is restricted to the defining file (e.g. C
``static`` functions/variables) -- this is the file-scoped/static indicator
we need for cross-file identity resolution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

_REQUIRED_FIELDS: tuple[str, ...] = ("name", "path", "line")


@dataclass(frozen=True, slots=True)
class CtagsDefinition:
    """A single symbol definition extracted from Universal Ctags JSON output."""

    name: str
    kind: str
    path: str
    line: int
    signature: str | None = None
    scope: str | None = None
    is_static: bool = False


@dataclass(frozen=True, slots=True)
class ParsedCtagsResult:
    """The outcome of parsing a stream of Universal Ctags JSON lines."""

    definitions: list[CtagsDefinition] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


def parse_ctags_jsonlines(lines: Iterable[str]) -> ParsedCtagsResult:
    """Parse Universal Ctags JSON-lines output, never raising on bad input.

    Blank/whitespace-only lines are silently ignored. Lines that parse as
    JSON but are not a tag record (e.g. Universal Ctags' ``ptag``
    program-metadata records), or are tag records missing a required field,
    are skipped and recorded in ``diagnostics``. Lines that fail to parse as
    JSON at all are likewise skipped and recorded, without raising.
    """
    definitions: list[CtagsDefinition] = []
    diagnostics: list[str] = []

    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            diagnostics.append(f"line {line_number}: invalid JSON ({exc})")
            continue

        if not isinstance(record, dict):
            # Valid JSON (e.g. `null`, `[]`, `42`, a bare string) but not a
            # tag object -- silently ignored rather than a diagnostic, since
            # this is not expected ctags output at all.
            continue

        if record.get("_type") != "tag":
            diagnostics.append(
                f"line {line_number}: skipped non-tag record "
                f"(_type={record.get('_type')!r})"
            )
            continue

        name = record.get("name")
        path = record.get("path")
        line_no = record.get("line")
        if not isinstance(name, str) or not isinstance(path, str) or not isinstance(line_no, int):
            diagnostics.append(
                f"line {line_number}: tag record missing required field(s) "
                f"among {_REQUIRED_FIELDS}"
            )
            continue

        definitions.append(
            CtagsDefinition(
                name=name,
                kind=str(record.get("kind", "")),
                path=path,
                line=line_no,
                signature=record.get("signature"),
                scope=record.get("scope"),
                is_static=bool(record.get("file", False)),
            )
        )

    return ParsedCtagsResult(definitions=definitions, diagnostics=diagnostics)
