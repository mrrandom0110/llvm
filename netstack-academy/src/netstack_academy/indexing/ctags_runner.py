"""Bounded, tolerant driver for the external Universal Ctags binary.

Ctags is treated as an optional, best-effort provider: it must never hang
the pipeline (all subprocess invocations use a finite timeout) and it must
never raise when the binary is missing, is a different ``ctags``
implementation (e.g. GNU Emacs' bundled ``ctags``), or misbehaves at
runtime. Every failure mode degrades to a typed :class:`CtagsRunResult`
status instead.

``subprocess`` is imported at module level (not ``from subprocess import
run``) so tests can monkeypatch ``ctags_runner.subprocess.run`` directly,
mirroring the existing ``repo_inspector`` convention.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from .ctags_parser import CtagsDefinition, parse_ctags_jsonlines

CTAGS_VERSION_TIMEOUT_SECONDS = 5.0
CTAGS_INDEX_TIMEOUT_SECONDS = 30.0

CtagsRunStatus = Literal["ok", "unavailable", "incompatible", "timeout", "error"]

# A curated, network-stack-focused subset of kernel source trees. Chosen to
# keep indexing fast and relevant, deliberately excluding unrelated trees
# such as `fs`, `arch`, the rest of `drivers/`, and the rest of
# `include/linux/`.
DEFAULT_INDEX_ROOTS: tuple[str, ...] = (
    "net",
    "include/net",
    "include/linux/skbuff.h",
    "include/linux/netdevice.h",
    "include/linux/inetdevice.h",
    "include/linux/socket.h",
    "include/linux/in.h",
    "include/linux/tcp.h",
    "include/linux/udp.h",
    "drivers/net/ethernet",
    "drivers/net/phy",
)


def default_index_roots() -> tuple[str, ...]:
    """Return the default set of network-relevant kernel source roots."""
    return DEFAULT_INDEX_ROOTS


@dataclass(frozen=True, slots=True)
class CtagsAvailability:
    """The outcome of probing a ``ctags`` executable via ``--version``."""

    available: bool
    is_universal: bool
    version: str | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class CtagsRunResult:
    """The outcome of a bounded ``run_ctags`` invocation."""

    status: CtagsRunStatus
    definitions: list[CtagsDefinition] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


def check_ctags_binary(
    executable: str = "ctags",
    *,
    timeout: float = CTAGS_VERSION_TIMEOUT_SECONDS,
) -> CtagsAvailability:
    """Probe ``executable --version`` and classify the result without raising."""
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return CtagsAvailability(
            available=False,
            is_universal=False,
            version=None,
            reason=f"ctags executable not found: {executable!r}",
        )
    except subprocess.TimeoutExpired:
        return CtagsAvailability(
            available=False,
            is_universal=False,
            version=None,
            reason="ctags --version timeout",
        )
    except OSError as exc:
        return CtagsAvailability(
            available=False,
            is_universal=False,
            version=None,
            reason=f"ctags --version failed: {exc}",
        )

    output = f"{result.stdout or ''}{result.stderr or ''}"

    if result.returncode != 0:
        return CtagsAvailability(
            available=False,
            is_universal=False,
            version=output.strip() or None,
            reason=f"ctags --version exited with code {result.returncode}",
        )

    version_text = output.strip()

    if "universal ctags" in output.lower():
        return CtagsAvailability(
            available=True,
            is_universal=True,
            version=version_text,
            reason=None,
        )

    if "emacs" in output.lower():
        return CtagsAvailability(
            available=True,
            is_universal=False,
            version=version_text,
            reason=(
                "Detected Emacs ctags, which does not support "
                "--output-format=json (Universal Ctags is required)"
            ),
        )

    return CtagsAvailability(
        available=True,
        is_universal=False,
        version=version_text,
        reason="ctags binary does not appear to be Universal Ctags",
    )


def _existing_roots(kernel_repo: Path, roots: Iterable[str]) -> list[str]:
    return [root for root in roots if (kernel_repo / root).exists()]


def run_ctags(
    kernel_repo: Path,
    *,
    executable: str = "ctags",
    roots: Iterable[str] | None = None,
    timeout: float = CTAGS_INDEX_TIMEOUT_SECONDS,
) -> CtagsRunResult:
    """Run Universal Ctags over ``roots`` (relative to ``kernel_repo``).

    The binary is verified first via :func:`check_ctags_binary`; the actual
    indexing subprocess is never invoked when it is missing or not Universal
    Ctags. Roots that do not currently exist under ``kernel_repo`` are
    silently skipped. Timeouts and OS-level failures degrade to a typed
    result instead of raising.
    """
    # The probe and the indexing subprocess below share the caller's
    # ``timeout`` rather than each defaulting independently, so a caller
    # with a tight overall latency budget has that budget honored
    # coherently by both invocations, not just the second one.
    availability = check_ctags_binary(executable, timeout=timeout)
    if not availability.available:
        return CtagsRunResult(
            status="unavailable",
            definitions=[],
            diagnostics=[availability.reason] if availability.reason else [],
        )
    if not availability.is_universal:
        return CtagsRunResult(
            status="incompatible",
            definitions=[],
            diagnostics=[availability.reason] if availability.reason else [],
        )

    selected_roots = list(roots) if roots is not None else list(default_index_roots())
    existing_roots = _existing_roots(kernel_repo, selected_roots)

    if not existing_roots:
        # No path operands to give ctags: `ctags -R` with zero path
        # operands still recursively scans `cwd`, which would silently
        # defeat the entire point of a curated, bounded root list. Degrade
        # to a benign, empty result instead of ever launching that
        # unbounded scan.
        return CtagsRunResult(
            status="ok",
            definitions=[],
            diagnostics=["no configured roots exist on disk; skipped ctags"],
        )

    args = [
        executable,
        "--languages=C",
        "--output-format=json",
        "--fields=+nS",
        "-R",
        *existing_roots,
    ]

    try:
        result = subprocess.run(
            args,
            cwd=kernel_repo,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return CtagsRunResult(
            status="timeout",
            definitions=[],
            diagnostics=["ctags indexing subprocess timeout"],
        )
    except OSError as exc:
        return CtagsRunResult(
            status="error",
            definitions=[],
            diagnostics=[f"ctags indexing subprocess failed: {exc}"],
        )

    if result.returncode != 0:
        return CtagsRunResult(
            status="error",
            definitions=[],
            diagnostics=[
                f"ctags exited with code {result.returncode}: {(result.stderr or '').strip()}"
            ],
        )

    parsed = parse_ctags_jsonlines((result.stdout or "").splitlines())
    return CtagsRunResult(
        status="ok",
        definitions=parsed.definitions,
        diagnostics=parsed.diagnostics,
    )
