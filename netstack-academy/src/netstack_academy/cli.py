"""Three commands, and nothing that runs what a caller typed.

``serve`` starts the local application, ``validate-content`` answers "is this
course loadable" for an author mid-edit, and ``index`` builds or refreshes the
symbol index from a terminal rather than from a request.

The command surface is deliberately closed. Every argument is a path, a port
or a flag; there is no option that takes a shell command, a program to run, or
a template to expand, and nothing here reaches a shell. That matters more in
this program than in most: a tool that teaches the kernel by *displaying* lab
commands is a tool whose content is full of shell, and the one thing it must
never do is execute it. A content root of ``; touch /tmp/x`` is therefore a
directory name that does not exist -- an ordinary "no such course" failure --
and never a command.

Three further properties are worth stating because each is a decision rather
than an accident.

**The bind address is checked by the parser, not by the server.** ``serve``
refuses anything that is not loopback before a database is opened, let alone
before a socket is bound. There is no authentication anywhere in this program,
so binding somewhere routable would publish a learner's notes and an
unauthenticated state-import endpoint to the local network;
:func:`~netstack_academy.web.app.is_loopback_host` is what says no, and it
says it as an ``argparse`` type so the refusal costs an exit code and nothing
else.

**Whatever a command opens, the command closes.** ``serve`` and ``index`` both
open an :class:`~netstack_academy.web.runtime.AcademyRuntime`, which holds two
SQLite connections, and both close it from a ``finally`` -- so serving that
ends because uvicorn returned, or because of Ctrl-C, or because the pipeline
raised, all leave the same clean state behind. ``validate-content`` opens no
runtime at all: asking whether Markdown parses is a read of the filesystem,
and it must not create a database or need a kernel checkout to answer.

**Both injectable collaborators are the ones that would otherwise block or
shell out.** ``server_runner`` defaults to :func:`uvicorn.run`, which never
returns until the server stops; ``session_runner`` defaults to
:func:`~netstack_academy.indexing.composition.run_indexing_session`, which
starts ``ctags`` and ``clangd``. Injecting them is what makes
:func:`main` testable without a socket or a toolchain.

There is deliberately no ``--reload``. uvicorn's reloader takes an import
string rather than an application object and re-executes the process to pick
up changes, which would hand the runtime's two open SQLite connections to a
process that did not open them and will not close them. Editing this program's
own source is a developer activity, and ``uvicorn`` is on the path for it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import uvicorn

from netstack_academy.curriculum.loader import (
    CurriculumError,
    CurriculumValidationError,
    load_curriculum,
)
from netstack_academy.curriculum.models import Curriculum
from netstack_academy.indexing.composition import run_indexing_session
from netstack_academy.indexing.orchestrator import IndexRunResult
from netstack_academy.settings import Settings
from netstack_academy.web.app import (
    DEFAULT_PORT,
    LOOPBACK_HOST,
    create_web_app,
    is_loopback_host,
)
from netstack_academy.web.runtime import (
    AcademyRuntime,
    SessionRunner,
    StateDirectoryInsideRepositoryError,
    resolve_content_root,
)

PROGRAM_NAME = "netstack-academy"

#: Nothing went wrong.
EXIT_OK = 0

#: The command ran and reported a problem with the content or the index. Not
#: the same thing as being asked something impossible, which ``argparse``
#: answers with 2.
EXIT_FAILURE = 1

#: Ctrl-C, by the shell convention of 128 plus the signal number.
EXIT_INTERRUPTED = 130

#: What kernel changelogs abbreviate a commit to, and what a page shows.
SHORT_HASH_LENGTH = 12

#: Ports below 1024 need root on every platform this runs on, and needing
#: root to read a kernel checkout would be an odd thing for a learning tool
#: to ask for.
MIN_PORT = 1024
MAX_PORT = 65535


class ServerRunner(Protocol):
    """The one call ``serve`` makes to start a server.

    Narrower than :func:`uvicorn.run`'s real signature on purpose: an
    application object, a host and a port is the whole of what this program is
    allowed to decide, and a test can substitute a recorder that proves where
    it would have bound without binding anywhere.
    """

    def __call__(self, app: Any, *, host: str, port: int) -> None: ...


# ----------------------------------------------------------------------
# Argument types
# ----------------------------------------------------------------------


def _loopback_host(value: str) -> str:
    """Accept any spelling of "this machine", and nothing else.

    Returned unchanged rather than normalised to :data:`LOOPBACK_HOST`,
    because ``localhost`` and ``::1`` are different bind addresses and a
    caller who names one means it.
    """
    if not is_loopback_host(value):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a loopback address; this application is only ever "
            "served to the machine it runs on."
        )
    return value


def _unprivileged_port(value: str) -> int:
    """A port number, and only a port number.

    ``int()`` is also what makes ``--port '9123; touch /tmp/x'`` a usage error
    instead of anything more interesting.
    """
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a port number.") from None
    if not MIN_PORT <= port <= MAX_PORT:
        raise argparse.ArgumentTypeError(
            f"{port} is outside the unprivileged port range {MIN_PORT}-{MAX_PORT}."
        )
    return port


def build_parser() -> argparse.ArgumentParser:
    """The whole command surface, in one place so it can be read at a glance."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description="Read the Linux network stack, one packet at a time.",
    )
    # Required, so a bare invocation prints usage instead of guessing that the
    # caller meant the one command that opens a socket.
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    serve = commands.add_parser(
        "serve", help="Serve the academy on this machine and nowhere else."
    )
    serve.add_argument(
        "--host",
        type=_loopback_host,
        default=LOOPBACK_HOST,
        help=f"Loopback address to bind (default: {LOOPBACK_HOST}).",
    )
    serve.add_argument(
        "--port",
        type=_unprivileged_port,
        default=DEFAULT_PORT,
        help=f"Port to bind (default: {DEFAULT_PORT}).",
    )

    validate = commands.add_parser(
        "validate-content", help="Check that the course loads, and report every problem."
    )
    validate.add_argument(
        "--content-root",
        default=None,
        help="Course directory to check (default: the configured or packaged course).",
    )

    index = commands.add_parser(
        "index", help="Build or refresh the symbol index for the configured checkout."
    )
    index.add_argument(
        "--force",
        action="store_true",
        help="Reindex even when the persisted index already matches HEAD.",
    )

    return parser


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def _short_hash(value: str | None) -> str | None:
    return None if not value else value[:SHORT_HASH_LENGTH]


def _local_url(host: str, port: int) -> str:
    # An IPv6 literal has to be bracketed to be a URL at all: "http://::1:8765"
    # has no unambiguous reading.
    shown = f"[{host}]" if ":" in host else host
    return f"http://{shown}:{port}"


def _print_curriculum_errors(errors: Sequence[CurriculumError]) -> None:
    """Every problem, one per line, located precisely enough to fix.

    On stdout rather than stderr because producing this list *is* the command's
    output -- an author redirecting it to a file wants the problems in the
    file. The one-line verdict goes to stderr so a script watching only that
    stream still learns the run failed.
    """
    print(f"{len(errors)} problem(s) found:")
    for error in errors:
        location = error.path or "<content root>"
        field = f" [{error.field}]" if error.field else ""
        print(f"  {location}{field}: {error.message}")


def _print_curriculum(curriculum: Curriculum) -> None:
    lesson_count = sum(len(module.lessons) for module in curriculum.modules)
    for module in curriculum.modules:
        print(
            f"  {module.slug}: {module.title} "
            f"({len(module.lessons)} lesson(s))"
        )
    print(
        f"{len(curriculum.modules)} module(s), {lesson_count} lesson(s): "
        "content is valid"
    )


def _print_index_run(result: IndexRunResult) -> None:
    """What one pipeline run did, or why it did nothing.

    A failed run reports its reason and nothing else: there is no head to name
    and no counts to give, and the previously persisted generation is
    deliberately still there -- a failed reindex does not destroy a working
    index.
    """
    if result.status == "failed":
        print(
            f"index: failed ({result.reason or 'no reason reported'})",
            file=sys.stderr,
        )
    else:
        head = _short_hash(result.head) or "an unnamed commit"
        print(
            f"index: {result.status} at {head} "
            f"({result.symbol_count} symbol(s), {result.edge_count} edge(s))"
        )

    if result.provider_diagnostics:
        described = [
            f"{diagnostic.provider_name} "
            + ("available" if diagnostic.available else "unavailable")
            + (f" ({diagnostic.reason})" if diagnostic.reason else "")
            for diagnostic in result.provider_diagnostics
        ]
        print(f"providers: {'; '.join(described)}")

    # Partial failures inside an otherwise successful run: one symbol that
    # timed out during semantic enrichment explains a call graph with a gap in
    # it, and is worth saying out loud rather than leaving to be noticed.
    if result.diagnostics:
        print("notes:")
        for note in result.diagnostics:
            print(f"  {note}")


def _report_startup_failure(exception: Exception) -> int:
    """Turn a refused configuration into a verdict rather than a traceback.

    Both cases a runtime can refuse to open for are the caller's to fix: a
    state directory inside the kernel checkout, or a course that does not
    validate. Neither is a bug in this program, so neither gets a stack trace.
    """
    if isinstance(exception, CurriculumValidationError):
        _print_curriculum_errors(exception.errors)
        print("the course does not load; nothing was started", file=sys.stderr)
    else:
        print(str(exception), file=sys.stderr)
    return EXIT_FAILURE


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------


def _serve(
    args: argparse.Namespace,
    *,
    server_runner: ServerRunner,
    session_runner: SessionRunner,
) -> int:
    settings = Settings()
    try:
        runtime = AcademyRuntime.open(settings, session_runner=session_runner)
    except (CurriculumValidationError, StateDirectoryInsideRepositoryError) as exc:
        return _report_startup_failure(exc)

    try:
        application = create_web_app(runtime.context)
        print(f"{PROGRAM_NAME} serving {_local_url(args.host, args.port)}")
        print(f"course: {runtime.content_root}")
        print(f"state:  {runtime.state_dir}")
        # Returns when the server stops. The index is not built here: the
        # first request that needs symbols builds it, so starting up stays
        # measured in milliseconds rather than in minutes of ctags.
        server_runner(application, host=args.host, port=args.port)
    finally:
        runtime.close()
    return EXIT_OK


def _validate_content(args: argparse.Namespace) -> int:
    """Answer "does this course load", touching nothing.

    Loaded without ``strict``, which validates identically -- the strict flag
    only chooses whether the aggregated errors arrive as an exception or as a
    value -- and a command whose entire job is to print all of them wants the
    value.
    """
    if args.content_root is not None:
        content_root = Path(args.content_root).expanduser().resolve()
    else:
        content_root = resolve_content_root(Settings())

    print(f"content root: {content_root}")
    result = load_curriculum(content_root, strict=False)

    if result.errors:
        _print_curriculum_errors(result.errors)
        print("content is not valid", file=sys.stderr)
        return EXIT_FAILURE

    assert result.curriculum is not None  # no errors means a curriculum
    _print_curriculum(result.curriculum)
    return EXIT_OK


def _index(args: argparse.Namespace, *, session_runner: SessionRunner) -> int:
    settings = Settings()
    try:
        runtime = AcademyRuntime.open(settings, session_runner=session_runner)
    except (CurriculumValidationError, StateDirectoryInsideRepositoryError) as exc:
        return _report_startup_failure(exc)

    try:
        print(f"kernel repo: {settings.kernel_repo}")
        print(f"state:       {runtime.state_dir}")
        result = runtime.index.ensure(force=args.force)
        _print_index_run(result)
        return EXIT_FAILURE if result.status == "failed" else EXIT_OK
    finally:
        # A second ``index`` in the same shell must not meet a database this
        # one is still holding.
        runtime.close()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    *,
    server_runner: ServerRunner = uvicorn.run,
    session_runner: SessionRunner = run_indexing_session,
) -> int:
    """Run one command and return its exit status.

    ``argv`` defaults to the real command line so this doubles as the console
    script's entry point, and is a parameter so a test never has to touch
    :data:`sys.argv`. A usage error exits 2 from inside ``argparse``, which is
    the one status this function does not return itself.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "serve":
            return _serve(
                args, server_runner=server_runner, session_runner=session_runner
            )
        if args.command == "validate-content":
            return _validate_content(args)
        # ``argparse`` admits only the three registered commands.
        return _index(args, session_runner=session_runner)
    except KeyboardInterrupt:
        # Ctrl-C is how ``serve`` is meant to end. The runtime is already
        # closed by the ``finally`` above; all that is left is to not print a
        # traceback at someone who stopped the thing on purpose.
        print("stopped", file=sys.stderr)
        return EXIT_INTERRUPTED


__all__ = [
    "DEFAULT_PORT",
    "EXIT_FAILURE",
    "EXIT_INTERRUPTED",
    "EXIT_OK",
    "LOOPBACK_HOST",
    "ServerRunner",
    "build_parser",
    "main",
]
