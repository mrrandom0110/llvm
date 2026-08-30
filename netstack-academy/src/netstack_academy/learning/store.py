"""SQLite-backed store for everything a learner accumulates.

Schema overview
---------------

- ``schema_meta`` -- single-row table recording the applied schema version,
  so migrations are idempotent and a reopened database is never rebuilt.
- ``lesson_progress`` -- one row per lesson the learner has touched, and the
  anchor every other table hangs off. ``quiz_attempts`` and ``review_cards``
  carry a foreign key to it, so recording either brings the progress row
  into existence first rather than leaving a dangling reference.
- ``notes`` -- lesson notes and symbol notes in one table, keyed by
  ``(target_type, target_key, relative_path)``. Two ``static`` functions can
  share a name, so a symbol note is identified by name *and* file.
- ``quiz_attempts`` / ``quiz_attempt_responses`` -- one row per submission
  plus its per-question choices, split out so a response set is queryable
  rather than an opaque blob.
- ``review_cards`` -- the Leitner box and next due date for each lesson.

Conventions that the rest of the module depends on:

**Timestamps are canonical UTC text.** Every stored instant is written as
``isoformat(timespec="microseconds")`` in UTC, which is fixed width -- so
lexicographic ordering on the text column *is* chronological ordering, and
``ORDER BY``/``WHERE next_due <= ?`` need no conversion. Naive datetimes are
rejected at the boundary rather than silently interpreted as local time.

**The clock is injected.** ``LearningStore.open(..., clock=...)`` takes a
zero-argument callable returning an aware ``datetime``; tests supply a
controllable one, and nothing in the module calls ``datetime.now`` directly.

**Status transitions are explicit.** ``not_started -> in_progress ->
completed`` and nothing else. Starting an already-started lesson is the
common case (reopening it) and is idempotent, preserving the original start
time; every other move raises, because "complete a lesson I never opened"
is a caller bug, not a shortcut worth papering over with an invented start
time.

**Quiz scoring is server-side.** :meth:`LearningStore.record_quiz_attempt`
takes the loaded lesson and grades it through
:func:`netstack_academy.learning.quiz.grade_quiz`; the score is never read
from the submission.

**"No file" is the empty string, not NULL.** SQLite treats NULLs as
distinct in a UNIQUE index, so a nullable ``relative_path`` would make
``upsert_symbol_note("napi_poll", ...)`` insert a second row every time
instead of updating the first. The column is ``NOT NULL DEFAULT ''`` and
``''`` means "no file"; it is unambiguous because ``''`` is not a valid
relative path and is rejected on the way in. Exports translate it back to
``null``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from netstack_academy.curriculum.models import Lesson
from netstack_academy.indexing.paths import is_safe_relative_path

from .quiz import grade_quiz
from .review import MAX_LEITNER_LEVEL, schedule_review

#: Applied database schema version, recorded in ``schema_meta``.
SCHEMA_VERSION = 1

#: Version of the portable export/import document. Independent of
#: ``SCHEMA_VERSION``: the on-disk layout may change without changing the
#: interchange format, and vice versa.
STATE_VERSION = 1

LessonStatus = Literal["not_started", "in_progress", "completed"]

LESSON_STATUSES: tuple[str, ...] = ("not_started", "in_progress", "completed")

NoteTarget = Literal["lesson", "symbol"]

NOTE_TARGETS: tuple[str, ...] = ("lesson", "symbol")

#: Sentinel stored in ``notes.relative_path`` for a note with no file.
_NO_PATH = ""

Clock = Callable[[], datetime]

_STATE_SECTIONS: tuple[str, ...] = ("progress", "notes", "quiz_attempts", "review_cards")
_STATE_KEYS = frozenset({"version", *_STATE_SECTIONS})
_PROGRESS_FIELDS = frozenset({"lesson_id", "status", "started_at", "completed_at"})
_NOTE_FIELDS = frozenset(
    {"target_type", "target_key", "relative_path", "body", "created_at", "updated_at"}
)
_ATTEMPT_FIELDS = frozenset(
    {
        "lesson_id",
        "score",
        "correct_count",
        "question_count",
        "responses",
        "created_at",
    }
)
_CARD_FIELDS = frozenset({"lesson_id", "level", "next_due", "last_reviewed_at"})


class InvalidStatusTransitionError(ValueError):
    """Raised for any lesson status move outside the explicit machine."""


class UnsafeNotePathError(ValueError):
    """Raised when a symbol note names a path that escapes the repository."""


class StateImportError(ValueError):
    """Raised when an imported state document fails validation.

    Always raised *before* anything is written: a rejected import leaves
    the store byte-identical to what it was.
    """


@dataclass(frozen=True, slots=True)
class LessonProgress:
    """Where the learner stands on one lesson."""

    lesson_id: str
    status: LessonStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class Note:
    """A note about a lesson or about a kernel symbol.

    ``relative_path`` is only meaningful for symbol notes, where it
    distinguishes same-named ``static`` functions in different files.
    """

    id: int
    target_type: NoteTarget
    target_key: str
    relative_path: str | None
    body: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class QuizAttempt:
    """One graded submission. ``score`` is always computed server-side."""

    id: int
    lesson_id: str
    score: float
    correct_count: int
    question_count: int
    responses: dict[str, str]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ReviewCard:
    """A lesson's position on the Leitner ladder."""

    lesson_id: str
    level: int
    next_due: datetime
    last_reviewed_at: datetime | None = None


_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS lesson_progress (
        lesson_id TEXT PRIMARY KEY,
        status TEXT NOT NULL
            CHECK (status IN ('not_started', 'in_progress', 'completed')),
        started_at TEXT,
        completed_at TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        target_type TEXT NOT NULL CHECK (target_type IN ('lesson', 'symbol')),
        target_key TEXT NOT NULL,
        relative_path TEXT NOT NULL DEFAULT '',
        body TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (target_type, target_key, relative_path)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS quiz_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lesson_id TEXT NOT NULL
            REFERENCES lesson_progress(lesson_id) ON DELETE CASCADE,
        score REAL NOT NULL,
        correct_count INTEGER NOT NULL,
        question_count INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS quiz_attempt_responses (
        attempt_id INTEGER NOT NULL
            REFERENCES quiz_attempts(id) ON DELETE CASCADE,
        question_id TEXT NOT NULL,
        option_id TEXT NOT NULL,
        PRIMARY KEY (attempt_id, question_id)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS review_cards (
        lesson_id TEXT PRIMARY KEY
            REFERENCES lesson_progress(lesson_id) ON DELETE CASCADE,
        level INTEGER NOT NULL CHECK (level BETWEEN 0 AND {MAX_LEITNER_LEVEL}),
        next_due TEXT NOT NULL,
        last_reviewed_at TEXT
    )
    """,
    # Attempts are always read for one lesson, and the review queue always
    # asks "what is due at or before now"; both would otherwise scan.
    "CREATE INDEX IF NOT EXISTS idx_quiz_attempts_lesson ON quiz_attempts(lesson_id)",
    "CREATE INDEX IF NOT EXISTS idx_review_cards_next_due ON review_cards(next_due)",
)


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


def _require_aware(moment: datetime, *, what: str) -> datetime:
    if not isinstance(moment, datetime):
        raise ValueError(f"{what} must be a datetime, got {moment!r}")
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"{what} must be timezone-aware; state is stored in UTC")
    return moment.astimezone(timezone.utc)


def _to_text(moment: datetime) -> str:
    """Canonical, fixed-width UTC text form (sorts chronologically)."""
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _from_text(text: str | None) -> datetime | None:
    if text is None:
        return None
    return datetime.fromisoformat(text)


class LearningStore:
    """The learner's durable state for one academy installation."""

    def __init__(self, connection: sqlite3.Connection, *, clock: Clock | None = None) -> None:
        self._connection = connection
        self._clock: Clock = clock if clock is not None else _default_clock

    @classmethod
    def open(cls, db_path: str | Path, *, clock: Clock | None = None) -> "LearningStore":
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # ``check_same_thread=False`` because the store is opened by whoever
        # composes the application and then used from wherever a request is
        # served, which is not the same thread. What makes that safe is that
        # every caller uses one connection from one thread *at a time*: the
        # web layer serves every handler that touches this store on its
        # event-loop thread, and no handler awaits in the middle of a
        # transaction. Concurrent use of a single connection would still be a
        # bug, and this flag does not make it one less of a bug.
        connection = sqlite3.connect(
            db_path, isolation_level=None, check_same_thread=False
        )
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")

        store = cls(connection, clock=clock)
        store._run_migrations()
        return store

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT version FROM schema_meta WHERE id = 1"
        ).fetchone()
        return row[0] if row is not None else 0

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "LearningStore":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _run_migrations(self) -> None:
        connection = self._connection
        connection.execute("BEGIN")
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    version INTEGER NOT NULL
                )
                """
            )
            row = connection.execute(
                "SELECT version FROM schema_meta WHERE id = 1"
            ).fetchone()
            current_version = row[0] if row is not None else 0

            if current_version < SCHEMA_VERSION:
                for statement in _SCHEMA_STATEMENTS:
                    connection.execute(statement)
                connection.execute(
                    """
                    INSERT INTO schema_meta (id, version) VALUES (1, ?)
                    ON CONFLICT(id) DO UPDATE SET version = excluded.version
                    """,
                    (SCHEMA_VERSION,),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connection
        connection.execute("BEGIN")
        try:
            yield connection
        except Exception:
            connection.rollback()
            raise
        connection.commit()

    def _now(self) -> datetime:
        return _require_aware(self._clock(), what="The injected clock reading")

    # ------------------------------------------------------------------
    # Lesson progress
    # ------------------------------------------------------------------

    def get_progress(self, lesson_id: str) -> LessonProgress:
        """Progress for ``lesson_id``; ``not_started`` when never touched.

        Reading progress never creates a row: a lesson the learner has only
        looked at from the dashboard is not "started".
        """
        row = self._connection.execute(
            "SELECT lesson_id, status, started_at, completed_at "
            "FROM lesson_progress WHERE lesson_id = ?",
            (lesson_id,),
        ).fetchone()
        if row is None:
            return LessonProgress(lesson_id=lesson_id, status="not_started")
        return _row_to_progress(row)

    def list_progress(self) -> list[LessonProgress]:
        rows = self._connection.execute(
            "SELECT lesson_id, status, started_at, completed_at "
            "FROM lesson_progress ORDER BY lesson_id"
        ).fetchall()
        return [_row_to_progress(row) for row in rows]

    def start_lesson(self, lesson_id: str) -> LessonProgress:
        """Move a lesson to ``in_progress``; idempotent once started."""
        progress = self.get_progress(lesson_id)
        if progress.status == "in_progress":
            return progress
        if progress.status != "not_started":
            raise InvalidStatusTransitionError(
                f"Cannot move lesson {lesson_id!r} from {progress.status!r} to "
                "'in_progress'"
            )

        started_at = self._now()
        self._connection.execute(
            """
            INSERT INTO lesson_progress (lesson_id, status, started_at, completed_at)
            VALUES (?, 'in_progress', ?, NULL)
            ON CONFLICT(lesson_id) DO UPDATE SET
                status = 'in_progress',
                started_at = excluded.started_at
            """,
            (lesson_id, _to_text(started_at)),
        )
        return LessonProgress(
            lesson_id=lesson_id, status="in_progress", started_at=started_at
        )

    def complete_lesson(self, lesson_id: str) -> LessonProgress:
        """Move a started lesson to ``completed`` and stamp the time."""
        progress = self.get_progress(lesson_id)
        if progress.status != "in_progress":
            raise InvalidStatusTransitionError(
                f"Cannot move lesson {lesson_id!r} from {progress.status!r} to "
                "'completed'; a lesson must be 'in_progress' first"
            )

        completed_at = self._now()
        self._connection.execute(
            "UPDATE lesson_progress SET status = 'completed', completed_at = ? "
            "WHERE lesson_id = ?",
            (_to_text(completed_at), lesson_id),
        )
        return LessonProgress(
            lesson_id=lesson_id,
            status="completed",
            started_at=progress.started_at,
            completed_at=completed_at,
        )

    def _ensure_progress_row(self, lesson_id: str) -> None:
        """Create a ``not_started`` row so dependent rows have their anchor."""
        self._connection.execute(
            "INSERT OR IGNORE INTO lesson_progress (lesson_id, status) "
            "VALUES (?, 'not_started')",
            (lesson_id,),
        )

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------

    def upsert_lesson_note(self, lesson_id: str, body: str) -> Note:
        """Create or replace the learner's note on a lesson."""
        return self._upsert_note("lesson", lesson_id, body, _NO_PATH)

    def upsert_symbol_note(
        self, symbol_name: str, body: str, *, relative_path: str | None = None
    ) -> Note:
        """Create or replace the learner's note on a symbol.

        ``relative_path`` is untrusted input (it arrives from a URL or an
        imported document), so it is validated lexically before it is
        stored; nothing is written when it is rejected.
        """
        path = _NO_PATH
        if relative_path is not None:
            if not is_safe_relative_path(relative_path):
                raise UnsafeNotePathError(
                    f"Note path {relative_path!r} is not a safe repository-relative path"
                )
            path = relative_path
        return self._upsert_note("symbol", symbol_name, body, path)

    def _upsert_note(
        self, target_type: str, target_key: str, body: str, relative_path: str
    ) -> Note:
        if not body.strip():
            raise ValueError(
                "Note body must not be blank; delete the note instead of emptying it"
            )

        now = self._now()
        timestamp = _to_text(now)
        self._connection.execute(
            """
            INSERT INTO notes (
                target_type, target_key, relative_path, body, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(target_type, target_key, relative_path) DO UPDATE SET
                body = excluded.body,
                updated_at = excluded.updated_at
            """,
            (target_type, target_key, relative_path, body, timestamp, timestamp),
        )
        note = self._find_note(target_type, target_key, relative_path)
        assert note is not None  # just written
        return note

    def _find_note(
        self, target_type: str, target_key: str, relative_path: str
    ) -> Note | None:
        row = self._connection.execute(
            "SELECT id, target_type, target_key, relative_path, body, created_at, "
            "updated_at FROM notes "
            "WHERE target_type = ? AND target_key = ? AND relative_path = ?",
            (target_type, target_key, relative_path),
        ).fetchone()
        return _row_to_note(row) if row is not None else None

    def get_lesson_note(self, lesson_id: str) -> Note | None:
        return self._find_note("lesson", lesson_id, _NO_PATH)

    def get_symbol_note(
        self, symbol_name: str, *, relative_path: str | None = None
    ) -> Note | None:
        return self._find_note("symbol", symbol_name, relative_path or _NO_PATH)

    def delete_lesson_note(self, lesson_id: str) -> bool:
        """Delete a lesson note; ``True`` when one was actually removed."""
        return self._delete_note("lesson", lesson_id, _NO_PATH)

    def delete_symbol_note(
        self, symbol_name: str, *, relative_path: str | None = None
    ) -> bool:
        return self._delete_note("symbol", symbol_name, relative_path or _NO_PATH)

    def _delete_note(
        self, target_type: str, target_key: str, relative_path: str
    ) -> bool:
        cursor = self._connection.execute(
            "DELETE FROM notes "
            "WHERE target_type = ? AND target_key = ? AND relative_path = ?",
            (target_type, target_key, relative_path),
        )
        return cursor.rowcount > 0

    def list_notes(self) -> list[Note]:
        rows = self._connection.execute(
            "SELECT id, target_type, target_key, relative_path, body, created_at, "
            "updated_at FROM notes "
            "ORDER BY target_type, target_key, relative_path"
        ).fetchall()
        return [_row_to_note(row) for row in rows]

    # ------------------------------------------------------------------
    # Quiz attempts
    # ------------------------------------------------------------------

    def record_quiz_attempt(
        self, lesson: Lesson, responses: Mapping[str, str]
    ) -> QuizAttempt:
        """Grade ``responses`` against ``lesson`` and persist the attempt.

        Grading happens first and outside the transaction, so a submission
        naming an unknown question or option raises without leaving a
        half-written attempt behind.
        """
        grade = grade_quiz(lesson, responses)
        created_at = self._now()
        submitted = {str(key): str(value) for key, value in responses.items()}

        with self._transaction() as connection:
            self._ensure_progress_row(lesson.id)
            cursor = connection.execute(
                """
                INSERT INTO quiz_attempts (
                    lesson_id, score, correct_count, question_count, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    lesson.id,
                    grade.score,
                    grade.correct_count,
                    grade.question_count,
                    _to_text(created_at),
                ),
            )
            attempt_id = int(cursor.lastrowid)
            connection.executemany(
                "INSERT INTO quiz_attempt_responses (attempt_id, question_id, option_id) "
                "VALUES (?, ?, ?)",
                [
                    (attempt_id, question_id, option_id)
                    for question_id, option_id in sorted(submitted.items())
                ],
            )

        return QuizAttempt(
            id=attempt_id,
            lesson_id=lesson.id,
            score=grade.score,
            correct_count=grade.correct_count,
            question_count=grade.question_count,
            responses=submitted,
            created_at=created_at,
        )

    def list_quiz_attempts(self, lesson_id: str) -> list[QuizAttempt]:
        """Every attempt for a lesson, oldest first."""
        rows = self._connection.execute(
            "SELECT id, lesson_id, score, correct_count, question_count, created_at "
            "FROM quiz_attempts WHERE lesson_id = ? ORDER BY created_at, id",
            (lesson_id,),
        ).fetchall()

        attempts: list[QuizAttempt] = []
        for row in rows:
            attempts.append(
                QuizAttempt(
                    id=row[0],
                    lesson_id=row[1],
                    score=row[2],
                    correct_count=row[3],
                    question_count=row[4],
                    responses=self._attempt_responses(row[0]),
                    created_at=_from_text(row[5]),
                )
            )
        return attempts

    def _attempt_responses(self, attempt_id: int) -> dict[str, str]:
        rows = self._connection.execute(
            "SELECT question_id, option_id FROM quiz_attempt_responses "
            "WHERE attempt_id = ? ORDER BY question_id",
            (attempt_id,),
        ).fetchall()
        return {question_id: option_id for question_id, option_id in rows}

    # ------------------------------------------------------------------
    # Spaced review
    # ------------------------------------------------------------------

    def review_card(self, lesson_id: str) -> ReviewCard | None:
        row = self._connection.execute(
            "SELECT lesson_id, level, next_due, last_reviewed_at FROM review_cards "
            "WHERE lesson_id = ?",
            (lesson_id,),
        ).fetchone()
        return _row_to_card(row) if row is not None else None

    def record_review(self, lesson_id: str, *, correct: bool) -> ReviewCard:
        """Apply one review outcome to a lesson's card, creating it if needed."""
        existing = self.review_card(lesson_id)
        now = self._now()
        schedule = schedule_review(
            existing.level if existing is not None else 0, correct=correct, now=now
        )

        with self._transaction() as connection:
            self._ensure_progress_row(lesson_id)
            connection.execute(
                """
                INSERT INTO review_cards (lesson_id, level, next_due, last_reviewed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(lesson_id) DO UPDATE SET
                    level = excluded.level,
                    next_due = excluded.next_due,
                    last_reviewed_at = excluded.last_reviewed_at
                """,
                (lesson_id, schedule.level, _to_text(schedule.next_due), _to_text(now)),
            )

        return ReviewCard(
            lesson_id=lesson_id,
            level=schedule.level,
            next_due=schedule.next_due,
            last_reviewed_at=now,
        )

    def due_reviews(self, *, now: datetime | None = None) -> list[ReviewCard]:
        """Cards due at or before ``now`` (the injected clock by default)."""
        moment = self._now() if now is None else _require_aware(now, what="'now'")
        rows = self._connection.execute(
            "SELECT lesson_id, level, next_due, last_reviewed_at FROM review_cards "
            "WHERE next_due <= ? ORDER BY next_due, lesson_id",
            (_to_text(moment),),
        ).fetchall()
        return [_row_to_card(row) for row in rows]

    # ------------------------------------------------------------------
    # Portable state transfer
    # ------------------------------------------------------------------

    def export_state(self) -> dict[str, Any]:
        """The whole learner state as a JSON-serializable document."""
        return {
            "version": STATE_VERSION,
            "progress": [
                {
                    "lesson_id": progress.lesson_id,
                    "status": progress.status,
                    "started_at": _optional_text(progress.started_at),
                    "completed_at": _optional_text(progress.completed_at),
                }
                for progress in self.list_progress()
            ],
            "notes": [
                {
                    "target_type": note.target_type,
                    "target_key": note.target_key,
                    "relative_path": note.relative_path,
                    "body": note.body,
                    "created_at": _to_text(note.created_at),
                    "updated_at": _to_text(note.updated_at),
                }
                for note in self.list_notes()
            ],
            "quiz_attempts": [
                {
                    "lesson_id": attempt.lesson_id,
                    "score": attempt.score,
                    "correct_count": attempt.correct_count,
                    "question_count": attempt.question_count,
                    "responses": dict(attempt.responses),
                    "created_at": _to_text(attempt.created_at),
                }
                for attempt in self._all_attempts()
            ],
            "review_cards": [
                {
                    "lesson_id": card.lesson_id,
                    "level": card.level,
                    "next_due": _to_text(card.next_due),
                    "last_reviewed_at": _optional_text(card.last_reviewed_at),
                }
                for card in self._all_cards()
            ],
        }

    def import_state(self, payload: Mapping[str, Any]) -> None:
        """Replace the local state with ``payload``, or change nothing.

        The document is validated in full first, so a problem in the last
        record cannot leave the first ones applied. It is a *restore*, not a
        merge: anything not in the payload is gone afterwards, because
        reconciling two divergent histories would need conflict rules this
        program deliberately does not have.

        ``payload`` is only read, never modified.
        """
        state = _validate_state(payload)

        with self._transaction() as connection:
            connection.execute("DELETE FROM quiz_attempt_responses")
            connection.execute("DELETE FROM quiz_attempts")
            connection.execute("DELETE FROM review_cards")
            connection.execute("DELETE FROM notes")
            connection.execute("DELETE FROM lesson_progress")

            connection.executemany(
                "INSERT INTO lesson_progress (lesson_id, status, started_at, completed_at) "
                "VALUES (?, ?, ?, ?)",
                state.progress,
            )
            # Attempts and cards are anchored to a progress row; a payload
            # may legitimately carry an attempt for a lesson whose progress
            # row was never created (an anonymous quiz run), so the anchor
            # is materialized rather than treated as a validation failure.
            for lesson_id in state.anchor_lesson_ids:
                connection.execute(
                    "INSERT OR IGNORE INTO lesson_progress (lesson_id, status) "
                    "VALUES (?, 'not_started')",
                    (lesson_id,),
                )

            connection.executemany(
                "INSERT INTO notes (target_type, target_key, relative_path, body, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                state.notes,
            )

            for attempt, responses in state.quiz_attempts:
                cursor = connection.execute(
                    "INSERT INTO quiz_attempts (lesson_id, score, correct_count, "
                    "question_count, created_at) VALUES (?, ?, ?, ?, ?)",
                    attempt,
                )
                attempt_id = int(cursor.lastrowid)
                connection.executemany(
                    "INSERT INTO quiz_attempt_responses (attempt_id, question_id, "
                    "option_id) VALUES (?, ?, ?)",
                    [
                        (attempt_id, question_id, option_id)
                        for question_id, option_id in responses
                    ],
                )

            connection.executemany(
                "INSERT INTO review_cards (lesson_id, level, next_due, last_reviewed_at) "
                "VALUES (?, ?, ?, ?)",
                state.review_cards,
            )

    def _all_attempts(self) -> list[QuizAttempt]:
        rows = self._connection.execute(
            "SELECT id, lesson_id, score, correct_count, question_count, created_at "
            "FROM quiz_attempts ORDER BY created_at, id"
        ).fetchall()
        return [
            QuizAttempt(
                id=row[0],
                lesson_id=row[1],
                score=row[2],
                correct_count=row[3],
                question_count=row[4],
                responses=self._attempt_responses(row[0]),
                created_at=_from_text(row[5]),
            )
            for row in rows
        ]

    def _all_cards(self) -> list[ReviewCard]:
        rows = self._connection.execute(
            "SELECT lesson_id, level, next_due, last_reviewed_at FROM review_cards "
            "ORDER BY lesson_id"
        ).fetchall()
        return [_row_to_card(row) for row in rows]


def _row_to_progress(row: Sequence[Any]) -> LessonProgress:
    return LessonProgress(
        lesson_id=row[0],
        status=row[1],
        started_at=_from_text(row[2]),
        completed_at=_from_text(row[3]),
    )


def _row_to_note(row: Sequence[Any]) -> Note:
    return Note(
        id=row[0],
        target_type=row[1],
        target_key=row[2],
        relative_path=row[3] or None,
        body=row[4],
        created_at=_from_text(row[5]),
        updated_at=_from_text(row[6]),
    )


def _row_to_card(row: Sequence[Any]) -> ReviewCard:
    return ReviewCard(
        lesson_id=row[0],
        level=row[1],
        next_due=_from_text(row[2]),
        last_reviewed_at=_from_text(row[3]),
    )


def _optional_text(moment: datetime | None) -> str | None:
    return None if moment is None else _to_text(moment)


@dataclass(frozen=True, slots=True)
class _ValidatedState:
    """Row tuples ready to insert, produced before any write happens."""

    progress: tuple[tuple[str, str, str | None, str | None], ...]
    notes: tuple[tuple[str, str, str, str, str, str], ...]
    quiz_attempts: tuple[
        tuple[tuple[str, float, int, int, str], tuple[tuple[str, str], ...]], ...
    ]
    review_cards: tuple[tuple[str, int, str, str | None], ...]
    anchor_lesson_ids: tuple[str, ...]


def _validate_state(payload: Mapping[str, Any]) -> _ValidatedState:
    if not isinstance(payload, Mapping):
        raise StateImportError(f"State document must be a mapping, got {payload!r}")

    unknown = sorted(str(key) for key in payload if key not in _STATE_KEYS)
    if unknown:
        raise StateImportError(
            f"Unknown key(s) in state document: {', '.join(repr(key) for key in unknown)}"
        )

    if "version" not in payload:
        raise StateImportError("State document is missing 'version'")
    version = payload["version"]
    if version != STATE_VERSION:
        raise StateImportError(
            f"Unsupported state version {version!r} (this build reads {STATE_VERSION})"
        )

    progress = tuple(
        _validate_progress(record) for record in _section(payload, "progress")
    )
    _reject_duplicates(
        [record[0] for record in progress], what="progress lesson_id"
    )

    notes = tuple(_validate_note(record) for record in _section(payload, "notes"))
    _reject_duplicates(
        [(record[0], record[1], record[2]) for record in notes], what="note key"
    )

    attempts = tuple(
        _validate_attempt(record) for record in _section(payload, "quiz_attempts")
    )

    cards = tuple(_validate_card(record) for record in _section(payload, "review_cards"))
    _reject_duplicates([record[0] for record in cards], what="review card lesson_id")

    known_lessons = {record[0] for record in progress}
    referenced = {attempt[0][0] for attempt in attempts} | {card[0] for card in cards}
    anchors = sorted(referenced - known_lessons)

    return _ValidatedState(
        progress=progress,
        notes=notes,
        quiz_attempts=attempts,
        review_cards=cards,
        anchor_lesson_ids=tuple(anchors),
    )


def _section(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    records = payload.get(key, [])
    if not isinstance(records, list):
        raise StateImportError(f"State section {key!r} must be a list, got {records!r}")
    for record in records:
        if not isinstance(record, Mapping):
            raise StateImportError(
                f"State section {key!r} must contain mappings, got {record!r}"
            )
    return records


def _check_fields(
    record: Mapping[str, Any], allowed: frozenset[str], *, section: str
) -> None:
    unknown = sorted(str(key) for key in record if key not in allowed)
    if unknown:
        raise StateImportError(
            f"Unknown field(s) in {section} record: "
            f"{', '.join(repr(key) for key in unknown)}"
        )
    missing = sorted(allowed - set(record))
    if missing:
        raise StateImportError(
            f"Missing field(s) in {section} record: "
            f"{', '.join(repr(key) for key in missing)}"
        )


def _required_str(record: Mapping[str, Any], key: str, *, section: str) -> str:
    value = record[key]
    if not isinstance(value, str) or not value.strip():
        raise StateImportError(
            f"{section} field {key!r} must be a non-empty string, got {value!r}"
        )
    return value


def _timestamp(record: Mapping[str, Any], key: str, *, section: str) -> str:
    value = record[key]
    if not isinstance(value, str):
        raise StateImportError(
            f"{section} field {key!r} must be an ISO-8601 timestamp, got {value!r}"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise StateImportError(
            f"{section} field {key!r} is not a valid ISO-8601 timestamp: {value!r}"
        ) from exc
    if parsed.tzinfo is None:
        raise StateImportError(
            f"{section} field {key!r} must carry a UTC offset, got {value!r}"
        )
    return _to_text(parsed)


def _optional_timestamp(
    record: Mapping[str, Any], key: str, *, section: str
) -> str | None:
    if record[key] is None:
        return None
    return _timestamp(record, key, section=section)


def _validate_progress(record: Mapping[str, Any]) -> tuple[str, str, str | None, str | None]:
    _check_fields(record, _PROGRESS_FIELDS, section="progress")

    lesson_id = _required_str(record, "lesson_id", section="progress")
    status = record["status"]
    if status not in LESSON_STATUSES:
        raise StateImportError(
            f"Unknown lesson status {status!r} (expected one of "
            f"{', '.join(LESSON_STATUSES)})"
        )

    started_at = _optional_timestamp(record, "started_at", section="progress")
    completed_at = _optional_timestamp(record, "completed_at", section="progress")

    if status == "in_progress" and started_at is None:
        raise StateImportError(
            f"Lesson {lesson_id!r} is 'in_progress' but has no 'started_at'"
        )
    if status == "completed" and completed_at is None:
        raise StateImportError(
            f"Lesson {lesson_id!r} is 'completed' but has no 'completed_at'"
        )

    return lesson_id, status, started_at, completed_at


def _validate_note(record: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
    _check_fields(record, _NOTE_FIELDS, section="note")

    target_type = record["target_type"]
    if target_type not in NOTE_TARGETS:
        raise StateImportError(
            f"Unknown note target type {target_type!r} (expected one of "
            f"{', '.join(NOTE_TARGETS)})"
        )

    target_key = _required_str(record, "target_key", section="note")
    body = _required_str(record, "body", section="note")

    raw_path = record["relative_path"]
    if raw_path is None:
        relative_path = _NO_PATH
    elif not isinstance(raw_path, str) or not is_safe_relative_path(raw_path):
        raise StateImportError(
            f"Note path {raw_path!r} is not a safe repository-relative path"
        )
    else:
        relative_path = raw_path

    if target_type == "lesson" and relative_path != _NO_PATH:
        raise StateImportError(
            f"Lesson note {target_key!r} must not carry a file path, got {raw_path!r}"
        )

    return (
        target_type,
        target_key,
        relative_path,
        body,
        _timestamp(record, "created_at", section="note"),
        _timestamp(record, "updated_at", section="note"),
    )


def _validate_attempt(
    record: Mapping[str, Any],
) -> tuple[tuple[str, float, int, int, str], tuple[tuple[str, str], ...]]:
    _check_fields(record, _ATTEMPT_FIELDS, section="quiz attempt")

    lesson_id = _required_str(record, "lesson_id", section="quiz attempt")

    score = record["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise StateImportError(f"Quiz attempt score must be a number, got {score!r}")
    if not 0.0 <= float(score) <= 1.0:
        raise StateImportError(
            f"Quiz attempt score {score!r} is outside the 0..1 range"
        )

    correct_count = _non_negative_int(record, "correct_count", section="quiz attempt")
    question_count = _non_negative_int(record, "question_count", section="quiz attempt")
    if correct_count > question_count:
        raise StateImportError(
            f"Quiz attempt claims {correct_count} correct of {question_count} questions"
        )

    raw_responses = record["responses"]
    if not isinstance(raw_responses, Mapping):
        raise StateImportError(
            f"Quiz attempt responses must be a mapping, got {raw_responses!r}"
        )
    responses: list[tuple[str, str]] = []
    for question_id, option_id in sorted(raw_responses.items()):
        if not isinstance(question_id, str) or not isinstance(option_id, str):
            raise StateImportError(
                f"Quiz attempt response {question_id!r}: {option_id!r} must be strings"
            )
        responses.append((question_id, option_id))

    return (
        (
            lesson_id,
            float(score),
            correct_count,
            question_count,
            _timestamp(record, "created_at", section="quiz attempt"),
        ),
        tuple(responses),
    )


def _validate_card(record: Mapping[str, Any]) -> tuple[str, int, str, str | None]:
    _check_fields(record, _CARD_FIELDS, section="review card")

    lesson_id = _required_str(record, "lesson_id", section="review card")

    level = record["level"]
    if isinstance(level, bool) or not isinstance(level, int):
        raise StateImportError(f"Review level must be an integer, got {level!r}")
    if not 0 <= level <= MAX_LEITNER_LEVEL:
        raise StateImportError(
            f"Review level {level!r} is outside the ladder (0..{MAX_LEITNER_LEVEL})"
        )

    return (
        lesson_id,
        level,
        _timestamp(record, "next_due", section="review card"),
        _optional_timestamp(record, "last_reviewed_at", section="review card"),
    )


def _non_negative_int(record: Mapping[str, Any], key: str, *, section: str) -> int:
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StateImportError(
            f"{section} field {key!r} must be a non-negative integer, got {value!r}"
        )
    return value


def _reject_duplicates(keys: Sequence[Any], *, what: str) -> None:
    seen: set[Any] = set()
    for key in keys:
        if key in seen:
            raise StateImportError(f"Duplicate {what} in state document: {key!r}")
        seen.add(key)
