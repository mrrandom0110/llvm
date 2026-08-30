"""Contract for :mod:`netstack_academy.learning.services`.

The services layer is what a web handler, a CLI, or a test calls to get a
dashboard, a module page, a lesson page or a search result. It composes a
loaded curriculum with the learner's store and, optionally, the symbol
index -- and it does so without importing FastAPI, so the teaching logic
stays testable (and reusable) independently of how it is served.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from netstack_academy.learning.services import (
    LearningService,
    LessonNotFoundError,
    ModuleNotFoundError,
)
from netstack_academy.learning.store import LearningStore

from learning_fakes import ExplodingSymbolIndex, FakeClock, FakeSymbol, FakeSymbolIndex
from lesson_factory import make_curriculum, make_lesson, make_module

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def curriculum():
    rx = make_module(
        module_id="module-rx",
        slug="rx-path",
        title="Receive path",
        order=1,
        lessons=(
            make_lesson(
                lesson_id="lesson-napi-poll",
                slug="napi-poll",
                title="The NAPI poll loop",
                order=10,
                module_id="module-rx",
                summary="How the NAPI poll loop drains a device queue.",
            ),
            make_lesson(
                lesson_id="lesson-gro",
                slug="gro-coalescing",
                title="GRO coalescing",
                order=20,
                module_id="module-rx",
                summary="Merging segments before they reach the socket.",
                objectives=("Describe when GRO gives up",),
            ),
        ),
    )
    tx = make_module(
        module_id="module-tx",
        slug="tx-path",
        title="Transmit path",
        order=2,
        lessons=(
            make_lesson(
                lesson_id="lesson-qdisc",
                slug="qdisc-dequeue",
                title="Qdisc dequeue",
                order=10,
                module_id="module-tx",
                summary="How packets leave a queueing discipline.",
            ),
            make_lesson(
                lesson_id="lesson-draft",
                slug="draft-lesson",
                title="Draft: BQL",
                order=20,
                module_id="module-tx",
                status="draft",
                summary="",
                objectives=(),
                quiz=(),
            ),
        ),
    )
    return make_curriculum((rx, tx))


@pytest.fixture
def service(curriculum, store: LearningStore) -> LearningService:
    return LearningService(curriculum, store)


def test_services_module_does_not_import_fastapi() -> None:
    """A learning service that reaches for FastAPI cannot be reused from a
    CLI or a background job, and drags a web framework into every test that
    touches teaching logic.
    """
    probe = (
        "import sys; import netstack_academy.learning.services; "
        "print('fastapi' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "False"


def test_dashboard_lists_modules_in_curriculum_order(service: LearningService) -> None:
    dashboard = service.dashboard()

    assert [module.slug for module in dashboard.modules] == ["rx-path", "tx-path"]
    assert [lesson.slug for lesson in dashboard.modules[0].lessons] == [
        "napi-poll",
        "gro-coalescing",
    ]


def test_dashboard_counts_progress_across_the_curriculum(
    service: LearningService, store: LearningStore
) -> None:
    store.start_lesson("lesson-napi-poll")
    store.complete_lesson("lesson-napi-poll")
    store.start_lesson("lesson-gro")

    dashboard = service.dashboard()

    assert dashboard.lesson_count == 4
    assert dashboard.completed_count == 1
    assert dashboard.in_progress_count == 1


def test_dashboard_next_lesson_is_the_first_unfinished_published_lesson(
    service: LearningService, store: LearningStore
) -> None:
    store.start_lesson("lesson-napi-poll")
    store.complete_lesson("lesson-napi-poll")

    assert service.dashboard().next_lesson.id == "lesson-gro"


def test_dashboard_next_lesson_skips_drafts(
    service: LearningService, store: LearningStore
) -> None:
    """A draft is visible to its author but is not something to send a
    learner to next.
    """
    for lesson_id in ("lesson-napi-poll", "lesson-gro", "lesson-qdisc"):
        store.start_lesson(lesson_id)
        store.complete_lesson(lesson_id)

    assert service.dashboard().next_lesson is None


def test_dashboard_counts_due_reviews(
    service: LearningService, store: LearningStore, clock: FakeClock
) -> None:
    store.record_review("lesson-napi-poll", correct=False)
    store.record_review("lesson-gro", correct=True)

    assert service.dashboard().due_review_count == 1

    clock.advance(days=1)
    assert service.dashboard().due_review_count == 2


def test_module_view_reports_progress_for_each_lesson(
    service: LearningService, store: LearningStore
) -> None:
    store.start_lesson("lesson-napi-poll")

    module = service.module_view("rx-path")

    assert module.title == "Receive path"
    statuses = {lesson.slug: lesson.progress_status for lesson in module.lessons}
    assert statuses == {"napi-poll": "in_progress", "gro-coalescing": "not_started"}


def test_module_view_rejects_an_unknown_slug(service: LearningService) -> None:
    with pytest.raises(ModuleNotFoundError):
        service.module_view("no-such-module")


def test_lesson_view_exposes_the_rendered_body_and_progress(
    service: LearningService, store: LearningStore
) -> None:
    store.start_lesson("lesson-napi-poll")

    lesson = service.lesson_view("lesson-napi-poll")

    assert lesson.title == "The NAPI poll loop"
    assert lesson.module_slug == "rx-path"
    assert "<p>" in lesson.body_html
    assert lesson.progress_status == "in_progress"


def test_lesson_view_quiz_never_carries_the_answer_key(
    service: LearningService,
) -> None:
    lesson = service.lesson_view("lesson-napi-poll")

    assert [question.id for question in lesson.quiz] == ["q-context"]
    assert not hasattr(lesson.quiz[0], "answer")
    assert not hasattr(lesson.quiz[0], "explanation")


def test_lesson_view_includes_the_learners_note(
    service: LearningService, store: LearningStore
) -> None:
    store.upsert_lesson_note("lesson-napi-poll", "Budget is per poll.")

    assert service.lesson_view("lesson-napi-poll").note == "Budget is per poll."


def test_lesson_view_note_is_none_when_unwritten(service: LearningService) -> None:
    assert service.lesson_view("lesson-napi-poll").note is None


def test_lesson_view_includes_the_review_state(
    service: LearningService, store: LearningStore, clock: FakeClock
) -> None:
    store.record_review("lesson-napi-poll", correct=True)

    lesson = service.lesson_view("lesson-napi-poll")

    assert lesson.review_level == 1
    assert lesson.review_due_at > clock.now


def test_lesson_view_can_be_addressed_by_slug(service: LearningService) -> None:
    assert service.lesson_view("napi-poll").id == "lesson-napi-poll"


def test_lesson_view_of_a_draft_still_renders(service: LearningService) -> None:
    lesson = service.lesson_view("lesson-draft")

    assert lesson.status == "draft"
    assert lesson.quiz == ()


def test_lesson_view_rejects_an_unknown_lesson(service: LearningService) -> None:
    with pytest.raises(LessonNotFoundError):
        service.lesson_view("lesson-nope")


def test_search_matches_lesson_titles_case_insensitively(
    service: LearningService,
) -> None:
    results = service.search("NAPI POLL")

    assert [hit.lesson_id for hit in results.lessons] == ["lesson-napi-poll"]


def test_search_matches_summaries_and_objectives(service: LearningService) -> None:
    assert [hit.lesson_id for hit in service.search("queueing discipline").lessons] == [
        "lesson-qdisc"
    ]
    assert [hit.lesson_id for hit in service.search("gives up").lessons] == [
        "lesson-gro"
    ]


def test_search_matches_lesson_body_text(service: LearningService) -> None:
    assert [hit.lesson_id for hit in service.search("sk_buff").lessons] != []


def test_search_results_are_in_curriculum_order(service: LearningService) -> None:
    results = service.search("sk_buff")

    assert [hit.lesson_id for hit in results.lessons] == [
        "lesson-napi-poll",
        "lesson-gro",
        "lesson-qdisc",
        "lesson-draft",
    ]


def test_search_hits_name_their_module(service: LearningService) -> None:
    hit = service.search("NAPI poll").lessons[0]

    assert hit.module_slug == "rx-path"
    assert hit.title == "The NAPI poll loop"


def test_search_returns_nothing_for_a_blank_query(
    curriculum, store: LearningStore
) -> None:
    """A blank query must not degenerate into "match everything", and must
    not spend a symbol-index round trip either.
    """
    service = LearningService(curriculum, store, symbol_index=ExplodingSymbolIndex())

    results = service.search("   ")

    assert results.lessons == ()
    assert results.symbols == ()


def test_search_includes_symbol_index_results_when_one_is_configured(
    curriculum, store: LearningStore
) -> None:
    index = FakeSymbolIndex(
        [FakeSymbol(id=1, name="napi_poll", kind="function", relative_path="net/core/dev.c", line=6500)]
    )
    service = LearningService(curriculum, store, symbol_index=index)

    results = service.search("napi_poll")

    assert [symbol.name for symbol in results.symbols] == ["napi_poll"]
    assert results.symbols[0].relative_path == "net/core/dev.c"
    assert results.symbols[0].line == 6500


def test_search_without_a_symbol_index_returns_only_lesson_hits(
    service: LearningService,
) -> None:
    results = service.search("napi_poll")

    assert results.symbols == ()
    assert results.lessons != ()


def test_search_forwards_the_query_and_limit_to_the_symbol_index(
    curriculum, store: LearningStore
) -> None:
    index = FakeSymbolIndex()
    service = LearningService(curriculum, store, symbol_index=index)

    service.search("napi_poll", limit=7)

    assert index.calls == [("napi_poll", {"limit": 7})]


def test_search_bounds_the_number_of_lesson_hits(service: LearningService) -> None:
    results = service.search("sk_buff", limit=2)

    assert len(results.lessons) == 2


def test_search_echoes_the_query(service: LearningService) -> None:
    assert service.search("napi_poll").query == "napi_poll"
