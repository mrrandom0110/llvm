"""Builders for curriculum model objects used by the learning-side tests.

The learning store, quiz grader and services all consume already-loaded
:class:`~netstack_academy.curriculum.models.Lesson` objects, never files, so
these tests construct models directly instead of round-tripping content
through the loader -- a loader bug should fail the loader's own tests, not
every test in this directory.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from netstack_academy.curriculum.models import (
    Curriculum,
    Lab,
    Lesson,
    MasteryGate,
    Module,
    QuizOption,
    QuizQuestion,
    StructureReference,
    SymbolReference,
)

SCHEMA_VERSION = 1


def make_question(
    *,
    question_id: str = "q-context",
    prompt: str = "In which context does napi_poll run?",
    options: Sequence[tuple[str, str]] = (("a", "Hard IRQ"), ("b", "Softirq")),
    answer: str = "b",
    explanation: str = "NAPI polling is deferred to NET_RX_SOFTIRQ.",
) -> QuizQuestion:
    return QuizQuestion(
        id=question_id,
        prompt=prompt,
        options=tuple(QuizOption(id=option_id, text=text) for option_id, text in options),
        answer=answer,
        explanation=explanation,
    )


def make_lesson(
    *,
    lesson_id: str = "lesson-napi-poll",
    slug: str = "napi-poll",
    title: str = "The NAPI poll loop",
    order: int = 10,
    module_id: str = "module-rx",
    status: str = "published",
    source_path: str | None = None,
    summary: str = "How the NAPI poll loop drains a device queue.",
    objectives: Iterable[str] = ("Explain when napi_poll runs",),
    quiz: Iterable[QuizQuestion] | None = None,
    body_markdown: str = "The driver hands the sk_buff to the stack.",
    body_html: str = "<p>The driver hands the sk_buff to the stack.</p>",
    **overrides: object,
) -> Lesson:
    fields: dict[str, object] = {
        "id": lesson_id,
        "slug": slug,
        "title": title,
        "order": order,
        "module_id": module_id,
        "status": status,
        "source_path": source_path if source_path is not None else f"rx/{slug}.md",
        "summary": summary,
        "objectives": tuple(objectives),
        "prerequisites": (),
        "packet_stage": "rx-softirq",
        "execution_context": "softirq",
        "ownership": "The NAPI instance is owned by the device driver.",
        "locking": "NAPI_STATE_SCHED bit serializes pollers.",
        "rcu": "rcu_read_lock() is held across the receive handler.",
        "structures": (
            StructureReference(name="struct napi_struct", fields=("poll", "weight")),
        ),
        "config_caveats": ("CONFIG_RPS moves work to a remote CPU.",),
        "version_caveats": ("Budget accounting changed in v5.15.",),
        "tracepoints": ("napi:napi_poll",),
        "source_symbols": (SymbolReference(name="napi_poll", path="net/core/dev.c"),),
        "lab": Lab(
            commands=("cat /proc/net/softnet_stat",),
            expected_observations=("The second column stays at zero.",),
            cleanup=("true",),
        ),
        "quiz": tuple(quiz) if quiz is not None else (make_question(),),
        "mastery_gate": MasteryGate(min_quiz_score=0.8, required_review_level=2),
        "body_markdown": body_markdown,
        "body_html": body_html,
    }
    fields.update(overrides)
    return Lesson(**fields)  # type: ignore[arg-type]


def make_module(
    *,
    module_id: str = "module-rx",
    slug: str = "rx-path",
    title: str = "Receive path",
    order: int = 1,
    summary: str = "How a frame becomes an sk_buff.",
    lessons: Iterable[Lesson] | None = None,
) -> Module:
    return Module(
        id=module_id,
        slug=slug,
        title=title,
        order=order,
        summary=summary,
        lessons=tuple(lessons) if lessons is not None else (make_lesson(),),
    )


def make_curriculum(modules: Iterable[Module] | None = None) -> Curriculum:
    return Curriculum(
        schema_version=SCHEMA_VERSION,
        modules=tuple(modules) if modules is not None else (make_module(),),
    )
