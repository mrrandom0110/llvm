"""Authored course content: models, the Markdown loader, and safe rendering."""

from .loader import (
    CurriculumError,
    CurriculumLoadResult,
    CurriculumValidationError,
    load_curriculum,
)
from .models import (
    SUPPORTED_SCHEMA_VERSION,
    Curriculum,
    Lab,
    Lesson,
    MasteryGate,
    Module,
    PublicQuizOption,
    PublicQuizQuestion,
    QuizOption,
    QuizQuestion,
    StructureReference,
    SymbolReference,
    public_quiz,
)
from .rendering import render_markdown

__all__ = [
    "SUPPORTED_SCHEMA_VERSION",
    "Curriculum",
    "CurriculumError",
    "CurriculumLoadResult",
    "CurriculumValidationError",
    "Lab",
    "Lesson",
    "MasteryGate",
    "Module",
    "PublicQuizOption",
    "PublicQuizQuestion",
    "QuizOption",
    "QuizQuestion",
    "StructureReference",
    "SymbolReference",
    "load_curriculum",
    "public_quiz",
    "render_markdown",
]
