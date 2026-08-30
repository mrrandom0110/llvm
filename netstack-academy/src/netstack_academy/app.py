from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from netstack_academy.deep_link import build_editor_deep_link
from netstack_academy.repo_inspector import inspect_repository
from netstack_academy.settings import Settings


def _test_symbol_payload(settings: Settings) -> dict[str, object]:
    if settings.test_symbol_path is None:
        return {
            "resolvable": False,
            "deep_link": None,
            "reason": "Test symbol path not configured",
        }

    symbol_path = Path(settings.test_symbol_path)
    if not symbol_path.is_file():
        return {
            "resolvable": False,
            "deep_link": None,
            "reason": "Test symbol file not found",
        }

    try:
        deep_link = build_editor_deep_link(
            file_path=symbol_path,
            line=settings.test_symbol_line,
            column=settings.test_symbol_column,
            kernel_repo=settings.kernel_repo,
            wsl_distro=settings.wsl_distro,
            editor_scheme=settings.editor_scheme,
        )
    except ValueError as exc:
        return {
            "resolvable": False,
            "deep_link": None,
            "reason": str(exc),
        }

    return {
        "resolvable": True,
        "deep_link": deep_link,
        "reason": None,
    }


def create_app() -> FastAPI:
    app = FastAPI(title="netstack-academy")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/diagnostics")
    def diagnostics() -> dict[str, object]:
        settings = Settings()
        repository = inspect_repository(settings.kernel_repo)
        return {
            "kernel_repo": str(settings.kernel_repo.resolve()),
            "repository": {
                "available": repository.available,
                "head": repository.head,
                "reason": repository.reason,
            },
            "wsl_distro": settings.wsl_distro,
            "editor_scheme": settings.editor_scheme,
            "test_symbol": _test_symbol_payload(settings),
        }

    return app
