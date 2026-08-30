from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import ValidationError

from netstack_academy.deep_link import build_editor_deep_link
from netstack_academy.repo_inspector import inspect_repository
from netstack_academy.settings import Settings

_SETTINGS_ENV_FIELDS = {
    "kernel_repo": "KERNEL_REPO",
    "editor_scheme": "EDITOR_SCHEME",
    "wsl_distro": "WSL_DISTRO_NAME",
    "test_symbol_path": "TEST_SYMBOL_PATH",
    "test_symbol_line": "TEST_SYMBOL_LINE",
    "test_symbol_column": "TEST_SYMBOL_COLUMN",
}


def _invalid_configuration_payload(exc: ValidationError) -> dict[str, object]:
    errors: list[dict[str, str]] = []
    for error in exc.errors():
        location = error.get("loc", ())
        field_name = str(location[0]) if location else "configuration"
        env_field = _SETTINGS_ENV_FIELDS.get(field_name, field_name.upper())
        errors.append(
            {
                "field": env_field,
                "message": str(error.get("msg", "Invalid value")),
            }
        )

    return {
        "configuration": {
            "valid": False,
            "errors": errors,
        },
        "test_symbol": {
            "resolvable": False,
            "deep_link": None,
            "reason": "Invalid configuration",
        },
    }


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
        try:
            settings = Settings()
        except ValidationError as exc:
            return _invalid_configuration_payload(exc)

        repository = inspect_repository(settings.kernel_repo)
        return {
            "configuration": {
                "valid": True,
                "errors": [],
            },
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
