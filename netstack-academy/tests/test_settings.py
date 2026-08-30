from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from netstack_academy.settings import Settings


def test_settings_default_kernel_repo_is_erickurbanov_linux_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KERNEL_REPO", raising=False)
    settings = Settings()
    assert settings.kernel_repo == Path("/home/erickurbanov/linux")


def test_settings_default_editor_scheme_is_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EDITOR_SCHEME", raising=False)
    settings = Settings()
    assert settings.editor_scheme == "cursor"


def test_settings_default_wsl_distro_is_ubuntu_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    settings = Settings()
    assert settings.wsl_distro == "Ubuntu"


def test_settings_wsl_distro_reads_wsl_distro_name_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    settings = Settings()
    assert settings.wsl_distro == "Ubuntu-22.04"


def test_settings_explicit_env_overrides_kernel_repo_default(
    monkeypatch: pytest.MonkeyPatch,
    git_repository: Path,
) -> None:
    monkeypatch.setenv("KERNEL_REPO", str(git_repository))
    settings = Settings()
    assert settings.kernel_repo == git_repository.resolve()


def test_settings_explicit_env_overrides_editor_scheme_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EDITOR_SCHEME", "vscode")
    settings = Settings()
    assert settings.editor_scheme == "vscode"


def test_settings_rejects_unsupported_editor_scheme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EDITOR_SCHEME", "emacs")
    with pytest.raises(ValidationError):
        Settings()
