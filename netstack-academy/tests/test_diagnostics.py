from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import ANY

import pytest
from fastapi.testclient import TestClient

from netstack_academy.app import create_app


@pytest.fixture
def diagnostics_client(
    monkeypatch: pytest.MonkeyPatch,
    git_repository: Path,
) -> TestClient:
    monkeypatch.setenv("KERNEL_REPO", str(git_repository))
    monkeypatch.setenv("EDITOR_SCHEME", "cursor")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setenv(
        "TEST_SYMBOL_PATH",
        str(git_repository / "net" / "ipv4" / "tcp_input.c"),
    )
    monkeypatch.setenv("TEST_SYMBOL_LINE", "1")
    monkeypatch.setenv("TEST_SYMBOL_COLUMN", "1")
    return TestClient(create_app())


def test_health_endpoint_returns_ok(diagnostics_client: TestClient) -> None:
    response = diagnostics_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_diagnostics_exposes_configured_kernel_repo(
    diagnostics_client: TestClient,
    git_repository: Path,
) -> None:
    response = diagnostics_client.get("/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["kernel_repo"] == str(git_repository.resolve())


def test_diagnostics_exposes_git_head_state(
    diagnostics_client: TestClient,
    git_repository: Path,
) -> None:
    expected_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=git_repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    response = diagnostics_client.get("/diagnostics")
    payload = response.json()

    assert payload["repository"]["available"] is True
    assert payload["repository"]["head"] == expected_head


def test_diagnostics_exposes_wsl_distro_and_editor_scheme(
    diagnostics_client: TestClient,
) -> None:
    response = diagnostics_client.get("/diagnostics")
    payload = response.json()

    assert payload["wsl_distro"] == "Ubuntu"
    assert payload["editor_scheme"] == "cursor"


def test_diagnostics_includes_deep_link_when_test_symbol_resolvable(
    diagnostics_client: TestClient,
    git_repository: Path,
) -> None:
    target = (git_repository / "net" / "ipv4" / "tcp_input.c").resolve()

    response = diagnostics_client.get("/diagnostics")
    payload = response.json()

    assert payload["test_symbol"]["resolvable"] is True
    assert payload["test_symbol"]["deep_link"] == (
        f"cursor://vscode-remote/wsl+Ubuntu{target.as_posix()}:1:1"
    )


def test_diagnostics_reports_unresolvable_test_symbol_without_deep_link(
    monkeypatch: pytest.MonkeyPatch,
    git_repository: Path,
) -> None:
    monkeypatch.setenv("KERNEL_REPO", str(git_repository))
    monkeypatch.setenv("TEST_SYMBOL_PATH", str(git_repository / "missing.c"))
    monkeypatch.setenv("TEST_SYMBOL_LINE", "1")
    monkeypatch.setenv("TEST_SYMBOL_COLUMN", "1")

    client = TestClient(create_app())
    payload = client.get("/diagnostics").json()

    assert payload["test_symbol"]["resolvable"] is False
    assert payload["test_symbol"]["deep_link"] is None
    assert payload["test_symbol"]["reason"] is not None


def test_diagnostics_returns_200_with_invalid_test_symbol_line_configuration(
    monkeypatch: pytest.MonkeyPatch,
    git_repository: Path,
) -> None:
    monkeypatch.setenv("KERNEL_REPO", str(git_repository))
    monkeypatch.setenv(
        "TEST_SYMBOL_PATH",
        str(git_repository / "net" / "ipv4" / "tcp_input.c"),
    )
    monkeypatch.setenv("TEST_SYMBOL_LINE", "not-a-number")
    monkeypatch.setenv("TEST_SYMBOL_COLUMN", "1")

    response = TestClient(create_app()).get("/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["configuration"]["valid"] is False
    assert payload["configuration"]["errors"] == [
        {"field": "TEST_SYMBOL_LINE", "message": ANY},
    ]
    assert payload["test_symbol"]["resolvable"] is False
    assert payload["test_symbol"]["deep_link"] is None
    assert "invalid" in payload["test_symbol"]["reason"].lower()


def test_diagnostics_returns_200_with_invalid_test_symbol_column_configuration(
    monkeypatch: pytest.MonkeyPatch,
    git_repository: Path,
) -> None:
    monkeypatch.setenv("KERNEL_REPO", str(git_repository))
    monkeypatch.setenv(
        "TEST_SYMBOL_PATH",
        str(git_repository / "net" / "ipv4" / "tcp_input.c"),
    )
    monkeypatch.setenv("TEST_SYMBOL_LINE", "1")
    monkeypatch.setenv("TEST_SYMBOL_COLUMN", "zero")

    response = TestClient(create_app()).get("/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["configuration"]["valid"] is False
    assert payload["configuration"]["errors"] == [
        {"field": "TEST_SYMBOL_COLUMN", "message": ANY},
    ]
    assert payload["test_symbol"]["resolvable"] is False
    assert payload["test_symbol"]["deep_link"] is None
    assert "invalid" in payload["test_symbol"]["reason"].lower()
