"""Contract for :mod:`netstack_academy.web.app`.

The app is composed from an explicit :class:`~netstack_academy.web.context.AcademyContext`
rather than from module-level state, because everything it needs -- the
curriculum, the learner's database, the symbol index, the editor scheme --
is exactly the sort of thing a test, a second instance, or a future second
kernel checkout needs to differ on.

It is also a *local* application. It is only ever bound to the loopback
interface, and it does not publish an interactive API console: there is no
authentication anywhere in this program, so a schema browser is a map of
unauthenticated mutations for anything else that can reach the port.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from netstack_academy.learning.store import LearningStore
from netstack_academy.web import app as app_module
from netstack_academy.web.app import (
    DEFAULT_PORT,
    LOOPBACK_HOST,
    create_web_app,
    is_loopback_host,
)
from netstack_academy.web.context import AcademyContext


def test_create_web_app_returns_a_fastapi_application(
    context: AcademyContext,
) -> None:
    assert isinstance(create_web_app(context), FastAPI)


def test_app_exposes_the_context_it_was_built_from(
    context: AcademyContext,
) -> None:
    """The composition root has to be reachable to be closed.

    The CLI builds a runtime, hands its context to the app, and has to shut
    the same databases down when serving ends.
    """
    app = create_web_app(context)

    assert app.state.context is context


def test_loopback_host_is_the_documented_bind_address() -> None:
    assert LOOPBACK_HOST == "127.0.0.1"
    assert "127.0.0.1" in (app_module.__doc__ or "")


def test_default_port_is_a_fixed_unprivileged_port() -> None:
    assert isinstance(DEFAULT_PORT, int)
    assert 1024 < DEFAULT_PORT < 65536


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_hosts_are_accepted(host: str) -> None:
    assert is_loopback_host(host) is True


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "192.168.1.10", "10.0.0.1", "::", "example.com", ""],
)
def test_non_loopback_hosts_are_rejected(host: str) -> None:
    """A learning tool holding a learner's notes has no business listening
    on a routable address, and there is nothing in this program that would
    stop a caller once it can reach the port.
    """
    assert is_loopback_host(host) is False


def test_interactive_api_docs_are_disabled(client: TestClient) -> None:
    app = client.app

    assert app.docs_url is None
    assert app.redoc_url is None
    assert app.openapi_url is None


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_docs_routes_are_not_served(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 404


def test_health_endpoint_reports_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_progress_reflects_the_injected_store(
    client: TestClient, store: LearningStore
) -> None:
    """Mutations made through the store the app was handed are what the app
    reports -- there is no second, hidden source of learner state.
    """
    before = client.get("/api/progress").json()
    store.start_lesson("lesson-napi-poll")
    after = client.get("/api/progress").json()

    assert before["in_progress_count"] == 0
    assert after["in_progress_count"] == 1


def test_app_ignores_the_process_environment_for_its_kernel_repo(
    client: TestClient,
    kernel_head: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Configuration arrives through the injected settings only.

    An app that re-reads ``KERNEL_REPO`` per request cannot be tested, and
    silently changes behaviour when something else in the process exports a
    variable.
    """
    monkeypatch.setenv("KERNEL_REPO", str(tmp_path / "not-a-repo"))

    payload = client.get("/api/index/status").json()

    assert payload["repository_head"] == kernel_head


def test_two_apps_can_disagree_about_the_editor_scheme(make_client) -> None:
    cursor_client = make_client(editor_scheme="cursor")
    vscode_client = make_client(editor_scheme="vscode")

    cursor_link = cursor_client.get("/api/symbols/napi_poll").json()["deep_link"]
    vscode_link = vscode_client.get("/api/symbols/napi_poll").json()["deep_link"]

    assert cursor_link.startswith("cursor://")
    assert vscode_link.startswith("vscode://")


def test_static_assets_are_mounted_locally(client: TestClient) -> None:
    css = client.get("/static/css/academy.css")
    js = client.get("/static/js/academy.js")

    assert css.status_code == 200
    assert js.status_code == 200


def test_unknown_paths_return_a_not_found_page(client: TestClient) -> None:
    response = client.get("/no-such-page")

    assert response.status_code == 404
