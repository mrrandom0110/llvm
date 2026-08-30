"""Contract for the stylesheet and the script.

Both are hand-written, local, and small enough to read. There is no build
step, no bundler and no framework, which is a deliberate constraint rather
than an omission: a learning tool that needs ``npm install`` before it renders
a lesson is a tool that stops working the week its toolchain moves on.

What the CSS owes a reader: a layout that survives a narrow window, a visible
focus ring for anyone navigating by keyboard, and respect for
``prefers-reduced-motion``.

What the JavaScript owes them: search, progress, notes, quiz submission and
the call graph, plus a visible message when a request fails -- and nothing
that a page's content could turn into code.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import netstack_academy.web as web_package

STATIC_ROOT = Path(web_package.__file__).resolve().parent / "static"
CSS_PATH = STATIC_ROOT / "css" / "academy.css"
JS_PATH = STATIC_ROOT / "js" / "academy.js"

#: Ways to turn a string into code. None of them belong in this application:
#: every value the script handles came from a lesson, a note or the index.
FORBIDDEN_JS = ["eval(", "new Function(", "document.write", "innerHTML +="]

#: Every asset is served from this application, so nothing here loads from an
#: origin at all.
FORBIDDEN_REMOTE = ["url(http", "@import url(", 'from "http', "fetch('http", 'fetch("http']


def test_the_stylesheet_ships_with_the_package() -> None:
    assert CSS_PATH.is_file()


def test_the_script_ships_with_the_package() -> None:
    assert JS_PATH.is_file()


def test_the_stylesheet_is_served(client: TestClient) -> None:
    response = client.get("/static/css/academy.css")

    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]


def test_the_script_is_served(client: TestClient) -> None:
    response = client.get("/static/js/academy.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]


def test_static_files_cannot_escape_the_static_directory(
    client: TestClient,
) -> None:
    response = client.get("/static/../settings.py")

    assert response.status_code in (400, 404)


def test_the_layout_is_responsive() -> None:
    css = CSS_PATH.read_text(encoding="utf-8")

    assert "@media" in css
    assert "max-width" in css or "min-width" in css


def test_focus_is_visible_for_keyboard_users() -> None:
    """Kernel reading is link-heavy. Losing the focus ring makes the whole
    application unusable without a mouse.
    """
    css = CSS_PATH.read_text(encoding="utf-8")

    assert ":focus-visible" in css
    assert "outline" in css


def test_motion_is_optional() -> None:
    css = CSS_PATH.read_text(encoding="utf-8")

    assert "prefers-reduced-motion" in css


def test_the_stylesheet_defines_the_graph_and_status_styling() -> None:
    css = CSS_PATH.read_text(encoding="utf-8")

    assert "data-symbol-graph" in css or "symbol-graph" in css


@pytest.mark.parametrize("forbidden", FORBIDDEN_REMOTE)
def test_the_stylesheet_loads_nothing_remote(forbidden: str) -> None:
    css = CSS_PATH.read_text(encoding="utf-8")

    assert forbidden not in css


@pytest.mark.parametrize("forbidden", FORBIDDEN_REMOTE)
def test_the_script_loads_nothing_remote(forbidden: str) -> None:
    javascript = JS_PATH.read_text(encoding="utf-8")

    assert forbidden not in javascript


@pytest.mark.parametrize("forbidden", FORBIDDEN_JS)
def test_the_script_never_turns_data_into_code(forbidden: str) -> None:
    javascript = JS_PATH.read_text(encoding="utf-8")

    assert forbidden not in javascript


@pytest.mark.parametrize(
    "endpoint",
    ["/api/search", "/api/progress", "/api/lessons/", "/api/symbols/"],
)
def test_the_script_talks_to_the_documented_endpoints(endpoint: str) -> None:
    javascript = JS_PATH.read_text(encoding="utf-8")

    assert endpoint in javascript


@pytest.mark.parametrize(
    "hook",
    [
        "data-search-form",
        "data-note-form",
        "data-quiz-form",
        "data-progress-action",
        "data-symbol-graph",
    ],
)
def test_the_script_binds_to_the_rendered_hooks(hook: str) -> None:
    """The page and the script agree on data attributes rather than on class
    names, so restyling cannot silently unbind behaviour.
    """
    javascript = JS_PATH.read_text(encoding="utf-8")

    assert hook in javascript


def test_the_script_reports_failed_requests() -> None:
    """A learning tool that silently swallows a failed save loses notes."""
    javascript = JS_PATH.read_text(encoding="utf-8")

    assert "catch" in javascript
    assert 'role="alert"' in javascript or "data-error" in javascript


def test_the_script_is_deferred_so_content_renders_first(
    client: TestClient,
) -> None:
    html = client.get("/lessons/napi-poll").text

    assert "defer" in html or 'type="module"' in html
