"""Assertions shared by the page tests.

These check *structure*, not markup byte-for-byte. A test that compares a
whole rendered page against a snapshot fails on every wording change and
passes when a landmark quietly disappears, which is the wrong way round: the
things worth pinning here are the ones a screen reader, a keyboard, or a
browser's content security policy actually depends on.
"""

from __future__ import annotations

import re

#: ``<script src="...">`` and ``<link href="...">``, whatever the attribute
#: order.
_SCRIPT_SRC_RE = re.compile(r"<script[^>]*\ssrc=\"([^\"]*)\"", re.IGNORECASE)
_LINK_HREF_RE = re.compile(r"<link[^>]*\shref=\"([^\"]*)\"", re.IGNORECASE)
_INLINE_SCRIPT_RE = re.compile(
    r"<script(?![^>]*\ssrc=)[^>]*>(.*?)</script>", re.IGNORECASE | re.DOTALL
)


def assert_local_assets_only(html: str) -> None:
    """No CDN, no analytics, no webfont host: every asset is served locally.

    This is both a privacy property (a local learning tool has no business
    talking to anyone) and an offline one: a kernel reading session on a
    laptop with no network must render exactly the same page.
    """
    for source in _SCRIPT_SRC_RE.findall(html) + _LINK_HREF_RE.findall(html):
        assert source.startswith("/static/"), f"non-local asset: {source!r}"


def assert_no_inline_scripts(html: str) -> None:
    """Behaviour lives in ``/static/js``, never in the page.

    An inline script is what forces ``script-src 'unsafe-inline'`` into the
    content security policy, and once that is there the policy stops
    defending against injected markup at all.
    """
    for body in _INLINE_SCRIPT_RE.findall(html):
        assert not body.strip(), f"inline script found: {body.strip()[:80]!r}"


def assert_responsive_shell(html: str) -> None:
    """The document scales on a phone and starts with a real landmark."""
    assert 'name="viewport"' in html
    assert "width=device-width" in html
    assert "<main" in html
    assert "<h1" in html


def assert_keyboard_reachable(html: str) -> None:
    """A skip link is the cheapest thing that makes a page keyboard-usable."""
    assert 'href="#main"' in html
    assert 'id="main"' in html


def assert_labelled_navigation(html: str) -> None:
    """More than one ``<nav>`` on a page needs names to tell them apart."""
    navigations = re.findall(r"<nav[^>]*>", html, re.IGNORECASE)
    assert navigations, "no navigation landmark"
    for navigation in navigations:
        assert "aria-label" in navigation or "aria-labelledby" in navigation, (
            f"unlabelled navigation: {navigation!r}"
        )


def assert_page_shell(html: str) -> None:
    """Everything every page in the application owes its reader."""
    assert_responsive_shell(html)
    assert_keyboard_reachable(html)
    assert_labelled_navigation(html)
    assert_local_assets_only(html)
    assert_no_inline_scripts(html)


def position_of(html: str, needle: str) -> int:
    """Index of ``needle``, asserting it is present (for order assertions)."""
    index = html.find(needle)
    assert index >= 0, f"{needle!r} is not on the page"
    return index


def region(html: str, hook: str, *, size: int = 1500) -> str:
    """The markup that follows ``hook``, for section-scoped assertions.

    Lesson pages repeat words across sections -- ``weight`` is a structure
    field *and* a word in the body -- so "is this in the structures section"
    has to be asked of the section, not of the page.
    """
    index = position_of(html, hook)
    return html[index : index + size]
