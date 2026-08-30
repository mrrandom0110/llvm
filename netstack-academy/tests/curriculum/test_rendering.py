"""Contract for :mod:`netstack_academy.curriculum.rendering`.

Lesson bodies are authored Markdown that ends up injected into a page, so
``render_markdown`` has to do two jobs that pull in opposite directions:
keep everything a kernel lesson genuinely needs (fenced code with its
punctuation intact, tables, ordinary links, inline code), while removing
everything that can execute (``<script>``, inline event handlers,
``javascript:`` URLs, embedded frames).
"""

from __future__ import annotations

import pytest

from netstack_academy.curriculum.rendering import render_markdown


def test_script_tags_and_their_contents_are_removed() -> None:
    html = render_markdown("Before.\n\n<script>steal(document.cookie)</script>\n\nAfter.\n")

    assert "<script" not in html.lower()
    assert "steal" not in html
    assert "Before." in html
    assert "After." in html


def test_inline_event_handler_attributes_are_removed() -> None:
    html = render_markdown('<p onclick="steal()">Click me</p>\n')

    assert "onclick" not in html.lower()
    assert "steal()" not in html
    assert "Click me" in html


def test_javascript_urls_are_removed_but_the_link_text_survives() -> None:
    html = render_markdown("[Read more](javascript:steal())\n")

    assert "javascript:" not in html.lower()
    assert "Read more" in html


def test_image_error_handlers_are_removed() -> None:
    html = render_markdown('<img src="x" onerror="steal()">\n')

    assert "onerror" not in html.lower()
    assert "steal()" not in html


def test_embedded_frames_are_removed() -> None:
    html = render_markdown('<iframe src="https://example.com/evil"></iframe>\n')

    assert "<iframe" not in html.lower()


def test_fenced_code_blocks_survive_sanitization() -> None:
    html = render_markdown("```c\nif (skb->len < 0)\n        return -EINVAL;\n```\n")

    assert "<pre" in html
    assert "<code" in html
    assert "return -EINVAL;" in html


def test_code_block_angle_brackets_are_escaped_not_dropped() -> None:
    """Kernel code is full of ``<linux/skbuff.h>``-style includes and ``->``
    dereferences; sanitizing must escape them, not swallow them as tags.
    """
    html = render_markdown("```c\n#include <linux/skbuff.h>\n```\n")

    assert "linux/skbuff.h" in html
    assert "&lt;linux/skbuff.h&gt;" in html


def test_tables_survive_sanitization() -> None:
    markdown = "| Field | Meaning |\n| --- | --- |\n| weight | poll budget |\n"

    html = render_markdown(markdown)

    assert "<table" in html
    assert "weight" in html
    assert "poll budget" in html


def test_http_links_keep_their_href() -> None:
    html = render_markdown("[NAPI](https://docs.kernel.org/networking/napi.html)\n")

    assert 'href="https://docs.kernel.org/networking/napi.html"' in html
    assert "NAPI" in html


def test_relative_links_keep_their_href() -> None:
    html = render_markdown("[Next lesson](/lessons/qdisc-dequeue)\n")

    assert 'href="/lessons/qdisc-dequeue"' in html


def test_inline_code_and_emphasis_survive() -> None:
    html = render_markdown("Call `napi_poll()` **before** the budget expires.\n")

    assert "<code>napi_poll()</code>" in html
    assert "<strong>before</strong>" in html


def test_headings_and_lists_survive() -> None:
    html = render_markdown("## Steps\n\n- First\n- Second\n")

    assert "<h2" in html
    assert "<li>First</li>" in html


def test_rendering_is_deterministic() -> None:
    markdown = "## Title\n\n`code` and [link](https://example.com/a)\n"

    assert render_markdown(markdown) == render_markdown(markdown)


def test_empty_body_renders_to_empty_html() -> None:
    assert render_markdown("").strip() == ""


@pytest.mark.parametrize(
    "hostile",
    [
        '<a href="javascript:alert(1)">x</a>',
        "<svg onload=alert(1)></svg>",
        '<body onload="alert(1)">x</body>',
        "<object data=\"data:text/html;base64,PHNjcmlwdD4=\"></object>",
        "<script src=\"https://example.com/evil.js\"></script>",
    ],
)
def test_hostile_markup_never_survives_rendering(hostile: str) -> None:
    html = render_markdown(hostile + "\n").lower()

    assert "javascript:" not in html
    assert "onload" not in html
    assert "<script" not in html
    assert "<object" not in html
