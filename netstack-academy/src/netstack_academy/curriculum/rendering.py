"""Render authored Markdown into HTML that is safe to embed in a page.

Lesson bodies are trusted-ish (they come from this repository) but they are
still text that ends up inside a page, and kernel lessons need exactly the
constructs that make naive sanitizers useless: fenced code full of ``<``,
``>`` and ``->``, tables of structure fields, and links. So this module
renders permissively and sanitizes afterwards, rather than trying to render
defensively.

Two decisions are load-bearing and non-obvious:

- **Raw HTML is enabled in the Markdown parser** (``html=True``). Disabling
  it would escape ``<script>`` into visible text instead of removing it,
  which is worse than useless in a lesson body: the reader sees the attack
  payload as content. Everything is instead handed to :mod:`nh3` (Rust
  ammonia), which drops disallowed elements -- and the *contents* of
  ``<script>``/``<style>`` -- while keeping the surrounding prose.
- **Link validation is delegated entirely to the sanitizer.**
  markdown-it's own ``validateLink`` rejects a ``javascript:`` destination
  by refusing to build the link at all, which makes it fall back to
  emitting the raw source text -- so ``[Read more](javascript:steal())``
  would render as the literal string, ``javascript:`` payload and all.
  Allowing the link to be built and then stripping the unsafe attribute
  leaves the reader with the link text and no URL, and keeps exactly one
  component (this module's allowlist) responsible for URL safety.

:func:`render_markdown` is deterministic: the same input always produces
the same output, which the curriculum loader relies on to compare two
loads of the same content root for equality.
"""

from __future__ import annotations

import re

import nh3
from markdown_it import MarkdownIt

#: Schemes permitted on ``href``/``src``. Deliberately narrower than
#: ammonia's default set: lesson content has no reason to link to ``ftp:``
#: or embed ``data:`` payloads.
ALLOWED_URL_SCHEMES = frozenset({"http", "https", "mailto"})

#: Ammonia's curated element allowlist, which already excludes every
#: scripting and embedding vector the tests exercise (``script``, ``style``,
#: ``iframe``, ``object``, ``embed``, ``form``, ``svg``, ``body``) while
#: keeping ``pre``/``code``, the table elements, and ordinary prose markup.
ALLOWED_TAGS = frozenset(nh3.ALLOWED_TAGS)

#: Same rationale: ammonia's per-tag attribute allowlist keeps ``a[href]``
#: and ``img[src]`` and admits no ``on*`` event handler.
ALLOWED_ATTRIBUTES = {tag: set(names) for tag, names in nh3.ALLOWED_ATTRIBUTES.items()}

#: Attributes whose value is a URL and therefore has to pass
#: :func:`_is_safe_url` before it is allowed through.
_URL_ATTRIBUTES = frozenset({"href", "src"})

#: A leading ``scheme:``. Anything without one is relative (or an anchor),
#: which is safe by construction: it can only address this application.
_SCHEME_RE = re.compile(r"^([a-z][a-z0-9+.\-]*):", re.IGNORECASE)

#: Whitespace and C0/C1 control characters are ignored by browsers when
#: they parse a URL, so ``java\tscript:x`` is a live ``javascript:`` URL.
#: They are removed before the scheme is read, never after.
_URL_IGNORED_CHARS_RE = re.compile(r"[\s\x00-\x20\x7f-\xa0]+")


def _is_safe_url(value: str) -> bool:
    """True when ``value`` is relative or carries an allowed scheme."""
    candidate = _URL_IGNORED_CHARS_RE.sub("", value)
    match = _SCHEME_RE.match(candidate)
    if match is None:
        return True
    return match.group(1).lower() in ALLOWED_URL_SCHEMES


def _filter_attribute(element: str, attribute: str, value: str) -> str | None:
    """Drop URL attributes whose scheme is not allowed, keep the rest."""
    if attribute in _URL_ATTRIBUTES and not _is_safe_url(value):
        return None
    return value


def _build_parser() -> MarkdownIt:
    parser = MarkdownIt(
        "js-default",
        {"html": True, "linkify": False, "typographer": False},
    )
    # See the module docstring: refusing a URL here would degrade the link
    # into raw source text, so every destination is built and the
    # sanitizer below decides which ones keep their href.
    parser.validateLink = lambda url: True
    return parser


_PARSER = _build_parser()


def render_markdown(text: str) -> str:
    """Render ``text`` as Markdown and return sanitized HTML."""
    if not text.strip():
        return ""

    return nh3.clean(
        _PARSER.render(text),
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        attribute_filter=_filter_attribute,
        url_schemes=set(ALLOWED_URL_SCHEMES),
        # Lessons link to each other with site-relative paths, so relative
        # URLs are passed through rather than denied or rewritten against
        # a base this layer does not know.
        url_relative="pass_through",
    )
