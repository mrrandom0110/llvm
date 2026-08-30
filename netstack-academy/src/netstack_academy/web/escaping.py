"""Escaping for the two kinds of text this application does not author.

Everything on a page comes from one of three places. Lesson bodies are
authored Markdown, sanitized once at load time, and arrive here as HTML that
is already safe to emit. Curriculum metadata -- titles, objectives, caveats
-- is authored plain text that Jinja's autoescaping handles completely. And
then there are *notes* and *search queries*: free text written by whoever is
at the keyboard, or, in the query's case, by whoever wrote the link they
clicked.

Jinja's autoescaping is not quite enough for that third kind, and the reason
is worth being precise about. Autoescaping neutralises the five characters
that let text *become markup* -- ``& < > " '`` -- which is exactly right for
element content. But the query is also echoed into an attribute
(``value="..."``), and a value of ``" autofocus onfocus="steal()`` escapes to
``&#34; autofocus onfocus=&#34;steal()``: no longer able to close the
attribute, but still carrying the literal text ``onfocus=`` into the
document. That is not exploitable on its own. It becomes exploitable the
moment anything downstream unescapes once too often -- a template that marks
a value safe by mistake, a script that assigns to ``innerHTML``, a future
partial that drops the value into an unquoted attribute. Defence in depth
here costs one function.

So :func:`untrusted_text` escapes the five HTML characters *and* three more,
each for a specific reason rather than out of general caution:

``=``
    An attribute is ``name=value``. Without an ``=`` there is no attribute
    to inject, quoted context or not.
``:``
    A URL scheme is ``scheme:rest``. Without a ``:`` there is no
    ``javascript:`` to smuggle into an ``href``.
``` ` ```
    Internet Explorer treated a backtick as an attribute-value delimiter,
    which is the documented reason OWASP recommends escaping it.

Escaping more than the minimum is invisible to the reader: a browser decodes
``&#61;`` to ``=`` before painting it, and a ``<textarea>`` hands the decoded
original back when the note is edited again. The learner sees exactly what
they typed. Only the document source differs, which is the only place an
injection could have lived.

What this is *not* is a sanitizer: it removes nothing. It is the last
render-time step for text that is stored verbatim, because storing verbatim
is what makes a note about ``<linux/skbuff.h>`` still say that when it comes
back.
"""

from __future__ import annotations

from markupsafe import Markup

#: Character to replacement. The first five are ordinary HTML escaping; the
#: last three are the structural characters explained in the module
#: docstring. ``&`` is first in this order deliberately -- see
#: :func:`untrusted_text`.
_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("&", "&amp;"),
    ("<", "&lt;"),
    (">", "&gt;"),
    ('"', "&#34;"),
    ("'", "&#39;"),
    ("=", "&#61;"),
    (":", "&#58;"),
    ("`", "&#96;"),
)


def untrusted_text(value: str | None) -> Markup:
    """Escape free text for any HTML context, element or attribute.

    Returns :class:`~markupsafe.Markup` so a template renders it once and
    Jinja does not escape the ampersands of these entities a second time.
    ``None`` becomes empty, so a lesson with no note needs no conditional at
    the call site.
    """
    if not value:
        return Markup()

    escaped = value
    for character, replacement in _REPLACEMENTS:
        # ``&`` is replaced first, so the ampersands introduced by every
        # later replacement come after it and are not escaped twice.
        escaped = escaped.replace(character, replacement)
    return Markup(escaped)


__all__ = ["untrusted_text"]
