"""Shared block-level text helpers and regexes.

Leaf module: the small predicates and patterns that several pipeline stages
(``merge``, ``classify``, ``assemble``) all need to inspect a block's HTML —
strip tags, recognise a plain ``<p>``, a heading, or a caption label.
"""

from __future__ import annotations

import re

_STRIP_TAGS_RE = re.compile(r"<[^>]+>")
_SENTENCE_END_RE = re.compile(r"[.!?;:]\s*$")
# Paragraphs that open with a bold label ("Keywords:", "Abbreviations:", "Note:")
# are structured metadata, never mid-sentence continuations.
_BOLD_LABEL_RE = re.compile(r"^<strong>[^<]+:</strong>")
_HEADING_TAG_RE = re.compile(r"^<h([1-6])>(.*)</h\1>$", re.DOTALL)

# A caption opens with a figure/table label ("FIG. 2 …", "**Table 1.** …").
_CAPTION_RE = re.compile(
    r"^\*{0,2}(?:fig(?:ure|\.|\b)|table|scheme|supplement)", re.IGNORECASE
)
# A figure placeholder only ever owns a *figure* caption; a "Table …" block sat
# beside it belongs to its table (see _colocate_table_captions), so the
# figure-caption test deliberately excludes the table label.
_FIGURE_CAPTION_RE = re.compile(r"^\*{0,2}(?:fig(?:ure|\.|\b)|scheme)", re.IGNORECASE)
# A table caption ("Table 1 …", "Supplementary Table 2 …").  Matched against a
# block's *visible* text so it's recognised through a <strong> wrapper.  After
# the "Table <id>" label a true caption is followed by punctuation, a
# capitalised title word, or nothing — *not* a lowercase word, which marks a
# running reference sentence ("Table 1 summarizes …") that must stay in the body.
# A pipe ("TABLE 2 | Kinetic parameters …") is the Frontiers caption separator, but
# only when a title follows it — "Table 1 | 2 | 3" is a stray prose row, not a
# caption — so the pipe branch carries the same capitalised-title requirement as the
# space-separated one rather than accepting any character after the pipe.  Only
# "table"/"supplementary" are case-folded (OCR casing is unreliable); the
# capitalised-title test stays case-sensitive, as that is the whole signal.
_TABLE_CAPTION_RE = re.compile(
    r"^\*{0,2}\s*"
    r"(?i:supp(?:l(?:ementary)?)?\.?\s+)?"
    r"(?i:table)\b\s*"
    r"\w+"
    r"(?:\s*[.:)–—-]|\s*\|\s*[A-Z(]|\s+[A-Z(]|\s*\*{0,2}\s*$)"
)


_BLOCK_SPLIT_RE = re.compile(r"\n[ \t]*\n")


def _split_md_blocks(md: str) -> list[str]:
    """Split markdown into blocks on blank lines (a run of whitespace-only lines).

    The single definition of "block" shared by the page-block parser and the
    figure-recovery crop parser, so both segment a page identically."""
    return _BLOCK_SPLIT_RE.split(md.strip())


def _visible_text(html: str) -> str:
    """The text a reader sees: ``html`` with every tag removed."""
    return _STRIP_TAGS_RE.sub("", html)


def _visible_text_folded(html: str) -> str:
    """``_visible_text`` normalized for label comparison — trimmed and lowercased.

    OCR casing is unreliable, so document-type and metadata-heading labels are
    matched case-insensitively against this form.
    """
    return _visible_text(html).strip().lower()


def _plain_p_text(s: str) -> str | None:
    """Return the inner content of a plain ``<p>…</p>`` block, or ``None``.

    Returns ``None`` for footnote/class paragraphs, multi-paragraph strings
    (reference lists), headings, tables, figures, and any other element.
    """
    if s.startswith("<p>") and s.endswith("</p>") and s.count("</p>") == 1:
        return s[3:-4]
    return None


def _heading_inner(part: str) -> tuple[int, str] | None:
    m = _HEADING_TAG_RE.match(part)
    return (int(m.group(1)), m.group(2).strip()) if m else None


def _looks_like_figure_caption(block: str) -> bool:
    return bool(_FIGURE_CAPTION_RE.match(block.strip()))


def _opens_with_caption_label(text: str) -> bool:
    """True when text's visible content opens with a figure/table caption label,
    even when wrapped in inline markup (``<strong>Table 1</strong> …``).

    A caption is a merge barrier: gluing it onto an adjacent paragraph both
    garbles the sentence and strands the caption away from its float, so the
    label must be recognised through the ``<strong>`` the model often wraps it
    in, which a raw match of ``_CAPTION_RE`` against the markdown would miss."""
    return bool(_CAPTION_RE.match(_visible_text(text).lstrip()))


def _opens_with_table_label(text: str) -> bool:
    """True when text's visible content opens with a *table* caption label
    ("Table 2 …", "TABLE 2 | …") — the figure-caption case excluded."""
    return bool(_TABLE_CAPTION_RE.match(_visible_text(text).lstrip()))
