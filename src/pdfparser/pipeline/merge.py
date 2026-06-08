"""Block-stream stitching: re-join paragraphs split by two-column PDF layout and
fold free-standing table captions into their ``<table>``.

Pure, no GPU.  Operates on the flat list of block-HTML strings produced per page.
"""

from __future__ import annotations

import re

from pdfparser.pipeline.text import (
    _BOLD_LABEL_RE,
    _SENTENCE_END_RE,
    _STRIP_TAGS_RE,
    _TABLE_CAPTION_RE,
    _opens_with_caption_label,
    _plain_p_text,
)

_FLOAT_RE = re.compile(r"^<(?:table|figure)[\s>]", re.IGNORECASE)
_HYPHEN_BREAK_RE = re.compile(r"-\s*$")
_ENUM_RE = re.compile(
    r"^\s*(?:\d+[.)]\s|\[\d|[•\-]\s|\([a-z0-9ivx]+\)\s)", re.IGNORECASE
)
# A fragment ending with a function word is *definitively* grammatically
# incomplete: its continuation must be a predicate, object, or complement,
# which in normal prose starts lowercase.  If the next region starts with an
# uppercase letter in this context the real continuation was likely dropped by
# OCR, so we refuse the merge rather than joining unrelated sentences.
_FUNCTION_WORD_END_RE = re.compile(
    r"\b(?:a|an|the|is|are|was|were|be|been|being|have|has|had|"
    r"will|would|can|could|should|may|might|must|do|does|did|"
    r"of|in|on|at|by|for|with|to|from|and|or|but|nor|"
    r"that|which|who|whom|this|these|those)\s*$",
    re.IGNORECASE,
)
# An all-caps acronym ("TRII", "DNA", "NAD") opening the continuation is part
# of the same sentence, not a new-sentence capital, so it must not trip the
# function-word guard — otherwise a clause split across a column/page break
# ("…TRI and" / "TRII compete…") is wrongly left as two paragraphs.
_ACRONYM_HEAD_RE = re.compile(r"^[A-Z]{2,}[0-9]*\b")
_MAX_FLOATS_TO_SKIP = 2

_TABLE_OPEN_RE = re.compile(r"^<table[\s>]", re.IGNORECASE)
_TABLE_OPEN_TAG_RE = re.compile(r"^<table\b[^>]*>", re.IGNORECASE)
_FIGURE_OPEN_RE = re.compile(r"^<figure[\s>]", re.IGNORECASE)


def _merge_split_paragraphs(parts: list[str]) -> list[str]:
    """Stitch paragraph fragments broken by two-column PDF layout.

    When a plain ``<p>`` ends without terminal punctuation the next plain
    ``<p>`` is treated as a continuation.  Intervening tables and figures
    (up to ``_MAX_FLOATS_TO_SKIP``) are collected and re-emitted *after*
    the merged paragraph so the float stays near its reference text.

    Headings, footnote paragraphs, enumeration items, and figure/table
    captions act as merge barriers and are never absorbed into an adjacent
    paragraph.
    """
    out: list[str] = []
    i = 0
    while i < len(parts):
        part = parts[i]
        inner = _plain_p_text(part)
        if inner is not None:
            stripped = inner.rstrip()
            if (
                not _SENTENCE_END_RE.search(stripped)
                and not _BOLD_LABEL_RE.match(inner)
                and not _opens_with_caption_label(inner)
            ):
                j = i + 1
                floats: list[str] = []
                while (
                    j < len(parts)
                    and _FLOAT_RE.match(parts[j])
                    and len(floats) < _MAX_FLOATS_TO_SKIP
                ):
                    floats.append(parts[j])
                    j += 1
                if j < len(parts):
                    cont = _plain_p_text(parts[j])
                    if (
                        cont is not None
                        and not _ENUM_RE.match(cont)
                        and not _BOLD_LABEL_RE.match(cont)
                        and not _opens_with_caption_label(cont)
                        and not (
                            _FUNCTION_WORD_END_RE.search(stripped)
                            and cont[:1].isupper()
                            and not _ACRONYM_HEAD_RE.match(cont)
                        )
                    ):
                        dehyphenated, n = _HYPHEN_BREAK_RE.subn("", stripped)
                        joined = dehyphenated + ("" if n else " ") + cont.lstrip()
                        out.append(f"<p>{joined}</p>")
                        out.extend(floats)
                        i = j + 1
                        continue
        out.append(part)
        i += 1
    return out


def _is_table_caption(part: str) -> bool:
    """True when a block is a stand-alone ``<p>`` table caption ("Table 1 …")."""
    inner = _plain_p_text(part)
    return inner is not None and bool(
        _TABLE_CAPTION_RE.match(_STRIP_TAGS_RE.sub("", inner).lstrip())
    )


def _inject_table_caption(table_html: str, caption_part: str) -> str:
    """Insert the caption's inner HTML as a ``<caption>`` first child of the table.

    A lambda replacement keeps any backslash in the caption text (e.g. a literal
    ``\\frac``) from being read as a regex backreference."""
    inner = _plain_p_text(caption_part)
    if inner is None:
        return table_html
    # Insert after the *whole* opening tag so attributes ("<table class=…>")
    # aren't split, which would orphan them and break the table.
    return _TABLE_OPEN_TAG_RE.sub(
        lambda m: f"{m.group(0)}<caption>{inner}</caption>", table_html, count=1
    )


def _colocate_table_captions(parts: list[str]) -> list[str]:
    """Fold a free-standing "Table N …" caption into its ``<table>`` as a
    ``<caption>`` first child so it renders with the table rather than drifting
    in the block stream.

    Each captionless table claims the nearest unused table caption — preferring
    one just *before* it (the usual convention), else just *after* — skipping
    only intervening figures and never reaching across prose or another table.
    A caption with no table nearby is left untouched as its own block.
    """
    n = len(parts)
    is_caption = [_is_table_caption(p) for p in parts]
    is_figure = [bool(_FIGURE_OPEN_RE.match(p)) for p in parts]
    needs_caption = [
        bool(_TABLE_OPEN_RE.match(p)) and "<caption" not in p.lower() for p in parts
    ]
    used = [False] * n
    attached: dict[int, int] = {}

    def claim(table_idx: int, step: int) -> int | None:
        k = table_idx + step
        while 0 <= k < n:
            if is_caption[k] and not used[k]:
                return k
            if is_figure[k]:
                k += step
                continue
            return None
        return None

    tables = [t for t in range(n) if needs_caption[t]]
    # Backward first for every table, so each secures its own leading caption
    # before a neighbouring table can forward-grab it: a caption between two
    # tables belongs to the one it precedes, not the one it follows.
    for t in tables:
        c = claim(t, -1)
        if c is not None:
            used[c] = True
            attached[t] = c
    for t in tables:
        if t not in attached and (c := claim(t, 1)) is not None:
            used[c] = True
            attached[t] = c

    out: list[str] = []
    for idx, part in enumerate(parts):
        if used[idx]:
            continue
        out.append(
            _inject_table_caption(part, parts[attached[idx]])
            if idx in attached
            else part
        )
    return out
