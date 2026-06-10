"""Block-stream stitching: re-join paragraphs split by two-column PDF layout and
fold free-standing table captions into their ``<table>``.

Pure, no GPU.  Operates on the flat list of block-HTML strings produced per page.
"""

from __future__ import annotations

import re

from pdfparser.pipeline.classify import _is_stray_metadata
from pdfparser.pipeline.text import (
    _BOLD_LABEL_RE,
    _SENTENCE_END_RE,
    _TABLE_CAPTION_RE,
    _opens_with_caption_label,
    _plain_p_text,
    _visible_text,
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
# A trailing comma arms the guard only after a clause-introducer
# (that/which/…) — "revealed that, with no additives present, …" — where the
# comma opens a fronted parenthetical and the real continuation is still
# lowercase.  After a preposition/conjunction a trailing comma before a capital
# is more often a genuine (proper-noun) continuation, so it must not arm the
# guard.
_FUNCTION_WORD_END_RE = re.compile(
    r"\b(?:a|an|the|is|are|was|were|be|been|being|have|has|had|"
    r"will|would|can|could|should|may|might|must|do|does|did|"
    r"of|in|on|at|by|for|with|to|from|and|or|but|nor)\s*$"
    r"|\b(?:that|which|who|whom|this|these|those)[\s,;]*$",
    re.IGNORECASE,
)
# An all-caps acronym ("TRII", "DNA", "NAD") opening the continuation is part
# of the same sentence, not a new-sentence capital, so it must not trip the
# function-word guard — otherwise a clause split across a column/page break
# ("…TRI and" / "TRII compete…") is wrongly left as two paragraphs.
_ACRONYM_HEAD_RE = re.compile(r"^[A-Z]{2,}[0-9]*\b")
# A column/page break can strand a paragraph's continuation behind a whole float
# cluster — observed as two figures plus a table between the two halves — so the
# skip window must clear such a cluster.  The grammatical guards below (the
# fragment lacks terminal punctuation, the continuation isn't a new sentence /
# caption / enumeration) are what keep the merge honest; this bound only limits
# how far floats are relocated after the joined paragraph.
_MAX_FLOATS_TO_SKIP = 3

_TABLE_OPEN_RE = re.compile(r"^<table[\s>]", re.IGNORECASE)
_TABLE_OPEN_TAG_RE = re.compile(r"^<table\b[^>]*>", re.IGNORECASE)
_FIGURE_OPEN_RE = re.compile(r"^<figure[\s>]", re.IGNORECASE)
# A *bare* table label: the "Table <id>" caption header with no descriptive text
# after it ("TABLE I", "Table 1.", "Supplementary Table 2:").  Unlike
# _TABLE_CAPTION_RE this rejects a label carrying its own title ("Table 4 X"),
# because only a header with the title stranded in the *next* paragraph needs
# rejoining.
_BARE_TABLE_LABEL_RE = re.compile(
    r"^\*{0,2}\s*"
    r"(?i:supp(?:l(?:ementary)?)?\.?\s+)?"
    r"(?i:table)\b\s*"
    r"\w+"
    r"\s*[.:]?\s*\*{0,2}\s*$"
)


def _join_split_table_caption_labels(parts: list[str]) -> list[str]:
    """Rejoin a table caption OCR split into a bare label paragraph and its title.

    A caption rendered as ``<p>TABLE I</p>`` followed by ``<p>Selected
    substrates…</p>`` leaves the title stranded as its own block: colocation
    folds a caption into its ``<table>`` only when the caption is one block
    adjacent to the table, and the cross-table paragraph merge can't see past
    the stray ``<p>`` either.  A bare label (nothing after the table identifier)
    is never a body sentence, so the following plain paragraph is its title and
    the two are one caption.
    """
    out: list[str] = []
    i = 0
    while i < len(parts):
        inner = _plain_p_text(parts[i])
        if (
            inner is not None
            and _BARE_TABLE_LABEL_RE.match(_visible_text(inner).strip())
            and i + 1 < len(parts)
        ):
            title = _plain_p_text(parts[i + 1])
            if title is not None and not _opens_with_caption_label(title):
                out.append(f"<p>{inner.rstrip()} {title.lstrip()}</p>")
                i += 2
                continue
        out.append(parts[i])
        i += 1
    return out


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
            # Terminal punctuation can sit *inside* a closing inline tag
            # ("…carboxylase.</em>"), so the sentence-end test must run on the
            # visible text — matching the raw HTML would miss the period behind
            # the tag and wrongly treat a finished caption as a fragment.
            visible = _visible_text(stripped).rstrip()
            if (
                not _SENTENCE_END_RE.search(visible)
                and not _BOLD_LABEL_RE.match(inner)
                and not _opens_with_caption_label(inner)
                # A self-contained metadata footer (DOI / "Published online …" /
                # correspondence) is complete even when it ends without terminal
                # punctuation, so it must not be absorbed as a fragment with the
                # following body prose glued on.  Backstops the page-0 stray sweep.
                and not _is_stray_metadata(part)
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
                            _FUNCTION_WORD_END_RE.search(visible)
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


def _merge_split_paragraphs_stable(parts: list[str]) -> list[str]:
    """Run ``_merge_split_paragraphs`` until it reaches a fixpoint.

    A single pass joins each fragment to its immediate continuation, but a
    paragraph split into three-plus pieces by column/page breaks only collapses
    fully once the earlier joins expose the next adjacency.  Iterating to a
    fixpoint stitches the whole chain regardless of how many pieces it was in.
    """
    while True:
        merged = _merge_split_paragraphs(parts)
        if merged == parts:
            return merged
        parts = merged


def _is_table_caption(part: str) -> bool:
    """True when a block is a stand-alone ``<p>`` table caption ("Table 1 …")."""
    inner = _plain_p_text(part)
    return inner is not None and bool(
        _TABLE_CAPTION_RE.match(_visible_text(inner).lstrip())
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


# A table note is a single line ("Molecule structures are shown in Fig. 3.") before
# the superscript-marker lines; bounding the leading non-marker run to one keeps a
# body paragraph an OCR mis-order may have stranded after a table from being
# swallowed as a note.
_MAX_TABLE_NOTE_LINES = 1
# A footnote marker: a SHORT superscript ("<sup>a</sup>", "<sup>*</sup>").  Same
# bound as classify's _SUP_MARKER_RE, but capturing the label so a trailing
# footnote can be matched to the marker it annotates inside the table.
_SUP_LABEL_RE = re.compile(r"<sup>([^<]{1,3})</sup>")
_FOOTNOTE_SYMBOLS = frozenset("*†‡§¶#")


def _is_footnote_label(label: str) -> bool:
    """True for a footnote-style superscript label (letter or footnote symbol).

    A purely numeric/sign superscript inside a table is an exponent or charge
    ("cm²", "10⁻¹"), not a footnote referent, so it must not let a numbered
    article footnote that follows the table be mistaken for the table's own.
    """
    return any(c.isalpha() or c in _FOOTNOTE_SYMBOLS for c in label)


def _leading_sup_label(inner: str) -> str | None:
    """The footnote-marker label a block opens with ("a" for ``<sup>a</sup>…``)."""
    m = _SUP_LABEL_RE.match(inner)
    return m.group(1).strip() if m else None


def _table_sup_labels(table_html: str) -> set[str]:
    """The footnote-style superscript labels a table carries (exponents excluded)."""
    return {
        label
        for m in _SUP_LABEL_RE.finditer(table_html)
        if _is_footnote_label(label := m.group(1).strip())
    }


def _as_table_footnote(part: str) -> str:
    inner = _plain_p_text(part)
    return f'<p class="footnote">{inner}</p>' if inner is not None else part


def _colocate_table_footnotes(parts: list[str]) -> list[str]:
    """Absorb a table's trailing footnote run into its ``<table>`` block.

    Right after ``</table>`` the model emits the table's footnotes — superscript
    marker lines (``<sup>a</sup> …``) and any short note sentence wedged between
    the table and those markers — as free-standing paragraphs.  Folding them onto
    the end of the table block keeps them rendering under their table, stops the
    classifier from sweeping them into the article's footnote section, and lets
    the cross-table paragraph merge skip the table (now a single float) to rejoin
    prose split across it.

    A marker line is only this table's footnote when its label is one the table
    actually carries (``<sup>a</sup>`` inside a header/cell) — otherwise the
    superscript line is an article footnote that merely follows the table, and is
    left for the classifier to route to the footnote section.  A plain paragraph
    is taken as a note only while it sits *before* the first marker; once the
    markers end, the next paragraph is the body resuming.  A run with no matching
    marker is not a footnote run and is left untouched.
    """
    n = len(parts)
    out: list[str] = []
    i = 0
    while i < n:
        part = parts[i]
        if not _TABLE_OPEN_RE.match(part):
            out.append(part)
            i += 1
            continue
        labels = _table_sup_labels(part)
        run: list[str] = []
        seen_marker = False
        leading = 0
        j = i + 1
        while j < n:
            inner = _plain_p_text(parts[j])
            if inner is None:
                break
            marker = _leading_sup_label(inner)
            if marker is not None:
                if marker not in labels:
                    break
                run.append(parts[j])
                seen_marker = True
                j += 1
                continue
            if seen_marker or leading >= _MAX_TABLE_NOTE_LINES:
                break
            run.append(parts[j])
            leading += 1
            j += 1
        if seen_marker:
            out.append(part + "".join(_as_table_footnote(b) for b in run))
            i = j
        else:
            out.append(part)
            i += 1
    return out
