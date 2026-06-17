"""Table re-OCR: recover content LightOnOCR drops from dense full-page tables.

The full-page OCR pass silently drops small in-table content — e.g. a column-
spanning subheader like ``Relative activity (%)`` — at *every* render resolution,
yet re-OCRing the table on its own as a tight crop recovers it (and produces the
``colspan``/``rowspan`` structure the full-page pass omits).  So for each table
region we locate it on the page, re-OCR just that crop, and substitute the result.

Localization is Option 3 (see plans / design.rst): the PDF text layer is used for
**geometry only** — we match the cells the full-page pass *did* capture against the
text layer to seed a bounding box, then flood it out to the connected text block.
The text layer's *content* is never read into the document; that stays the model's
job, preserving design B-prime's "OCR reads, nothing else does" split.

Crops are planned for the whole document first (text-layer localization plus a
coverage gate that skips regions the page already captured in full), then re-OCR'd
in one batched ``ocr_regions`` call so the server processes them concurrently.

Leaf module: touches the PDF (render + text layer) and the GPU (via the injected
``ocr_regions`` callback), so the pure ``_assemble_html`` core stays model-free.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable  # noqa: TC003 — beartype reads annotations
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 — beartype reads annotations at runtime

import pypdfium2 as pdfium
from PIL import Image  # noqa: TC002 — beartype reads annotations at runtime

from pdfparser.pipeline.markdown import _render_inline
from pdfparser.pipeline.render import _downscale_to_long_side
from pdfparser.pipeline.text import _visible_text

_TABLE_RE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)
_OPEN_TABLE_RE = re.compile(r"<table\b", re.IGNORECASE)
_CLOSE_TABLE_RE = re.compile(r"</table\s*>", re.IGNORECASE)
_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
_CAPTION_RE = re.compile(r"<caption\b[^>]*>.*?</caption>", re.DOTALL | re.IGNORECASE)
_ROW_RE = re.compile(r"<tr\b.*?</tr>", re.DOTALL | re.IGNORECASE)
_CELL_OPEN_RE = re.compile(r"<t[dh]\b([^>]*)>", re.IGNORECASE)
_COLSPAN_RE = re.compile(r"colspan\s*=\s*[\"']?(\d+)", re.IGNORECASE)
_THEAD_OPEN_RE = re.compile(r"<thead\b[^>]*>", re.IGNORECASE)
_TABLE_OPEN_RE = re.compile(r"\s*<table\b[^>]*>", re.IGNORECASE)
# Sub-table labels ("A. Effect…", "B. ICP-MS analysis") that the crop re-OCR lifts
# out as level-2+ markdown headings; level-1 is the table's overall caption, which
# the pipeline already carries as its own block, so it is left out here.
_SUBHEADING_RE = re.compile(r"^#{2,6}\s+(.*\S)\s*$", re.MULTILINE)

# A normalized cell must be at least this many chars to seed localization — short
# numeric/symbol cells ("1 mM", "None") recur in prose and would match the wrong
# spot, so only distinctive multi-word cells anchor the box; the flood then fills
# the rest of the table around them.
_MIN_ANCHOR_LEN = 4

# The coverage gate ignores one-letter/one-symbol noise, but a multi-char numeric
# token ("100", "26621") is real table data — a dropped data column is exactly the
# kind of loss the gate must not skip over — so digit-bearing tokens count as
# evidence from length 2, while plain words must reach _COVERAGE_MIN_TOKEN_LEN
# (shorter words like "of"/"the" recur in prose and would misjudge completeness).
_COVERAGE_MIN_TOKEN_LEN = 4
_COVERAGE_MIN_NUMERIC_LEN = 2

# A table's caption (above) and legend (below) are short; a block longer than this
# is body prose the located bbox happened to overrun, and folding it into the gate's
# "captured" set would mask a genuine drop, so it is not treated as adjacent.
_ADJACENT_PARA_MAX_LEN = 300

# Vertical growth stops when the gap to the next text line exceeds this multiple
# of the table's own line spacing — large enough to step over inter-row gaps,
# small enough to halt at the wider margin separating the table from body prose.
# Derived from spacing (not a fixed size) so it adapts to font and render scale.
_GAP_FACTOR = 1.5
# A table legend sits one line below the body at a gap wider than the row gaps but
# narrower than the margin to prose; this factor (vs _GAP_FACTOR) reaches it.
_LEGEND_GAP_FACTOR = 2.5
_MAX_LEGEND_LINES = 1
_LEGEND_MAX_LEN = 200  # a recovered legend/source note is short, not body prose
_PAD_PT = 4.0  # small margin so glyph edges are not clipped by the crop

# Crop render scale: aim for this long side in px, clamped to a sane DPI band.
_TARGET_CROP_PX = 1500
_MIN_SCALE = 200 / 72
_MAX_SCALE = 600 / 72

_Box = tuple[float, float, float, float]  # PDF points: left, bottom, right, top


def _close_unclosed_tables(md: str) -> str:
    """Append a closing tag for any ``<table>`` the page OCR left open.

    LightOnOCR transcribes a table row by row and, when the table runs past the
    bottom of the page, stops mid-table without emitting ``</table>``.  Left open,
    the table absorbs everything after it — most visibly the *next* page's opening
    prose, which renders inside the table.  Closing the surplus opens per page (the
    unit at which the model transcribes) keeps following content out.  A balanced
    page is returned unchanged, so this is idempotent."""
    missing = len(_OPEN_TABLE_RE.findall(md)) - len(_CLOSE_TABLE_RE.findall(md))
    if missing <= 0:
        return md
    return md.rstrip() + "</table>" * missing


def _cell_texts(table_html: str) -> list[str]:
    """Visible text of every non-empty cell in a ``<table>`` block."""
    return [t for m in _CELL_RE.findall(table_html) if (t := _visible_text(m).strip())]


def _nonempty_cell_count(table_html: str) -> int:
    return len(_cell_texts(table_html))


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Fold text to matchable form and map each output char to its source index.

    NFKD plus alnum-only (everything else collapses to a single space) erases the
    encoding gaps between the OCR'd cell and the text layer — superscripts
    (``²⁺`` → ``2``), the micro sign (``µ`` vs Greek ``μ``), and the assorted
    dashes NFKD leaves as a U+2212 minus where the text layer has ASCII ``-``.
    The index map lets a match in the folded string recover the original char
    range, and thus the glyph boxes."""
    out: list[str] = []
    idx_map: list[int] = []
    prev_space = True
    for i, ch in enumerate(text):
        for d in unicodedata.normalize("NFKD", ch):
            if d.isalnum():
                out.append(d.lower())
                idx_map.append(i)
                prev_space = False
            elif not prev_space:
                out.append(" ")
                idx_map.append(i)
                prev_space = True
    while out and out[-1] == " ":
        out.pop()
        idx_map.pop()
    return "".join(out), idx_map


def _normalize(text: str) -> str:
    return _normalize_with_map(text)[0]


def _anchor_texts(cell_texts: list[str]) -> list[str]:
    """Normalized cell strings distinctive enough to seed localization."""
    seen: set[str] = set()
    anchors: list[str] = []
    for cell in cell_texts:
        norm = _normalize(cell)
        if len(norm) >= _MIN_ANCHOR_LEN and norm not in seen:
            seen.add(norm)
            anchors.append(norm)
    return anchors


def _union(boxes: list[_Box]) -> _Box:
    ls, bs, rs, ts = zip(*boxes, strict=True)
    return (min(ls), min(bs), max(rs), max(ts))


def _group_lines(char_boxes: list[_Box | None]) -> list[_Box]:
    """Cluster glyph boxes into text lines, top of page first.

    Boxes whose vertical intervals overlap are the same line; a clean vertical gap
    starts a new one.  Each returned box is one line's bounds (PDF points)."""
    boxes = sorted((b for b in char_boxes if b is not None), key=lambda b: -b[3])
    lines: list[list[float]] = []
    for left, bottom, right, top in boxes:
        if lines and bottom < lines[-1][3] and top > lines[-1][1]:  # overlaps line
            ln = lines[-1]
            ln[0], ln[1] = min(ln[0], left), min(ln[1], bottom)
            ln[2], ln[3] = max(ln[2], right), max(ln[3], top)
        else:
            lines.append([left, bottom, right, top])
    return [tuple(ln) for ln in lines]  # type: ignore[misc]


def _median(values: list[float]) -> float:
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _locate_bbox(
    anchors: list[str],
    page_norm: str,
    idx_map: list[int],
    char_boxes: list[_Box | None],
    page_size: tuple[float, float],
) -> _Box | None:
    """Bounding box (PDF points) of the table region, or ``None`` if unlocatable.

    Anchors that occur exactly once in the folded page text locate the table's
    lines (a repeated anchor is ambiguous — it may be prose — so it is skipped).
    The line run is then grown up and down while the inter-line gap stays within
    ``_GAP_FACTOR`` of the table's own spacing, so the box reaches rows no anchor
    covered (e.g. trailing data rows) yet halts at the wider gap to body prose.
    Horizontal extent is the union of the included lines."""
    seeds: list[_Box] = []
    for anchor in anchors:
        first = page_norm.find(anchor)
        if first == -1 or page_norm.find(anchor, first + 1) != -1:
            continue  # absent, or ambiguous (recurs elsewhere on the page)
        i0, i1 = idx_map[first], idx_map[first + len(anchor) - 1]
        spanned = [b for b in char_boxes[i0 : i1 + 1] if b is not None]
        if spanned:
            seeds.append(_union(spanned))
    if not seeds:
        return None

    lines = _group_lines(char_boxes)
    if not lines:
        return None
    gaps = [lines[i][1] - lines[i + 1][3] for i in range(len(lines) - 1)]

    seed_box = _union(seeds)
    touched = [
        i for i, ln in enumerate(lines) if ln[1] < seed_box[3] and ln[3] > seed_box[1]
    ]
    if not touched:
        return None
    lo, hi = min(touched), max(touched)

    # Median (not max) of the table's own line gaps: robust to a single wide
    # internal separation (a row-group break) that would otherwise inflate the
    # threshold enough to swallow the following section.
    anchor_gaps = gaps[lo:hi]
    spacing = _median(anchor_gaps) if anchor_gaps else (_median(gaps) if gaps else 0.0)
    threshold = spacing * _GAP_FACTOR
    while lo > 0 and gaps[lo - 1] <= threshold:
        lo -= 1
    while hi < len(lines) - 1 and gaps[hi] <= threshold:
        hi += 1

    # Legend allowance: a table's footnote/legend ("MW: molecular weight, …") sits
    # one line below the body, set off by a gap a little wider than the row gaps —
    # past the body threshold but well short of the margin to prose.  Pull in that
    # single line so the crop re-OCR can recover a legend the page OCR truncated.
    legend_threshold = spacing * _LEGEND_GAP_FACTOR
    added = 0
    while (
        hi < len(lines) - 1
        and added < _MAX_LEGEND_LINES
        and gaps[hi] <= legend_threshold
    ):
        hi += 1
        added += 1

    left, bottom, right, top = _union(list(lines[lo : hi + 1]))
    page_w, page_h = page_size
    return (
        max(0.0, left - _PAD_PT),
        max(0.0, bottom - _PAD_PT),
        min(page_w, right + _PAD_PT),
        min(page_h, top + _PAD_PT),
    )


def _table_columns(table_html: str) -> int:
    """Widest row's column count (honoring ``colspan``), at least 1."""
    best = 0
    for row in _ROW_RE.findall(table_html):
        cols = sum(
            int(m.group(1)) if (m := _COLSPAN_RE.search(attrs)) else 1
            for attrs in _CELL_OPEN_RE.findall(row)
        )
        best = max(best, cols)
    return best or 1


def _fold_subheading(heading: str, table_html: str) -> str:
    """Prepend ``heading`` as a full-width spanning header row.

    The crop re-OCR lifts a sub-table label out of the table into a heading; the
    full-page pass had it as a spanning ``<th>`` row, so we restore it as one — both
    keeping the label and the cell it contributes (so the substitution guard sees no
    regression)."""
    row = f'<tr><th colspan="{_table_columns(table_html)}">{heading}</th></tr>'
    thead = _THEAD_OPEN_RE.search(table_html)
    if thead:
        return table_html[: thead.end()] + row + table_html[thead.end() :]
    open_tag = _TABLE_OPEN_RE.match(table_html)
    at = open_tag.end() if open_tag else 0
    return f"{table_html[:at]}<thead>{row}</thead>{table_html[at:]}"


def _extract_tables(md: str) -> list[str]:
    """``<table>`` blocks from re-OCR output, inner ``<caption>`` stripped (the
    pipeline colocates captions separately, so an inline one would double), and any
    immediately-preceding sub-table label folded back in as a spanning header row."""
    tables: list[str] = []
    prev_end = 0
    for m in _TABLE_RE.finditer(md):
        table = _CAPTION_RE.sub("", m.group()).strip()
        labels = _SUBHEADING_RE.findall(md[prev_end : m.start()])
        if labels:
            table = _fold_subheading(labels[-1], table)
        tables.append(table)
        prev_end = m.end()
    return tables


def _table_regions(md: str) -> list[tuple[int, int, list[str]]]:
    """Group the page's ``<table>`` blocks into regions of consecutive tables
    (separated by whitespace only).  Returns ``(start, end, [table_html, ...])``
    char spans, so stacked tables the model split (e.g. a Table 2 "A"/"B") are
    re-OCR'd together from one crop."""
    regions: list[tuple[int, int, list[str]]] = []
    start = end = -1
    tables: list[str] = []
    for m in _TABLE_RE.finditer(md):
        if tables and md[end : m.start()].strip() == "":
            end = m.end()
            tables.append(m.group())
        else:
            if tables:
                regions.append((start, end, tables))
            start, end, tables = m.start(), m.end(), [m.group()]
    if tables:
        regions.append((start, end, tables))
    return regions


def _scaled_crop(
    page: pdfium.PdfPage, bbox: _Box, page_size: tuple[float, float]
) -> Image.Image:
    """Render only ``bbox`` of the page, scaled for a clean table re-OCR."""
    page_w, page_h = page_size
    left, bottom, right, top = bbox
    long_pt = max(right - left, top - bottom)
    scale = min(_MAX_SCALE, max(_MIN_SCALE, _TARGET_CROP_PX / long_pt))
    img: Image.Image = (
        page.render(scale=scale, crop=(left, bottom, page_w - right, page_h - top))
        .to_pil()
        .convert("RGB")
    )
    return _downscale_to_long_side(img)


def _legend_footnote_html(legend: str) -> str:
    """Render a table legend the crop recovered as a footnote paragraph.

    The legend is raw OCR markup — a ``<sup>a</sup>`` footnote marker, ``*emphasis*``
    (organism names), an inline ``$…$`` script — rendered tag-aware: blanket
    ``html.escape`` would surface a literal ``<sup>a</sup>``, while a raw passthrough
    would let a stray ``<``/``&`` (a footnote ``n<5``) start a bogus tag."""
    return f'<p class="footnote">{_render_inline(legend)}</p>'


def _crop_trailing(crop_md: str) -> str:
    """Text the crop re-OCR emitted *after* the last table — a legend the page OCR
    placed below the table (and may have truncated)."""
    idx = crop_md.rfind("</table>")
    return crop_md[idx + len("</table>") :].strip() if idx != -1 else ""


def _splice_region(md: str, start: int, end: int, new_tables: list[str]) -> str:
    return md[:start] + "\n\n".join(new_tables) + md[end:]


def _page_text_and_boxes(
    textpage: pdfium.PdfTextPage,
) -> tuple[str, list[_Box | None]]:
    """Page text paired with one glyph box per character, index-aligned.

    Built char by char in the same index domain so that position *p* in the
    returned text always indexes ``boxes[p]``.  ``get_text_range()`` (the text
    view) and ``count_chars()`` (the char-array) disagree on real PDFs — pdfium
    drops or inserts characters between the two — so deriving the text from one
    and the boxes from the other would silently misalign every glyph past the
    first dropped char, locating the table from the wrong region."""
    parts: list[str] = []
    boxes: list[_Box | None] = []
    for i in range(textpage.count_chars()):
        ch = textpage.get_text_range(i, 1)
        try:
            left, bottom, right, top = textpage.get_charbox(i)
            box: _Box | None = (
                (left, bottom, right, top) if right > left and top > bottom else None
            )
        except Exception:
            box = None
        parts.append(ch)
        boxes.extend([box] * len(ch))  # a glyph may decode to several text chars
    return "".join(parts), boxes


def _glyph_centers(
    page_norm: str, idx_map: list[int], char_boxes: list[_Box | None]
) -> list[tuple[float, float] | None]:
    """Per-``page_norm``-position glyph centre (``None`` for a space or box-less
    char), computed once per page so the coverage gate's per-region containment test
    is a cheap lookup rather than re-deriving every centre for every table region."""
    centers: list[tuple[float, float] | None] = []
    for pos, ch in enumerate(page_norm):
        box = None if ch == " " else char_boxes[idx_map[pos]]
        centers.append(
            None if box is None else ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        )
    return centers


def _in_bbox_tokens(
    bbox: _Box, page_norm: str, centers: list[tuple[float, float] | None]
) -> list[str]:
    """Folded text-layer tokens whose glyphs fall inside ``bbox``.

    Walks ``page_norm`` (its single spaces preserved as word boundaries) and keeps
    an alnum char only when its precomputed glyph centre lies in ``bbox``, so words
    outside the box drop out without gluing onto their in-box neighbours.  The
    boundary must come from ``page_norm`` rather than from filtering glyph boxes
    directly: a text-layer space carries no glyph box (pdfium reports none), so
    dropping box-less chars would weld adjacent words into one unmatchable token."""
    left, bottom, right, top = bbox
    kept: list[str] = []
    for ch, center in zip(page_norm, centers, strict=True):
        if (
            ch != " "
            and center is not None
            and left <= center[0] <= right
            and bottom <= center[1] <= top
        ):
            kept.append(ch)
        else:
            kept.append(" ")
    return "".join(kept).split()


def _adjacent_para_tokens(md: str, start: int, end: int) -> set[str]:
    """Folded tokens of the short paragraphs flanking a table region — the caption
    above and any legend below.  The located bbox grows to include these lines, yet
    they are captured in the page markdown though not in the cells, so the coverage
    gate must count them as already-present; otherwise every captioned table would
    look like it had dropped content and the gate would never fire.

    Only a genuinely short flanking block (a caption/legend) is folded in: a long
    block — or the whole pre-table content when no blank line separates it — is body
    prose the bbox overran, and adding it to ``captured`` would mask a real drop."""
    before = md[:start].rstrip().rsplit("\n\n", 1)[-1]
    after = md[end:].lstrip().split("\n\n", 1)[0]
    tokens: set[str] = set()
    for chunk in (before, after):
        text = _visible_text(chunk)
        if len(text) <= _ADJACENT_PARA_MAX_LEN:
            tokens.update(_normalize(text).split())
    return tokens


def _is_distinctive(token: str) -> bool:
    """A token strong enough to judge table completeness: a real word
    (``_COVERAGE_MIN_TOKEN_LEN``+) or a multi-char number (table data)."""
    return len(token) >= _COVERAGE_MIN_TOKEN_LEN or (
        len(token) >= _COVERAGE_MIN_NUMERIC_LEN and any(c.isdigit() for c in token)
    )


def _region_fully_captured(
    bbox: _Box,
    page_norm: str,
    centers: list[tuple[float, float] | None],
    captured: set[str],
) -> bool:
    """Whether the page already captured every distinctive text-layer token inside
    the region — in which case the crop re-OCR has nothing to recover and is skipped.

    Conservative by construction: only the *provably complete* case skips (no
    distinctive in-box token missing from ``captured``).  Distinctive covers real
    words and multi-char numbers, so a dropped data column counts and is not skipped.
    A scanned or empty text layer yields no distinctive tokens, so it returns
    ``False`` and the region is re-OCR'd as before — the gate is an opt-out for
    proven completeness, never an opt-out on absence of evidence.  This is
    deliberately strict: the cross-fixture data shows the located bbox pulls in body
    prose, so a loose threshold cannot separate a genuine drop (often a single
    token) from that prose noise; only the zero-missing case is safe to skip (see
    optimize-pipeline-performance.md §2)."""
    distinctive = [
        t for t in _in_bbox_tokens(bbox, page_norm, centers) if _is_distinctive(t)
    ]
    if not distinctive:
        return False
    return all(t in captured for t in distinctive)


@dataclass(frozen=True)
class _RegionPlan:
    """A table region whose crop re-OCR is worth attempting: its char span in the
    page's markdown, the original table blocks, and the rendered crop image.  Held
    per page (the caller keeps the page partition), so it carries no page index, and
    the re-OCR markdown flows as a separate value rather than mutating the plan."""

    start: int
    end: int
    tables: list[str]
    crop: Image.Image


def _plan_page_tables(md: str, page: pdfium.PdfPage) -> tuple[str, list[_RegionPlan]]:
    """Localize each table region on the page and render its re-OCR crop, dropping
    regions the page already captured in full (the coverage gate) or that cannot be
    localized.  Pure CPU work — no OCR.  Returns the page markdown with open tables
    closed (so the apply pass splices into the same string) and the crop plans.

    Closing unclosed tables first means a table that overran the page (and was thus
    truncated mid-row) is a closed region the crop re-OCR can locate and recover in
    full, not an unmatched fragment."""
    md = _close_unclosed_tables(md)
    regions = _table_regions(md)
    if not regions:
        return md, []

    page_text, char_boxes = _page_text_and_boxes(page.get_textpage())
    page_norm, idx_map = _normalize_with_map(page_text)
    centers = _glyph_centers(page_norm, idx_map, char_boxes)
    page_size = page.get_size()

    plans: list[_RegionPlan] = []
    for start, end, tables in regions:
        cells = [c for tbl in tables for c in _cell_texts(tbl)]
        bbox = _locate_bbox(
            _anchor_texts(cells), page_norm, idx_map, char_boxes, page_size
        )
        if bbox is None:
            continue
        captured = {t for c in cells for t in _normalize(c).split()}
        captured |= _adjacent_para_tokens(md, start, end)
        if _region_fully_captured(bbox, page_norm, centers, captured):
            continue
        crop = _scaled_crop(page, bbox, page_size)
        plans.append(_RegionPlan(start, end, tables, crop))
    return md, plans


def _apply_page_results(md: str, results: list[tuple[_RegionPlan, str]]) -> str:
    """Splice each planned region's re-OCR markup back into the page markdown.

    Each ``(plan, crop_md)`` pairs a region with its batched re-OCR output.  A region
    is replaced only when its re-OCR yields tables with at least as many non-empty
    cells as the originals, so a crop that came back worse (or empty) leaves the
    full-page transcription untouched.  Splices run from the back so earlier char
    offsets stay valid."""
    source_norm = _normalize(md)  # to tell a recovered legend from one already in md
    for plan, crop_md in sorted(results, key=lambda pc: pc[0].start, reverse=True):
        new_tables = _extract_tables(crop_md)
        if not new_tables:
            continue
        old_cells = sum(_nonempty_cell_count(t) for t in plan.tables)
        new_cells = sum(_nonempty_cell_count(t) for t in new_tables)
        if new_cells < old_cells:
            continue
        # Fold a legend the crop recovered from below the table into the table block
        # (so it rides as one float and stays under its table) — but only when it is
        # not already in the page markdown, else a normal table would duplicate the
        # following content the crop happens to also see.
        legend = _crop_trailing(crop_md)
        if (
            legend
            and len(legend) <= _LEGEND_MAX_LEN
            and _normalize(legend) not in source_norm
        ):
            new_tables[-1] += "\n" + _legend_footnote_html(legend)
        md = _splice_region(md, plan.start, plan.end, new_tables)
    return md


def _recover_dropped_tables(
    pdf_path: Path | str,
    pages_md: list[str],
    ocr_regions: Callable[[list[Image.Image]], list[str]],
) -> list[str]:
    """Re-OCR the table regions the page pass may have under-captured and splice the
    richer markup back in; return updated markdown, one entry per input page.

    Every page's crops are planned first (text-layer localization + the coverage
    gate, pure CPU work), then OCR'd in **one batched call** so the server can
    process them concurrently, then spliced back per page.  Pages without a table —
    and regions the gate found already complete — incur no OCR at all.  Order and
    length match ``pages_md`` so the caller's page-to-image alignment is preserved."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        closed_md: list[str] = []
        per_page_plans: list[list[_RegionPlan]] = []
        for i, md in enumerate(pages_md):
            if "<table" not in md.lower():
                closed_md.append(md)
                per_page_plans.append([])
                continue
            page_md, plans = _plan_page_tables(md, pdf[i])
            closed_md.append(page_md)
            per_page_plans.append(plans)

        flat = [plan for plans in per_page_plans for plan in plans]
        if not flat:
            return closed_md

        crop_mds = iter(ocr_regions([plan.crop for plan in flat]))
        for i, plans in enumerate(per_page_plans):
            if plans:
                pairs = [(plan, next(crop_mds)) for plan in plans]
                closed_md[i] = _apply_page_results(closed_md[i], pairs)
        return closed_md
    finally:
        pdf.close()
