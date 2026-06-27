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

import html
import re
from collections import Counter
from collections.abc import Callable  # noqa: TC003 — beartype reads annotations
from dataclasses import dataclass

import pypdfium2 as pdfium  # noqa: TC002 — beartype reads annotations at runtime
from PIL import Image  # noqa: TC002 — beartype reads annotations at runtime

from pdfparser.pipeline.layers import (
    _Box,
    _DocumentLayers,
    _normalize,
    _PageLayer,
)
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
# More than this many byte-identical consecutive rows is a decode repetition loop,
# not real data — a genuine table never repeats one row this many times in a row.
_MAX_IDENTICAL_ROW_RUN = 3
_PAD_PT = 4.0  # small margin so glyph edges are not clipped by the crop

# Crop render scale: aim for this long side in px, clamped to a sane DPI band.
_TARGET_CROP_PX = 1500
_MIN_SCALE = 200 / 72
_MAX_SCALE = 600 / 72

# Quarter-turn rotations whose text reads along the *vertical* page axis — a table
# at either is "sideways" and localized along that axis; 0°/180° keep horizontal rows.
_SIDEWAYS_ROTATIONS = frozenset({90, 270})


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


def _group_lines(char_boxes: list[_Box | None], vertical: bool = False) -> list[_Box]:
    """Cluster glyph boxes into text lines, in reading order.

    Boxes whose cross-flow intervals overlap are the same line; a clean gap starts a
    new one.  Each returned box is one line's bounds (PDF points).  For upright text
    a line is a horizontal run (boxes sharing a vertical interval), ordered top of
    page first.  For a sideways table (text at 90°/270°) the reading lines run
    *vertically* — a line is a column-strip of glyphs sharing a horizontal interval —
    so clustering and ordering switch to the x-axis (left of page first)."""
    lo_i, hi_i = (0, 2) if vertical else (1, 3)
    key = (lambda b: b[0]) if vertical else (lambda b: -b[3])
    boxes = sorted((b for b in char_boxes if b is not None), key=key)
    lines: list[list[float]] = []
    for box in boxes:
        if lines and box[lo_i] < lines[-1][hi_i] and box[hi_i] > lines[-1][lo_i]:
            ln = lines[-1]
            ln[0], ln[1] = min(ln[0], box[0]), min(ln[1], box[1])
            ln[2], ln[3] = max(ln[2], box[2]), max(ln[3], box[3])
        else:
            lines.append(list(box))
    return [tuple(ln) for ln in lines]  # type: ignore[misc]


def _median(values: list[float]) -> float:
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def _seed_rotation(rotations: list[int]) -> int:
    """The dominant quarter-turn among a region's seed glyphs (0 when none rotate).

    A sideways table carries the same rotation on every glyph, so the mode of the
    seed glyphs' rotations identifies it; ties and an empty input fall back to 0 (the
    upright path), so a stray rotated glyph can never flip an upright table sideways."""
    if not rotations:
        return 0
    counts = Counter(rotations)
    best = max(counts.values())
    return 0 if counts[0] == best else max(counts, key=lambda r: (counts[r], -r))


def _collect_seeds(
    anchors: list[str],
    page_norm: str,
    idx_map: list[int],
    char_boxes: list[_Box | None],
    char_rotations: list[int | None],
) -> tuple[list[_Box], list[int]]:
    """Seed boxes and their rotations from each anchor that occurs exactly once
    (a repeated anchor is ambiguous — it may be prose — so it is skipped)."""
    seeds: list[_Box] = []
    seed_rotations: list[int] = []
    for anchor in anchors:
        first = page_norm.find(anchor)
        if first == -1 or page_norm.find(anchor, first + 1) != -1:
            continue  # absent, or ambiguous (recurs elsewhere on the page)
        i0, i1 = idx_map[first], idx_map[first + len(anchor) - 1]
        spanned = [b for b in char_boxes[i0 : i1 + 1] if b is not None]
        if spanned:
            seeds.append(_union(spanned))
            seed_rotations.extend(
                r for r in char_rotations[i0 : i1 + 1] if r is not None
            )
    return seeds, seed_rotations


def _locate_bbox(
    anchors: list[str],
    page_norm: str,
    idx_map: list[int],
    char_boxes: list[_Box | None],
    char_rotations: list[int | None],
    page_size: tuple[float, float],
) -> tuple[_Box, int] | None:
    """Bounding box (PDF points) and rotation of the table region, or ``None``.

    Anchors that occur exactly once in the folded page text locate the table's
    lines (a repeated anchor is ambiguous — it may be prose — so it is skipped).
    The line run is then grown along the reading axis while the inter-line gap stays
    within ``_GAP_FACTOR`` of the table's own spacing, so the box reaches rows no
    anchor covered (e.g. trailing data rows) yet halts at the wider gap to body
    prose.  Cross-axis extent is the union of the included lines.

    A sideways table (its anchor glyphs rotated 90°/270°) is localized along its
    *true* reading axis: lines run vertically, growth runs left↔right, and only
    glyphs sharing the table's rotation are clustered — so the box grows over the
    table's columns and stops at the gutter to the upright body text beside it
    instead of sweeping a neighbouring column's heading into the crop."""
    seeds, seed_rotations = _collect_seeds(
        anchors, page_norm, idx_map, char_boxes, char_rotations
    )
    if not seeds:
        return None

    rot = _seed_rotation(seed_rotations)
    vertical = rot in _SIDEWAYS_ROTATIONS
    lo_i, hi_i = (0, 2) if vertical else (1, 3)
    # For a sideways table, cluster only the sideways glyphs so the upright body text
    # beside it can't bridge the column gutter into the box.  Keep *both* 90° and
    # 270° (not just the modal rotation): a uniformly-rotated table can still have a
    # cell glyph pdfium snaps to the opposite quarter-turn, and dropping it would
    # clip that column from the crop.  Upright (0°) and inverted (180°) body text is
    # still excluded.
    boxes = (
        [
            b
            for b, r in zip(char_boxes, char_rotations, strict=True)
            if r in _SIDEWAYS_ROTATIONS
        ]
        if vertical
        else char_boxes
    )
    lines = _group_lines(boxes, vertical)
    if not lines:
        return None
    if vertical:
        gaps = [lines[i + 1][0] - lines[i][2] for i in range(len(lines) - 1)]
    else:
        gaps = [lines[i][1] - lines[i + 1][3] for i in range(len(lines) - 1)]

    seed_box = _union(seeds)
    touched = [
        i
        for i, ln in enumerate(lines)
        if ln[lo_i] < seed_box[hi_i] and ln[hi_i] > seed_box[lo_i]
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
    # Skipped for a sideways table: its legend is colocated as a separate caption,
    # and the line-gap model that anchors it assumes upright rows.
    legend_threshold = spacing * _LEGEND_GAP_FACTOR
    added = 0
    while (
        not vertical
        and hi < len(lines) - 1
        and added < _MAX_LEGEND_LINES
        and gaps[hi] <= legend_threshold
    ):
        hi += 1
        added += 1

    left, bottom, right, top = _union(list(lines[lo : hi + 1]))
    page_w, page_h = page_size
    bbox = (
        max(0.0, left - _PAD_PT),
        max(0.0, bottom - _PAD_PT),
        min(page_w, right + _PAD_PT),
        min(page_h, top + _PAD_PT),
    )
    return bbox, rot


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


def _collapse_repeated_rows(table_html: str) -> str:
    """Collapse a degenerate run of identical consecutive ``<tr>`` rows to one.

    A tight table crop sometimes drives the model into a decode repetition loop:
    it emits one row over and over (the "RAMS Deviations" explosion — dozens of
    identical rows trailing a real table).  A run of more than
    ``_MAX_IDENTICAL_ROW_RUN`` byte-identical adjacent rows is that pathology, never
    real tabular data, so it is reduced to its first occurrence.  Left in, the loop's
    cell count also beats the truncated original and wins the substitution gate."""
    matches = list(_ROW_RE.finditer(table_html))
    drop = [False] * len(matches)
    i = 0
    while i < len(matches):
        key = matches[i].group()
        j = i + 1
        while j < len(matches) and matches[j].group() == key:
            j += 1
        if j - i > _MAX_IDENTICAL_ROW_RUN:
            for k in range(i + 1, j):
                drop[k] = True
        i = j
    if not any(drop):
        return table_html
    out: list[str] = []
    cursor = 0
    for m, dropped in zip(matches, drop, strict=True):
        if dropped:
            gap = table_html[cursor : m.start()]
            if gap.strip():
                out.append(gap)
            cursor = m.end()
    out.append(table_html[cursor:])
    return "".join(out)


def _collapse_repeated_rows_md(md: str) -> str:
    """Collapse decode-loop row runs in every ``<table>`` of a page's markdown.

    The crop re-OCR path collapses its own substitutions, but the *page* pass has
    its own re-OCR (``_ocr_page`` retries a length-truncated page over the full
    context window) that can decode-loop on a dense table and land the explosion
    straight in ``pages_md`` — a path the crop guard never sees.  Running the
    collapse over the assembled page markdown catches the loop whatever its source.
    Idempotent: a table with no degenerate run is returned byte-for-byte."""
    return _TABLE_RE.sub(lambda m: _collapse_repeated_rows(m.group()), md)


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
    """A token strong enough to judge table completeness: a real word (length
    ``_COVERAGE_MIN_TOKEN_LEN`` or more) or a multi-char number (table data)."""
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


def _plan_page_tables(
    md: str, layers: _DocumentLayers, page_index: int
) -> tuple[str, list[_RegionPlan]]:
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

    layer = layers.page_layer(page_index)
    centers = _glyph_centers(layer.norm, layer.idx_map, layer.char_boxes)
    page = layers.pdf[page_index]
    page_size = page.get_size()

    plans: list[_RegionPlan] = []
    for start, end, tables in regions:
        cells = [c for tbl in tables for c in _cell_texts(tbl)]
        located = _locate_bbox(
            _anchor_texts(cells),
            layer.norm,
            layer.idx_map,
            layer.char_boxes,
            layer.char_rotations,
            page_size,
        )
        if located is None:
            continue
        bbox, rot = located
        captured = {t for c in cells for t in _normalize(c).split()}
        captured |= _adjacent_para_tokens(md, start, end)
        if _region_fully_captured(bbox, layer.norm, centers, captured):
            continue
        crop = _scaled_crop(page, bbox, page_size)
        # Turn a sideways table upright before re-OCR: the model reads an upright
        # table's column structure correctly, but mis-groups a rotated one (a 4-wide
        # column header collapses to colspan=2).  rotate() is CCW and the glyph angle
        # is the CCW rotation, so rotating by it cancels the page rotation.
        if rot:
            crop = crop.rotate(rot, expand=True)
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
        new_tables = [_collapse_repeated_rows(t) for t in _extract_tables(crop_md)]
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


# --- Text-layer table repair ---------------------------------------------------------
# A dense two-column statistics table (crystallography refinement stats) is mangled by
# the OCR in a way it can't self-correct: it drops the empty top-left header cell,
# shifting the first rows off by one and *losing* a value, and truncates the tail —
# and the failure is deterministic (same every run).  The PDF text layer holds the
# table's full content at fixed positions, so for such a table we rebuild the
# *structure* from the text layer (rows by y, label|value by the column gap) and keep
# the OCR's good cell *formatting* by matching each cell's text — recovering the dropped
# value/header with correct sub/superscripts, deterministically.  Crosses the usual
# "text layer = geometry only" rule, scoped to a two-column table the OCR got wrong.
_RECON_X_PAD = 6.0
_RECON_ROW_TOL = 6.0  # merge super/subscript-shifted glyphs into their row
_RECON_MIN_SEEDS = 4  # unique anchors needed to localize the column confidently
_RECON_SPACE_FRAC = 0.3  # insert a space when a glyph gap exceeds this * glyph width
_RECON_LABEL_GAP = 10.0  # min label|value separation (pt)
_RECON_VALUE_OFFSET = 25.0  # a row starting this far right of the labels has no label
_RECON_FOOTNOTE_LEN = 34  # a long value-less row past the body is a footnote: stop
# The space heuristic over-splits a digit run (a narrow "1" reads as a gap); rejoin a
# space that sits between two digits/decimal points.
_RECON_NUM_SPACE = re.compile(r"(?<=[\d.])\s+(?=[\d.])")
_Glyph = tuple[str, float, float, float]  # char, x0, x1, y-center


def _glyph_cell_text(seg: list[_Glyph]) -> str:
    """Join a glyph run into text — the text layer drops space glyphs, so insert a space
    at each inter-glyph gap wider than a fraction of the glyph width, then rejoin a
    digit run the heuristic over-split."""
    if not seg:
        return ""
    widths = sorted(g[2] - g[1] for g in seg)
    cw = widths[len(widths) // 2] or 1.0
    out = seg[0][0]
    for prev, cur in zip(seg, seg[1:], strict=False):
        if cur[1] - prev[2] > _RECON_SPACE_FRAC * cw:
            out += " "
        out += cur[0]
    return _RECON_NUM_SPACE.sub("", out.strip())


def _group_glyph_rows(glyphs: list[_Glyph]) -> list[list[_Glyph]]:
    """Group glyphs into rows top-to-bottom, merging a baseline-shifted super/subscript
    glyph into its row (within ``_RECON_ROW_TOL``); each row is sorted left to right."""
    rows: list[list[_Glyph]] = []
    anchors: list[float] = []
    for g in sorted(glyphs, key=lambda g: (-g[3], g[1])):
        if anchors and abs(anchors[-1] - g[3]) <= _RECON_ROW_TOL:
            rows[-1].append(g)
        else:
            rows.append([g])
            anchors.append(g[3])
    return [sorted(row, key=lambda g: g[1]) for row in rows]


def _rows_to_cells(rows: list[list[_Glyph]]) -> list[tuple[str, str]]:
    """Turn text-layer rows into ``(label, value)`` cells.  A row that starts in the
    value column has an empty label (the data-column header, or a centered divider); a
    row with a clear gap splits there; a gapless row is a single cell.  Stops at the
    first footnote (a long value-less row below the table body)."""
    if not rows:
        return []
    label_left = min(
        (
            row[0][1]
            for row in rows
            if any(
                row[i + 1][1] - row[i][2] >= _RECON_LABEL_GAP
                for i in range(len(row) - 1)
            )
        ),
        default=rows[0][0][1],
    )
    cells: list[tuple[str, str]] = []
    seen_data = 0
    for row in rows:
        gaps = [(row[i + 1][1] - row[i][2], i) for i in range(len(row) - 1)]
        maxgap, gi = max(gaps, default=(0.0, -1))
        if row[0][1] > label_left + _RECON_VALUE_OFFSET:
            label, value = "", _glyph_cell_text(row)
        elif maxgap >= _RECON_LABEL_GAP:
            label = _glyph_cell_text(row[: gi + 1])
            value = _glyph_cell_text(row[gi + 1 :])
        else:
            label, value = _glyph_cell_text(row), ""
        # A long value-less row is the table's title (above the body) or a footnote
        # (below it): skip it above the body, stop at it once data rows have started.
        if not value and len(label) > _RECON_FOOTNOTE_LEN:
            if seen_data >= 3:
                break
            continue
        # skip a stray rule glyph (a lone "_" the header underline leaves behind)
        if not label and len(value) <= 1 and not value.isalnum():
            continue
        if label or value:
            cells.append((label, value))
            seen_data += bool(value)
    return cells


def _cell_format_map(table_html: str) -> dict[str, str]:
    """Map each OCR cell's space-stripped normalized text to its formatted inner HTML,
    so a reconstructed cell can reuse the OCR's sub/superscript markup."""
    fmt: dict[str, str] = {}
    for inner in _CELL_RE.findall(table_html):
        key = _normalize(_visible_text(inner)).replace(" ", "")
        if key and key not in fmt:
            fmt[key] = inner.strip()
    return fmt


def _format_cell(plain: str, fmt: dict[str, str]) -> str:
    # The format-map branch is already escaped HTML from the OCR cell; only the raw
    # text-layer fallback needs escaping, or a value carrying '<'/'>'/'&' (a "<0.001"
    # statistic, an "&") injects markup / breaks the <td>.
    return fmt.get(_normalize(plain).replace(" ", "")) or html.escape(plain)


def _leading_caption_rows(table_html: str) -> tuple[list[str], str]:
    """The OCR table's leading spanning rows (the ``<th colspan>`` title/caption) and
    their combined normalized text — kept verbatim (they carry the table title) and used
    to drop the title fragments the text-layer pass picks up from the wide centered
    caption."""
    rows: list[str] = []
    for row in _ROW_RE.findall(table_html):
        if len(_CELL_RE.findall(row)) == 1 and _COLSPAN_RE.search(row):
            rows.append(row.strip())
        else:
            break
    return rows, _normalize(_visible_text("".join(rows)))


def _build_reconstructed_table(
    cells: list[tuple[str, str]], fmt: dict[str, str], caption_rows: list[str]
) -> str:
    rows: list[str] = list(caption_rows)
    for label, value in cells:
        if label and not value:
            rows.append(f'<tr><td colspan="2">{_format_cell(label, fmt)}</td></tr>')
        else:
            rows.append(
                f"<tr><td>{_format_cell(label, fmt)}</td>"
                f"<td>{_format_cell(value, fmt)}</td></tr>"
            )
    return "<table>\n" + "\n".join(rows) + "\n</table>"


def _reconstruct_table_from_text_layer(
    layer: _PageLayer, table_html: str
) -> str | None:
    """Rebuild a two-column table from the page text layer, or ``None`` if it can't be
    localized.  Localizes the table *column* from the OCR cells used as anchors, so the
    body prose in the other column of a two-column page is excluded."""
    seeds, _ = _collect_seeds(
        _anchor_texts(_cell_texts(table_html)),
        layer.norm,
        layer.idx_map,
        layer.char_boxes,
        layer.char_rotations,
    )
    if len(seeds) < _RECON_MIN_SEEDS:
        return None
    col_left = min(s[0] for s in seeds) - _RECON_X_PAD
    col_right = max(s[2] for s in seeds) + _RECON_X_PAD
    y_top = max(s[3] for s in seeds) + _RECON_X_PAD
    glyphs = [
        (ch, b[0], b[2], (b[1] + b[3]) / 2)
        for ch, b in zip(layer.page_text, layer.char_boxes, strict=True)
        if b is not None
        and not ch.isspace()
        and col_left <= (b[0] + b[2]) / 2 <= col_right
        and (b[1] + b[3]) / 2 <= y_top
    ]
    caption_rows, caption_text = _leading_caption_rows(table_html)
    # Drop the wrapped title fragments the wide centered caption leaves in the column,
    # but only *above* the table body — a value-less row mid-table is a real divider
    # whose label can legitimately be a substring of the caption (e.g. "Refinement").
    cells: list[tuple[str, str]] = []
    seen_data = False
    for label, value in _rows_to_cells(_group_glyph_rows(glyphs)):
        seen_data = seen_data or bool(value)
        if (
            not seen_data
            and not value
            and _normalize(label)
            and (_normalize(label) in caption_text)
        ):
            continue
        cells.append((label, value))
    if not cells:
        return None
    return _build_reconstructed_table(cells, _cell_format_map(table_html), caption_rows)


def _repair_tables_from_text_layer(
    layers: _DocumentLayers, pages_md: list[str]
) -> list[str]:
    """Replace a two-column table the OCR mangled with a deterministic text-layer
    reconstruction, when that recovers more rows (the OCR was truncated/off-by-one); a
    healthy table keeps its better-formatted OCR markup.  One entry per input page."""
    out: list[str] = []
    for i, md in enumerate(pages_md):
        if "<table" not in md.lower():
            out.append(md)
            continue
        out.append(_repair_page_tables(md, layers, i))
    return out


def _repair_page_tables(md: str, layers: _DocumentLayers, page_index: int) -> str:
    # Extract the page text layer lazily and at most once: _TABLE_RE.sub calls repair
    # for every complete <table>, and the extraction is thousands of native calls — so
    # share it (via the document-level cache) across a page's 2-column tables, but skip
    # it entirely on a page whose tables are all non-2-column (the common case) or whose
    # <table> is unclosed (no _TABLE_RE match), where no reconstruction is ever
    # attempted.
    def repair(m: re.Match[str]) -> str:
        table = m.group()
        if _table_columns(table) != 2:  # the repair only understands label|value tables
            return table
        layer = layers.page_layer(page_index)
        recon = _reconstruct_table_from_text_layer(layer, table)
        # Compare against the *collapsed* OCR table: a decode-loop explosion inflates
        # its raw <tr> count (collapsed only later in _assemble_html), which would
        # otherwise mask a genuinely shorter (off-by-one/truncated) table.
        if recon is None or recon.count("<tr") <= _collapse_repeated_rows(table).count(
            "<tr"
        ):
            return table  # couldn't localize, or the OCR table is no shorter
        return recon

    return _TABLE_RE.sub(repair, md)


def _recover_dropped_tables(
    layers: _DocumentLayers,
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
    closed_md: list[str] = []
    per_page_plans: list[list[_RegionPlan]] = []
    for i, md in enumerate(pages_md):
        if "<table" not in md.lower():
            closed_md.append(md)
            per_page_plans.append([])
            continue
        page_md, plans = _plan_page_tables(md, layers, i)
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
