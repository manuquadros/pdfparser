"""Figure extraction: parse LightOnOCR's bbox placeholders, recover the true
crop from the rendered page, merge over-segmented boxes, and emit ``<figure>``.

Pure image-processing, no GPU.  All boxes are LightOnOCR's ``[0, 1000]``-normalized
``x0,y0,x1,y1`` until ``_denormalize_bbox`` scales them to page pixels.
"""

from __future__ import annotations

import base64
import io
import re

import numpy as np
from PIL import Image  # noqa: TC002 — beartype reads annotations at runtime

from pdfparser.pipeline.latex import _inline_md_to_html

# LightOnOCR-bbox emits figures as a markdown image placeholder with the crop
# box appended as bare ``x0,y0,x1,y1`` integers **normalized to [0, 1000]** (per
# the model card), e.g. ``![image](image_1.png)122,89,877,614``.  The base
# variant omits the coordinates, so they are optional.
_BBOX_NORM_MAX = 1000
_FIGURE_PLACEHOLDER_RE = re.compile(
    r"^!\[[^\]]*\]\([^)]*\)"
    r"(?:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+))?\s*$"
)

_MIN_FIGURE_HEIGHT = 50  # pixels — gaps smaller than this are not figures
# The model's box often clips the bottom of a figure.  Rather than pad blindly
# (which grabs the caption when the box was already correct), grow the bottom
# edge over figure content and stop at the whitespace gap before the caption.
# A row is *blank* when fewer than this fraction of its pixels are ink — kept
# low (≈empty) so a sparse figure row (a thin axis, or content narrower than the
# box) still counts as content rather than ending growth early.  A vertical run
# of blank rows this tall is the gap that ends growth; growth is capped at this
# fraction of the page.
_FIGURE_BLANK_ROW_FRAC = 0.005
_FIGURE_INK_LEVEL = 250  # pixel value below which a grayscale pixel is "ink"
_FIGURE_GAP_FRAC = 0.012
_FIGURE_MAX_GROW_FRAC = 0.10
# The model sometimes over-segments one figure into stacked boxes (a tall figure
# split into a main box + a thin strip).  Two boxes are the same figure when they
# share a column (substantial horizontal overlap) and are vertically adjacent.
_FIGURE_MERGE_GAP_FRAC = 0.03  # of page height


def _parse_figure_placeholder(line: str) -> tuple[int, int, int, int] | None | bool:
    """Classify a single markdown line as a figure placeholder.

    Returns the ``(x0, y0, x1, y1)`` crop box (normalized to ``[0, 1000]``) when
    present, ``True`` for a bbox-less placeholder (base-variant fallback), or
    ``None`` when the line is not a figure placeholder at all.
    """
    m = _FIGURE_PLACEHOLDER_RE.match(line.strip())
    if m is None:
        return None
    if m.group(1) is None:
        return True
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))


def _figure_html(crop: Image.Image, caption_text: str | None) -> str:
    """Encode a figure crop as a base64 PNG and return a <figure> element."""
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    data_uri = f"data:image/png;base64,{b64}"
    caption_html = (
        f"<figcaption>{_inline_md_to_html(caption_text)}</figcaption>"
        if caption_text
        else ""
    )
    return f'<figure><img src="{data_uri}" alt="">{caption_html}</figure>'


def _denormalize_bbox(
    bbox: tuple[int, int, int, int], image: Image.Image
) -> tuple[int, int, int, int]:
    """Scale a ``[0, 1000]``-normalized box to ``image``'s pixel coordinates."""
    w, h = image.size
    return (
        round(bbox[0] / _BBOX_NORM_MAX * w),
        round(bbox[1] / _BBOX_NORM_MAX * h),
        round(bbox[2] / _BBOX_NORM_MAX * w),
        round(bbox[3] / _BBOX_NORM_MAX * h),
    )


def _extend_bottom_to_content(image: Image.Image, x0: int, x1: int, y1: int) -> int:
    """Grow ``y1`` downward to recover a figure bottom the box clipped.

    Only grows when the ink below ``y1`` ends in a clear whitespace gap (the
    space before the caption): then ``y1`` moves to that figure bottom.  If the
    ink runs to the search cap with no gap — ambiguous, and usually caption/body
    text below a correct box — the box is left unchanged so no text is pulled in.
    """
    h = image.size[1]
    limit = min(h, y1 + round(_FIGURE_MAX_GROW_FRAC * h))
    if y1 >= limit:
        return y1
    strip = np.asarray(image.crop((x0, y1, x1, limit)).convert("L"))
    ink_per_row = (strip < _FIGURE_INK_LEVEL).mean(axis=1)
    gap = max(1, round(_FIGURE_GAP_FRAC * h))
    last_ink = 0  # rows past y1, exclusive
    blank_run = 0
    found_gap = False
    for offset, fraction in enumerate(ink_per_row, start=1):
        if fraction < _FIGURE_BLANK_ROW_FRAC:
            blank_run += 1
            if blank_run >= gap:
                found_gap = True
                break
        else:
            blank_run = 0
            last_ink = offset
    return y1 + last_ink if found_gap else y1


def _safe_crop(
    image: Image.Image, bbox: tuple[int, int, int, int]
) -> Image.Image | None:
    """Crop a pixel-space ``bbox`` from ``image``, clamped to bounds and grown
    down to the figure's true bottom edge; ``None`` if degenerate."""
    w, h = image.size
    x0 = max(0, min(bbox[0], w))
    y0 = max(0, min(bbox[1], h))
    x1 = max(0, min(bbox[2], w))
    y1 = max(0, min(bbox[3], h))
    if x1 - x0 < _MIN_FIGURE_HEIGHT or y1 - y0 < _MIN_FIGURE_HEIGHT:
        return None
    y1 = _extend_bottom_to_content(image, x0, x1, y1)
    return image.crop((x0, y0, x1, y1))


def _union_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _figures_same(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: float
) -> bool:
    x_overlap = min(a[2], b[2]) - max(a[0], b[0])
    if x_overlap < 0.5 * min(a[2] - a[0], b[2] - b[0]):
        return False
    y_gap = max(a[1], b[1]) - min(a[3], b[3])  # ≤ 0 when the boxes overlap
    return y_gap <= gap


def _cluster_figure_boxes(
    boxes: list[tuple[int, int, int, int]], gap: float
) -> list[list[int]]:
    """Group indices of boxes that belong to the same figure (transitive)."""
    parent = list(range(len(boxes)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _figures_same(boxes[i], boxes[j], gap):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())
