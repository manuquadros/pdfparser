"""Document assembly: per-page block parts → cleaned, classified HTML shell.

Design B-prime (see plans/replace-falcon-with-lightonocr.md):
render → per-page markdown → block HTML → cleanup/merge → document shell.

``_assemble_html`` is the pure (no-GPU) core — render-free, model-free — so it is
unit-testable by feeding synthetic markdown + images.  ``lightonocr_pdf_to_html``
wires in rendering and OCR around it.
"""

from __future__ import annotations

import html as _html
import re
from pathlib import Path
from typing import Any

from PIL import Image  # noqa: TC002 — beartype reads annotations at runtime

from pdfparser.pipeline.classify import (
    _REF_SECTION_RE,
    _classify_parts,
    _extract_front_matter,
    _leading_pages_to_skip_md,
    _strip_running_furniture,
)
from pdfparser.pipeline.figures import (
    _FIGURE_MERGE_GAP_FRAC,
    _cluster_figure_boxes,
    _denormalize_bbox,
    _figure_html,
    _parse_figure_placeholder,
    _safe_crop,
    _union_box,
)
from pdfparser.pipeline.latex import _inline_md_to_html, _latex_to_html
from pdfparser.pipeline.markdown import _md_to_html_blocks
from pdfparser.pipeline.merge import _colocate_table_captions, _merge_split_paragraphs
from pdfparser.pipeline.model import OcrModel, _ocr_page, load_ocr_model
from pdfparser.pipeline.render import _render_page_images
from pdfparser.pipeline.text import _STRIP_TAGS_RE, _looks_like_figure_caption

_WRAPPER_CSS = """
body {
    font-family: Georgia, serif;
    max-width: 860px;
    margin: 2rem auto;
    padding: 0 1.5rem;
    color: #222;
    line-height: 1.7;
    font-size: 16px;
}
header {
    border-bottom: 1px solid #ccc; margin-bottom: 2rem; padding-bottom: 1rem; }
header h1 { font-size: 1.4rem; margin: 0 0 .4rem; }
h1 { font-size: 1.3rem; margin: 1.5em 0 .4em; }
h2 { font-size: 1.15rem; margin: 1.5em 0 .4em; border-bottom: 1px solid #e0e0e0; }
h3 { font-size: 1rem; margin: 1.2em 0 .3em; }
p  { margin: .6em 0; }
section.abstract { background: #f7f7f7; padding: 1em 1.2em; border-radius: 4px;
    margin: 1.5em 0; }
details.metadata { margin: 1.5em 0; border: 1px solid #e0e0e0; border-radius: 4px;
    padding: .3em 1em; background: #fafafa; font-size: .9rem; color: #555; }
details.metadata > summary { cursor: pointer; font-weight: bold; color: #333;
    list-style: disclosure-closed; }
details.metadata[open] > summary { list-style: disclosure-open;
    margin-bottom: .4em; }
figure { margin: 1.5em 0; }
figcaption { font-size: .875em; color: #555; }
p.footnote { font-size: .8rem; color: #666; border-top: 1px solid #eee;
    padding-top: .3em; margin-top: .3em; }
table { border-collapse: collapse; width: 100%; overflow-x: auto;
    display: block; font-size: .9rem; margin: 1em 0; }
caption { font-weight: bold; text-align: left; padding: .4em 0 .6em;
    font-size: .9rem; color: #333; }
td, th { padding: .4em .7em; border: 1px solid #ccc; }
hr { border: none; border-top: 1px solid #ddd; margin: 2rem 0; }
"""


def _page_to_html_parts(md: str, image: Image.Image) -> list[str]:
    """Convert one page's markdown to block HTML, replacing each ``![image]``
    placeholder with a cropped <figure> (caption from the placeholder's own
    trailing text, else the following caption-like block).  Figure boxes the
    model split across stacked crops are unioned into a single figure."""
    # Parse into ordered items: ("md", text) or ("fig", box_px | None, caption).
    items: list[tuple[Any, ...]] = []
    k = 0
    raw_blocks = re.split(r"\n[ \t]*\n", md.strip())
    while k < len(raw_blocks):
        block = raw_blocks[k].strip()
        lines = block.splitlines()
        fig = _parse_figure_placeholder(lines[0]) if lines else None
        if fig is None:
            items.append(("md", block))
            k += 1
            continue
        rest = "\n".join(lines[1:]).strip()
        caption: str | None = rest or None
        if (
            caption is None
            and k + 1 < len(raw_blocks)
            and _looks_like_figure_caption(raw_blocks[k + 1])
        ):
            k += 1
            caption = raw_blocks[k].strip()
        box = _denormalize_bbox(fig, image) if isinstance(fig, tuple) else None
        items.append(("fig", box, caption))
        k += 1

    positions = [
        i for i, it in enumerate(items) if it[0] == "fig" and it[1] is not None
    ]
    boxes = [items[i][1] for i in positions]
    gap = _FIGURE_MERGE_GAP_FRAC * image.size[1]
    union_at: dict[int, tuple[int, int, int, int]] = {}
    caption_at: dict[int, str | None] = {}
    drop: set[int] = set()
    for group in _cluster_figure_boxes(boxes, gap):
        members = sorted(positions[g] for g in group)
        union_at[members[0]] = _union_box([boxes[g] for g in group])
        caption_at[members[0]] = next(
            (items[m][2] for m in members if items[m][2]), None
        )
        drop.update(members[1:])

    parts: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            parts.extend(_md_to_html_blocks("\n\n".join(pending)))
            pending.clear()

    for i, it in enumerate(items):
        if it[0] == "md":
            pending.append(it[1])
            continue
        flush()
        if i in drop:
            continue
        box = union_at.get(i, it[1])
        caption = caption_at.get(i, it[2])
        caption_html = _latex_to_html(caption) if caption else None
        crop = _safe_crop(image, box) if box is not None else None
        if crop is not None:
            parts.append(_figure_html(crop, caption_html))
        elif caption_html is not None:
            parts.append(
                f"<figure><figcaption>{_inline_md_to_html(caption_html)}"
                "</figcaption></figure>"
            )
    flush()
    return parts


def _assemble_html(pages_md: list[str], images: list[Image.Image]) -> str:
    start = _leading_pages_to_skip_md(pages_md)
    pages_md = pages_md[start:]
    images = images[start:]

    parts: list[str] = []
    for md, img in zip(pages_md, images, strict=True):
        parts.extend(_page_to_html_parts(md, img))
    parts = _colocate_table_captions(parts)

    meta = _classify_parts(parts)

    title_html = meta.title_html or "Untitled"
    title_safe = _html.escape(_STRIP_TAGS_RE.sub("", title_html))
    byline_safe = _html.escape(meta.byline_html)

    abstract_html = (
        "<section class='abstract'>\n"
        + "\n".join(_merge_split_paragraphs(_merge_split_paragraphs(meta.abstract)))
        + "\n</section>"
        if meta.abstract
        else ""
    )

    body = _merge_split_paragraphs(
        _merge_split_paragraphs(_strip_running_furniture(meta.body))
    )
    if meta.footnotes:
        ref_idx = next(
            (i for i, p in enumerate(body) if _REF_SECTION_RE.match(p)), len(body)
        )
        body[ref_idx:ref_idx] = meta.footnotes
    metadata, body = _extract_front_matter(body)
    # Front matter is kept right after the abstract, collapsed by default in a
    # toggleable <details> panel, so the body opens with prose.
    metadata_html = (
        "<details class='metadata'>\n<summary>Metadata</summary>\n"
        + "\n".join(metadata)
        + "\n</details>"
        if metadata
        else ""
    )
    body_html = "\n".join(body)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title_safe}</title>
<style>{_WRAPPER_CSS}</style>
</head>
<body>
<header>
  <h1>{title_html}</h1>
  <p>{byline_safe}</p>
</header>
{abstract_html}
{metadata_html}
<div class="body">
{body_html}
</div>
</body>
</html>"""


def lightonocr_pdf_to_html(
    pdf_path: Path | str,
    *,
    ocr: OcrModel | None = None,
    device: str | None = None,
) -> str:
    """Convert a PDF to self-contained HTML with LightOnOCR-2-1B-bbox.

    Args:
        pdf_path: Path to the input PDF.
        ocr: Pre-loaded model bundle.  ``None`` calls ``load_ocr_model()``.
        device: Torch device string.  Only used when ``ocr`` is ``None``.
    """
    if ocr is None:
        ocr = load_ocr_model(device)
    images = _render_page_images(Path(pdf_path))
    pages_md = [_ocr_page(img, ocr) for img in images]
    return _assemble_html(pages_md, images)
