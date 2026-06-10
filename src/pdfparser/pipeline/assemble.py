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
from dataclasses import dataclass
from pathlib import Path

from PIL import Image  # noqa: TC002 — beartype reads annotations at runtime

from pdfparser.pipeline.classify import (
    _REF_SECTION_RE,
    _classify_parts,
    _extract_front_matter,
    _extract_named_metadata_sections,
    _extract_stray_metadata,
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
from pdfparser.pipeline.merge import (
    _colocate_table_captions,
    _colocate_table_footnotes,
    _join_split_table_caption_labels,
    _merge_split_paragraphs_stable,
)
from pdfparser.pipeline.model import OcrModel, _ocr_page, load_ocr_model
from pdfparser.pipeline.render import _render_page_images
from pdfparser.pipeline.text import _looks_like_figure_caption, _visible_text

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


@dataclass
class _MdBlock:
    """A run of page markdown destined for ``_md_to_html_blocks``."""

    text: str


@dataclass
class _FigBlock:
    """A figure placeholder: its ``[0, 1000]``-normalized box (``None`` for the
    bbox-less base-variant placeholder) and the caption markdown, if any."""

    bbox_norm: tuple[int, int, int, int] | None
    caption: str | None


_Block = _MdBlock | _FigBlock


def _parse_page_blocks(md: str) -> list[_Block]:
    """Split a page's markdown into an ordered stream of prose and figure blocks.

    Pure: no image needed.  A figure's caption is taken from the placeholder's
    own trailing text, else the following caption-like block (which is then
    consumed)."""
    blocks: list[_Block] = []
    raw_blocks = re.split(r"\n[ \t]*\n", md.strip())
    k = 0
    while k < len(raw_blocks):
        block = raw_blocks[k].strip()
        lines = block.splitlines()
        fig = _parse_figure_placeholder(lines[0]) if lines else None
        if fig is None:
            blocks.append(_MdBlock(block))
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
        blocks.append(_FigBlock(fig if isinstance(fig, tuple) else None, caption))
        k += 1
    return blocks


def _resolve_figure_clusters(
    blocks: list[_Block], image: Image.Image
) -> tuple[dict[int, tuple[int, int, int, int]], dict[int, str | None], set[int]]:
    """Resolve figure boxes the model over-segmented into stacked crops.

    Pure (reads only ``image.size``, never crops).  Returns, keyed by figure
    block index: ``box_at`` the pixel crop box (the union for a merged cluster),
    ``caption_at`` the cluster's caption (its first non-empty member's), both
    held on the surviving lowest index, and ``drop`` the merged-away members.
    Figure blocks without a bbox are absent from all three, so the caller leaves
    them be."""
    figures = [
        (i, b.caption, _denormalize_bbox(b.bbox_norm, image))
        for i, b in enumerate(blocks)
        if isinstance(b, _FigBlock) and b.bbox_norm is not None
    ]
    boxes = [box for _, _, box in figures]
    gap = _FIGURE_MERGE_GAP_FRAC * image.size[1]

    box_at: dict[int, tuple[int, int, int, int]] = {}
    caption_at: dict[int, str | None] = {}
    drop: set[int] = set()
    for group in _cluster_figure_boxes(boxes, gap):
        members = sorted(group, key=lambda g: figures[g][0])
        survivor = figures[members[0]][0]
        box_at[survivor] = _union_box([boxes[g] for g in members])
        caption_at[survivor] = next(
            (figures[g][1] for g in members if figures[g][1]), None
        )
        drop.update(figures[g][0] for g in members[1:])
    return box_at, caption_at, drop


def _figcaption_only(caption_html: str) -> str:
    return (
        f"<figure><figcaption>{_inline_md_to_html(caption_html)}</figcaption></figure>"
    )


def _page_to_html_parts(md: str, image: Image.Image) -> list[str]:
    """Convert one page's markdown to block HTML, replacing each ``![image]``
    placeholder with a cropped ``<figure>`` and stitching consecutive prose into
    a single markdown render."""
    blocks = _parse_page_blocks(md)
    box_at, caption_at, drop = _resolve_figure_clusters(blocks, image)

    parts: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            parts.extend(_md_to_html_blocks("\n\n".join(pending)))
            pending.clear()

    for i, block in enumerate(blocks):
        if isinstance(block, _MdBlock):
            pending.append(block.text)
            continue
        flush()
        if i in drop:
            continue
        box = box_at.get(i)
        caption = caption_at.get(i, block.caption)
        caption_html = _latex_to_html(caption) if caption else None
        crop = _safe_crop(image, box) if box is not None else None
        if crop is not None:
            parts.append(_figure_html(crop, caption_html))
        elif caption_html is not None:
            parts.append(_figcaption_only(caption_html))
    flush()
    return parts


def _abstract_section(abstract: list[str]) -> str:
    if not abstract:
        return ""
    return "<section class='abstract'>\n" + "\n".join(abstract) + "\n</section>"


def _metadata_panel(metadata: list[str]) -> str:
    # Front matter is kept right after the abstract, collapsed by default in a
    # toggleable <details> panel, so the body opens with prose.
    if not metadata:
        return ""
    return (
        "<details class='metadata'>\n<summary>Metadata</summary>\n"
        + "\n".join(metadata)
        + "\n</details>"
    )


def _insert_footnotes_before_refs(body: list[str], footnotes: list[str]) -> list[str]:
    """Splice footnote paragraphs in just before the references section (or at the
    end, if there is none) so they read after the prose but before the bibliography."""
    if not footnotes:
        return body
    ref_idx = next(
        (i for i, p in enumerate(body) if _REF_SECTION_RE.match(p)), len(body)
    )
    return body[:ref_idx] + footnotes + body[ref_idx:]


def _document_shell(
    *, title_html: str, byline_html: str, abstract: str, metadata: str, body: str
) -> str:
    title_safe = _html.escape(_visible_text(title_html))
    byline_safe = _html.escape(byline_html)
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
{abstract}
{metadata}
<div class="body">
{body}
</div>
</body>
</html>"""


def _assemble_html(pages_md: list[str], images: list[Image.Image]) -> str:
    start = _leading_pages_to_skip_md(pages_md)
    pages_md = pages_md[start:]
    images = images[start:]

    per_page_parts = [
        _page_to_html_parts(md, img) for md, img in zip(pages_md, images, strict=True)
    ]
    # Front matter the OCR scattered into the article's first page — a glossary
    # section (Abbreviations / Nomenclature) past the leading-run scan, and
    # self-contained footer lines (correspondence, submission/DOI, supporting-info
    # notes) — is pulled here, scoped to that first page.  The stray-line sweep runs
    # before the paragraph-merge so a footer line ending in ")" is removed before
    # the merge can glue following body prose onto it.
    named_metadata: list[str] = []
    stray_metadata: list[str] = []
    if per_page_parts:
        named_metadata, per_page_parts[0] = _extract_named_metadata_sections(
            per_page_parts[0]
        )
        stray_metadata, per_page_parts[0] = _extract_stray_metadata(per_page_parts[0])

    parts = [part for page in per_page_parts for part in page]
    parts = _join_split_table_caption_labels(parts)
    parts = _colocate_table_captions(parts)
    # Before classify so a table's footnotes stay with it rather than being swept
    # into the article footnote section, and before merge so the table is a
    # single float the cross-table paragraph merge can step over.
    parts = _colocate_table_footnotes(parts)

    meta = _classify_parts(parts)

    abstract = _merge_split_paragraphs_stable(meta.abstract)
    body = _merge_split_paragraphs_stable(_strip_running_furniture(meta.body))
    body = _insert_footnotes_before_refs(body, meta.footnotes)
    leading_metadata, body = _extract_front_matter(body)
    metadata = leading_metadata + named_metadata + stray_metadata

    return _document_shell(
        title_html=meta.title_html or "Untitled",
        byline_html=meta.byline_html,
        abstract=_abstract_section(abstract),
        metadata=_metadata_panel(metadata),
        body="\n".join(body),
    )


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
