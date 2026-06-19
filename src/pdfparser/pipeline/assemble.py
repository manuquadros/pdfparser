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
from collections.abc import Callable  # noqa: TC003 — beartype reads annotations
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
    _base64_src,
    _cluster_figure_boxes,
    _denormalize_bbox,
    _figure_html,
    _file_image_writer,
    _is_bare_figure_label,
    _is_panel_label,
    _opens_with_panel_label,
    _parse_figure_placeholder,
    _safe_crop,
    _union_box,
)
from pdfparser.pipeline.latex import _inline_md_to_html, _latex_to_html
from pdfparser.pipeline.markdown import _md_to_html_blocks
from pdfparser.pipeline.merge import (
    _TABLE_OPEN_RE,
    _colocate_table_captions,
    _colocate_table_footnotes,
    _join_split_table_caption_labels,
    _merge_split_paragraphs_stable,
)
from pdfparser.pipeline.model import OcrModel, _ocr_page, _ocr_pages, load_ocr_model
from pdfparser.pipeline.render import _render_page_images
from pdfparser.pipeline.tables import _close_unclosed_tables, _recover_dropped_tables
from pdfparser.pipeline.text import (
    _looks_like_figure_caption,
    _opens_with_caption_label,
    _opens_with_table_label,
    _visible_text,
)

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


def _starts_figure(block: str) -> bool:
    """True when ``block``'s first line is a figure placeholder."""
    block = block.strip()
    return bool(block) and _parse_figure_placeholder(block.splitlines()[0]) is not None


def _is_caption_continuation(block: str) -> bool:
    """A block that continues a bare figure label's caption: running prose, not the
    start of a new structural element (another float, a heading, a table)."""
    block = block.strip()
    if not block:
        return False
    first = block.splitlines()[0].lstrip()
    if _starts_figure(block) or first.startswith(("#", "|")):
        return False
    return not _opens_with_caption_label(block)


def _parse_page_blocks(md: str) -> list[_Block]:
    """Split a page's markdown into an ordered stream of prose and figure blocks.

    Pure: no image needed.  A figure's caption is taken from the placeholder's
    own trailing text, else the following caption-like block (which is then
    consumed); a bare ``FIG. N`` label has its following descriptive block
    rejoined onto it.  Lone single-letter panel labels the model split out of a
    multi-panel figure are dropped when they abut a placeholder."""
    blocks: list[_Block] = []
    raw_blocks = re.split(r"\n[ \t]*\n", md.strip())
    k = 0
    while k < len(raw_blocks):
        block = raw_blocks[k].strip()
        lines = block.splitlines()
        fig = _parse_figure_placeholder(lines[0]) if lines else None
        if fig is None:
            next_is_fig = k + 1 < len(raw_blocks) and _starts_figure(raw_blocks[k + 1])
            prev_is_fig = bool(blocks) and isinstance(blocks[-1], _FigBlock)
            if _is_panel_label(block) and (next_is_fig or prev_is_fig):
                k += 1
                continue
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
        if (
            caption is not None
            and _is_bare_figure_label(caption)
            and k + 1 < len(raw_blocks)
            and _is_caption_continuation(raw_blocks[k + 1])
        ):
            k += 1
            caption = f"{caption} {raw_blocks[k].strip()}"
        # Multi-panel sub-descriptions the model split into their own paragraph(s)
        # ("(A) … (B) … (C) …") are caption text the OCR detached from the caption
        # header, not body prose — fold each onto the caption (starting one if the
        # figure had no header block of its own).
        while k + 1 < len(raw_blocks) and _opens_with_panel_label(raw_blocks[k + 1]):
            k += 1
            panel = raw_blocks[k].strip()
            caption = f"{caption} {panel}" if caption else panel
        blocks.append(_FigBlock(fig if isinstance(fig, tuple) else None, caption))
        k += 1
    return blocks


def _is_table_md(block: _Block | None) -> bool:
    return isinstance(block, _MdBlock) and _TABLE_OPEN_RE.match(block.text) is not None


def _is_table_caption_md(block: _Block | None) -> bool:
    """A markdown block that is a free-standing "TABLE N …" caption (not the table
    itself) — the form the model emits when it doesn't bake the caption onto the
    figure placeholder line."""
    return (
        isinstance(block, _MdBlock)
        and not _is_table_md(block)
        and _opens_with_table_label(block.text)
    )


def _dedup_table_figures(blocks: list[_Block]) -> list[_Block]:
    """Drop a figure that is really a table boxed as an image.

    A model whose bbox head reads a sideways or graphical table as a picture emits
    *both* a figure placeholder and the ``<table>`` it also transcribed, so the table
    renders twice — once as a cropped image, once as the real table.  The figure is
    the duplicate when an actual ``<table>`` follows it, optionally with the table's
    "TABLE N …" caption in between (the model emits that caption either baked onto the
    placeholder line — held on the ``_FigBlock`` — or as its own following block).

    The image is dropped; the caption is preserved as a standalone block so
    :func:`_colocate_table_captions` folds it into the real table as its
    ``<caption>``.  When the caption already stands as its own block it is left in
    place; only when it rode on the placeholder is it re-emitted."""
    out: list[_Block] = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if isinstance(block, _FigBlock):
            j = i + 1
            standalone_caption = _is_table_caption_md(
                blocks[j] if j < len(blocks) else None
            )
            if standalone_caption:
                j += 1
            baked_caption = (
                block.caption
                if block.caption is not None and _opens_with_table_label(block.caption)
                else None
            )
            table_follows = _is_table_md(blocks[j] if j < len(blocks) else None)
            if table_follows and (standalone_caption or baked_caption is not None):
                if baked_caption is not None:
                    # The caption rode on the placeholder line, not as its own block;
                    # re-emit it so _colocate_table_captions can fold it in.
                    out.append(_MdBlock(baked_caption))
                i += 1  # drop the figure image; keep the caption/table blocks after it
                continue
        out.append(block)
        i += 1
    return out


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


def _page_to_html_parts(
    md: str,
    image: Image.Image,
    ocr_region: Callable[[Image.Image], str] | None = None,
    encode_image: Callable[[Image.Image], str] = _base64_src,
) -> list[str]:
    """Convert one page's markdown to block HTML, replacing each ``![image]``
    placeholder with a cropped ``<figure>`` and stitching consecutive prose into
    a single markdown render.  ``ocr_region`` (re-OCR a sub-image), when supplied,
    lets the crop trim a caption the model baked into a text-bodied figure box.
    ``encode_image`` turns each crop into its ``<img src>`` (inline data URI by
    default; a sidecar PNG path when a file writer is supplied)."""
    blocks = _dedup_table_figures(_parse_page_blocks(md))
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
        crop = (
            _safe_crop(image, box, caption_text=caption, ocr_region=ocr_region)
            if box is not None
            else None
        )
        if crop is not None:
            parts.append(_figure_html(crop, caption_html, encode_image))
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


def _assemble_html(
    pages_md: list[str],
    images: list[Image.Image],
    ocr_region: Callable[[Image.Image], str] | None = None,
    encode_image: Callable[[Image.Image], str] = _base64_src,
) -> str:
    start = _leading_pages_to_skip_md(pages_md)
    pages_md = pages_md[start:]
    images = images[start:]

    # Close any table the OCR left open at a page's bottom before block-splitting,
    # so a table that overran the page does not swallow the next page's prose.
    per_page_parts = [
        _page_to_html_parts(_close_unclosed_tables(md), img, ocr_region, encode_image)
        for md, img in zip(pages_md, images, strict=True)
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
    base_url: str | None = None,
    model: str | None = None,
    image_dir: Path | None = None,
) -> str:
    """Convert a PDF to HTML with LightOnOCR-2-1B-bbox.

    Args:
        pdf_path: Path to the input PDF.
        ocr: Open connection bundle.  ``None`` calls ``load_ocr_model()``.
        base_url: vLLM endpoint root, used only when ``ocr`` is ``None``.
        model: Served model name, used only when ``ocr`` is ``None``.
        image_dir: When given, figure crops are written here as sidecar PNGs and
            referenced by a path relative to its parent (so the HTML, written into
            that parent, links them) instead of inlined as base64 — quicker to
            regenerate and live-editable in a browser.  ``None`` (default) inlines
            the images, keeping the HTML self-contained.
    """
    owns_ocr = ocr is None
    if ocr is None:
        ocr = load_ocr_model(base_url=base_url, model=model)
    try:
        images = _render_page_images(Path(pdf_path))
        pages_md = _ocr_pages(images, ocr)
        # Single-region re-OCR for figure caption-trimming; batched (concurrent)
        # re-OCR for the document's table crops.
        ocr_region = lambda region: _ocr_page(region, ocr)  # noqa: E731
        ocr_regions = lambda regions: _ocr_pages(regions, ocr)  # noqa: E731
        pages_md = _recover_dropped_tables(pdf_path, pages_md, ocr_regions)
        encode_image = (
            _file_image_writer(image_dir) if image_dir is not None else _base64_src
        )
        return _assemble_html(pages_md, images, ocr_region, encode_image)
    finally:
        if owns_ocr:
            ocr.close()
