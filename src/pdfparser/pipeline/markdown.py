"""Markdown → top-level block-HTML conversion.

LightOnOCR emits whole-page markdown mixed with raw HTML (``<table>``, ``<sup>``)
and inline LaTeX.  CommonMark + the table plugin, with raw-HTML passthrough,
covers the structure; LaTeX sub/superscripts are converted to HTML beforehand by
``latex._latex_to_html``.
"""

from __future__ import annotations

from markdown_it import MarkdownIt

from pdfparser.pipeline.latex import _latex_to_html

_MD = MarkdownIt("commonmark", {"html": True}).enable("table")


def _md_to_html_blocks(md_text: str) -> list[str]:
    """Convert a page's markdown to a list of top-level block HTML strings.

    One string per top-level block (heading, paragraph, list, raw-HTML table…),
    so downstream cleanup (merge, header/footer strip, footnote reordering) can
    operate block-by-block.  Thematic breaks (``---``) are dropped.
    """
    tokens = _MD.parse(_latex_to_html(md_text))
    blocks: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.level != 0:
            i += 1
            continue
        if token.nesting == 1:
            depth, j = 0, i
            while j < len(tokens):
                depth += tokens[j].nesting
                j += 1
                if depth == 0:
                    break
            group, i = tokens[i:j], j
        else:
            group, i = [token], i + 1
        html = _MD.renderer.render(group, _MD.options, {}).strip()
        if html and not html.startswith("<hr"):
            blocks.append(html)
    return blocks
