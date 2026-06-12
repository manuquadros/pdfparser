"""Markdown → top-level block-HTML conversion.

LightOnOCR emits whole-page markdown mixed with raw HTML (``<table>``, ``<sup>``)
and inline LaTeX.  CommonMark + the table plugin, with raw-HTML passthrough,
covers the structure; LaTeX sub/superscripts are converted to HTML beforehand by
``latex._latex_to_html``.
"""

from __future__ import annotations

import re

from markdown_it import MarkdownIt

from pdfparser.pipeline.latex import _latex_to_html

_MD = MarkdownIt("commonmark", {"html": True}).enable("table")

# LightOnOCR emits tables as raw HTML, which CommonMark passes through verbatim —
# so ``*emphasis*`` the model leaves inside a cell (organism names: ``*Klebsiella
# aerogenes*``) never reaches the inline parser.  Render it cell-by-cell instead.
_TABLE_CELL_RE = re.compile(
    r"(<t[dh]\b[^>]*>)(.*?)(</t[dh]>)", re.DOTALL | re.IGNORECASE
)


def _render_inline(text: str) -> str:
    """Render a fragment of OCR markup to HTML through CommonMark's *inline*
    parser, the tag-aware counterpart to a regex sub.

    It passes valid inline HTML through (``<sup>a</sup>`` markers, ``<sub>`` from
    ``_latex_to_html``), escapes a stray ``<``/``&`` (so a footnote ``n<5`` stays
    safe rather than starting a bogus tag), and forms emphasis only on properly
    flanked ``*`` (so a cell ``5 * 10 * 3`` is not spuriously italicised).
    """
    rendered: str = _MD.renderInline(_latex_to_html(text))
    return rendered.strip()


def _render_cell_markdown(table_html: str) -> str:
    return _TABLE_CELL_RE.sub(
        lambda m: m.group(1) + _render_inline(m.group(2)) + m.group(3),
        table_html,
    )


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
            if "<td" in html or "<th" in html:
                html = _render_cell_markdown(html)
            blocks.append(html)
    return blocks
