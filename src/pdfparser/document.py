"""PDF → HTML document conversion using GROBID + gmft.

Combines GROBID's full-text extraction with gmft table detection to produce
a self-contained HTML document with tables injected at their original positions.
"""

from __future__ import annotations

import html as _html
import re
from pathlib import Path

from lxml import html as lxhtml

from pdfparser._html import strip_html_tags
from pdfparser._tei import _TABLE_PLACEHOLDER_RE, tei_to_parts
from pdfparser.grobid import DEFAULT_GROBID_URL, pdf_to_tei
from pdfparser.tables import (
    DEFAULT_DEVICE,
    DEFAULT_THRESHOLD,
    ExtractedTable,
    extract_tables,
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
header p  { margin: 0; color: #555; font-size: .9rem; }
.chunk-body section { margin: 1.5em 0; }
.chunk-body section.abstract {
    background: #f7f7f7; padding: 1em 1.2em; border-radius: 4px; }
.chunk-body h2 {
    font-size: 1.15rem; margin: 1.5em 0 .4em;
    border-bottom: 1px solid #e0e0e0; }
.chunk-body h3 { font-size: 1rem; margin: 1.2em 0 .3em; }
.chunk-body p  { margin: .6em 0; }
.chunk-body figure { margin: 1.5em 0; text-align: center; }
.chunk-body figure img {
    max-width: 100%; height: auto; display: block; margin: 0 auto; }
.chunk-body figure span.label {
    display: block; margin-bottom: .25em; font-size: .875em;
    font-weight: bold; color: #666; text-align: left; }
.chunk-body figure figcaption {
    margin-top: .5em; font-size: .875em; color: #666; text-align: left; }
.chunk-body table {
    border-collapse: collapse; width: 100%; overflow-x: auto;
    display: block; font-size: .9rem; }
.chunk-body td, .chunk-body th { padding: .4em .7em; border: 1px solid #ccc; }
hr { border: none; border-top: 1px solid #ddd; margin: 2rem 0; }
"""


def pdf_to_html(
    pdf_path: Path | str,
    *,
    grobid_url: str = DEFAULT_GROBID_URL,
    threshold: float = DEFAULT_THRESHOLD,
    device: str = DEFAULT_DEVICE,
    ffill_spanning: bool = False,
) -> str:
    """Convert a PDF to a self-contained HTML document.

    GROBID provides the document structure and text; gmft extracts table content.
    Table figures in the GROBID output are replaced with the corresponding
    gmft-rendered HTML tables, matched by document order.

    Args:
        pdf_path: Path to the input PDF.
        grobid_url: Base URL of the GROBID service (must be running).
        threshold: gmft detection confidence threshold (default 0.99).
        device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
        ffill_spanning: Forward-fill NaN runs within table columns, useful for
            PDFs with merged/spanning cells.

    Returns:
        Self-contained HTML document string.

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
        RuntimeError: If GROBID is unreachable or returns an error.
    """
    from xmlparser import transform_article

    pdf_path = Path(pdf_path)

    tei_xml = pdf_to_tei(pdf_path, grobid_url)
    tables = extract_tables(
        pdf_path,
        threshold=threshold,
        device=device,
        ffill_spanning=ffill_spanning,
    )

    abstract_xml, body_xml, meta = tei_to_parts(tei_xml)

    body_envelope = (
        "<article><front><article-meta></article-meta></front>"
        f"{body_xml or ''}</article>"
    )

    abstract_html = transform_article(abstract_xml) if abstract_xml else None
    body_html: str = transform_article(body_envelope)
    body_html = _inject_tables(body_html, tables)
    body_html = _place_unplaced_tables(body_html, tables)

    return _build_html(meta, abstract_html, body_html)


def _inject_tables(html: str, tables: list[ExtractedTable]) -> str:
    """Replace ``[[TABLE_N]]`` placeholders with gmft HTML tables.

    Placeholders whose index exceeds the number of detected tables are removed.
    """

    def _replace(m: re.Match[str]) -> str:
        n = int(m.group(1))
        return tables[n].html if n < len(tables) else ""

    return re.sub(_TABLE_PLACEHOLDER_RE, _replace, html)


_TABLE_PARA_RE = re.compile(
    r"<p[^>]*>\s*TABLE\s+(?:[IVXLCDM]+|\d+)(.*?)</p>",
    re.IGNORECASE | re.DOTALL,
)


def _place_unplaced_tables(body_html: str, tables: list[ExtractedTable]) -> str:
    """Replace raw TABLE-N paragraphs with properly placed gmft tables.

    When GROBID emits table content as plain text rather than
    ``<figure type="table">`` elements, ``_inject_tables`` has no placeholders
    to fill.  This function finds those paragraphs by their label prefix,
    pairs each with the next unplaced gmft table in document order, and
    replaces the paragraph with an optional caption ``<p>``, the gmft
    ``<table>``, and an optional legend ``<p>``.

    Tables beyond the last matching paragraph are appended at the end.
    """
    n_placed = body_html.count("<table>")
    unplaced = tables[n_placed:]
    if not unplaced:
        return body_html

    idx = 0

    def _replace(m: re.Match[str]) -> str:
        nonlocal idx
        if idx >= len(unplaced):
            return m.group(0)
        table = unplaced[idx]
        idx += 1

        plain = strip_html_tags(m.group(1)).strip()
        caption, legend = _split_caption_legend(plain, table)

        parts: list[str] = []
        if caption:
            parts.append(f"<p>{_html.escape(caption)}</p>")
        parts.append(table.html)
        if legend:
            parts.append(f"<p>{_html.escape(legend)}</p>")
        return "\n".join(parts)

    result = _TABLE_PARA_RE.sub(_replace, body_html)
    result += "".join(t.html for t in unplaced[idx:])
    return result


def _split_caption_legend(text: str, table: ExtractedTable) -> tuple[str, str]:
    """Split a table paragraph's body text into ``(caption, legend)``.

    Builds a set of bigrams (consecutive whitespace-delimited token pairs,
    ≥ 8 chars) from every gmft cell and searches for them verbatim in
    ``text``.  The text before the first match is the caption; the text after
    the last match is the legend.  If no bigrams match, the whole text is
    treated as the caption.
    """
    table_root = lxhtml.fragment_fromstring(table.html)
    anchors: set[str] = set()
    for cell in table_root.iter("th", "td"):
        tokens = [t for t in re.split(r"\s+", cell.text_content().strip()) if t]
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            if len(bigram) >= 8:
                anchors.add(bigram)

    if not anchors:
        return text, ""

    pattern = re.compile(
        "|".join(re.escape(b) for b in sorted(anchors, key=len, reverse=True)),
        re.IGNORECASE,
    )
    first_start = last_end = None
    for m in pattern.finditer(text):
        if first_start is None:
            first_start = m.start()
        last_end = m.end()

    if first_start is None:
        return text, ""
    return text[:first_start].strip(), text[last_end:].strip()


def _build_html(
    meta: dict[str, str],
    abstract_html: str | None,
    body_html: str,
) -> str:
    title = meta["title"]
    authors = meta["authors"]
    year = meta["year"]
    byline = "; ".join(filter(None, [authors, year]))

    abstract_section = (
        f"<div class='chunk-body'>{abstract_html}</div><hr>" if abstract_html else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{_WRAPPER_CSS}</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <p>{byline}</p>
</header>
{abstract_section}
<div class='chunk-body'>{body_html}</div>
</body>
</html>"""
