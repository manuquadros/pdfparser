"""PDF → HTML pipeline using Falcon-OCR

Falcon-OCR (tiiuae/Falcon-OCR) runs PP-DocLayoutV3 to detect page regions,
then generates text for each crop.  Table regions come back as HTML directly;
all other regions use light markdown (``**bold**``, ``*italic*``).

Typical use::

    model = load_model()
    html = falcon_pdf_to_html("paper.pdf", model=model)
    Path("out.html").write_text(html)
"""

from __future__ import annotations

import bisect
import html as _html
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch._dynamo

# flex_attention inside Falcon-OCR compiles with fullgraph=True and recompiles
# as sequence length grows during autoregressive decoding.
torch._dynamo.config.recompile_limit = 64

MODEL_ID = "tiiuae/Falcon-OCR"
_RENDER_SCALE = 200 / 72
_MAX_LONG_SIDE = 1024
_DEFAULT_OCR_BATCH_SIZE = 2

_HTML_CATS = frozenset({"table", "vision_footnote"})
_SKIP_CATS = frozenset({"header", "page-header", "footer", "page-footer"})

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


def load_model(device: str | None = None) -> Any:
    """Load Falcon-OCR, applying the project's paddle flags first.

    Args:
        device: Torch device string.  Defaults to ``"cuda"`` if available.

    Returns:
        Loaded ``AutoModelForCausalLM`` instance.
    """
    import paddle

    paddle.set_flags(
        {
            "FLAGS_use_mkldnn": False,
            "FLAGS_enable_pir_api": False,
            "FLAGS_allocator_strategy": "auto_growth",
        }
    )
    paddle.set_device("cpu")

    from transformers import AutoModelForCausalLM

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)

    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
    )


def _render_pages(pdf_path: Path) -> list[Any]:
    import pypdfium2 as pdfium
    from PIL import Image

    with pdfium.PdfDocument(str(pdf_path)) as pdf:
        pages = []
        for page in pdf:
            img = page.render(scale=_RENDER_SCALE).to_pil().convert("RGB")
            long_side = max(img.size)
            if long_side > _MAX_LONG_SIDE:
                ratio = _MAX_LONG_SIDE / long_side
                img = img.resize(
                    (int(img.size[0] * ratio), int(img.size[1] * ratio)),
                    Image.LANCZOS,
                )
            pages.append(img)
    return pages


def _sort_regions(regions: list[dict], page_width: float) -> list[dict]:
    """Sort regions into reading order for two-column PDF layouts.

    Full-width regions (spanning > 55 % of the page) act as section
    boundaries.  Within each section, left-column content (col 1) is read
    before right-column content (col 2); within a column, regions are ordered
    top-to-bottom by y0.

    This correctly handles paragraphs that start near the bottom of the left
    column and continue near the top of the right column: the left fragment
    always precedes the right fragment within the same section, regardless of
    their absolute y positions.
    """
    half = page_width / 2

    def classify(r: dict) -> tuple[int, float]:
        bbox = r.get("bbox")
        if not bbox:
            return 1, 0.0  # treat bbox-less regions as left-column at top
        x0, y0, x1, _ = bbox
        cx = (x0 + x1) / 2
        if (x1 - x0) > 0.55 * page_width:
            return 0, float(y0)
        return (2 if cx > half else 1), float(y0)

    # Classify once per region; reuse results for both boundaries and sort key.
    classified = [(classify(r), r) for r in regions]

    # y-positions of full-width elements divide the page into sections.
    # bisect_right(boundaries, y0) gives the section index: 0 = before the
    # first full-width element, 1 = after it but before the second, etc.
    # Full-width elements themselves get the section AFTER their own y0,
    # so they sort first within that section (col 0 < col 1 < col 2).
    boundaries: list[float] = sorted(y0 for (col, y0), _ in classified if col == 0)

    def key(col_y0: tuple[int, float]) -> tuple[int, int, float]:
        col, y0 = col_y0
        return (bisect.bisect_right(boundaries, y0), col, y0)

    return [r for _, r in sorted(classified, key=lambda item: key(item[0]))]


# re.DOTALL intentionally omitted: italic spans in academic text don't cross
# line boundaries, and DOTALL would cause two stray footnote asterisks anywhere
# in a multi-line region to wrap the entire intervening content in <em>.
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_ITALIC_RE = re.compile(r"\*(.+?)\*")
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+")
_REF_LIST_RE = re.compile(r"^\[1\]")
_REF_SPLIT_RE = re.compile(r"\n(?=\[\d+\])")
_SENTENCE_END_RE = re.compile(r"[.!?;:]\s*$")
_FLOAT_RE = re.compile(r"^<(?:table|figure)[\s>]", re.IGNORECASE)
_ENUM_RE = re.compile(
    r"^\s*(?:\d+[.)]\s|\[\d|[•\-]\s|\([a-z0-9ivx]+\)\s)", re.IGNORECASE
)
# A fragment ending with a function word is *definitively* grammatically
# incomplete: its continuation must be a predicate, object, or complement,
# which in normal prose starts lowercase.  If the next region starts with an
# uppercase letter in this context the real continuation was likely dropped by
# OCR, so we refuse the merge rather than joining unrelated sentences.
_FUNCTION_WORD_END_RE = re.compile(
    r"\b(?:a|an|the|is|are|was|were|be|been|being|have|has|had|"
    r"will|would|can|could|should|may|might|must|do|does|did|"
    r"of|in|on|at|by|for|with|to|from|and|or|but|nor|"
    r"that|which|who|whom|this|these|those)\s*$",
    re.IGNORECASE,
)
_MAX_FLOATS_TO_SKIP = 2
_DOCUMENT_TYPE_LABELS = frozenset(
    {
        "article",
        "research article",
        "original article",
        "letter",
        "review",
        "communication",
        "report",
        "brief communication",
        "short communication",
    }
)


def _plain_p_text(s: str) -> str | None:
    """Return the inner content of a plain ``<p>…</p>`` block, or ``None``.

    Returns ``None`` for footnote/class paragraphs, multi-paragraph strings
    (reference lists), headings, tables, figures, and any other element.
    """
    if s.startswith("<p>") and s.endswith("</p>") and s.count("</p>") == 1:
        return s[3:-4]
    return None


def _merge_split_paragraphs(parts: list[str]) -> list[str]:
    """Stitch paragraph fragments broken by two-column PDF layout.

    When a plain ``<p>`` ends without terminal punctuation the next plain
    ``<p>`` is treated as a continuation.  Intervening tables and figures
    (up to ``_MAX_FLOATS_TO_SKIP``) are collected and re-emitted *after*
    the merged paragraph so the float stays near its reference text.

    Headings, footnote paragraphs, and apparent enumeration items act as
    merge barriers and are never absorbed into an adjacent paragraph.
    """
    out: list[str] = []
    i = 0
    while i < len(parts):
        part = parts[i]
        inner = _plain_p_text(part)
        if inner is not None and not _SENTENCE_END_RE.search(inner.rstrip()):
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
                    and not (
                        _FUNCTION_WORD_END_RE.search(inner.rstrip())
                        and cont[:1].isupper()
                    )
                ):
                    out.append(f"<p>{inner.rstrip()} {cont.lstrip()}</p>")
                    out.extend(floats)
                    i = j + 1
                    continue
        out.append(part)
        i += 1
    return out


_RUNNING_HEADER_MAX_LEN = 200


def _remove_repeated_short_paragraphs(parts: list[str]) -> list[str]:
    """Legitimate prose never repeats verbatim; repeated identical short paragraphs
    are always structural artefacts (running headers, footers, page labels) that
    the layout model mis-classified as text.
    """
    counts: Counter[str] = Counter()
    for p in parts:
        inner = _plain_p_text(p)
        if inner is not None and len(inner) <= _RUNNING_HEADER_MAX_LEN:
            counts[p] += 1
    repeated = {p for p, n in counts.items() if n > 1}
    return [p for p in parts if p not in repeated]


def _inline_md_to_html(text: str) -> str:
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = _ITALIC_RE.sub(r"<em>\1</em>", text)
    return text.strip()


def _heading_html(text: str, default_level: int = 2) -> str:
    """Convert a paragraph_title to an ``<h2>`` or ``<h3>`` element.

    Falcon sometimes emits ``##`` markdown prefixes inside paragraph_title
    regions to signal sub-headings.  Strip the prefix and use the level.
    """
    m = _MD_HEADING_RE.match(text)
    if m:
        level = min(len(m.group(1)) + 1, 4)  # ## → h3, ### → h4
        text = text[m.end() :]
    else:
        level = default_level
    return f"<h{level}>{_inline_md_to_html(text)}</h{level}>"


def _region_to_html(r: dict) -> str:
    cat = r.get("category", "text")
    text = r.get("text", "").strip()
    if not text:
        return ""

    if cat in _HTML_CATS:
        # Sanity-check table regions: if the model returned plain text instead
        # of HTML (failure path or version change), fall back to <pre> rather
        # than injecting raw text into the document structure.
        if cat == "table" and "<table" not in text.lower():
            return f"<pre>{_html.escape(text)}</pre>"
        # vision_footnote text is inline HTML (e.g. "<sup>a</sup> note…") but
        # arrives without a wrapper element, so we supply one.
        if cat == "vision_footnote":
            return f'<p class="footnote">{text}</p>'
        return text

    if cat == "paragraph_title":
        return _heading_html(text)

    if cat == "figure_title":
        return f"<figure><figcaption>{_inline_md_to_html(text)}</figcaption></figure>"

    if cat == "footnote":
        return f'<p class="footnote">{_inline_md_to_html(text)}</p>'

    if _REF_LIST_RE.match(text):
        refs = _REF_SPLIT_RE.split(text)
        return "\n".join(
            f"<p>{_inline_md_to_html(ref.strip())}</p>" for ref in refs if ref.strip()
        )

    # Table footnotes rendered by Falcon as inline HTML (e.g. "<sup>a</sup> …")
    # arrive via the text category when the model emits them outside a table.
    if text.startswith("<sup>"):
        return f'<p class="footnote">{text}</p>'

    return f"<p>{_inline_md_to_html(text)}</p>"


def _inject_caption(table_html: str, caption_text: str) -> str:
    """Insert a <caption> element immediately after the opening <table> tag."""
    caption_html = f"<caption>{_inline_md_to_html(caption_text)}</caption>"
    return re.sub(
        r"(<table(?:\s[^>]*)?>)",
        lambda m: m.group(1) + caption_html,
        table_html,
        count=1,
    )


def _extract_meta(page0_regions: list[dict], page_width: float) -> dict[str, str]:
    """Heuristically pull title and author from the first page's regions."""
    title = ""
    author = ""
    # Sort into reading order so the first short text after the title is the
    # author line, not an interceding journal label or correspondence fragment.
    for r in _sort_regions(page0_regions, page_width):
        cat = r.get("category", "")
        text = r.get("text", "").strip()
        if not text:
            continue
        if cat == "doc_title" and not title:
            title = re.sub(r"<[^>]+>", "", text).strip()
        elif (
            cat == "text"
            and not author
            and title
            and text.count("\n") <= 2
            and len(text) < 200
        ):
            author = re.sub(r"<[^>]+>", "", text).strip()
    return {"title": title or "Untitled", "authors": author, "year": ""}


def falcon_pdf_to_html(
    pdf_path: Path | str,
    *,
    model: Any = None,
    device: str | None = None,
    ocr_batch_size: int = _DEFAULT_OCR_BATCH_SIZE,
) -> str:
    """Convert a PDF to a self-contained HTML document using Falcon-OCR.

    No GROBID or gmft dependency.  All text and table content comes from the
    Falcon-OCR model.

    Args:
        pdf_path: Path to the input PDF.
        model: Pre-loaded Falcon-OCR model.  If ``None``, ``load_model()`` is
            called automatically (slow first call).
        device: Torch device string.  Only used when ``model`` is ``None``.
        ocr_batch_size: Number of region crops to batch per model call.

    Returns:
        Self-contained HTML document string.
    """
    pdf_path = Path(pdf_path)
    if model is None:
        model = load_model(device)

    images = _render_pages(pdf_path)

    all_regions: list[list[dict]] = []
    for img in images:
        with torch.inference_mode():
            results = model.generate_with_layout([img], ocr_batch_size=ocr_batch_size)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Guard against blank/image-only pages that return no regions.
        all_regions.append(results[0] if results else [])

    page0_width = float(images[0].size[0]) if images else 800.0
    meta = (
        _extract_meta(all_regions[0], page0_width)
        if all_regions
        else {"title": "Untitled", "authors": "", "year": ""}
    )

    abstract_parts: list[str] = []
    body_parts: list[str] = []
    # figure_title regions are buffered so the next table can absorb them as
    # <caption> rather than emitting a detached <figure><figcaption>.
    pending_fig_title: str | None = None

    _punct_re = re.compile(r"[^\w\s]")
    title_norm = re.sub(r"\s+", " ", _punct_re.sub("", meta["title"])).lower().strip()

    def _flush_fig_title() -> None:
        nonlocal pending_fig_title
        if pending_fig_title:
            body_parts.append(
                f"<figure><figcaption>{_inline_md_to_html(pending_fig_title)}</figcaption></figure>"
            )
            pending_fig_title = None

    for regions, img in zip(all_regions, images, strict=True):
        # Use the rendered image width as the authoritative page width so that
        # single-column pages (where max(bbox.x1) ≪ page width) don't cause
        # the column-split heuristic to misclassify left-column content.
        pw = float(img.size[0])
        for r in _sort_regions(regions, pw):
            cat = r.get("category", "text")
            text = r.get("text", "").strip()
            if not text or cat in _SKIP_CATS:
                continue
            if cat == "doc_title":
                continue  # already captured in meta; skip body duplicate

            if cat == "abstract":
                abstract_parts.append(f"<p>{_inline_md_to_html(text)}</p>")
                continue

            if cat == "paragraph_title" and text.lower() in _DOCUMENT_TYPE_LABELS:
                continue

            if cat == "text" and title_norm:
                text_norm = re.sub(r"\s+", " ", _punct_re.sub("", text)).lower().strip()
                if text_norm == title_norm:
                    continue

            if cat == "figure_title":
                _flush_fig_title()
                pending_fig_title = text
                continue

            html_chunk = _region_to_html(r)

            if cat == "table" and pending_fig_title:
                html_chunk = _inject_caption(html_chunk, pending_fig_title)
                pending_fig_title = None
            else:
                _flush_fig_title()

            body_parts.append(html_chunk)

    _flush_fig_title()

    raw_title = meta["title"]
    title_safe = _html.escape(re.sub(r"\*+", "", raw_title))
    title_display = _inline_md_to_html(raw_title)
    byline_safe = _html.escape("; ".join(filter(None, [meta["authors"], meta["year"]])))

    abstract_html = (
        "<section class='abstract'>\n"
        + "\n".join(_merge_split_paragraphs(_merge_split_paragraphs(abstract_parts)))
        + "\n</section>"
        if abstract_parts
        else ""
    )
    body_html = "\n".join(
        _merge_split_paragraphs(
            _merge_split_paragraphs(_remove_repeated_short_paragraphs(body_parts))
        )
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title_safe}</title>
<style>{_WRAPPER_CSS}</style>
</head>
<body>
<header>
  <h1>{title_display}</h1>
  <p>{byline_safe}</p>
</header>
{abstract_html}
<div class="body">
{body_html}
</div>
</body>
</html>"""
