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

import base64
import bisect
import html as _html
import io
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import torch
import torch._dynamo
from PIL import Image
from transformers import AutoModelForCausalLM

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

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
    )

    # Falcon-OCR's _load_layout_model() short-circuits on a `_layout_model`
    # attribute it never actually assigns, so PP-DocLayoutV3 is reloaded from
    # disk on every generate_with_layout() call. Prime it once and set the
    # sentinel the guard checks for, so later calls reuse the loaded detector.
    model._load_layout_model()
    model._layout_model = model._layout_det_model
    return model


def _render_page(page: pdfium.PdfPage) -> Image.Image:
    img: Image.Image = page.render(scale=_RENDER_SCALE).to_pil().convert("RGB")
    long_side = max(img.size)
    if long_side > _MAX_LONG_SIDE:
        ratio = _MAX_LONG_SIDE / long_side
        img = img.resize(
            (int(img.size[0] * ratio), int(img.size[1] * ratio)),
            Image.LANCZOS,
        )
    return img


def _render_pages(pdf_path: Path) -> list[Image.Image]:
    with pdfium.PdfDocument(str(pdf_path)) as pdf:
        return [_render_page(page) for page in pdf]


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


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
# re.DOTALL intentionally omitted: italic spans in academic text don't cross
# line boundaries, and DOTALL would cause two stray footnote asterisks anywhere
# in a multi-line region to wrap the entire intervening content in <em>.
_ITALIC_RE = re.compile(r"\*(.+?)\*")
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+")
_REF_LIST_RE = re.compile(r"^\[1\]")
_REF_SPLIT_RE = re.compile(r"\n(?=\[\d+\])")
_SENTENCE_END_RE = re.compile(r"[.!?;:]\s*$")
_FLOAT_RE = re.compile(r"^<(?:table|figure)[\s>]", re.IGNORECASE)
_HYPHEN_BREAK_RE = re.compile(r"-\s*$")
_ENUM_RE = re.compile(
    r"^\s*(?:\d+[.)]\s|\[\d|[•\-]\s|\([a-z0-9ivx]+\)\s)", re.IGNORECASE
)
# Paragraphs that open with a bold label ("Keywords:", "Abbreviations:", "Note:")
# are structured metadata, never mid-sentence continuations.
_BOLD_LABEL_RE = re.compile(r"^<strong>[^<]+:</strong>")
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
# An all-caps acronym ("TRII", "DNA", "NAD") opening the continuation is part
# of the same sentence, not a new-sentence capital, so it must not trip the
# function-word guard — otherwise a clause split across a column/page break
# ("…TRI and" / "TRII compete…") is wrongly left as two paragraphs.
_ACRONYM_HEAD_RE = re.compile(r"^[A-Z]{2,}[0-9]*\b")
_MAX_FLOATS_TO_SKIP = 2
_DOCUMENT_TYPE_LABELS = frozenset(
    {
        "abstract",
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
# re.IGNORECASE: OCR output casing is unreliable ("REFERENCES", "References").
# <h\d[^>]*> tolerates class/id attributes the model may inject.
_REF_SECTION_RE = re.compile(
    r"^(?:<h\d[^>]*>\s*References\s*</h\d>|<p>\[1\])", re.IGNORECASE
)

# A page begins the article if it carries the paper title or abstract, or an
# "Abstract"/"Introduction" *heading* — the word must head a paragraph_title,
# not merely appear in body or advertising copy, so a cover ad mentioning
# "introduction" isn't mistaken for the article start.
_ARTICLE_PAGE_CATS = frozenset({"abstract", "doc_title"})
_ARTICLE_HEADING_RE = re.compile(
    r"^\s*(?:\d+[.)]?\s+)?(?:abstract|introduction)\b", re.IGNORECASE
)
# Categories the layout model emits only for genuine article content; a leading
# page carrying any of these is real content, never a droppable cover/masthead.
_CONTENT_CATS = frozenset({"abstract", "doc_title", "figure_title"})


def _is_article_page(regions: list[dict]) -> bool:
    for r in regions:
        cat = r.get("category")
        if cat in _ARTICLE_PAGE_CATS:
            return True
        if cat == "paragraph_title" and _ARTICLE_HEADING_RE.match(r.get("text") or ""):
            return True
    return False


def _has_structural_content(regions: list[dict]) -> bool:
    return any(r.get("category") in _CONTENT_CATS for r in regions)


def _leading_pages_to_skip(all_regions: list[list[dict]]) -> int:
    """Number of leading non-article pages (cover ads, mastheads) to drop.

    A leading page is dropped only when *no* page before the article start
    carries structural content of its own, so a real first page the layout
    model under-tagged (its title/abstract missed) is never discarded just
    because a later page has an "Introduction" heading.
    """
    first_article = next(
        (i for i, regions in enumerate(all_regions) if _is_article_page(regions)),
        0,
    )
    if any(_has_structural_content(regions) for regions in all_regions[:first_article]):
        return 0
    return first_article


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
        if inner is not None:
            stripped = inner.rstrip()
            if not _SENTENCE_END_RE.search(stripped) and not _BOLD_LABEL_RE.match(
                inner
            ):
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
                        and not _BOLD_LABEL_RE.match(cont)
                        and not (
                            _FUNCTION_WORD_END_RE.search(stripped)
                            and cont[:1].isupper()
                            and not _ACRONYM_HEAD_RE.match(cont)
                        )
                    ):
                        dehyphenated, n = _HYPHEN_BREAK_RE.subn("", stripped)
                        joined = dehyphenated + ("" if n else " ") + cont.lstrip()
                        out.append(f"<p>{joined}</p>")
                        out.extend(floats)
                        i = j + 1
                        continue
        out.append(part)
        i += 1
    return out


_RUNNING_HEADER_MAX_LEN = 200
_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")
_STRIP_TAGS_RE = re.compile(r"<[^>]+>")


def _remove_repeated_short_paragraphs(parts: list[str]) -> list[str]:
    """Drop repeated short paragraphs that are structural artefacts (running
    headers, footers, page labels) the layout model mis-classified as text.

    Only sentence-*fragment* repeats are removed: running headers carry no
    terminal punctuation, whereas a legitimately repeated short sentence does,
    so requiring the absence of sentence-ending punctuation preserves real
    prose that happens to recur verbatim.
    """
    counts: Counter[str] = Counter(
        p
        for p in parts
        if (inner := _plain_p_text(p)) is not None
        and len(inner) <= _RUNNING_HEADER_MAX_LEN
        and not _SENTENCE_END_RE.search(inner.rstrip())
    )
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


_MIN_FIGURE_HEIGHT = 50  # pixels — gaps smaller than this are not figures


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


def _infer_figure_crop(
    fig_title_region: dict,
    all_regions: list[dict],
    img: Image.Image,
) -> Image.Image | None:
    """Crop the figure area adjacent to a caption region from the page image.

    The figure shares the caption's column.  On a two-column page the crop is
    confined to that column (left or right half) and the vertical gap is
    measured only against same-column regions, so a figure never absorbs the
    neighbouring column's text.  Single-column pages and figures whose caption
    spans the page use the full page width.

    Finds the nearest same-column content boundary above and below the caption,
    then crops whichever gap is larger.  Returns None if neither gap reaches
    _MIN_FIGURE_HEIGHT.
    """
    bbox = fig_title_region.get("bbox")
    if not bbox:
        return None

    fy0, fy1 = float(bbox[1]), float(bbox[3])
    page_w, page_h = float(img.size[0]), float(img.size[1])
    half = page_w / 2

    def is_full_width(rb: list[float | int]) -> bool:
        return (float(rb[2]) - float(rb[0])) > 0.55 * page_w

    def left_of_gutter(rb: list[float | int]) -> bool:
        return (float(rb[0]) + float(rb[2])) / 2 <= half

    # A two-column page has narrow regions on both sides of the gutter; only
    # then is it safe to confine the crop to a single column.
    sides = {
        left_of_gutter(rb)
        for r in all_regions
        if (rb := r.get("bbox")) and not is_full_width(rb)
    }
    confine_to_column = len(sides) == 2 and not is_full_width(bbox)
    caption_left = left_of_gutter(bbox)

    def same_column(rb: list[float | int]) -> bool:
        # Full-width regions (section headings, rules) bound the figure
        # vertically regardless of which column it sits in.
        return (
            not confine_to_column
            or is_full_width(rb)
            or left_of_gutter(rb) == caption_left
        )

    above_y1 = 0.0
    below_y0 = page_h
    for r in all_regions:
        if r is fig_title_region:
            continue
        rb = r.get("bbox")
        if not rb or not same_column(rb):
            continue
        ry0, ry1 = float(rb[1]), float(rb[3])
        if ry1 <= fy0:
            above_y1 = max(above_y1, ry1)
        if ry0 >= fy1:
            below_y0 = min(below_y0, ry0)

    gap_above = fy0 - above_y1
    gap_below = below_y0 - fy1
    if max(gap_above, gap_below) < _MIN_FIGURE_HEIGHT:
        return None

    gy0, gy1 = (above_y1, fy0) if gap_above >= gap_below else (fy1, below_y0)
    if confine_to_column:
        cx0, cx1 = (0.0, half) if caption_left else (half, page_w)
    else:
        cx0, cx1 = 0.0, page_w
    return img.crop((int(cx0), int(gy0), int(cx1), int(gy1)))


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
            title = _STRIP_TAGS_RE.sub("", text).strip()
        elif (
            cat == "text"
            and not author
            and title
            and text.count("\n") <= 2
            and len(text) < 200
        ):
            author = _STRIP_TAGS_RE.sub("", text).strip()
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

    cuda_available = torch.cuda.is_available()
    all_regions: list[list[dict]] = []
    for img in images:
        with torch.inference_mode():
            results = model.generate_with_layout([img], ocr_batch_size=ocr_batch_size)
        if cuda_available:
            torch.cuda.empty_cache()
        # Guard against blank/image-only pages that return no regions.
        all_regions.append(results[0] if results else [])

    start = _leading_pages_to_skip(all_regions)
    if start:
        all_regions = all_regions[start:]
        images = images[start:]

    page0_width = float(images[0].size[0]) if images else 800.0
    meta = (
        _extract_meta(all_regions[0], page0_width)
        if all_regions
        else {"title": "Untitled", "authors": "", "year": ""}
    )

    abstract_parts: list[str] = []
    body_parts: list[str] = []
    footnote_parts: list[str] = []
    # figure_title regions are buffered so the next table can absorb them as
    # <caption> rather than emitting a detached <figure><figcaption>.
    pending_fig_title: str | None = None
    pending_fig_crop: Image.Image | None = None

    title_norm = (
        _WHITESPACE_RE.sub(" ", _PUNCT_RE.sub("", meta["title"])).lower().strip()
    )

    def _flush_fig_title() -> None:
        nonlocal pending_fig_title, pending_fig_crop
        if pending_fig_title is not None:
            if pending_fig_crop is not None:
                body_parts.append(_figure_html(pending_fig_crop, pending_fig_title))
            else:
                body_parts.append(
                    f"<figure><figcaption>{_inline_md_to_html(pending_fig_title)}</figcaption></figure>"
                )
            pending_fig_title = None
            pending_fig_crop = None

    for regions, img in zip(all_regions, images, strict=True):
        # Use the rendered image width as the authoritative page width so that
        # single-column pages (where max(bbox.x1) ≪ page width) don't cause
        # the column-split heuristic to misclassify left-column content.
        pw = float(img.size[0])

        for r in _sort_regions(regions, pw):
            cat = r.get("category", "text")
            text = r.get("text", "").strip()
            if (not text and cat != "figure") or cat in _SKIP_CATS:
                continue
            if cat == "doc_title":
                continue  # already captured in meta; skip body duplicate

            if cat == "abstract":
                abstract_parts.append(f"<p>{_inline_md_to_html(text)}</p>")
                continue

            if cat == "paragraph_title" and text.lower() in _DOCUMENT_TYPE_LABELS:
                continue

            if cat == "text" and title_norm:
                text_norm = (
                    _WHITESPACE_RE.sub(" ", _PUNCT_RE.sub("", text)).lower().strip()
                )
                if text_norm == title_norm:
                    continue

            if cat == "figure_title":
                if pending_fig_title is None:
                    pending_fig_crop = _infer_figure_crop(r, regions, img)
                    pending_fig_title = text
                else:
                    pending_fig_title = pending_fig_title + " " + text
                continue

            if cat == "figure":
                bbox = r.get("bbox")
                if bbox:
                    x0, y0, x1, y1 = (
                        int(bbox[0]),
                        int(bbox[1]),
                        int(bbox[2]),
                        int(bbox[3]),
                    )
                    body_parts.append(
                        _figure_html(img.crop((x0, y0, x1, y1)), pending_fig_title)
                    )
                    pending_fig_title = None
                    pending_fig_crop = None
                else:
                    _flush_fig_title()
                continue

            html_chunk = _region_to_html(r)

            if cat == "footnote":
                # Footnotes don't break a pending figure_title → table pairing.
                footnote_parts.append(html_chunk)
                continue

            if cat == "table" and pending_fig_title is not None:
                html_chunk = _inject_caption(html_chunk, pending_fig_title)
                pending_fig_title = None
                pending_fig_crop = None
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
    processed_body = _merge_split_paragraphs(
        _merge_split_paragraphs(_remove_repeated_short_paragraphs(body_parts))
    )
    if footnote_parts:
        ref_idx = next(
            (i for i, p in enumerate(processed_body) if _REF_SECTION_RE.match(p)),
            len(processed_body),
        )
        processed_body[ref_idx:ref_idx] = footnote_parts
    body_html = "\n".join(processed_body)

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
