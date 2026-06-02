"""PDF → HTML pipeline using LightOnOCR-2-1B-bbox.

LightOnOCR (lightonai/LightOnOCR-2-1B-bbox, Apache-2.0) is an end-to-end VLM that
reconstructs each page as markdown — reading order, emphasis, ``<table>`` HTML,
LaTeX math, and figure crop boxes appended to ``![image]`` placeholders.  The
pipeline OCRs every page, converts the markdown to HTML, crops figures from the
rendered page, and assembles a document shell.  (The module keeps the ``falcon``
filename for import stability; the GRM/Falcon + Heron + text-layer engine it
replaced is gone — see plans/replace-falcon-with-lightonocr.md.)

Typical use::

    ocr = load_ocr_model()
    html = lightonocr_pdf_to_html("paper.pdf", ocr=ocr)
    Path("out.html").write_text(html)
"""

from __future__ import annotations

import base64
import html as _html
import io
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import torch
from markdown_it import MarkdownIt
from PIL import Image

# transformers' type stubs lag the LightOnOCR classes shipped at runtime (5.9+).
from transformers import (  # type: ignore[attr-defined]
    LightOnOcrForConditionalGeneration,
    LightOnOcrProcessor,
)

MODEL_ID_BBOX = "lightonai/LightOnOCR-2-1B-bbox"
_OCR_MAX_NEW_TOKENS = 2048
_RENDER_SCALE = 200 / 72  # 200 DPI per the model card

# Unicode superscript forms.  A LaTeX ``$^{…}$`` run is rendered with these
# glyphs when every character has one (so "NAD$^+$" → "NAD⁺"); otherwise it falls
# back to an HTML <sup> wrapper so nothing is lost.
_SUPERSCRIPT_MAP = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "−": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "a": "ᵃ",
    "b": "ᵇ",
    "c": "ᶜ",
    "d": "ᵈ",
    "e": "ᵉ",
    "f": "ᶠ",
    "g": "ᵍ",
    "h": "ʰ",
    "i": "ⁱ",
    "j": "ʲ",
    "k": "ᵏ",
    "l": "ˡ",
    "m": "ᵐ",
    "n": "ⁿ",
    "o": "ᵒ",
    "p": "ᵖ",
    "r": "ʳ",
    "s": "ˢ",
    "t": "ᵗ",
    "u": "ᵘ",
    "v": "ᵛ",
    "w": "ʷ",
    "x": "ˣ",
    "y": "ʸ",
    "z": "ᶻ",
}


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


@dataclass
class OcrModel:
    """LightOnOCR model + processor bundle and the device/dtype to run on."""

    model: Any
    processor: Any
    device: str
    dtype: torch.dtype


def load_ocr_model(device: str | None = None) -> OcrModel:
    """Load LightOnOCR-2-1B-bbox (model + processor) for whole-page OCR.

    Args:
        device: Torch device string.  Defaults to ``"cuda"`` if available.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        MODEL_ID_BBOX, torch_dtype=dtype
    ).to(device)
    processor = LightOnOcrProcessor.from_pretrained(MODEL_ID_BBOX)
    return OcrModel(model=model, processor=processor, device=device, dtype=dtype)


def _ocr_page(
    image: Image.Image, ocr: OcrModel, max_new_tokens: int = _OCR_MAX_NEW_TOKENS
) -> str:
    """Run LightOnOCR on a single page image and return its markdown."""
    conversation = [{"role": "user", "content": [{"type": "image", "image": image}]}]
    inputs = ocr.processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {
        k: (
            v.to(device=ocr.device, dtype=ocr.dtype)
            if v.is_floating_point()
            else v.to(ocr.device)
        )
        for k, v in inputs.items()
    }
    with torch.inference_mode():
        output_ids = ocr.model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    text: str = ocr.processor.decode(generated, skip_special_tokens=True)
    del inputs, output_ids, generated
    if ocr.device == "cuda":
        torch.cuda.empty_cache()
    return text


# LightOnOCR-bbox emits figures as a markdown image placeholder with the crop
# box appended as bare ``x0,y0,x1,y1`` integers **normalized to [0, 1000]** (per
# the model card), e.g. ``![image](image_1.png)122,89,877,614``.  The base
# variant omits the coordinates, so they are optional.
_BBOX_NORM_MAX = 1000
_FIGURE_PLACEHOLDER_RE = re.compile(
    r"^!\[[^\]]*\]\([^)]*\)"
    r"(?:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+))?\s*$"
)


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


def _to_superscript(core: str) -> str:
    if all(ch in _SUPERSCRIPT_MAP or ch.isspace() for ch in core):
        return "".join(_SUPERSCRIPT_MAP.get(ch, ch) for ch in core)
    return f"<sup>{core}</sup>"


_BOLDITALIC_RE = re.compile(r"\*\*\*(.+?)\*\*\*", re.DOTALL)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
# re.DOTALL intentionally omitted: italic spans in academic text don't cross
# line boundaries, and DOTALL would cause two stray footnote asterisks anywhere
# in a multi-line region to wrap the entire intervening content in <em>.
_ITALIC_RE = re.compile(r"\*(.+?)\*")
# A leading footnote marker is a SHORT superscript ("<sup>a</sup>", "<sup>1</sup>").
# Bounding the marker length stops a body paragraph that merely opens with a
# reconstructed multi-character superscript from being mistaken for a footnote.
_SUP_MARKER_RE = re.compile(r"^<sup>[^<]{1,3}</sup>")
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

# The article starts at the first page carrying an "Abstract"/"Introduction"
# heading; a cover ad / masthead has neither.
_ARTICLE_HEADING_RE = re.compile(
    r"^\s*(?:\d+[.)]?\s+)?(?:abstract|introduction)\b", re.IGNORECASE
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


_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")
_STRIP_TAGS_RE = re.compile(r"<[^>]+>")


def _inline_md_to_html(text: str) -> str:
    text = _BOLDITALIC_RE.sub(r"<strong><em>\1</em></strong>", text)
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = _ITALIC_RE.sub(r"<em>\1</em>", text)
    return text.strip()


# LightOnOCR emits whole-page markdown mixed with raw HTML (<table>, <sup>) and
# inline LaTeX.  CommonMark + the table plugin, with raw-HTML passthrough, covers
# the structure; LaTeX sub/superscripts are converted to HTML beforehand.
_MD = MarkdownIt("commonmark", {"html": True}).enable("table")

# An inline math span: $…$ not preceded by a backslash, shortest match.
_LATEX_SPAN_RE = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$", re.DOTALL)
# Sub/superscript inside a math span: ^{multi} / ^x and _{multi} / _x.
_LATEX_SUP_RE = re.compile(r"\^\{([^{}]*)\}|\^(\S)")
_LATEX_SUB_RE = re.compile(r"_\{([^{}]*)\}|_(\S)")
# Font/style wrappers (\text{…}, \mathrm{…}) carry no semantics here — unwrap to
# their content so the inner sub/superscript handling sees plain text.
_LATEX_WRAP_RE = re.compile(
    r"\\(?:text|mathrm|mathit|mathbf|mathsf|operatorname)\{([^{}]*)\}"
)


def _latex_span_to_html(content: str) -> str:
    """Convert the inside of a ``$…$`` span: sub/superscripts to HTML, then drop
    residual TeX syntax.  Full math is out of scope (a later MathJax option)."""
    content = _LATEX_WRAP_RE.sub(r"\1", content)
    content = _LATEX_SUP_RE.sub(
        lambda m: _to_superscript(m.group(1) if m.group(1) is not None else m.group(2)),
        content,
    )
    content = _LATEX_SUB_RE.sub(
        lambda m: f"<sub>{m.group(1) if m.group(1) is not None else m.group(2)}</sub>",
        content,
    )
    return content.replace("\\,", " ").replace("{", "").replace("}", "")


def _latex_to_html(text: str) -> str:
    """Replace every inline ``$…$`` math span with deterministic HTML.

    Runs on the markdown *before* parsing so the emitted ``<sub>``/``<sup>`` pass
    through as raw HTML and the ``_`` inside ``V_{max}`` isn't read as emphasis.
    """
    return _LATEX_SPAN_RE.sub(lambda m: _latex_span_to_html(m.group(1)), text)


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


# ---------------------------------------------------------------------------
# LightOnOCR markdown pipeline (design B-prime — see
# plans/replace-falcon-with-lightonocr.md).  render → per-page markdown →
# block HTML → cleanup/merge → document shell.
# ---------------------------------------------------------------------------

_OCR_MAX_LONG_SIDE = 1540  # model-card target; VRAM ≈ 2.7/6.1 GiB at this size

# A caption opens with a figure/table label ("FIG. 2 …", "**Table 1.** …").
_CAPTION_RE = re.compile(
    r"^\*{0,2}(?:fig(?:ure|\.|\b)|table|scheme|supplement)", re.IGNORECASE
)
_HEADING_TAG_RE = re.compile(r"^<h([1-6])>(.*)</h\1>$", re.DOTALL)
_ABSTRACT_HEADING_RE = re.compile(r"^\s*abstract\b", re.IGNORECASE)
# Running header/footer: a short, terminal-punctuation-free line that recurs
# across pages.  Page numbers vary per page, so they are stripped before the
# recurrence is counted (e.g. "Biotechnology … 601" / "… 602" share a key).
_FURNITURE_MAX_LEN = 120
_DIGITS_RE = re.compile(r"\d+")


def _render_page_images(pdf_path: Path) -> list[Image.Image]:
    """Render every page to an RGB image, long side ≤ ``_OCR_MAX_LONG_SIDE``.

    ``convert("RGB")`` detaches each image from the pdfium bitmap, so the
    document can be closed before the images are consumed.
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        images: list[Image.Image] = []
        for page in pdf:
            img = page.render(scale=_RENDER_SCALE).to_pil().convert("RGB")
            long_side = max(img.size)
            if long_side > _OCR_MAX_LONG_SIDE:
                ratio = _OCR_MAX_LONG_SIDE / long_side
                img = img.resize(
                    (int(img.size[0] * ratio), int(img.size[1] * ratio)),
                    Image.Resampling.LANCZOS,
                )
            images.append(img)
        return images
    finally:
        pdf.close()


def _looks_like_caption(block: str) -> bool:
    return bool(_CAPTION_RE.match(block.strip()))


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


def _safe_crop(
    image: Image.Image, bbox: tuple[int, int, int, int]
) -> Image.Image | None:
    """Crop a pixel-space ``bbox`` from ``image``, clamped to bounds; ``None`` if
    degenerate."""
    w, h = image.size
    x0 = max(0, min(bbox[0], w))
    y0 = max(0, min(bbox[1], h))
    x1 = max(0, min(bbox[2], w))
    y1 = max(0, min(bbox[3], h))
    if x1 - x0 < _MIN_FIGURE_HEIGHT or y1 - y0 < _MIN_FIGURE_HEIGHT:
        return None
    return image.crop((x0, y0, x1, y1))


def _page_to_html_parts(md: str, image: Image.Image) -> list[str]:
    """Convert one page's markdown to block HTML, replacing each ``![image]``
    placeholder with a cropped <figure> (caption from the placeholder's own
    trailing text, else the following caption-like block)."""
    parts: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            parts.extend(_md_to_html_blocks("\n\n".join(pending)))
            pending.clear()

    raw_blocks = re.split(r"\n[ \t]*\n", md.strip())
    k = 0
    while k < len(raw_blocks):
        block = raw_blocks[k].strip()
        lines = block.splitlines()
        fig = _parse_figure_placeholder(lines[0]) if lines else None
        if fig is None:
            pending.append(block)
            k += 1
            continue
        flush()
        rest = "\n".join(lines[1:]).strip()
        caption: str | None = rest or None
        if (
            caption is None
            and k + 1 < len(raw_blocks)
            and _looks_like_caption(raw_blocks[k + 1])
        ):
            k += 1
            caption = raw_blocks[k].strip()
        caption_html = _latex_to_html(caption) if caption else None
        crop = (
            _safe_crop(image, _denormalize_bbox(fig, image))
            if isinstance(fig, tuple)
            else None
        )
        if crop is not None:
            parts.append(_figure_html(crop, caption_html))
        elif caption_html is not None:
            parts.append(
                f"<figure><figcaption>{_inline_md_to_html(caption_html)}"
                "</figcaption></figure>"
            )
        k += 1
    flush()
    return parts


# A footer/header is identified by its digit-stripped text recurring across
# pages ("… 601" / "… 602" share a key).  Require that text to be substantial so
# stripping the digits can't collapse short enumerated labels ("Fig 1" / "Fig 2",
# "Step 1" / "Step 2") into one key and delete them as furniture.
_MIN_FURNITURE_KEY_LEN = 12


def _furniture_key(inner: str) -> str:
    text = _DIGITS_RE.sub("", _STRIP_TAGS_RE.sub("", inner))
    return _WHITESPACE_RE.sub(" ", _PUNCT_RE.sub("", text)).strip().lower()


def _is_furniture_candidate(part: str) -> str | None:
    inner = _plain_p_text(part)
    if inner is None:
        return None
    plain = _STRIP_TAGS_RE.sub("", inner)
    if len(plain) > _FURNITURE_MAX_LEN or _SENTENCE_END_RE.search(plain.rstrip()):
        return None
    key = _furniture_key(inner)
    return key if len(key) >= _MIN_FURNITURE_KEY_LEN else None


def _strip_running_furniture(parts: list[str]) -> list[str]:
    """Drop short, recurring header/footer lines (page-number-insensitive)."""
    counts: Counter[str] = Counter(
        key for part in parts if (key := _is_furniture_candidate(part)) is not None
    )
    repeated = {key for key, n in counts.items() if n > 1}
    return [p for p in parts if _is_furniture_candidate(p) not in repeated]


# Even though LightOnOCR-bbox usually boxes figures (so they never reach the text
# stream), a diagram it misses can still be OCRed into one label repeated dozens
# of times ("AaTRI, AaTRI, …") — many tokens, almost no diversity.  This drops
# such a paragraph from the body; real prose (even with some repetition) stays.
_MIN_REPEAT_TOKENS = 8
_MAX_REPEAT_SHARE = 0.6
_TOKEN_RE = re.compile(r"\w+")


def _is_degenerate_repetition(text: str) -> bool:
    tokens = _TOKEN_RE.findall(_STRIP_TAGS_RE.sub("", text))
    if len(tokens) < _MIN_REPEAT_TOKENS:
        return False
    top = Counter(tokens).most_common(1)[0][1]
    return top / len(tokens) >= _MAX_REPEAT_SHARE


def _heading_inner(part: str) -> tuple[int, str] | None:
    m = _HEADING_TAG_RE.match(part)
    return (int(m.group(1)), m.group(2).strip()) if m else None


def _is_title_heading(inner: str) -> bool:
    plain = _STRIP_TAGS_RE.sub("", inner).strip().lower()
    return plain not in _DOCUMENT_TYPE_LABELS and not _ARTICLE_HEADING_RE.match(plain)


def _is_byline(inner: str) -> bool:
    plain = _STRIP_TAGS_RE.sub("", inner).strip()
    return (
        bool(plain)
        and len(plain) < 400
        and not _SENTENCE_END_RE.search(plain)
        and not _BOLD_LABEL_RE.match(inner)
    )


def _byline_text(inner: str) -> str:
    return _STRIP_TAGS_RE.sub("", re.sub(r"<br\s*/?>", "; ", inner)).strip()


def _is_article_page_md(md: str) -> bool:
    """A page is the article start if it carries an Abstract/Introduction
    heading (a cover ad / masthead has neither)."""
    for line in md.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)", line.strip())
        if m and _ARTICLE_HEADING_RE.match(_STRIP_TAGS_RE.sub("", m.group(1)).strip()):
            return True
    return False


def _leading_pages_to_skip_md(pages_md: list[str]) -> int:
    return next(
        (i for i, md in enumerate(pages_md) if _is_article_page_md(md)),
        0,
    )


@dataclass
class _Meta:
    title_html: str
    byline_html: str
    abstract: list[str]
    body: list[str]
    footnotes: list[str]


def _classify_parts(parts: list[str]) -> _Meta:
    """Single pass: pull the title, byline, abstract and footnotes out of the
    flat block list; everything else is body."""
    title_html = ""
    byline_html = ""
    abstract: list[str] = []
    body: list[str] = []
    footnotes: list[str] = []
    # The byline is only the block *immediately* after the title heading; this
    # window closes at the next block so a body sentence is never mistaken for it.
    expect_byline = False
    in_abstract = False

    for part in parts:
        heading = _heading_inner(part)
        if heading is not None:
            _, inner = heading
            if not title_html and _is_title_heading(inner):
                title_html = inner
                expect_byline = True
                continue
            expect_byline = False
            if _ABSTRACT_HEADING_RE.match(_STRIP_TAGS_RE.sub("", inner)):
                in_abstract = True
                continue
            # A document-type label heading ("Article") is dropped entirely.
            if _STRIP_TAGS_RE.sub("", inner).strip().lower() in _DOCUMENT_TYPE_LABELS:
                continue
            in_abstract = False
            body.append(part)
            continue

        inner_p = _plain_p_text(part)
        if expect_byline:
            expect_byline = False
            if inner_p is not None and _is_byline(inner_p):
                byline_html = _byline_text(inner_p)
                continue
        if in_abstract:
            if inner_p is not None and not _BOLD_LABEL_RE.match(inner_p):
                abstract.append(part)
                continue
            in_abstract = False
        if inner_p is not None and _SUP_MARKER_RE.match(inner_p):
            footnotes.append(f'<p class="footnote">{inner_p}</p>')
            continue
        if inner_p is not None and _is_degenerate_repetition(inner_p):
            continue
        body.append(part)

    return _Meta(title_html, byline_html, abstract, body, footnotes)


def _assemble_html(pages_md: list[str], images: list[Image.Image]) -> str:
    start = _leading_pages_to_skip_md(pages_md)
    pages_md = pages_md[start:]
    images = images[start:]

    parts: list[str] = []
    for md, img in zip(pages_md, images, strict=True):
        parts.extend(_page_to_html_parts(md, img))

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
