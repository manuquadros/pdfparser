"""PDF page rendering: a PDF path → one RGB PIL image per page."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — beartype reads annotations at runtime

import pypdfium2 as pdfium
from PIL import Image

_RENDER_SCALE = 200 / 72  # 200 DPI per the model card
_OCR_MAX_LONG_SIDE = 1540  # model-card target; VRAM ≈ 2.7/6.1 GiB at this size


def _downscale_to_long_side(img: Image.Image) -> Image.Image:
    """Shrink ``img`` so its long side is ≤ ``_OCR_MAX_LONG_SIDE`` (no-op if it
    already fits).  Shared by full-page render and the table-crop render so both
    honour the model's long-side budget from one place."""
    long_side = max(img.size)
    if long_side <= _OCR_MAX_LONG_SIDE:
        return img
    ratio = _OCR_MAX_LONG_SIDE / long_side
    return img.resize(
        (int(img.size[0] * ratio), int(img.size[1] * ratio)),
        Image.Resampling.LANCZOS,
    )


def _render_page_images(pdf_path: Path) -> list[Image.Image]:
    """Render every page to an RGB image, long side ≤ ``_OCR_MAX_LONG_SIDE``.

    ``convert("RGB")`` detaches each image from the pdfium bitmap, so the
    document can be closed before the images are consumed.
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return [
            _downscale_to_long_side(
                page.render(scale=_RENDER_SCALE).to_pil().convert("RGB")
            )
            for page in pdf
        ]
    finally:
        pdf.close()
