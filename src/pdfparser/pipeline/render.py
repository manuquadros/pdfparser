"""PDF page rendering: a PDF path → one RGB PIL image per page."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — beartype reads annotations at runtime

import pypdfium2 as pdfium
from PIL import Image

_RENDER_SCALE = 200 / 72  # 200 DPI per the model card
_OCR_MAX_LONG_SIDE = 1540  # model-card target; VRAM ≈ 2.7/6.1 GiB at this size


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
