"""LightOnOCR model seam: load the weights and OCR one page image to markdown.

The only GPU-touching part of the pipeline.  ``OcrModel`` bundles the model,
processor, device and dtype so the rest of the pipeline can stay model-free and
unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image  # noqa: TC002 — beartype reads annotations at runtime

# transformers' type stubs lag the LightOnOCR classes shipped at runtime (5.9+).
from transformers import (  # type: ignore[attr-defined]
    LightOnOcrForConditionalGeneration,
    LightOnOcrProcessor,
)

MODEL_ID_BBOX = "lightonai/LightOnOCR-2-1B-bbox"
_OCR_MAX_NEW_TOKENS = 2048


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
        # Greedy decoding: OCR wants the most-likely transcription, and a
        # deterministic decode avoids run-to-run drift (e.g. a figure box
        # occasionally over-segmenting into two stacked crops).
        output_ids = ocr.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    text: str = ocr.processor.decode(generated, skip_special_tokens=True)
    del inputs, output_ids, generated
    if ocr.device == "cuda":
        torch.cuda.empty_cache()
    return text
