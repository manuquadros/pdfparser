"""LightOnOCR seam: OCR one page image to markdown via a vLLM server.

The GPU work runs in a vLLM OpenAI-compatible server (see ``deploy/vllm/``), so
this module is a thin HTTP client rather than an in-process model load.  It keeps
the seam contract the rest of the pipeline depends on — ``load_ocr_model()``
returns an ``OcrModel`` bundle and ``_ocr_page(image, ocr) -> str`` — so nothing
downstream changes, and the package no longer depends on torch or transformers.
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass

import httpx
from PIL import Image  # noqa: TC002 — beartype reads annotations at runtime

_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
_DEFAULT_MODEL = "lightonocr"
_OCR_MAX_NEW_TOKENS = 2048
# A cold page can take tens of seconds on a small GPU; httpx's 5 s default would
# abort mid-decode, so OCR requests use a generous per-request budget.
_REQUEST_TIMEOUT_S = 600.0
# The reachability probe must fail fast — it shares the client but not the long
# generation budget, so a wedged server doesn't block startup for ten minutes.
_HEALTH_TIMEOUT_S = 10.0


@dataclass
class OcrModel:
    """Open connection to a vLLM server hosting LightOnOCR + the served name.

    Holds an ``httpx.Client`` (a connection pool), so close it when done — or use
    it as a context manager.  ``lightonocr_pdf_to_html`` closes the bundle it
    creates internally; a caller that passes its own ``ocr`` owns its lifecycle.
    """

    client: httpx.Client
    base_url: str
    model: str

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> OcrModel:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def load_ocr_model(
    base_url: str | None = None,
    model: str | None = None,
    timeout: float = _REQUEST_TIMEOUT_S,
) -> OcrModel:
    """Open a client to the vLLM server and verify it is reachable.

    Args:
        base_url: OpenAI-compatible endpoint root.  Defaults to the
            ``PDFPARSER_VLLM_URL`` env var, then ``http://127.0.0.1:8000/v1``.
        model: Served model name.  Defaults to ``PDFPARSER_VLLM_MODEL``, then
            ``lightonocr``.
        timeout: Per-request timeout in seconds.

    Raises:
        httpx.HTTPError: If the server is unreachable or unhealthy.  Callers that
            want to degrade gracefully (e.g. the integration fixture) catch this.
    """
    base_url = (
        base_url or os.environ.get("PDFPARSER_VLLM_URL", _DEFAULT_BASE_URL)
    ).rstrip("/")
    model = model or os.environ.get("PDFPARSER_VLLM_MODEL", _DEFAULT_MODEL)
    client = httpx.Client(timeout=timeout)
    try:
        client.get(f"{base_url}/models", timeout=_HEALTH_TIMEOUT_S).raise_for_status()
    except httpx.HTTPError:
        client.close()  # don't leak the pool when the probe fails
        raise
    return OcrModel(client=client, base_url=base_url, model=model)


def _ocr_page(
    image: Image.Image, ocr: OcrModel, max_new_tokens: int = _OCR_MAX_NEW_TOKENS
) -> str:
    """OCR a single page image to markdown via the vLLM chat endpoint.

    Greedy (``temperature=0``) so the result is the most-likely transcription and
    reproducible run-to-run, matching the former in-process decode.  The model
    ignores any text prompt, so the request carries only the image.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    response = ocr.client.post(
        f"{ocr.base_url}/chat/completions",
        json={
            "model": ocr.model,
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded}"},
                        }
                    ],
                }
            ],
        },
    )
    response.raise_for_status()
    payload = response.json()
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"unexpected OCR response shape: {payload}") from exc
    # A degenerate page can yield null/empty content; return "" (as the former
    # in-process decode did) rather than tripping the str return contract.
    text: str = content or ""
    return text
