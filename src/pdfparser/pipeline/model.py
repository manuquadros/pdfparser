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
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx
from PIL import Image  # noqa: TC002 — beartype reads annotations at runtime

_DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
_DEFAULT_MODEL = "lightonocr"
_OCR_MAX_NEW_TOKENS = 2048
# Fallback context window — input (page-image) tokens plus generated tokens — used
# only when the server's ``/models`` response omits ``max_model_len`` *and* the
# ``PDFPARSER_VLLM_MAX_MODEL_LEN`` override is unset (see ``_resolve_context_len``).
# It sizes the retry budget when a page's generation truncates: a dense page (a large
# table plus prose) can exceed ``_OCR_MAX_NEW_TOKENS`` and get cut off mid-output,
# silently dropping the rest of the table and everything after it.
_DEFAULT_MODEL_CONTEXT_LEN = 8192
# Leave a little of the window unclaimed so the retry's ``max_tokens`` can't tip
# prompt+output past the server limit (vLLM rejects such a request outright).
_CONTEXT_SAFETY_MARGIN = 64
# Pages OCR independently, so the client issues several requests at once to let
# the vLLM server's continuous batching engage — a serial caller pins
# ``num_requests_running`` at 1, leaving most of the GPU idle.  Bounded because
# the card is small and shared; override with ``PDFPARSER_OCR_CONCURRENCY``.
_DEFAULT_OCR_CONCURRENCY = 4
# A cold page can take tens of seconds on a small GPU; httpx's 5 s default would
# abort mid-decode, so OCR requests use a generous per-request budget.
_REQUEST_TIMEOUT_S = 600.0
# The reachability probe must fail fast — it shares the client but not the long
# generation budget, so a wedged server doesn't block startup for ten minutes.
_HEALTH_TIMEOUT_S = 10.0
# A page POST runs concurrently against a small shared GPU, so a transient
# connection blip or vLLM overload status on any one page would otherwise abort the
# whole multi-page document via the propagated exception.  Absorb those with a
# bounded backoff retry.  Deliberately narrow (see _is_retryable_ocr_error): a 4xx
# is a caller bug that re-fails identically, and a *timeout* already spent the full
# per-request budget, so neither is retried.
_MAX_OCR_RETRIES = 2
_RETRY_BACKOFF_BASE_S = 0.5
_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})
# Cap a server-sent Retry-After so a pathological header can't pin a pool worker.
_MAX_RETRY_AFTER_S = 30.0


@dataclass
class OcrModel:
    """Open connection to a vLLM server hosting LightOnOCR + the served name.

    Holds an ``httpx.Client`` (a connection pool), so close it when done — or use
    it as a context manager.  ``lightonocr_pdf_to_html`` closes the bundle it
    creates internally; a caller that passes its own ``ocr`` owns its lifecycle.

    ``context_len`` is the server's token window (input + output), resolved once at
    ``load_ocr_model`` time from the ``/models`` response (or the env override), so
    the truncation-retry budget in ``_ocr_page`` is correct for the actual server
    without a per-page ``os.environ`` re-read.
    """

    client: httpx.Client
    base_url: str
    model: str
    context_len: int = _DEFAULT_MODEL_CONTEXT_LEN

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> OcrModel:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _resolve_base_url(base_url: str | None) -> str:
    """The vLLM endpoint root: the explicit arg, else ``PDFPARSER_VLLM_URL``, else the
    built-in default; trailing slash stripped so ``f"{base_url}/…"`` stays clean."""
    return (base_url or os.environ.get("PDFPARSER_VLLM_URL", _DEFAULT_BASE_URL)).rstrip(
        "/"
    )


def _client_limits() -> httpx.Limits:
    """Size the connection pool to the resolved OCR concurrency so a raised
    ``PDFPARSER_OCR_CONCURRENCY`` isn't capped below the worker count by httpx's
    default ``max_connections`` — the page pass issues that many POSTs at once over
    the single shared client."""
    workers = _resolve_ocr_concurrency()
    return httpx.Limits(max_connections=workers, max_keepalive_connections=workers)


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
    base_url = _resolve_base_url(base_url)
    model = model or os.environ.get("PDFPARSER_VLLM_MODEL", _DEFAULT_MODEL)
    client = httpx.Client(timeout=timeout, limits=_client_limits())
    try:
        response = client.get(f"{base_url}/models", timeout=_HEALTH_TIMEOUT_S)
        response.raise_for_status()
    except httpx.HTTPError:
        client.close()  # don't leak the pool when the probe fails
        raise
    # The probe doubles as the context-window query (vLLM reports max_model_len in
    # /models), so the truncation-retry budget matches the live server with no extra
    # round trip.  A non-JSON body (a bare-200 mock / non-vLLM endpoint) falls back.
    try:
        payload: object = response.json()
    except ValueError:
        payload = None
    context_len = _resolve_context_len(_parse_server_context_len(payload))
    return OcrModel(
        client=client, base_url=base_url, model=model, context_len=context_len
    )


def _env_int(name: str, default: int) -> int:
    """Read a positive-int env var, defaulting (with a warning) on a bad value."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        # A misconfigured deployment should be visible, not silently defaulted.
        warnings.warn(
            f"{name}={raw!r} is not an integer; falling back to {default}",
            stacklevel=3,  # past this helper to the real caller
        )
        return default


def _parse_server_context_len(payload: object) -> int | None:
    """Extract ``max_model_len`` from a vLLM ``/models`` response body, or ``None`` if
    the shape doesn't carry it (an older server, or a non-vLLM OpenAI-compatible
    endpoint) so the caller falls back to the env override / default.  vLLM reports it
    per served model under ``data[0]``."""
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    value = first.get("max_model_len")
    return value if isinstance(value, int) and value > 0 else None


def _resolve_context_len(reported: int | None) -> int:
    """The server context window (the truncation-retry token budget): the
    ``PDFPARSER_VLLM_MAX_MODEL_LEN`` override wins, else the value the server reported
    via ``/models`` (``reported``), else the built-in default.  The override lets a
    deployment whose server can't report ``max_model_len`` still tune the budget."""
    if os.environ.get("PDFPARSER_VLLM_MAX_MODEL_LEN") is not None:
        return _env_int(
            "PDFPARSER_VLLM_MAX_MODEL_LEN",
            reported if reported is not None else _DEFAULT_MODEL_CONTEXT_LEN,
        )
    return reported if reported is not None else _DEFAULT_MODEL_CONTEXT_LEN


def _is_retryable_ocr_error(exc: httpx.HTTPError) -> bool:
    """Whether a failed page POST is worth re-issuing.

    Retryable: a connection-level blip (``httpx.NetworkError`` — connect refused/reset,
    a dropped read) or a vLLM overload/gateway status (``_RETRYABLE_STATUS``).  A
    *timeout* (``httpx.TimeoutException``: read/connect/pool) is excluded on purpose —
    it already spent the full per-request budget, so retrying re-spends it, and
    ``_ocr_pages`` relies on a wedged page failing *without* retry so the pool tears
    down promptly rather than after ~3× the timeout.  A 4xx (caller bug) is excluded
    too — it re-fails identically.
    """
    if isinstance(exc, httpx.NetworkError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    return False


def _retry_delay(exc: httpx.HTTPError, attempt: int) -> float:
    """Seconds to wait before the next attempt: honor a server-sent ``Retry-After`` on
    an overload status (so the client doesn't pile load back onto a busy server),
    capped by ``_MAX_RETRY_AFTER_S``; otherwise exponential backoff."""
    if isinstance(exc, httpx.HTTPStatusError):
        raw = exc.response.headers.get("Retry-After")
        if raw is not None:
            try:
                return min(max(0.0, float(raw)), _MAX_RETRY_AFTER_S)
            except ValueError:
                pass  # HTTP-date form (rare from vLLM) → fall back to backoff
    return _RETRY_BACKOFF_BASE_S * (2.0**attempt)


def _request_ocr(encoded: str, ocr: OcrModel, max_new_tokens: int) -> httpx.Response:
    """POST one page to the chat endpoint, retrying transient failures.

    A connection-level blip or a retryable vLLM status (see ``_is_retryable_ocr_error``)
    is retried with backoff (``_retry_delay``) up to ``_MAX_OCR_RETRIES`` times; any
    other error — and a final exhausted retry — re-raises unchanged, so a genuine
    failure (and a slow-timeout wedge) still surfaces promptly.
    """
    body = {
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
    }
    for attempt in range(_MAX_OCR_RETRIES + 1):
        try:
            response = ocr.client.post(f"{ocr.base_url}/chat/completions", json=body)
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            if not _is_retryable_ocr_error(exc) or attempt >= _MAX_OCR_RETRIES:
                raise
            time.sleep(_retry_delay(exc, attempt))
    raise AssertionError("unreachable: OCR retry loop exited without return/raise")


def _post_ocr_page(
    encoded: str, ocr: OcrModel, max_new_tokens: int
) -> tuple[str, str | None, int | None]:
    """One OCR request; returns ``(markdown, finish_reason, prompt_tokens)``.

    ``finish_reason`` is ``"length"`` when the generation hit ``max_new_tokens``
    (the page was truncated) and ``"stop"`` on a natural end; ``prompt_tokens`` (the
    image's token cost, from the response ``usage``) sizes a retry.  Either may be
    ``None`` for a server/mock that omits them — the caller then skips the retry.
    """
    response = _request_ocr(encoded, ocr, max_new_tokens)
    payload = response.json()
    try:
        choice = payload["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        # TypeError covers a null-valued key ("choices": null -> None[0]), not just
        # a missing one, so a malformed shape always surfaces as this clear error.
        raise RuntimeError(f"unexpected OCR response shape: {payload}") from exc
    finish_reason = choice.get("finish_reason")
    usage = payload.get("usage")
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    # A degenerate page yields null content; return "" (as the former in-process
    # decode did) rather than tripping the str return contract.  A non-string,
    # non-null content is a structurally different response (e.g. OpenAI content
    # parts) the pipeline can't consume — fail loudly rather than silently drop the
    # page's text as empty.
    if content is None:
        return "", finish_reason, prompt_tokens
    if not isinstance(content, str):
        raise RuntimeError(f"unexpected OCR content type: {payload}")
    return content, finish_reason, prompt_tokens


def _ocr_page(
    image: Image.Image, ocr: OcrModel, max_new_tokens: int = _OCR_MAX_NEW_TOKENS
) -> str:
    """OCR a single page image to markdown via the vLLM chat endpoint.

    Greedy (``temperature=0``) so the result is the most-likely transcription and
    reproducible run-to-run, matching the former in-process decode.  The model
    ignores any text prompt, so the request carries only the image.

    A page dense enough to exceed ``max_new_tokens`` (a large table plus prose)
    truncates mid-output, dropping the rest of the table and everything after it.
    On that signal (``finish_reason == "length"``) the page is re-OCR'd once with
    the entire remaining context window — greedy decode reproduces the prefix and
    continues past the cut.  If even the full window is too small, or the retry
    comes back degenerate (empty/shorter than the first response), the best-effort
    truncated text is kept rather than dropped.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    content, finish_reason, prompt_tokens = _post_ocr_page(encoded, ocr, max_new_tokens)
    if finish_reason != "length" or prompt_tokens is None:
        return content
    retry_budget = ocr.context_len - prompt_tokens - _CONTEXT_SAFETY_MARGIN
    if retry_budget <= max_new_tokens:
        return content
    retried, _, _ = _post_ocr_page(encoded, ocr, retry_budget)
    # A degenerate retry (null content -> "", or a non-deterministic shorter decode)
    # must not discard the good-but-truncated first response; keep whichever
    # recovered more of the page.
    return retried if len(retried) >= len(content) else content


def _resolve_ocr_concurrency() -> int:
    return _env_int("PDFPARSER_OCR_CONCURRENCY", _DEFAULT_OCR_CONCURRENCY)


def _ocr_pages(
    images: list[Image.Image],
    ocr: OcrModel,
    max_new_tokens: int = _OCR_MAX_NEW_TOKENS,
    concurrency: int | None = None,
) -> list[str]:
    """OCR page images via the vLLM server, returning their markdown in page order.

    Requests are issued with up to ``concurrency`` in flight at once (``None`` ->
    ``PDFPARSER_OCR_CONCURRENCY``/``_DEFAULT_OCR_CONCURRENCY``; an explicit value is
    floored at 1, so ``concurrency=0`` means serial, not "use the default") over the
    shared, thread-safe ``httpx.Client``.  Per-page OCR is independent, so results
    are gathered back in input order and the page↔image alignment the pipeline
    depends on is preserved.

    On a per-page error, the first failing page's exception propagates and the pool
    is torn down without waiting on still-running siblings (queued requests are
    cancelled), so one quick failure can't be stalled for minutes behind a wedged
    request near the long per-request timeout.

    Output is semantically equivalent to the serial path but not byte-identical
    across runs: continuous batching means batch composition now varies with
    request timing, jittering the low-order bits (mostly figure bbox digits).  The
    acceptance gate is substring/structural for exactly this reason — see
    ``spike_results/vllm_determinism.md``.
    """
    if not images:
        return []
    limit = _resolve_ocr_concurrency() if concurrency is None else max(1, concurrency)
    pool = ThreadPoolExecutor(max_workers=min(limit, len(images)))
    try:
        futures = [pool.submit(_ocr_page, img, ocr, max_new_tokens) for img in images]
        return [future.result() for future in futures]
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
