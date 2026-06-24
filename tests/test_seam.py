"""Tests for the OCR seam, concurrency resolution, and CLI."""

import base64
import io
from pathlib import Path

import pytest
from helpers import (
    _fake_image,
)
from PIL import Image


class TestOcrSeam:
    """The model seam is now an HTTP client to the vLLM server, so it is
    unit-testable with a mock transport — no GPU, no model load."""

    def test_ocr_page_request_shape_and_parsing(self) -> None:
        import json

        import httpx

        from pdfparser.pipeline.model import OcrModel, _ocr_page

        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "# OCR markdown"}}]}
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        result = _ocr_page(_fake_image(8, 8), ocr)

        assert result == "# OCR markdown"
        assert captured["url"] == "http://srv/v1/chat/completions"
        body = captured["body"]
        assert isinstance(body, dict)
        # Greedy decode (matches the former in-process do_sample=False) against
        # the served model name.
        assert body["temperature"] == 0.0
        assert body["model"] == "lightonocr"
        part = body["messages"][0]["content"][0]
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_ocr_page_raises_on_server_error(self) -> None:
        import httpx

        from pdfparser.pipeline.model import OcrModel, _ocr_page

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        with pytest.raises(httpx.HTTPStatusError):
            _ocr_page(_fake_image(8, 8), ocr)

    def test_ocr_page_null_content_returns_empty_string(self) -> None:
        import httpx

        from pdfparser.pipeline.model import OcrModel, _ocr_page

        def handler(request: httpx.Request) -> httpx.Response:
            body = {"choices": [{"message": {"content": None}}]}
            return httpx.Response(200, json=body)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        # A degenerate page must yield "" (a str), not trip the return contract.
        assert _ocr_page(_fake_image(8, 8), ocr) == ""

    def test_ocr_page_raises_on_malformed_response(self) -> None:
        import httpx

        from pdfparser.pipeline.model import OcrModel, _ocr_page

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": []})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        with pytest.raises(RuntimeError, match="unexpected OCR response"):
            _ocr_page(_fake_image(8, 8), ocr)

    def test_ocr_page_null_choices_raises_runtime_error(self) -> None:
        import httpx

        from pdfparser.pipeline.model import OcrModel, _ocr_page

        # "choices": null is a null-valued key (None[0] -> TypeError), not a missing
        # one; it must still surface as the clear "unexpected OCR response" error.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": None})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        with pytest.raises(RuntimeError, match="unexpected OCR response"):
            _ocr_page(_fake_image(8, 8), ocr)

    def test_ocr_page_non_string_content_raises(self) -> None:
        import httpx

        from pdfparser.pipeline.model import OcrModel, _ocr_page

        # Structured (non-null, non-string) content — e.g. OpenAI content parts —
        # is a response the pipeline can't consume; fail loudly rather than silently
        # transcribe the page as empty.
        def handler(request: httpx.Request) -> httpx.Response:
            body = {"choices": [{"message": {"content": [{"text": "x"}]}}]}
            return httpx.Response(200, json=body)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        with pytest.raises(RuntimeError, match="unexpected OCR content type"):
            _ocr_page(_fake_image(8, 8), ocr)

    def test_truncated_page_is_reocrd_with_full_context(self) -> None:
        import json

        import httpx

        from pdfparser.pipeline.model import OcrModel, _ocr_page

        # First response truncates (finish_reason "length"); the seam must retry once
        # with the whole remaining context window (max-model-len 8192 − prompt − a
        # small margin) so the page's dropped tail is recovered.
        calls: list[int] = []
        truncated = "<table>the page up to the"
        # The full re-OCR reproduces the prefix and continues, so it is longer than
        # the truncated first response.
        full = truncated + " cut and the rest</table>"

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            calls.append(body["max_tokens"])
            content, reason = (
                (truncated, "length") if len(calls) == 1 else (full, "stop")
            )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": content}, "finish_reason": reason}
                    ],
                    "usage": {"prompt_tokens": 2500},
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        assert _ocr_page(_fake_image(8, 8), ocr) == full
        assert len(calls) == 2
        assert calls[0] == 2048
        # The retry claims the full remaining window, not just another fixed block.
        assert calls[1] == 8192 - 2500 - 64

    def test_natural_finish_does_not_retry(self) -> None:
        import httpx

        from pdfparser.pipeline.model import OcrModel, _ocr_page

        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "done"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 2500},
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        assert _ocr_page(_fake_image(8, 8), ocr) == "done"
        assert len(calls) == 1

    def test_truncation_without_headroom_keeps_best_effort(self) -> None:
        import httpx

        from pdfparser.pipeline.model import OcrModel, _ocr_page

        # The prompt already fills the window, so a retry has no more room than the
        # first call — keep the truncated text rather than re-OCRing for nothing.
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "partial"}, "finish_reason": "length"}
                    ],
                    "usage": {"prompt_tokens": 8000},
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        assert _ocr_page(_fake_image(8, 8), ocr) == "partial"
        assert len(calls) == 1

    def test_degenerate_retry_keeps_first_truncated_content(self) -> None:
        import json

        import httpx

        from pdfparser.pipeline.model import OcrModel, _ocr_page

        # First call truncates with usable text; the retry comes back degenerate
        # (null content -> ""). The good-but-truncated first response must be kept,
        # not replaced by the empty retry — otherwise the whole page is dropped.
        contents = ["a real but truncated page of markdown", None]

        def handler(request: httpx.Request) -> httpx.Response:
            json.loads(request.content)
            content = contents.pop(0)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": content}, "finish_reason": "length"}
                    ],
                    "usage": {"prompt_tokens": 2500},
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        assert (
            _ocr_page(_fake_image(8, 8), ocr) == "a real but truncated page of markdown"
        )

    def test_ocr_pages_preserves_order_under_concurrency(self) -> None:
        import json

        import httpx

        from pdfparser.pipeline.model import OcrModel, _ocr_pages

        # Each page carries a distinct width so the handler can echo its identity;
        # concurrent completion must still gather back in input (page) order.
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            url = body["messages"][0]["content"][0]["image_url"]["url"]
            png = base64.b64decode(url.split(",", 1)[1])
            width = Image.open(io.BytesIO(png)).size[0]
            return httpx.Response(
                200, json={"choices": [{"message": {"content": f"page-{width}"}}]}
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        images = [_fake_image(w, 8) for w in range(10, 18)]
        result = _ocr_pages(images, ocr, concurrency=4)

        assert result == [f"page-{w}" for w in range(10, 18)]

    def test_ocr_pages_empty_input_returns_empty(self) -> None:
        import httpx

        from pdfparser.pipeline.model import OcrModel, _ocr_pages

        # No images → no requests, no thread pool; returns [] without touching the
        # client (a handler that would fail the test if called).
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request should be issued for empty input")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        assert _ocr_pages([], ocr) == []

    def test_ocr_pages_propagates_page_error(self) -> None:
        import httpx

        from pdfparser.pipeline.model import OcrModel, _ocr_pages

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        ocr = OcrModel(client=client, base_url="http://srv/v1", model="lightonocr")
        with pytest.raises(httpx.HTTPStatusError):
            _ocr_pages([_fake_image(8, 8), _fake_image(9, 8)], ocr, concurrency=2)

    def test_ocr_model_context_manager_closes_client(self) -> None:
        import httpx

        from pdfparser.pipeline.model import OcrModel

        transport = httpx.MockTransport(lambda r: httpx.Response(200))
        client = httpx.Client(transport=transport)
        with OcrModel(client=client, base_url="http://srv/v1", model="m") as ocr:
            assert not ocr.client.is_closed
        assert client.is_closed

    @staticmethod
    def _patch_client(monkeypatch: pytest.MonkeyPatch, handler: object) -> list[object]:
        """Make ``model.load_ocr_model``'s internal ``httpx.Client(...)`` use a mock
        transport; return the list of clients it builds so a test can assert on the
        pool's lifecycle."""
        import httpx

        from pdfparser.pipeline import model

        built: list[object] = []
        real_client = httpx.Client

        def fake_client(**kwargs: object) -> httpx.Client:
            client = real_client(transport=httpx.MockTransport(handler), **kwargs)
            built.append(client)
            return client

        monkeypatch.setattr(model.httpx, "Client", fake_client)
        return built

    def test_load_ocr_model_probes_models_and_strips_trailing_slash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from pdfparser.pipeline.model import load_ocr_model

        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(200, json={"data": []})

        self._patch_client(monkeypatch, handler)
        ocr = load_ocr_model(base_url="http://srv/v1/", model="m")
        # the trailing slash is stripped and the reachability probe hits /models
        assert seen["url"] == "http://srv/v1/models"
        assert ocr.base_url == "http://srv/v1"
        assert ocr.model == "m"
        ocr.close()

    def test_load_ocr_model_reads_env_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from pdfparser.pipeline.model import load_ocr_model

        monkeypatch.setenv("PDFPARSER_VLLM_URL", "http://envhost:9/v1")
        monkeypatch.setenv("PDFPARSER_VLLM_MODEL", "envmodel")
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return httpx.Response(200)

        self._patch_client(monkeypatch, handler)
        ocr = load_ocr_model()
        assert seen["url"] == "http://envhost:9/v1/models"
        assert ocr.model == "envmodel"
        ocr.close()

    def test_load_ocr_model_closes_client_when_probe_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from pdfparser.pipeline.model import load_ocr_model

        built = self._patch_client(
            monkeypatch, lambda request: httpx.Response(503, json={"error": "down"})
        )
        with pytest.raises(httpx.HTTPStatusError):
            load_ocr_model(base_url="http://srv/v1")
        # the probe failed, so the pool must be torn down rather than leaked
        assert built and built[0].is_closed  # type: ignore[attr-defined]


class TestOcrConcurrencyResolution:
    """``PDFPARSER_OCR_CONCURRENCY`` parsing — defaulting, flooring, and the
    misconfiguration warning — without spinning up the thread pool."""

    def test_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pdfparser.pipeline.model import (
            _DEFAULT_OCR_CONCURRENCY,
            _resolve_ocr_concurrency,
        )

        monkeypatch.delenv("PDFPARSER_OCR_CONCURRENCY", raising=False)
        assert _resolve_ocr_concurrency() == _DEFAULT_OCR_CONCURRENCY

    def test_valid_integer_is_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pdfparser.pipeline.model import _resolve_ocr_concurrency

        monkeypatch.setenv("PDFPARSER_OCR_CONCURRENCY", "7")
        assert _resolve_ocr_concurrency() == 7

    def test_floored_at_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pdfparser.pipeline.model import _resolve_ocr_concurrency

        monkeypatch.setenv("PDFPARSER_OCR_CONCURRENCY", "0")
        assert _resolve_ocr_concurrency() == 1

    def test_non_integer_warns_and_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pdfparser.pipeline.model import (
            _DEFAULT_OCR_CONCURRENCY,
            _resolve_ocr_concurrency,
        )

        monkeypatch.setenv("PDFPARSER_OCR_CONCURRENCY", "lots")
        # a misconfigured deployment is surfaced, not silently defaulted
        with pytest.warns(UserWarning, match="not an integer"):
            assert _resolve_ocr_concurrency() == _DEFAULT_OCR_CONCURRENCY


class TestCli:
    """The ``python -m pdfparser`` entry point: output-path defaulting and option
    forwarding, with the conversion itself stubbed (no model, no rendering)."""

    @staticmethod
    def _make_pdf(tmp_path: Path) -> Path:
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        return pdf

    def test_default_output_path_derived_from_pdf_suffix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pdfparser.__main__ as cli

        pdf = self._make_pdf(tmp_path)
        captured: dict[str, object] = {}

        def fake_convert(p: Path, **kwargs: object) -> str:
            captured["pdf"] = p
            captured["kwargs"] = kwargs
            return "<html>ok</html>"

        monkeypatch.setattr(cli, "lightonocr_pdf_to_html", fake_convert)
        assert cli.main([str(pdf)]) == 0
        # default output is the input path with a .html suffix
        out = (tmp_path / "paper.html").read_text(encoding="utf-8")
        assert out == "<html>ok</html>"
        assert captured["pdf"] == pdf
        assert captured["kwargs"] == {
            "base_url": None,
            "model": None,
            "image_dir": None,
        }

    def test_explicit_output_and_options_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pdfparser.__main__ as cli

        pdf = self._make_pdf(tmp_path)
        out = tmp_path / "custom.html"
        image_dir = tmp_path / "imgs"
        captured: dict[str, object] = {}

        def fake_convert(p: Path, **kwargs: object) -> str:
            captured.update(kwargs)
            return "<html></html>"

        monkeypatch.setattr(cli, "lightonocr_pdf_to_html", fake_convert)
        rc = cli.main(
            [
                str(pdf),
                str(out),
                "--vllm-url",
                "http://h/v1",
                "--vllm-model",
                "m",
                "--image-dir",
                str(image_dir),
            ]
        )
        assert rc == 0
        assert out.read_text(encoding="utf-8") == "<html></html>"
        assert captured["base_url"] == "http://h/v1"
        assert captured["model"] == "m"
        assert captured["image_dir"] == image_dir

    def test_missing_pdf_errors_before_conversion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pdfparser.__main__ as cli

        called: list[int] = []
        monkeypatch.setattr(
            cli,
            "lightonocr_pdf_to_html",
            lambda *a, **k: called.append(1) or "",
        )
        # argparse's parser.error exits with code 2 and the convert is never reached
        with pytest.raises(SystemExit) as exc:
            cli.main([str(tmp_path / "absent.pdf")])
        assert exc.value.code == 2
        assert not called
