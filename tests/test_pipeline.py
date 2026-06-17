"""Tests for the pdfparser.pipeline package.  Unit tests load no model and render
no PDF; the integration tests (``@pytest.mark.integration``) run the real pipeline."""

import base64
import io
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _fake_image(width: int = 800, height: int = 1000) -> Image.Image:
    return Image.new("RGB", (width, height), color="white")


def _figure_sizes(html: str) -> list[tuple[int, int]]:
    """Decode every embedded ``data:image/png`` figure and return its (w, h)."""
    uris = re.findall(r"data:image/png;base64,([A-Za-z0-9+/=]+)", html)
    return [Image.open(io.BytesIO(base64.b64decode(u))).size for u in uris]


def _figure_size_by_caption(html: str, needle: str) -> tuple[int, int] | None:
    """Decode the embedded figure whose ``<figcaption>`` contains ``needle`` and
    return its ``(w, h)``; ``None`` if no such figure is present."""
    for fig in re.findall(r"<figure>.*?</figure>", html, re.DOTALL):
        cap = re.search(r"<figcaption>(.*?)</figcaption>", fig, re.DOTALL)
        uri = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", fig)
        if cap and uri and needle in cap.group(1):
            return Image.open(io.BytesIO(base64.b64decode(uri.group(1)))).size
    return None


def _run_lighton(pages_md: list[str], image: Image.Image | None = None) -> str:
    """Assemble HTML from synthetic per-page markdown (no model, no rendering)."""
    from pdfparser.pipeline.assemble import _assemble_html

    img = image or _fake_image(1190, 1540)
    return _assemble_html(pages_md, [img for _ in pages_md])


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


def _body(html: str) -> str:
    """Extract the content of the <div class="body"> element.

    Depth-aware so nested </div> inside model-emitted HTML doesn't truncate.
    """
    start = html.find('<div class="body">')
    assert start >= 0, "body div not found"
    pos = start
    depth = 0
    while pos < len(html):
        if html.startswith("<div", pos):
            depth += 1
            pos += 4
        elif html.startswith("</div>", pos):
            depth -= 1
            if depth == 0:
                return html[start:pos]
            pos += 6
        else:
            pos += 1
    raise AssertionError("unclosed body div")


def _tables_text(html: str) -> str:
    """Visible text rendered *inside* any <table> in the body, tag-stripped.

    Depth-aware (not a `<table>...</table>` regex) so an unclosed table — one the
    OCR left open at a page bottom — is seen to extend to the document end and thus
    captures any following prose it swallows.  Used both to assert real cell content
    is present and to assert post-table prose is *not* absorbed."""
    body = _body(html)
    out: list[str] = []
    depth = 0
    i = 0
    while i < len(body):
        if body[i : i + 6].lower() == "<table":
            depth += 1
            i += 6
        elif body[i : i + 8].lower() == "</table>":
            depth = max(0, depth - 1)
            i += 8
        else:
            if depth > 0:
                out.append(body[i])
            i += 1
    return re.sub(r"<[^>]+>", "", "".join(out))


def _metadata(html: str) -> str:
    """Content of the collapsible <details class='metadata'> panel."""
    start = html.find("<details class='metadata'>")
    assert start >= 0, "metadata panel not found"
    return html[start : html.find("</details>", start)]


class TestArticlePageDetection:
    """Cover ads / mastheads carry no Abstract or Introduction heading, so the
    article start is the first page that does."""

    def test_ad_page_is_not_article(self) -> None:
        from pdfparser.pipeline.classify import _is_article_page_md

        ad = "# Virtual Conference\n\n## Data integrity seminar\n\nRegister here."
        assert _is_article_page_md(ad) is False

    def test_abstract_page_is_article(self) -> None:
        from pdfparser.pipeline.classify import _is_article_page_md

        assert _is_article_page_md("# Title\n\n## Abstract\n\nWe did things.") is True

    def test_introduction_page_is_article(self) -> None:
        from pdfparser.pipeline.classify import _is_article_page_md

        assert _is_article_page_md("## 1. Introduction\n\nText.") is True

    def test_leading_ad_page_skipped(self) -> None:
        from pdfparser.pipeline.classify import _leading_pages_to_skip_md

        ad = "# Conference\n\nRegister here."
        article = "# Real Title\n\n## Abstract\n\nBody."
        assert _leading_pages_to_skip_md([ad, article]) == 1
        assert _leading_pages_to_skip_md([article]) == 0


class TestRunningFurniture:
    """Short header/footer lines that recur across pages — even with differing
    page numbers — are dropped; real repeated sentences are kept."""

    def test_page_numbered_footer_removed(self) -> None:
        from pdfparser.pipeline.classify import _strip_running_furniture

        parts = [
            "<p>Biotechnology and Applied Biochemistry 601</p>",
            "<p>Real body sentence one.</p>",
            "<p>Biotechnology and Applied Biochemistry 602</p>",
        ]
        out = _strip_running_furniture(parts)
        assert out == ["<p>Real body sentence one.</p>"]

    def test_repeated_real_sentence_kept(self) -> None:
        from pdfparser.pipeline.classify import _strip_running_furniture

        parts = ["<p>This is a sentence.</p>", "<p>This is a sentence.</p>"]
        assert _strip_running_furniture(parts) == parts

    def test_short_enumerated_labels_kept(self) -> None:
        # "Fig 1"/"Fig 2" share a digit-stripped key but must not be removed —
        # only substantial recurring text (a journal footer) is furniture.
        from pdfparser.pipeline.classify import _strip_running_furniture

        parts = ["<p>Fig 1</p>", "<p>body</p>", "<p>Fig 2</p>"]
        assert _strip_running_furniture(parts) == parts

    def test_short_digit_free_footer_removed(self) -> None:
        # A bare author-surname running foot ("Clark" on alternating pages) is
        # short but digit-free, so the digit-strip collision the length floor
        # guards against can't happen — it must still be recognised as furniture.
        from pdfparser.pipeline.classify import _strip_running_furniture

        parts = [
            "<p>Clark</p>",
            "<p>Real body sentence one.</p>",
            "<p>Clark</p>",
        ]
        assert _strip_running_furniture(parts) == ["<p>Real body sentence one.</p>"]

    def test_heading_form_footer_removed(self) -> None:
        # OCR transcribes the running journal line as a heading on a sparse page
        # (last page / after references); it must still count as furniture and be
        # stripped, not survive as an <h1>.
        from pdfparser.pipeline.classify import _strip_running_furniture

        parts = [
            "<p>Biotechnology and Applied Biochemistry 601</p>",
            "<p>Real body sentence one.</p>",
            "<h1>Biotechnology and Applied Biochemistry</h1>",
        ]
        out = _strip_running_furniture(parts)
        assert out == ["<p>Real body sentence one.</p>"]

    def test_abbreviation_terminated_running_head_removed(self) -> None:
        # A running head ending in an abbreviation ("… Sphingomonas sp.") only
        # looks like a finished sentence; recurring on 3+ pages, it is furniture
        # and must be stripped — otherwise it interleaves between a paragraph's
        # halves and blocks their cross-page merge.
        from pdfparser.pipeline.classify import _strip_running_furniture

        head = "<p>A ribitol dehydrogenase from <em>Sphingomonas</em> sp.</p>"
        parts = [head, "<p>Body one.</p>", head, "<p>Body two.</p>", head]
        body = ["<p>Body one.</p>", "<p>Body two.</p>"]
        assert _strip_running_furniture(parts) == body

    def test_twice_repeated_sentence_like_line_kept(self) -> None:
        # The same abbreviation-terminated line appearing only twice stays: two
        # occurrences are too few to outweigh its sentence-like shape, matching
        # the repeated-real-sentence guard.
        from pdfparser.pipeline.classify import _strip_running_furniture

        head = "<p>A ribitol dehydrogenase from <em>Sphingomonas</em> sp.</p>"
        parts = [head, "<p>Body one.</p>", head]
        assert _strip_running_furniture(parts) == parts

    def test_heading_repeated_only_as_heading_kept(self) -> None:
        # A section heading the article legitimately repeats ("Purification of
        # SpRDH" under both Methods and Results) recurs but never appears as a
        # plain paragraph, so it is structure, not a running header, and must
        # survive in both places.
        from pdfparser.pipeline.classify import _strip_running_furniture

        parts = [
            "<h3>Purification of SpRDH</h3>",
            "<p>Real body sentence one.</p>",
            "<h2>Purification of SpRDH</h2>",
        ]
        assert _strip_running_furniture(parts) == parts

    def test_standalone_page_number_removed(self) -> None:
        # OCR sometimes isolates the folio into its own block, away from the
        # journal line, so digit-stripped recurrence can't catch it; a number-only
        # block is the page number itself and must be dropped.
        from pdfparser.pipeline.classify import _strip_running_furniture

        parts = ["<p>601</p>", "<p>Real body sentence one.</p>", "<h2>602</h2>"]
        assert _strip_running_furniture(parts) == ["<p>Real body sentence one.</p>"]

    def test_section_number_kept(self) -> None:
        # A numbered section heading ("3.4 …") is not a bare folio and stays.
        from pdfparser.pipeline.classify import _strip_running_furniture

        parts = ["<h2>3.4 Enzymatic activities</h2>", "<p>4</p>"]
        assert _strip_running_furniture(parts) == ["<h2>3.4 Enzymatic activities</h2>"]


class TestByline:
    """The block after the title becomes the header byline only when it
    positively looks like authors; otherwise it stays in the body."""

    def test_marker_line_is_byline(self) -> None:
        from pdfparser.pipeline.classify import _is_byline

        assert _is_byline("Nianyang Wu¹") is True
        assert _is_byline("Daniel D. Clark <sup>*</sup>") is True

    def test_name_list_is_byline(self) -> None:
        from pdfparser.pipeline.classify import _is_byline

        assert _is_byline("Jane Doe and John Smith") is True

    def test_metadata_lines_are_not_byline(self) -> None:
        from pdfparser.pipeline.classify import _is_byline

        assert _is_byline("Received 26 March 2019") is False
        assert _is_byline("DOI: 10.1002/bab.1760") is False
        assert _is_byline("This is a complete sentence.") is False

    def test_unmarked_single_name_is_not_byline(self) -> None:
        # No marker and not a list → ambiguous → not promoted (stays in body).
        from pdfparser.pipeline.classify import _is_byline

        assert _is_byline("Jane Doe") is False

    def test_metadata_after_title_goes_to_metadata_panel(self) -> None:
        # A date line under the title is metadata: it must not be promoted into
        # the header (and must not be lost) — it belongs in the Metadata panel.
        md = (
            "# T\n\nReceived 26 March 2019\n\n## Abstract\n\nThe abstract.\n\n"
            "## Methods\n\nMethod text."
        )
        html = _run_lighton([md])
        header = html[html.find("<header>") : html.find("</header>")]
        assert "Received 26 March 2019" in _metadata(html)
        assert "Received 26 March 2019" not in header
        assert "Received 26 March 2019" not in _body(html)

    def test_marked_authors_after_title_are_promoted(self) -> None:
        md = "# T\n\nNianyang Wu¹, Xiaoqiang Liu¹*\n\n## Abstract\n\nA."
        html = _run_lighton([md])
        header = html[html.find("<header>") : html.find("</header>")]
        assert "Nianyang Wu" in header
        assert "Nianyang Wu" not in _body(html)


class TestDegenerateRepetition:
    """A figure the model fails to box can be OCRed into a repeated-token wall;
    such a paragraph is dropped from the body, real prose is kept."""

    def test_token_wall_flagged(self) -> None:
        from pdfparser.pipeline.classify import _is_degenerate_repetition

        assert _is_degenerate_repetition("AaTRI, " * 40) is True

    def test_real_prose_not_flagged(self) -> None:
        from pdfparser.pipeline.classify import _is_degenerate_repetition

        prose = (
            "The enzyme catalyzes the stereospecific oxidation of the substrate"
            " to the corresponding ketone under physiological conditions."
        )
        assert _is_degenerate_repetition(prose) is False

    def test_token_wall_dropped_from_body(self) -> None:
        wall = "AaTRI, " * 50
        md = f"# T\n\n## Abstract\n\nA.\n\n## Body\n\n{wall}\n\nReal sentence here."
        body = _body(_run_lighton([md]))
        assert "AaTRI" not in body
        assert "Real sentence here." in body


class TestLightonAssembly:
    """End-to-end markdown → HTML assembly (the new model-free seam)."""

    def test_title_skips_document_type_label(self) -> None:
        # "# Article" is a document-type label; the real title is the next heading.
        md = "# Article\n\n## The Real Title\n\nA. Author\n\n### Abstract\n\nText."
        html = _run_lighton([md])
        assert "<h1>The Real Title</h1>" in html
        assert "Article" not in _header_h1(html)

    def test_byline_extracted_and_dropped_from_body(self) -> None:
        md = "# A Study of Things\n\nJane Doe¹\n\n## Abstract\n\nThe abstract body."
        html = _run_lighton([md])
        header = html[html.find("<header>") : html.find("</header>")]
        assert "Jane Doe" in header
        assert "Jane Doe" not in _body(html)

    def test_abstract_wrapped_in_section(self) -> None:
        md = (
            "# T\n\nA. U.\n\n## Abstract\n\nThe abstract paragraph here.\n\n"
            "## Body\n\nProse."
        )
        html = _run_lighton([md])
        start = html.find("<section class='abstract'>")
        end = html.find("</section>", start)
        assert "The abstract paragraph here." in html[start:end]
        assert "Prose." not in html[start:end]

    def test_leading_superscript_routed_to_footnote_before_refs(self) -> None:
        md = (
            "# T\n\n## Abstract\n\nAbstract.\n\n## Body\n\n"
            "<sup>*</sup>To whom correspondence should be addressed.\n\n"
            "## References\n\n[1] A reference."
        )
        html = _run_lighton([md])
        fn = html.find("To whom correspondence")
        ref = html.find("[1] A reference")
        assert 0 < fn < ref
        assert 'class="footnote"' in html

    def test_frontmatter_moved_to_metadata_panel_after_abstract(self) -> None:
        # Affiliations, keywords and abbreviations are OCR'd between the abstract
        # and the body's first section; they are pulled into the collapsible
        # Metadata panel (after the abstract) so the body opens with prose.
        md = (
            "# A Study\n\nJane Doe¹\n\n"
            "¹Department of Examples, Example University\n\n"
            "## Abstract\n\nThe abstract.\n\n"
            "**Keywords:** alpha, beta, gamma\n\n"
            "## Abbreviations\n\nTRI, tropine reductase.\n\n"
            "## 1. Introduction\n\nThe study begins here.\n\n"
            "## References\n\n[1] A reference."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        # The front matter is in the Metadata panel, not the body.
        assert "<summary>Metadata</summary>" in meta
        for fragment in (
            "Department of Examples",
            "Keywords:",
            "TRI, tropine reductase.",
        ):
            assert fragment in meta
            assert fragment not in body
        # The panel sits after the abstract and before the body.
        assert (
            html.find("</section>")
            < html.find("<details class='metadata'>")
            < html.find('<div class="body">')
        )
        # The body opens with prose (the Introduction), not metadata.
        assert "<h2>1. Introduction</h2>" in body

    def test_frontmatter_boundary_is_name_agnostic(self) -> None:
        # The boundary is the first non-metadata heading, not a literal
        # "Introduction" — a body opening with "Background" works the same.
        md = (
            "# A Study\n\n"
            "**Keywords:** alpha, beta\n\n"
            "## Background\n\nThe study begins here.\n\n"
            "## References\n\n[1] A reference."
        )
        html = _run_lighton([md])
        assert "Keywords:" in _metadata(html)
        assert "Keywords:" not in _body(html)
        assert "<h2>Background</h2>" in _body(html)

    def test_frontmatter_unchanged_when_body_opens_with_section(self) -> None:
        # First block is already a body section heading → nothing precedes it →
        # order left intact.
        md = "# T\n\n## Abstract\n\nA.\n\n## Methods\n\nFirst.\n\nSecond."
        body = _body(_run_lighton([md]))
        assert body.find("First.") < body.find("Second.")

    def test_unlabeled_body_prose_not_relocated(self) -> None:
        # An article whose body opens with unlabelled prose (no "Introduction"
        # heading, first heading is "Methods") must not have that prose moved to
        # the end: only positively-recognised metadata is relocated.
        md = (
            "# T\n\nUnlabeled opening prose paragraph here.\n\n"
            "More opening prose follows.\n\n"
            "## Methods\n\nMethod text.\n\n## References\n\n[1] A reference."
        )
        body = _body(_run_lighton([md]))
        assert body.find("Unlabeled opening prose paragraph here.") < body.find(
            "<h2>Methods</h2>"
        )

    def test_prose_starting_with_metadata_keyword_not_hidden(self) -> None:
        # A leading body paragraph that merely begins with a metadata keyword
        # ("Published…") is a sentence, not a front-matter label — it must stay
        # visible in the body, not be hidden in the Metadata panel.  (A keywords
        # label closes the abstract so the prose is the next, leading body block.)
        md = (
            "# T\n\n## Abstract\n\nThe abstract.\n\n**Keywords:** alpha, beta\n\n"
            "Published studies have shown that the enzyme is active.\n\n"
            "## Methods\n\nMethod text."
        )
        html = _run_lighton([md])
        assert "Published studies have shown" in _body(html)
        assert "Published studies have shown" not in _metadata(html)

    def test_unheaded_prose_after_metadata_heading_not_hidden(self) -> None:
        # Sticky metadata-section capture must stop at a real prose paragraph, so
        # an unheaded opening section after "## Keywords" is not swallowed.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Keywords\n\nalpha, beta, gamma.\n\n"
            "The introduction begins here without its own heading and reads as"
            " ordinary prose.\n\n## Methods\n\nMethod text."
        )
        html = _run_lighton([md])
        assert "The introduction begins here" in _body(html)
        assert "The introduction begins here" not in _metadata(html)
        assert "alpha, beta, gamma." in _metadata(html)  # short keywords stay metadata

    def test_long_abbreviation_list_stays_in_panel(self) -> None:
        # A semicolon-separated abbreviation list ending in a period reads like a
        # sentence, but as the block directly under "## Abbreviations" it is the
        # heading's own content and belongs in the panel, not the body.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Abbreviations\n\n"
            "TAs, tropane alkaloids; TRI, tropine-forming reductase; "
            "GC-MS, gas chromatography-mass spectrometer.\n\n"
            "## 1. Introduction\n\nThe study begins here without abbreviation.\n\n"
            "## References\n\n[1] A reference."
        )
        html = _run_lighton([md])
        assert "TAs, tropane alkaloids" in _metadata(html)
        assert "TAs, tropane alkaloids" not in _body(html)
        assert "<h2>1. Introduction</h2>" in _body(html)

    def test_multi_block_abbreviation_list_fully_in_panel(self) -> None:
        # A long abbreviation list OCR-split across two paragraphs: both halves
        # are ";"-separated lists, so the heading owns the whole run and neither
        # half leaks into the body (the second block is not heading-adjacent).
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Abbreviations\n\n"
            "TAs, tropane alkaloids; TRI, tropine-forming reductase; "
            "TRII, pseudotropine-forming reductase; qPCR, quantitative PCR.\n\n"
            "PGK, protein kinase; SDRs, short-chain dehydrogenases; "
            "GC-MS, gas chromatography-mass spectrometer; HPLC, chromatography.\n\n"
            "## 1. Introduction\n\nThe study begins here without abbreviation."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        for fragment in ("TAs, tropane alkaloids", "PGK, protein kinase"):
            assert fragment in meta
            assert fragment not in body
        assert "<h2>1. Introduction</h2>" in body

    def test_keyword_led_metadata_after_section_pulled_into_panel(self) -> None:
        # Correspondence / received / DOI lines OCR'd after the abbreviation list
        # are keyword-led metadata; inside the headed front-matter region they are
        # pulled into the panel even though each ends like a sentence.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Abbreviations\n\n"
            "TAs, tropane alkaloids; GC-MS, gas chromatography-mass spectrometer.\n\n"
            "Address for correspondence: Jane Doe, Example University, City. "
            "Tel. 123; e-mail: jane@example.edu.\n\n"
            "Received 26 March 2019; accepted 29 April 2019 DOI: 10.1002/bab.1760.\n\n"
            "## 1. Introduction\n\nThe study begins here without abbreviation."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        for fragment in ("Address for correspondence", "Received 26 March 2019"):
            assert fragment in meta
            assert fragment not in body
        assert "<h2>1. Introduction</h2>" in body

    def test_keyword_led_body_prose_in_section_not_hidden(self) -> None:
        # A body paragraph that merely opens with a front-matter keyword
        # ("Published …"), appearing inside a headed metadata section, must stay
        # in the body — it carries no metadata token (e-mail/DOI/date), unlike a
        # genuine "Received … DOI: …" line, so it is not hidden in the panel.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Abbreviations\n\n"
            "TAs, tropane alkaloids; GC-MS, gas chromatography-mass spectrometer.\n\n"
            "Published reports indicate that the enzyme is active across many "
            "plant species and remains the focus of ongoing research.\n\n"
            "## Methods\n\nMethod text."
        )
        html = _run_lighton([md])
        assert "Published reports indicate" in _body(html)
        assert "Published reports indicate" not in _metadata(html)

    def test_midbody_abbreviations_section_pulled_into_panel(self) -> None:
        # OCR places the Abbreviations section *after* the Introduction prose on
        # the first page, out of the leading front-matter run.  Its heading and
        # ";"-separated glossary are pulled into the panel; the misfiled body
        # content that follows (a learning-objectives list, the next section) is
        # left visible.
        md = (
            "# A Study\n\n## Abstract\n\nThe abstract.\n\n"
            "## Introduction\n\nThe study begins here with ordinary prose.\n\n"
            "## Abbreviations\n\n"
            "2-HEC, 2-(2-hydroxyethylthio)ethanesulfonate; CoM, coenzyme M; "
            "EE, enantiomeric excess; HPC, hydroxypropyl thioether.\n\n"
            "1. Students improve their appreciation for kinetic data.\n\n"
            "## Methods\n\nMethod text."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        assert "<h2>Abbreviations</h2>" in meta
        assert "2-HEC, 2-(2-hydroxyethylthio)ethanesulfonate" in meta
        assert "2-HEC, 2-(2-hydroxyethylthio)ethanesulfonate" not in body
        # The Introduction prose and the misfiled learning-objective stay visible.
        assert "The study begins here" in body
        assert "Students improve their appreciation" in body
        assert "<h2>Methods</h2>" in body

    def test_midbody_named_metadata_scoped_to_first_page(self) -> None:
        # A same-named section deeper in the document (here on a later page, e.g. a
        # back-matter glossary) is left in the body — only the first article page
        # is scanned for misplaced metadata sections.
        page1 = "# A Study\n\n## Abstract\n\nThe abstract.\n\n## Introduction\n\nProse."
        page2 = (
            "## Nomenclature\n\n"
            "F, force in newtons; m, mass in kilograms; a, acceleration.\n\n"
            "## References\n\n[1] A reference."
        )
        body = _body(_run_lighton([page1, page2]))
        assert "<h2>Nomenclature</h2>" in body
        assert "F, force in newtons" in body

    def test_midbody_bare_named_heading_stays_in_body(self) -> None:
        # A "Nomenclature" heading that opens a real prose section (no glossary
        # content under it) is a section title, not front matter: leave it visible.
        md = (
            "# A Study\n\n## Abstract\n\nThe abstract.\n\n"
            "## Introduction\n\nProse opening the article body.\n\n"
            "## Nomenclature\n\n"
            "This section explains the naming conventions used throughout the "
            "paper in ordinary prose that reads as a real section.\n\n"
            "## Methods\n\nMethod text."
        )
        body = _body(_run_lighton([md]))
        assert "<h2>Nomenclature</h2>" in body
        assert "This section explains the naming conventions" in body

    def test_stray_footer_metadata_pulled_into_panel(self) -> None:
        # The first page's bottom-of-page footer (journal citation + correspondence,
        # a supporting-info note, a submission/DOI line) is OCR'd into the body
        # after the Introduction.  Each self-contained metadata line is relocated to
        # the panel; the body prose that follows stays visible.
        md = (
            "# A Study\n\n## Abstract\n\nThe abstract.\n\n"
            "## Introduction\n\nIntro paragraph one is reasonably long prose here.\n\n"
            "Volume 47, Number 2, March/April 2019, Pages 124-132 *To whom "
            "correspondence should be addressed. Daniel D. Clark, Tel.: "
            "(530)-898-5251. E-mail: ddclark@csuchico.edu.\n\n"
            "Additional Supporting Information may be found in the online version "
            "of this article.\n\n"
            "Received 19 June 2018; Revised 23 August 2018; Accepted 6 December "
            "2018 DOI 10.1002/bmb.21202 Published online 28 December 2018 in Wiley "
            "Online Library (wileyonlinelibrary.com)\n\n"
            "Herein, I propose that data from the characterization can augment the "
            "teaching of enzyme kinetics. The case study had five goals in mind.\n\n"
            "## Methods\n\nMethod text."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        for fragment in (
            "Volume 47",
            "Additional Supporting Information",
            "Received 19 June",
        ):
            assert fragment in meta
            assert fragment not in body
        # Running before the merge keeps the ")" -terminated footer line from
        # absorbing the body prose that follows it.
        assert "Herein, I propose" in body
        assert "Herein, I propose" not in meta

    def test_interleaved_publication_sidebar_pulled_into_panel(self) -> None:
        # 32639976 (PLOS ONE) page 0: the left-column metadata sidebar — a
        # postal-code-less affiliation plus a run of bold label-colon lines
        # (Citation/Editor/.../Competing interests) — is OCR'd into the body, the
        # affiliation as the body's first block and the labelled run stranded after
        # the Introduction heading and a column-break rule.  Every piece belongs in
        # the panel; the Introduction prose stays in the body.
        md = (
            "# Purification of a novel ribitol dehydrogenase\n\n"
            "**Kiet N. Tran, Nhung Pham, Sei-Heon Jang\\*, ChangWoo Lee\\***\n\n"
            "*Department of Biomedical Science and Center for Bio-Nanomaterials, "
            "Daegu University, Gyeongsan, South Korea*\n\n"
            "## Abstract\n\nThe abstract sentence is here.\n\n"
            "## Introduction\n\n"
            "Lichens have traditionally been considered a symbiotic association.\n\n"
            "---\n\n"
            "**Citation:** Tran KN, Lee C (2020) Purification. PLoS ONE 15(7): "
            "e0235718. https://doi.org/10.1371/journal.pone.0235718\n\n"
            "**Editor:** Leonidas Matsakas, Luleå University of Technology, SWEDEN\n\n"
            "**Received:** April 23, 2020\n\n"
            "**Copyright:** © 2020 Tran et al. This is an open access article.\n\n"
            "**Funding:** This work was supported by the NRF.\n\n"
            "**Competing interests:** The authors have declared that no competing "
            "interests exist.\n\n"
            "Polyols have a role in carbohydrate storage and stress protection."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        for fragment in (
            "Department of Biomedical Science",
            "<strong>Citation:</strong>",
            "<strong>Editor:</strong>",
            "<strong>Received:</strong>",
            "<strong>Copyright:</strong>",
            "<strong>Funding:</strong>",
            "<strong>Competing interests:</strong>",
        ):
            assert fragment in meta, fragment
            assert fragment not in body, fragment
        # The body keeps its prose and the section heading, with no stray rule.
        assert "Lichens have traditionally been considered" in body
        assert "Polyols have a role" in body
        assert "<hr" not in body

    def test_body_sentence_with_one_email_not_hidden(self) -> None:
        # The stray-metadata sweep must not hide a body sentence that merely embeds
        # a single address: one token is below the threshold.
        md = (
            "# A Study\n\n## Abstract\n\nThe abstract.\n\n"
            "## Introduction\n\n"
            "Raw data are available on request from the author at "
            "ddclark@csuchico.edu.\n\n"
            "## Methods\n\nMethod text."
        )
        body = _body(_run_lighton([md]))
        assert "Raw data are available on request" in body

    def test_stray_metadata_predicate(self) -> None:
        from pdfparser.pipeline.classify import _is_stray_metadata

        # Two tokens (tel + e-mail) → relocated.
        assert _is_stray_metadata(
            "<p>*To whom correspondence should be addressed. Tel.: (530)-898-5251. "
            "E-mail: ddclark@csuchico.edu.</p>"
        )
        # Boilerplate phrase, no token → relocated.
        assert _is_stray_metadata(
            "<p>Additional Supporting Information may be found in the online "
            "version of this article.</p>"
        )
        # Single-token publication lines the OCR splits off the page-bottom block,
        # each unambiguous on its own shape → relocated.
        assert _is_stray_metadata(
            "<p>Volume 47, Number 2, March/April 2019, Pages 124-132</p>"
        )
        assert _is_stray_metadata("<p>DOI 10.1002/bmb.21202</p>")
        assert _is_stray_metadata(
            "<p>Published online 28 December 2018 in Wiley Online Library "
            "(wileyonlinelibrary.com)</p>"
        )
        # A recognised journal-metadata bold label is relocated on the label alone,
        # even with no metadata token and even when the value runs long.
        assert _is_stray_metadata(
            "<p><strong>Competing interests:</strong> The authors have declared "
            "that no competing interests exist.</p>"
        )
        assert _is_stray_metadata(
            "<p><strong>Editor:</strong> Leonidas Matsakas, Luleå University of "
            "Technology, SWEDEN</p>"
        )
        assert _is_stray_metadata(
            "<p><strong>Funding:</strong> "
            + "This work was generously supported by many sources. " * 12
            + "</p>"
        )
        # A bold label-colon that is *not* a publishing-process label is ordinary
        # body emphasis, not metadata, and must not be relocated on the shape alone.
        assert not _is_stray_metadata(
            "<p><strong>Note:</strong> the reaction was repeated three times.</p>"
        )
        assert not _is_stray_metadata(
            "<p><strong>Results:</strong> the enzyme retained full activity.</p>"
        )
        # A single embedded e-mail is below the two-token bar → stays in body.
        assert not _is_stray_metadata(
            "<p>Raw data are available from the author at jane@example.edu.</p>"
        )
        # Prose that merely uses the words "volume"/"pages" or "published online"
        # — not the citation/publication shapes — stays in body.
        assert not _is_stray_metadata(
            "<p>The dataset volume reached 47 GB after we processed pages from "
            "124 to 132 of the log.</p>"
        )
        assert not _is_stray_metadata(
            "<p>These results were later published online for peer review.</p>"
        )
        # A body sentence that inline-cites a volume/pages reference is prose, not
        # a citation block: the journal shape is anchored at the block start, so
        # "Volume 47 … Pages 124" matches but a mid-sentence citation does not.
        assert not _is_stray_metadata(
            "<p>See volume 3, pages 45-67, for the original derivation.</p>"
        )
        assert not _is_stray_metadata(
            "<p>In volume 12, pages 8-9 of the proceedings, the method appeared.</p>"
        )
        assert not _is_stray_metadata(
            "<p>As shown in Vol. 5 pp. 30 onward, the trend continues.</p>"
        )
        # A long prose run with two tokens is not a footer line → stays in body.
        assert not _is_stray_metadata(
            "<p>" + "Lorem ipsum dolor sit amet. " * 20 + "Contact a@b.edu or "
            "c@d.edu.</p>"
        )

    def test_bare_affiliation_line_pulled_into_panel(self) -> None:
        # An author+affiliation line OCR'd between the title and the abstract,
        # without its author's superscript marker ("Name From the Department …,
        # City, Region, postcode"), is recognised structurally and pulled into the
        # panel, so it no longer breaks the leading run and strands the keywords.
        md = (
            "# A Study\n\n"
            "Daniel D. Clark From the Department of Chemistry and Biochemistry, "
            "California State University-Chico, Chico, California, 95929\n\n"
            "## Abstract\n\nThe abstract.\n\n"
            "**Keywords:** enzymology; enzyme kinetics; dehydrogenase\n\n"
            "## Introduction\n\nThe study begins here with ordinary prose.\n\n"
            "## References\n\n[1] A reference."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        for fragment in ("From the Department of Chemistry", "Keywords:"):
            assert fragment in meta
            assert fragment not in body
        assert "The study begins here" in body
        assert "<h2>Introduction</h2>" in body

    def test_body_sentence_mentioning_university_not_hidden(self) -> None:
        # The affiliation detector must not hide a body sentence that merely names
        # an institution: a terminal period marks it as prose, not an address.
        md = (
            "# A Study\n\n"
            "The work was carried out with the University of Example, the "
            "Department of Chemistry, and several partners.\n\n"
            "## Methods\n\nMethod text."
        )
        body = _body(_run_lighton([md]))
        assert "The work was carried out with the University of Example" in body

    def test_truncated_body_fragment_with_institution_not_hidden(self) -> None:
        # An OCR-truncated prose clause that names institutions and lacks terminal
        # punctuation must stay visible: without a postal-code tail it is not an
        # address.  (Guards against the false positive the no-punctuation rule
        # alone allowed.)
        md = (
            "# A Study\n\n"
            "In this work, conducted jointly with the Department of Biology, the "
            "School of Medicine, and several partner hospitals across the region\n\n"
            "## Methods\n\nMethod text."
        )
        body = _body(_run_lighton([md]))
        assert "In this work, conducted jointly" in body

    def test_affiliation_line_predicate(self) -> None:
        from pdfparser.pipeline.classify import _is_affiliation_line

        assert _is_affiliation_line(
            "Daniel D. Clark From the Department of Chemistry and Biochemistry, "
            "California State University-Chico, Chico, California, 95929"
        )
        # A terminal period marks prose, not an address.
        assert not _is_affiliation_line(
            "The work was done at the University of Example, City, Region."
        )
        # No institution keyword.
        assert not _is_affiliation_line("Jane Doe, John Smith, and Mary Major")
        # Too few comma-separated segments to be an address layout.
        assert not _is_affiliation_line("Department of Chemistry")
        # An institution-naming prose clause with no postal-code tail is not an
        # address, even without terminal punctuation — it does not *open* with the
        # institution name, so the head-anchored affiliation cue does not fire.
        assert not _is_affiliation_line(
            "In this work, conducted jointly with the Department of Biology, the "
            "School of Medicine, and several partner hospitals across the region"
        )
        # A number earlier in the line cannot stand in for the address tail.
        assert not _is_affiliation_line(
            "enrolled 250 patients from the Department of Cardiology, the ICU, "
            "and two partner clinics"
        )
        # An address that *opens* with the institution name and *closes* on a place
        # name is an affiliation even with no postal code — international addresses
        # often end on a country, not a code.  Both ends are load-bearing: the prose
        # clauses above open with the keyword too but run on into lowercase words
        # instead of closing on a city/country, so they stay in the body.
        assert _is_affiliation_line(
            "Department of Chemistry, University of Oxford, Oxford, United Kingdom"
        )
        assert _is_affiliation_line(
            "Department of Biomedical Science and Center for Bio-Nanomaterials, "
            "Daegu University, Gyeongsan, South Korea"
        )
        # International laboratory spellings (laboratoire/laboratorio) share the
        # stem, so the head cue fires for them as well.
        assert _is_affiliation_line(
            "Laboratoire de Biologie Moléculaire, Université de Paris, Paris, France"
        )
        assert _is_affiliation_line(
            "Laboratorio de Química, Universidad de Madrid, Madrid, Spain"
        )
        # A lowercase connector inside the closing place name ("Republic of Korea")
        # is allowed; the tail must only *open* on a capital.
        assert _is_affiliation_line(
            "Institute of Microbiology, Korea University, Seoul, Republic of Korea"
        )
        # Opening with the keyword is not enough when the line runs on into a
        # lowercase clause instead of closing on a place — kept visible in the body.
        assert not _is_affiliation_line(
            "University researchers, clinicians, and administrators worked across "
            "the region and beyond"
        )
        # The country/region tail is the language-independent signal: it fires even
        # when the institution word is in a language no stem covers (German "Labor"/
        # "Fakultät", Czech "Ústav"/"Katedra"), so these are recognised too.
        assert _is_affiliation_line(
            "Labor für Mikrobiologie, Klinikum der Universität, Berlin, Germany"
        )
        assert _is_affiliation_line("Fakultät für Chemie, Wien, Österreich")
        assert _is_affiliation_line(
            "Ústav organické chemie, Univerzita Karlova, Praha, Czech Republic"
        )
        assert _is_affiliation_line("Katedra biologie, Praha, Česko")
        # An uppercase country tail ("…, SWEDEN") still matches (OCR casing varies).
        assert _is_affiliation_line(
            "Department of Chemistry, Luleå University of Technology, SWEDEN"
        )
        # A prose clause that merely *mentions* countries but does not *close* on a
        # bare country name (the final segment is "and Spain", not "Spain") stays
        # in the body.
        assert not _is_affiliation_line("We ran trials in France, Germany, and Spain")

    def test_all_frontmatter_body_kept_visible(self) -> None:
        # If every body block looks like front matter, that signals misdetection,
        # not a metadata-only doc: keep it visible rather than emptying the body.
        md = (
            "# T\n\n## Abstract\n\nThe abstract.\n\n**Keywords:** alpha, beta\n\n"
            "¹Affiliation One, City\n\nReceived 26 March 2019"
        )
        html = _run_lighton([md])
        assert "Affiliation One" in _body(html)
        assert "<details class='metadata'>" not in html

    def test_figure_placeholder_becomes_cropped_figure(self) -> None:
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\nBefore.\n\n"
            "![image](image_1.png)100,100,900,900\n\n"
            "FIG. 1 A nice figure caption.\n\nAfter."
        )
        html = _run_lighton([md])
        assert "data:image/png;base64," in html
        assert "<figcaption>FIG. 1 A nice figure caption.</figcaption>" in html
        assert "100,100,900,900" not in html

    def test_figure_bbox_is_denormalized_to_full_extent(self) -> None:
        # A full box in [0,1000] space must crop the whole page, not a top-left
        # sliver (the truncation bug: coords are normalized, not pixels).
        img = _fake_image(1190, 1540)
        md = "# T\n\n## Abstract\n\nA.\n\n## Body\n\n![image](i.png)0,0,1000,1000"
        html = _run_lighton([md], image=img)
        assert _figure_sizes(html) == [(1190, 1540)]


class TestDenormalizeBbox:
    """[0,1000]-normalized model boxes scale to the image's pixel size."""

    def test_full_box_maps_to_full_image(self) -> None:
        from pdfparser.pipeline.figures import _denormalize_bbox

        assert _denormalize_bbox((0, 0, 1000, 1000), _fake_image(1190, 1540)) == (
            0,
            0,
            1190,
            1540,
        )

    def test_half_box(self) -> None:
        from pdfparser.pipeline.figures import _denormalize_bbox

        assert _denormalize_bbox((0, 0, 500, 500), _fake_image(1000, 2000)) == (
            0,
            0,
            500,
            1000,
        )


class TestFigureBoxMerge:
    """A figure the model over-segments into stacked boxes is unioned into one
    crop; genuinely separate figures stay separate."""

    def test_same_column_adjacent_boxes_merge(self) -> None:
        from pdfparser.pipeline.figures import _figures_same

        assert _figures_same((100, 100, 900, 500), (110, 500, 890, 560), 50.0) is True

    def test_vertically_separated_boxes_do_not_merge(self) -> None:
        from pdfparser.pipeline.figures import _figures_same

        assert _figures_same((100, 100, 900, 300), (100, 800, 900, 950), 50.0) is False

    def test_side_by_side_boxes_do_not_merge(self) -> None:
        from pdfparser.pipeline.figures import _figures_same

        assert _figures_same((0, 0, 100, 500), (200, 0, 300, 500), 50.0) is False

    def test_union_box(self) -> None:
        from pdfparser.pipeline.figures import _union_box

        assert _union_box([(100, 100, 900, 500), (120, 480, 880, 560)]) == (
            100,
            100,
            900,
            560,
        )

    def test_split_figure_emits_single_crop(self) -> None:
        img = _fake_image(1190, 1540)
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\n"
            "![image](a.png)100,100,900,500\n\n"
            "![image](b.png)100,500,900,560\n\n"
            "FIG. 1 One caption."
        )
        sizes = _figure_sizes(_run_lighton([md], image=img))
        assert len(sizes) == 1
        # The union spans both boxes (down to y≈862 px), not just the first.
        assert sizes[0][1] > 700

    def test_two_separated_figures_stay_separate(self) -> None:
        img = _fake_image(1190, 1540)
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\n"
            "![image](a.png)100,100,900,300\n\n"
            "Some intervening prose between the two figures.\n\n"
            "![image](b.png)100,800,900,950"
        )
        assert len(_figure_sizes(_run_lighton([md], image=img))) == 2


class TestFigureBottomGrowth:
    """The crop grows down over contiguous ink to the figure's true bottom and
    stops at the whitespace before the caption; a box already ending in
    whitespace grows nothing, so caption text is never pulled in."""

    @staticmethod
    def _image() -> Image.Image:
        # White page: figure block y[100,300), caption block y[360,380),
        # separated by a 60 px whitespace gap.
        img = Image.new("RGB", (400, 800), "white")
        img.paste(Image.new("RGB", (300, 200), "black"), (50, 100))
        img.paste(Image.new("RGB", (300, 20), "black"), (50, 360))
        return img

    def test_tight_box_grows_to_figure_bottom(self) -> None:
        from pdfparser.pipeline.figures import _extend_bottom_to_content

        assert _extend_bottom_to_content(self._image(), 50, 350, 250) == 300

    def test_box_at_bottom_does_not_grow(self) -> None:
        from pdfparser.pipeline.figures import _extend_bottom_to_content

        assert _extend_bottom_to_content(self._image(), 50, 350, 300) == 300

    def test_no_growth_when_ink_runs_without_gap(self) -> None:
        # Ink continues past the search window with no whitespace gap (caption /
        # body text below a correct box) → ambiguous → leave the box unchanged.
        from pdfparser.pipeline.figures import _extend_bottom_to_content

        img = Image.new("RGB", (400, 800), "white")
        img.paste(Image.new("RGB", (300, 300), "black"), (50, 100))
        assert _extend_bottom_to_content(img, 50, 350, 250) == 250

    def test_narrow_content_below_box_is_not_read_as_gap(self) -> None:
        # A figure tail narrower than the box (here 3 px of a 300 px-wide box,
        # ~1% ink) must count as content, not be mistaken for the whitespace gap
        # — otherwise the clipped bottom is dropped.
        from pdfparser.pipeline.figures import _extend_bottom_to_content

        img = Image.new("RGB", (400, 800), "white")
        img.paste(Image.new("RGB", (300, 150), "black"), (50, 100))  # y[100,250)
        img.paste(Image.new("RGB", (3, 40), "black"), (198, 250))  # narrow tail
        assert _extend_bottom_to_content(img, 50, 350, 270) == 290

    def test_growth_stops_before_caption(self) -> None:
        from pdfparser.pipeline.figures import _extend_bottom_to_content

        assert _extend_bottom_to_content(self._image(), 50, 350, 250) < 360

    def test_safe_crop_excludes_caption(self) -> None:
        from pdfparser.pipeline.figures import _safe_crop

        crop = _safe_crop(self._image(), (50, 100, 350, 250))
        assert crop is not None and crop.size == (300, 200)


class TestFigureRightGrowth:
    """The crop grows right over contiguous ink to the figure's true right edge
    and stops at the whitespace before the page margin or inter-column gutter; a
    box already ending in whitespace grows nothing, so a neighbouring column is
    never pulled in."""

    @staticmethod
    def _image() -> Image.Image:
        # White page: figure block x[100,300), a strip at x[360,380) (page-margin
        # neighbour), separated by a 60 px whitespace gap.
        img = Image.new("RGB", (800, 400), "white")
        img.paste(Image.new("RGB", (200, 300), "black"), (100, 50))
        img.paste(Image.new("RGB", (20, 300), "black"), (360, 50))
        return img

    def test_tight_box_grows_to_figure_right(self) -> None:
        from pdfparser.pipeline.figures import _extend_right_to_content

        assert _extend_right_to_content(self._image(), 50, 350, 250) == 300

    def test_box_at_right_does_not_grow(self) -> None:
        from pdfparser.pipeline.figures import _extend_right_to_content

        assert _extend_right_to_content(self._image(), 50, 350, 300) == 300

    def test_no_growth_when_ink_runs_without_gap(self) -> None:
        # Ink continues past the search window with no whitespace gap (a column
        # abutting a correct box) → ambiguous → leave the box unchanged.
        from pdfparser.pipeline.figures import _extend_right_to_content

        img = Image.new("RGB", (800, 400), "white")
        img.paste(Image.new("RGB", (300, 300), "black"), (100, 50))
        assert _extend_right_to_content(img, 50, 350, 250) == 250

    def test_narrow_content_right_of_box_is_not_read_as_gap(self) -> None:
        # A figure tail narrower than the box (here 3 px of a 300 px-tall box,
        # ~1% ink) must count as content, not be mistaken for the whitespace gap.
        from pdfparser.pipeline.figures import _extend_right_to_content

        img = Image.new("RGB", (800, 400), "white")
        img.paste(Image.new("RGB", (150, 300), "black"), (100, 50))  # x[100,250)
        img.paste(Image.new("RGB", (40, 3), "black"), (250, 198))  # narrow tail
        assert _extend_right_to_content(img, 50, 350, 270) == 290

    def test_gutter_stops_growth_before_next_column(self) -> None:
        # Left-column figure, a whitespace gutter, then right-column text: growth
        # recovers the clipped figure edge but stops at the gutter, never reaching
        # the next column.
        from pdfparser.pipeline.figures import _extend_right_to_content

        img = Image.new("RGB", (800, 400), "white")
        img.paste(Image.new("RGB", (200, 300), "black"), (50, 50))  # x[50,250)
        img.paste(Image.new("RGB", (200, 300), "black"), (310, 50))  # right column
        assert _extend_right_to_content(img, 50, 350, 200) == 250

    def test_safe_crop_excludes_neighbour_column(self) -> None:
        from pdfparser.pipeline.figures import _safe_crop

        crop = _safe_crop(self._image(), (100, 50, 250, 350))
        assert crop is not None and crop.size == (200, 300)


class TestFigureLeftGrowth:
    """The crop grows left over contiguous ink to the figure's true left edge and
    stops at the whitespace before the page margin or inter-column gutter; a box
    already ending in whitespace grows nothing, so a neighbouring column is never
    pulled in.  Mirror of :class:`TestFigureRightGrowth`."""

    @staticmethod
    def _image() -> Image.Image:
        # White page: a strip at x[20,40) (page-margin neighbour), a 60 px gap, then
        # the figure block x[100,300).
        img = Image.new("RGB", (800, 400), "white")
        img.paste(Image.new("RGB", (20, 300), "black"), (20, 50))
        img.paste(Image.new("RGB", (200, 300), "black"), (100, 50))
        return img

    def test_tight_box_grows_to_figure_left(self) -> None:
        from pdfparser.pipeline.figures import _extend_left_to_content

        # box left clipped 50 px into the figure → grows back out to x=100
        assert _extend_left_to_content(self._image(), 50, 350, 150) == 100

    def test_box_at_left_does_not_grow(self) -> None:
        from pdfparser.pipeline.figures import _extend_left_to_content

        assert _extend_left_to_content(self._image(), 50, 350, 100) == 100

    def test_no_growth_when_ink_runs_without_gap(self) -> None:
        # Ink continues left past the search window with no gap (a column abutting a
        # correct box) → ambiguous → leave the box unchanged.
        from pdfparser.pipeline.figures import _extend_left_to_content

        img = Image.new("RGB", (800, 400), "white")
        img.paste(Image.new("RGB", (300, 300), "black"), (100, 50))  # x[100,400)
        assert _extend_left_to_content(img, 50, 350, 350) == 350

    def test_gutter_stops_growth_before_previous_column(self) -> None:
        # Right-column figure, a whitespace gutter, then left-column text: growth
        # recovers the clipped figure edge but stops at the gutter, never reaching
        # the previous column.
        from pdfparser.pipeline.figures import _extend_left_to_content

        img = Image.new("RGB", (800, 400), "white")
        img.paste(Image.new("RGB", (200, 300), "black"), (50, 50))  # left column
        img.paste(Image.new("RGB", (200, 300), "black"), (360, 50))  # x[360,560)
        assert _extend_left_to_content(img, 50, 350, 410) == 360

    def test_safe_crop_recovers_clipped_left_edge(self) -> None:
        from pdfparser.pipeline.figures import _safe_crop

        # box left clipped to x=150; the crop grows back out to the figure's x=100.
        crop = _safe_crop(self._image(), (150, 50, 300, 350))
        assert crop is not None and crop.size == (200, 300)


class TestSwallowedCaptionTrim:
    """A bottom band that growth recovers is trimmed when it reads as the figure's
    caption (short prose ink-runs) but kept when it is figure content (long shaded
    runs); trimming happens only when a caption is known to follow the figure."""

    @staticmethod
    def _image(ink_run: int, gap_run: int) -> Image.Image:
        # 800x400 white; a solid figure body at y[50,200) and, contiguous below it,
        # a 30 px band at y[200,230) whose horizontal ink-run length is set by
        # (ink_run, gap_run) — short runs read as prose, long runs as figure.
        a = np.full((400, 800), 255, np.uint8)
        a[50:200, 50:750] = 0
        row = np.full(700, 255, np.uint8)
        for x in range(0, 700, ink_run + gap_run):
            row[x : x + ink_run] = 0
        a[200:230, 50:750] = np.tile(row, (30, 1))
        return Image.fromarray(a, "L").convert("RGB")

    def test_prose_band_trimmed_when_caption_present(self) -> None:
        from pdfparser.pipeline.figures import _safe_crop

        crop = _safe_crop(self._image(3, 6), (50, 50, 750, 200), caption_text="cap")
        assert crop is not None and crop.size == (700, 150)  # band dropped

    def test_prose_band_kept_without_caption(self) -> None:
        from pdfparser.pipeline.figures import _safe_crop

        crop = _safe_crop(self._image(3, 6), (50, 50, 750, 200), caption_text=None)
        assert crop is not None and crop.size == (700, 180)  # band recovered

    def test_figure_band_kept_even_with_caption(self) -> None:
        from pdfparser.pipeline.figures import _safe_crop

        crop = _safe_crop(self._image(600, 10), (50, 50, 750, 200), caption_text="cap")
        assert crop is not None and crop.size == (700, 180)  # dense band kept

    def test_prose_scores_below_figure_run_length(self) -> None:
        from pdfparser.pipeline.figures import _mean_norm_run_length

        width = 700
        prose = np.zeros((10, width), bool)
        prose[:, ::9] = True  # 1-px runs, 8-px gaps → letterform-like
        figure = np.zeros((10, width), bool)
        figure[:, :600] = True  # one long shaded run
        assert _mean_norm_run_length(prose) < 0.07 <= _mean_norm_run_length(figure)


class TestBakedCaptionTrim:
    """When the figure is itself text (an alignment), its caption is pixel-identical
    to it and the model can box it *inside* the figure.  The trailing text bands are
    re-OCRed and dropped when they reproduce the caption — guarded against a figure
    row the caption merely names, and against a repeated-token OCR wall."""

    _CAPTION = (
        "Fig 9. Multiple sequence alignments of widget and gadget proteins. "
        "Catalytic residues are marked in cyan and the binding site in orange."
    )

    def test_band_is_caption_matches_caption_words(self) -> None:
        from pdfparser.pipeline.figures import _WORD_RE, _band_is_caption

        words = set(_WORD_RE.findall(self._CAPTION.lower()))
        assert _band_is_caption(self._CAPTION, words)

    def test_band_is_caption_rejects_low_overlap(self) -> None:
        from pdfparser.pipeline.figures import _WORD_RE, _band_is_caption

        words = set(_WORD_RE.findall(self._CAPTION.lower()))
        # a figure row mentioning a few caption words amid mostly non-caption data:
        # enough distinct caption words to clear the wall guard, but the matched
        # fraction stays under the bar
        assert not _band_is_caption(
            "catalytic residues marked QWERTY ZXCVB ASDFG HJKL", words
        )

    def test_band_is_caption_rejects_repeated_token_wall(self) -> None:
        from pdfparser.pipeline.figures import _WORD_RE, _band_is_caption

        # a row the model fails to read collapses to one caption word repeated —
        # ~1.0 word-overlap but no diversity, so it must be rejected as degenerate
        words = set(_WORD_RE.findall((self._CAPTION + " bmsdh").lower()))
        assert not _band_is_caption("bmsdh " * 200, words)

    def test_band_is_caption_rejects_short_repeated_wall(self) -> None:
        from pdfparser.pipeline.figures import _WORD_RE, _band_is_caption

        # the wall need not be long: three identical caption words must still fail
        # (the old type-ratio guard let this through; the distinct-word floor stops it)
        words = set(_WORD_RE.findall((self._CAPTION + " panel").lower()))
        assert not _band_is_caption("panel panel panel", words)

    def test_band_is_caption_rejects_too_few_words(self) -> None:
        from pdfparser.pipeline.figures import _WORD_RE, _band_is_caption

        words = set(_WORD_RE.findall(self._CAPTION.lower()))
        assert not _band_is_caption("Fig 9", words)

    def test_ink_bands_split_on_gaps(self) -> None:
        from pdfparser.pipeline.figures import _ink_bands

        mask = np.zeros((200, 50), bool)
        mask[10:40] = True  # band 1
        mask[120:160] = True  # band 2, separated by a 80-row gap
        assert _ink_bands(mask, gap=12) == [(10, 40), (120, 160)]

    def test_ocr_band_pads_thin_band(self) -> None:
        from pdfparser.pipeline.figures import _FIGURE_OCR_MIN_BAND_PX, _ocr_band

        seen: list[tuple[int, int]] = []
        _ocr_band(
            Image.new("RGB", (700, 600), "white"),
            (0, 100, 700, 110),  # a 10-px band
            lambda im: seen.append(im.size) or "",
        )
        assert seen and seen[0][1] >= _FIGURE_OCR_MIN_BAND_PX

    @staticmethod
    def _striped(a: np.ndarray, y0: int, y1: int) -> None:
        a[y0:y1, 50:750:9] = 0  # text-like: thin ink columns (short runs)

    def _text_image(self) -> Image.Image:
        # figure (top), then a caption band and a note band low in the crop
        a = np.full((600, 800), 255, np.uint8)
        for y0, y1 in ((40, 300), (400, 470), (500, 530)):
            self._striped(a, y0, y1)
        return Image.fromarray(a, "L").convert("RGB")

    def test_trim_baked_caption_drops_caption_and_note(self) -> None:
        from pdfparser.pipeline.figures import _trim_baked_caption

        # scan runs bottom→top: note (DOI) → caption → figure band (re-OCRed once to
        # confirm the boundary, then the scan stops as it isn't caption)
        replies = iter(
            ["see https://doi.org/10.1/x", self._CAPTION, "unrelated figure axis tick"]
        )
        y1 = _trim_baked_caption(
            self._text_image(), 0, 800, 0, 530, self._CAPTION, lambda im: next(replies)
        )
        assert y1 == 400  # trimmed to the caption's top, note swept with it

    def test_safe_crop_without_ocr_region_keeps_baked_caption(self) -> None:
        from pdfparser.pipeline.figures import _safe_crop

        # no ocr_region → the OCR trim never runs, so a text-bodied baked caption
        # stays (the crop reaches the note band's bottom)
        crop = _safe_crop(
            self._text_image(), (0, 0, 800, 530), caption_text=self._CAPTION
        )
        assert crop is not None and crop.size[1] == 530


class TestCrossPageMerge:
    """A paragraph split across a page break is rejoined."""

    def test_cross_page_paragraph_merge(self) -> None:
        page1 = "# T\n\n## Abstract\n\nA.\n\n## Body\n\nThis suggests that TRI and"
        page2 = "TRII compete for the same substrate tropinone."
        html = _run_lighton([page1, page2])
        assert "This suggests that TRI and TRII compete for the same substrate" in html
        assert "This suggests that TRI and</p>" not in html

    def test_mixed_case_identifier_continuation_after_the(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        # the fragment ends in "The"; the continuation opens with a mixed-case
        # identifier ("SpRDH"), which is mid-sentence, not a new-sentence capital —
        # so the merge fires across the intervening table float
        parts = [
            "<p>enter the pentose phosphate pathway for carbon metabolism. The</p>",
            "<table><tr><td>x</td></tr></table>",
            "<p>SpRDH operon of the genome contains a transporter.</p>",
        ]
        merged = _merge_split_paragraphs(parts)
        assert any("metabolism. The SpRDH operon of the genome" in p for p in merged)

    def test_function_word_guard_still_refuses_new_sentence(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        # a plain capitalized word after "The" (no internal capital) signals a
        # dropped continuation — the guard must still refuse the merge
        parts = [
            "<p>they use the Entner-Doudoroff pathway for glucose metabolism. The</p>",
            "<p>Many sugar alcohols enter the pentose phosphate pathway.</p>",
        ]
        assert _merge_split_paragraphs(parts) == parts


class TestCaptionMergeBarrier:
    """A figure/table caption is never absorbed as a paragraph continuation,
    even across intervening floats and even when wrapped in <strong>."""

    def test_table_caption_after_floats_not_glued_to_fragment(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        parts = [
            "<p>PtTRII catalyzed the reduction of tropinone to form</p>",
            "<figure><img src='a' alt=''></figure>",
            "<figure><img src='b' alt=''></figure>",
            "<p><strong>TABLE 1</strong> Enzyme kinetics of PtTRI and PtTRII</p>",
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
        ]
        out = _merge_split_paragraphs(parts)
        # Caption stays its own block, immediately before its table; the
        # fragment and the floats keep their order.
        assert out == parts
        assert "to form <strong>TABLE 1</strong>" not in "".join(out)

    def test_real_continuation_still_merges(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        parts = [
            "<p>This suggests that TRI and</p>",
            "<p>TRII compete for the substrate.</p>",
        ]
        out = _merge_split_paragraphs(parts)
        assert out == [
            "<p>This suggests that TRI and TRII compete for the substrate.</p>"
        ]

    def test_function_word_with_trailing_comma_blocks_capital_continuation(
        self,
    ) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        # "revealed that," is grammatically incomplete; a capitalised new
        # sentence is not its continuation (here an OCR-misplaced figure caption),
        # so the trailing comma must not disarm the capital-letter guard.
        parts = [
            "<p>analyses of 2-butanol production revealed that,</p>",
            "<p>Molecule structures are shown in Fig. 3.</p>",
        ]
        assert _merge_split_paragraphs(parts) == parts

    def test_function_word_with_trailing_comma_still_merges_lowercase(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        # The genuine lowercase continuation of the same clause still joins.
        parts = [
            "<p>analyses of 2-butanol production revealed that,</p>",
            "<p>with no additives present, all forms preferred re-face addition.</p>",
        ]
        out = _merge_split_paragraphs(parts)
        assert out == [
            "<p>analyses of 2-butanol production revealed that, with no additives "
            "present, all forms preferred re-face addition.</p>"
        ]

    def test_preposition_comma_does_not_block_proper_noun_continuation(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        # The comma allowance is only for clause-introducers; after a preposition
        # a trailing comma before a capitalised proper noun is a genuine
        # continuation, so the merge must still join across the break.
        parts = [
            "<p>the epoxide-metabolising strains studied here consist of,</p>",
            "<p>Xanthobacter autotrophicus and related species.</p>",
        ]
        out = _merge_split_paragraphs(parts)
        assert out == [
            "<p>the epoxide-metabolising strains studied here consist of, "
            "Xanthobacter autotrophicus and related species.</p>"
        ]

    def test_metadata_line_not_merged_into_following_prose(self) -> None:
        # A self-contained footer-metadata line that ends without terminal
        # punctuation (here a ")") must not be treated as an incomplete paragraph
        # and glued to the body prose the OCR placed after it.
        from pdfparser.pipeline.merge import _merge_split_paragraphs_stable

        parts = [
            "<p>Published online 28 December 2018 in Wiley Online Library "
            "(wileyonlinelibrary.com)</p>",
            "<p>Herein, I propose that the method generalizes to other enzymes.</p>",
        ]
        assert _merge_split_paragraphs_stable(parts) == parts

    def test_continuation_after_two_figures_and_a_table_merges(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        # A column break stranded the continuation behind a figure+figure+table
        # cluster (the table's caption already folded in by colocation).
        parts = [
            "<p>PtTRII catalyzed the reduction of tropinone to form</p>",
            "<figure><img src='a' alt=''></figure>",
            "<figure><img src='b' alt=''></figure>",
            "<table><caption>TABLE 1</caption><tbody><tr><td>1</td></tr></tbody>"
            "</table>",
            "<p>pseudotropine with higher affinity to tropinone.</p>",
        ]
        out = _merge_split_paragraphs(parts)
        assert out[0] == (
            "<p>PtTRII catalyzed the reduction of tropinone to form "
            "pseudotropine with higher affinity to tropinone.</p>"
        )
        # The floats are relocated after the joined paragraph, order preserved.
        assert out[1:] == parts[1:4]


class TestTableCaptionColocation:
    """A free-standing "Table N …" caption is folded into its <table> as a
    <caption> first child so it renders with the table, not adrift."""

    def test_caption_before_table_folded(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        parts = [
            "<p><strong>TABLE 1</strong> Enzyme kinetics of PtTRI and PtTRII</p>",
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
        ]
        assert _colocate_table_captions(parts) == [
            "<table><caption><strong>TABLE 1</strong> Enzyme kinetics of PtTRI "
            "and PtTRII</caption><tbody><tr><td>1</td></tr></tbody></table>"
        ]

    def test_caption_after_table_folded(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        parts = [
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
            "<p>Table 2. Results.</p>",
        ]
        out = _colocate_table_captions(parts)
        assert out == [
            "<table><caption>Table 2. Results.</caption>"
            "<tbody><tr><td>1</td></tr></tbody></table>"
        ]

    def test_caption_separated_by_figure_folded(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        # The reported bug: a figure floats between the caption and its table.
        parts = [
            "<p>Table 3 Kinetic constants</p>",
            "<figure><img src='x' alt=''></figure>",
            "<table><tr><td>a</td></tr></table>",
        ]
        out = _colocate_table_captions(parts)
        assert out == [
            "<figure><img src='x' alt=''></figure>",
            "<table><caption>Table 3 Kinetic constants</caption>"
            "<tr><td>a</td></tr></table>",
        ]

    def test_caption_not_pulled_across_prose(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        # A real paragraph between caption and table breaks the association.
        parts = [
            "<p>Table 4 X</p>",
            "<p>Unrelated body sentence.</p>",
            "<table><tr><td>a</td></tr></table>",
        ]
        assert _colocate_table_captions(parts) == parts

    def test_orphan_caption_left_intact(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        parts = [
            "<p>Table 9 Orphan caption with no table near it.</p>",
            "<p>Prose.</p>",
        ]
        assert _colocate_table_captions(parts) == parts

    def test_two_tables_pair_with_own_captions(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        parts = [
            "<p>Table 1 A</p>",
            "<table><tr><td>1</td></tr></table>",
            "<p>Table 2 B</p>",
            "<table><tr><td>2</td></tr></table>",
        ]
        out = _colocate_table_captions(parts)
        assert "<caption>Table 1 A</caption>" in out[0]
        assert "<caption>Table 2 B</caption>" in out[1]
        assert len(out) == 2

    def test_existing_caption_not_duplicated(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        parts = [
            "<p>Table 5 Duplicate guard</p>",
            "<table><caption>existing</caption><tr><td>1</td></tr></table>",
        ]
        out = _colocate_table_captions(parts)
        # The table keeps its own caption; the stray block is left, not lost.
        assert out == parts

    def test_table_attributes_preserved(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        # The caption goes after the *whole* opening tag, not inside it.
        parts = [
            "<p>Table 1 Kinetics</p>",
            '<table class="data"><tbody><tr><td>1</td></tr></tbody></table>',
        ]
        out = _colocate_table_captions(parts)
        assert out == [
            '<table class="data"><caption>Table 1 Kinetics</caption>'
            "<tbody><tr><td>1</td></tr></tbody></table>"
        ]

    def test_caption_between_tables_pairs_with_following(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        # A caption sat between two tables precedes the second, so it belongs to
        # it — the first (captionless) table must not forward-steal it.
        parts = [
            "<table><tr><td>1</td></tr></table>",
            "<p>Table 2 Results</p>",
            "<table><tr><td>2</td></tr></table>",
        ]
        out = _colocate_table_captions(parts)
        assert out == [
            "<table><tr><td>1</td></tr></table>",
            "<table><caption>Table 2 Results</caption><tr><td>2</td></tr></table>",
        ]

    def test_reference_sentence_not_folded(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        # "Table N <lowercase verb> …" is a running reference, not a caption; it
        # must stay in the body, not be absorbed into the table.
        parts = [
            "<p>Table 1 summarizes the kinetic parameters of both enzymes.</p>",
            "<table><tr><td>1</td></tr></table>",
        ]
        assert _colocate_table_captions(parts) == parts

    def test_bare_label_rejoined_then_folded(self) -> None:
        from pdfparser.pipeline.merge import (
            _colocate_table_captions,
            _join_split_table_caption_labels,
        )

        # The reported bug: OCR split "TABLE I" from its title, stranding the
        # title between label and table so the caption never folds.
        parts = [
            "<p>TABLE I</p>",
            "<p>Selected substrates and inhibitors used to investigate.</p>",
            "<table><tbody><tr><td>a</td></tr></tbody></table>",
        ]
        out = _colocate_table_captions(_join_split_table_caption_labels(parts))
        assert out == [
            "<table><caption>TABLE I Selected substrates and inhibitors "
            "used to investigate.</caption><tbody><tr><td>a</td></tr></tbody></table>"
        ]

    def test_heading_label_rejoined_then_folded(self) -> None:
        from pdfparser.pipeline.merge import (
            _colocate_table_captions,
            _join_split_table_caption_labels,
        )

        # The OCR sometimes promotes the bare label to a heading
        # ("## TABLE IV" → <h2>), stranding the title below it just like the
        # <p>-form label; it must rejoin and fold the same way.
        parts = [
            "<h2>TABLE IV</h2>",
            "<p>Comparison of rR- and rS-HPCDH kinetic parameters.</p>",
            "<table><tbody><tr><td>a</td></tr></tbody></table>",
        ]
        out = _colocate_table_captions(_join_split_table_caption_labels(parts))
        assert out == [
            "<table><caption>TABLE IV Comparison of rR- and rS-HPCDH kinetic "
            "parameters.</caption><tbody><tr><td>a</td></tr></tbody></table>"
        ]

    def test_labelled_caption_not_rejoined(self) -> None:
        from pdfparser.pipeline.merge import _join_split_table_caption_labels

        # A label that already carries its title ("Table 4 X") is a complete
        # caption; the following block is unrelated prose and must stay separate.
        parts = [
            "<p>Table 4 X</p>",
            "<p>Unrelated body sentence.</p>",
        ]
        assert _join_split_table_caption_labels(parts) == parts

    def test_bare_label_before_table_not_rejoined_with_table(self) -> None:
        from pdfparser.pipeline.merge import _join_split_table_caption_labels

        # A bare label sitting directly on its table needs no rejoin (the next
        # block is the <table>, not a stray title paragraph).
        parts = [
            "<p>TABLE I</p>",
            "<table><tbody><tr><td>a</td></tr></tbody></table>",
        ]
        assert _join_split_table_caption_labels(parts) == parts

    def test_split_caption_lets_paragraph_merge_across_table(self) -> None:
        # End-to-end: a paragraph split across a captioned table rejoins once the
        # split caption is folded into the <table> (only the float remains between
        # the two halves).
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Results\n\n"
            "analyses of 2-butanol production revealed that,\n\n"
            "TABLE I\n\n"
            "Selected substrates and inhibitors used to investigate.\n\n"
            "<table><tbody><tr><td>Km</td></tr></tbody></table>\n\n"
            "with no additives present, all forms preferred a re-face addition."
        )
        body = _body(_run_lighton([md]))
        assert "revealed that, with no additives present, all forms preferred" in body
        assert "<caption>TABLE I Selected substrates" in body
        assert "revealed that,</p>" not in body

    def test_end_to_end_caption_heads_table(self) -> None:
        # Full assembly: the fragment must not absorb the caption, and the
        # caption must end up inside its table, not before the figures.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Results\n\n"
            "PtTRII catalyzed the reduction of tropinone to form\n\n"
            "![image](0,0,500,500)\n\n"
            "**TABLE 1** Enzyme kinetics of PtTRI and PtTRII\n\n"
            "<table><tbody><tr><td>Km</td></tr></tbody></table>"
        )
        body = _body(_run_lighton([md]))
        assert "<caption><strong>TABLE 1</strong> Enzyme kinetics" in body
        assert "to form <strong>TABLE 1</strong>" not in body
        # The caption no longer appears as a stand-alone paragraph.
        assert "<p><strong>TABLE 1</strong>" not in body

    def test_heading_label_title_not_absorbed_by_body_continuation(self) -> None:
        # 30592559 page 5→6: the OCR rendered "TABLE IV" as a heading with its
        # untitled-looking title stranded below.  Lacking terminal punctuation the
        # title was mistaken for a body fragment and the prose resuming after the
        # table ("reaction stereospecificity was not…") was glued onto the caption
        # instead of onto the page-5 paragraph it continues.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Implementation\n\n"
            "Although the general mechanism of a Zn-dependent alcohol "
            "dehydrogenase was covered, the\n\n"
            "## TABLE IV\n\n"
            "Comparison of rR- and rS-HPCDH kinetic parameters and stereoselectivity"
            "\n\n"
            "<table><tbody><tr><td>Km</td></tr></tbody></table>\n\n"
            "reaction stereospecificity was not. An additional prerequisite topic "
            "was prochirality."
        )
        body = _body(_run_lighton([md]))
        assert (
            "dehydrogenase was covered, the reaction stereospecificity was not." in body
        )
        assert "<caption>TABLE IV Comparison of rR- and rS-HPCDH" in body
        # The caption text must not have swallowed the body continuation.
        assert "stereoselectivity reaction stereospecificity" not in body


class TestTableFootnoteColocation:
    """A table's trailing footnote run — superscript-marker lines plus a note
    sentence wedged before them — is folded onto the table block, not left adrift
    or swept into the article footnote section."""

    def test_marker_footnotes_folded_onto_table(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # The table carries the a/b markers its footnotes annotate.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td><td>E<sup>b</sup></td></tr>"
            "</tbody></table>",
            "<p><sup>a</sup>Apparent K values.</p>",
            "<p><sup>b</sup>ND = not determined.</p>",
        ]
        assert _colocate_table_footnotes(parts) == [
            "<table><tbody><tr><td>K<sup>a</sup></td><td>E<sup>b</sup></td></tr>"
            "</tbody></table>"
            '<p class="footnote"><sup>a</sup>Apparent K values.</p>'
            '<p class="footnote"><sup>b</sup>ND = not determined.</p>'
        ]

    def test_note_sentence_before_markers_folded(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # The reported case: a note sentence sits between the table and its
        # superscript footnotes, so it rides along into the table block.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p>Molecule structures are shown in Fig. 3.</p>",
            "<p><sup>a</sup>Apparent K values.</p>",
        ]
        assert _colocate_table_footnotes(parts) == [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>"
            '<p class="footnote">Molecule structures are shown in Fig. 3.</p>'
            '<p class="footnote"><sup>a</sup>Apparent K values.</p>'
        ]

    def test_body_after_markers_stays_in_stream(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # The body paragraph that resumes after the footnotes is not absorbed.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p><sup>a</sup>Apparent K values.</p>",
            "<p>with no additives present, all forms preferred re-face.</p>",
        ]
        out = _colocate_table_footnotes(parts)
        assert out == [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>"
            '<p class="footnote"><sup>a</sup>Apparent K values.</p>',
            "<p>with no additives present, all forms preferred re-face.</p>",
        ]

    def test_article_footnote_marker_not_in_table_not_absorbed(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # The hardening: a superscript line whose label the table does NOT carry
        # is an article footnote that merely follows the table, not a table
        # footnote, so it is left for the classifier to route before references.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p><sup>*</sup>Corresponding author: a@b.com.</p>",
        ]
        assert _colocate_table_footnotes(parts) == parts

    def test_numeric_marker_matching_table_exponent_not_absorbed(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # A numbered article footnote whose digit collides with a table exponent
        # ("cm<sup>2</sup>") must not be folded — exponents are not footnote
        # referents, so the numeric label does not qualify as a table marker.
        parts = [
            "<table><tbody><tr><td>area cm<sup>2</sup></td></tr></tbody></table>",
            "<p><sup>2</sup>A numbered article footnote.</p>",
        ]
        assert _colocate_table_footnotes(parts) == parts

    def test_letter_marker_folded_despite_table_exponents(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # A letter footnote still folds when the table mixes exponents and an
        # 'a' referent; the exponent does not interfere.
        parts = [
            "<table><tbody><tr><td>cm<sup>2</sup></td><td>K<sup>a</sup></td></tr>"
            "</tbody></table>",
            "<p><sup>a</sup>Apparent K values.</p>",
        ]
        assert _colocate_table_footnotes(parts) == [
            "<table><tbody><tr><td>cm<sup>2</sup></td><td>K<sup>a</sup></td></tr>"
            "</tbody></table>"
            '<p class="footnote"><sup>a</sup>Apparent K values.</p>'
        ]

    def test_note_before_unmatched_marker_not_absorbed(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # A leading note rides along only when matching markers follow; an
        # unmatched marker abandons the run, so the note stays in the stream.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p>A note sentence.</p>",
            "<p><sup>*</sup>An article footnote.</p>",
        ]
        assert _colocate_table_footnotes(parts) == parts

    def test_note_without_markers_not_absorbed(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # A plain paragraph after a table with no footnote markers is body, not a
        # note, and is left untouched.
        parts = [
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
            "<p>This paragraph continues the discussion.</p>",
        ]
        assert _colocate_table_footnotes(parts) == parts

    def test_runaway_leading_prose_not_swallowed(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # More than a note's worth of prose before any marker is body, so the
        # run is abandoned and nothing is folded — even though the table carries
        # the late marker's label.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p>First body paragraph.</p>",
            "<p>Second body paragraph.</p>",
            "<p>Third body paragraph.</p>",
            "<p><sup>a</sup>A late marker.</p>",
        ]
        assert _colocate_table_footnotes(parts) == parts

    def test_second_leading_line_exceeds_note_cap(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # A table note is a single line; two non-marker lines before the marker
        # exceed the cap, so the run is abandoned rather than swallowing a second
        # (possibly body) line.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p>First note line.</p>",
            "<p>Second note line.</p>",
            "<p><sup>a</sup>Apparent K values.</p>",
        ]
        assert _colocate_table_footnotes(parts) == parts

    def test_trailing_source_note_after_markers_folded(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # The reported case: an attribution note with no superscript marker trails
        # the marker run, so the marker loop can't anchor it; its lexical shape
        # ("Data adapted from …") folds it onto the table.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p><sup>a</sup>Apparent K values.</p>",
            "<p>Data adapted from Clark et al. [7].</p>",
        ]
        assert _colocate_table_footnotes(parts) == [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>"
            '<p class="footnote"><sup>a</sup>Apparent K values.</p>'
            '<p class="footnote">Data adapted from Clark et al. [7].</p>'
        ]

    def test_standalone_source_note_without_markers_folded(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # A source note can also be the table's only footnote, with no markers at
        # all; the lexical cue still folds it onto the table.
        parts = [
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
            "<p>Data adapted from Clark et al. [7].</p>",
            "<p>The discussion continues in ordinary prose here.</p>",
        ]
        assert _colocate_table_footnotes(parts) == [
            "<table><tbody><tr><td>1</td></tr></tbody></table>"
            '<p class="footnote">Data adapted from Clark et al. [7].</p>',
            "<p>The discussion continues in ordinary prose here.</p>",
        ]

    def test_body_line_before_standalone_source_note_not_swallowed(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # A body line stranded between the table and a markerless source note must
        # not be dragged into the footnotes just because the source note folds: with
        # no anchoring markers, the ambiguous arrangement leaves both in the stream.
        parts = [
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
            "<p>This finding is discussed further in the next section.</p>",
            "<p>Data adapted from Clark et al. [7].</p>",
            "<p>More body prose follows here.</p>",
        ]
        assert _colocate_table_footnotes(parts) == parts

    def test_generic_verb_without_subject_not_a_source_note(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # A body sentence opening with a generic participle ("Obtained from …")
        # is not an attribution note, so it stays in the body after the table.
        parts = [
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
            "<p>Obtained from a commercial supplier, the reagents were used.</p>",
        ]
        assert _colocate_table_footnotes(parts) == parts

    def test_plain_body_after_markers_not_mistaken_for_source_note(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # The body resuming after the markers does not open with a source cue, so
        # the trailing-note sweep leaves it in the stream.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p><sup>a</sup>Apparent K values.</p>",
            "<p>Data presented here support the proposed mechanism in Fig. 2.</p>",
        ]
        assert _colocate_table_footnotes(parts) == [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>"
            '<p class="footnote"><sup>a</sup>Apparent K values.</p>',
            "<p>Data presented here support the proposed mechanism in Fig. 2.</p>",
        ]

    def test_end_to_end_table_footnotes_unblock_merge(self) -> None:
        # Full assembly: the table footnotes (a note sentence + marker lines) fold
        # into the table, so the paragraph split across the table rejoins and the
        # footnotes are not relocated to the article footnote section.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Results\n\n"
            "analyses of 2-butanol production revealed that,\n\n"
            "<table><tbody><tr><td>Km<sup>a</sup></td><td>E<sup>b</sup></td></tr>"
            "</tbody></table>\n\n"
            "Molecule structures are shown in Fig. 3.\n\n"
            "<sup>a</sup>Apparent K values.\n\n"
            "<sup>b</sup>ND = not determined.\n\n"
            "with no additives present, all forms preferred a re-face addition."
        )
        body = _body(_run_lighton([md]))
        assert "revealed that, with no additives present, all forms preferred" in body
        # The footnotes ride with the table inside the body, not before references.
        note = '<p class="footnote">Molecule structures are shown in Fig. 3.</p>'
        assert note in body
        assert '<p class="footnote"><sup>a</sup>Apparent K values.</p>' in body


class TestLatexToHtml:
    """Inline `$…$` math is converted to deterministic sub/superscript HTML
    before markdown parsing."""

    def test_simple_subscript(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        assert _latex_to_html("$K_m$") == "K<sub>m</sub>"

    def test_braced_subscript(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        assert _latex_to_html("$V_{max}$") == "V<sub>max</sub>"

    def test_superscript_becomes_unicode(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # All-mappable superscript chars collapse to Unicode (matches "NAD⁺").
        assert _latex_to_html("NAD$^+$") == "NAD⁺"

    def test_superscript_letters_fall_back_to_tag(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        assert _latex_to_html("pH$^{S}$") == "pH<sup>S</sup>"

    def test_ratio_of_kinetic_constants(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        assert _latex_to_html("$k_{cat}/K_m$") == "k<sub>cat</sub>/K<sub>m</sub>"

    def test_degree_command_superscript(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # ``$^\circ$`` is the LaTeX degree idiom; the single-char superscript
        # rule used to capture only the backslash, leaving a lone ``\`` inside
        # <sup> that markdown then mangled into "<sup></sup>circ".
        assert _latex_to_html(r"grown at 25 $\pm$ 1$^\circ$C") == "grown at 25 ± 1°C"

    def test_braced_degree_command(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        assert _latex_to_html(r"$^{\circ}$C") == "°C"

    def test_symbol_command_translated(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        assert _latex_to_html(r"$5 \times 10^{3}$ cells") == "5 × 10³ cells"

    def test_greek_command_as_subscript(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        assert _latex_to_html(r"$T_\alpha$") == "T<sub>α</sub>"

    def test_command_matched_as_whole_token(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # Commands are matched as maximal "\name" tokens and looked up whole, so
        # a short command never eats the head of a longer one ("\to" vs "\top",
        # "\sim" vs "\simeq") — each resolves to its own glyph.
        assert _latex_to_html(r"$\to$") == "→"
        assert _latex_to_html(r"$\top$") == "⊤"
        assert _latex_to_html(r"$A \simeq B$") == "A ≃ B"

    def test_command_still_terminated_by_digit(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        assert _latex_to_html(r"$\alpha2$") == "α2"

    def test_unknown_command_left_literal(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # pylatexenc returns "" for an unknown macro; we keep the literal rather
        # than silently dropping it.
        assert _latex_to_html(r"$x\notacommand y$") == r"x\notacommand y"

    def test_extended_symbol_coverage(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # Coverage we get for free from pylatexenc that the old hand map lacked.
        assert _latex_to_html(r"$T_\beta + \nabla$") == "T<sub>β</sub> + ∇"

    def test_arg_macro_that_raises_does_not_crash(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # pylatexenc raises when fed a bare arg-taking macro like \sqrt; the
        # exception must be swallowed and the span left intact, not propagated
        # up to crash the whole page conversion.
        assert _latex_to_html(r"$\sqrt{x}$") == r"\sqrtx"

    def test_arg_macro_template_not_leaked(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # \frac's substitution template ("%s/%s") must not reach the output; the
        # command stays literal so real math survives for a later MathJax pass.
        assert "%s" not in _latex_to_html(r"$\frac{a}{b}$")
        assert _latex_to_html(r"$\frac{a}{b}$") == r"\fracab"

    def test_extended_symbol_coverage_via_command_helper(self) -> None:
        from pdfparser.pipeline.latex import _latex_command_to_unicode

        assert _latex_command_to_unicode(r"\sqrt") == r"\sqrt"
        assert _latex_command_to_unicode(r"\frac") == r"\frac"
        assert _latex_command_to_unicode(r"\alpha") == "α"

    def test_plain_text_untouched(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        assert _latex_to_html("no math here") == "no math here"

    def test_currency_dollars_left_alone(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # No TeX markup between the '$' → not math; must not be stripped/merged.
        assert _latex_to_html("costs $5 and $10 total") == "costs $5 and $10 total"

    def test_math_wrapped_bare_number_unwrapped(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # A lone number in a math span is just a value the model wrapped — drop
        # the '$' delimiters but keep the number (and the surrounding spaces).
        assert _latex_to_html("was $42.26$ Sec") == "was 42.26 Sec"

    def test_script_span_reattaches_to_preceding_token(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # The model writes a unit and its exponent with a gap ("Sec $^{-1}$"); a
        # span opening with a script attaches to the previous token, no space.
        assert _latex_to_html("Sec $^{-1}$ mM $^{-1}$") == "Sec⁻¹ mM⁻¹"


class TestMdToHtmlBlocks:
    """A page's markdown becomes one HTML string per top-level block, with raw
    HTML (tables, <sup>) passed through and thematic breaks dropped."""

    def test_heading_and_paragraph_split(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        blocks = _md_to_html_blocks("## Introduction\n\nSome prose here.")
        assert blocks == ["<h2>Introduction</h2>", "<p>Some prose here.</p>"]

    def test_emphasis_rendered(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        (block,) = _md_to_html_blocks("*Przewalskia tangutica* is **rare**.")
        assert (
            block == "<p><em>Przewalskia tangutica</em> is <strong>rare</strong>.</p>"
        )

    def test_table_passthrough(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        table = "<table><tbody><tr><td>1</td></tr></tbody></table>"
        assert _md_to_html_blocks(table) == [table]

    def test_table_cell_emphasis_rendered(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        # Raw-HTML table cells carry the model's ``*emphasis*`` unparsed; organism
        # names in a cell must still italicise instead of showing bare asterisks.
        table = (
            "<table><tbody><tr><td>*Sphingomonas* sp. PAMC 26621</td></tr>"
            "<tr><th>*Klebsiella aerogenes*</th></tr></tbody></table>"
        )
        (block,) = _md_to_html_blocks(table)
        assert "<td><em>Sphingomonas</em> sp. PAMC 26621</td>" in block
        assert "<th><em>Klebsiella aerogenes</em></th>" in block
        assert "*" not in block

    def test_table_cell_lone_asterisk_kept_literal(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        # A single footnote-marker asterisk is not an emphasis span — it must stay.
        (block,) = _md_to_html_blocks(
            "<table><tbody><tr><td>100*</td></tr></tbody></table>"
        )
        assert "<td>100*</td>" in block

    def test_table_cell_spaced_asterisks_not_emphasis(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        # Asterisks flanked by spaces (multiplication, paired footnote daggers) are
        # not CommonMark emphasis — they must stay literal, not wrap an <em>.
        (block,) = _md_to_html_blocks(
            "<table><tbody><tr><td>5 * 10 * 3</td></tr></tbody></table>"
        )
        assert "<td>5 * 10 * 3</td>" in block
        assert "<em>" not in block

    def test_table_cell_stray_lt_and_amp_escaped(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        # A bare "<" / "&" / ">" inside a cell must be escaped, not left to start a
        # bogus tag or a broken entity.
        (block,) = _md_to_html_blocks(
            "<table><tbody><tr><td>n<5 & p>0.05</td></tr></tbody></table>"
        )
        assert "<td>n&lt;5 &amp; p&gt;0.05</td>" in block

    def test_sup_passthrough(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        (block,) = _md_to_html_blocks("NAD<sup>+</sup> dependent.")
        assert block == "<p>NAD<sup>+</sup> dependent.</p>"

    def test_degree_does_not_bleed_into_superscript(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        # Regression: "$^\circ$" once produced "<sup>\</sup>", whose lone "\<"
        # markdown escaped into "&lt;/sup&gt;", swallowing the rest of the
        # sentence into a superscript.  No <sup> must survive here.
        (block,) = _md_to_html_blocks(
            r"grown at 25 $\pm$ 1$^\circ$C under 16 H of light."
        )
        assert block == "<p>grown at 25 ± 1°C under 16 H of light.</p>"

    def test_thematic_break_dropped(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        assert _md_to_html_blocks("A.\n\n---\n\nB.") == ["<p>A.</p>", "<p>B.</p>"]

    def test_list_kept_as_one_block(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        (block,) = _md_to_html_blocks("- one\n- two")
        assert block.startswith("<ul>")
        assert "<li>one</li>" in block and "<li>two</li>" in block


class TestParseFigurePlaceholder:
    """LightOnOCR-bbox emits figures as `![image](...)x0,y0,x1,y1`; the parser
    must recover the crop box, recognise a bbox-less placeholder, and reject
    ordinary prose."""

    def test_box_extracted(self) -> None:
        from pdfparser.pipeline.figures import _parse_figure_placeholder

        assert _parse_figure_placeholder("![image](image_1.png)122,89,877,614") == (
            122,
            89,
            877,
            614,
        )

    def test_box_with_surrounding_whitespace(self) -> None:
        from pdfparser.pipeline.figures import _parse_figure_placeholder

        assert _parse_figure_placeholder("  ![image](img.png) 10, 20, 30, 40 ") == (
            10,
            20,
            30,
            40,
        )

    def test_bboxless_placeholder_returns_true(self) -> None:
        from pdfparser.pipeline.figures import _parse_figure_placeholder

        assert _parse_figure_placeholder("![image](image_1.png)") is True

    def test_caption_line_is_not_a_placeholder(self) -> None:
        from pdfparser.pipeline.figures import _parse_figure_placeholder

        assert _parse_figure_placeholder("FIG. 2 Protein alignments of TRI.") is None

    def test_inline_image_in_prose_is_not_a_placeholder(self) -> None:
        from pdfparser.pipeline.figures import _parse_figure_placeholder

        line = "Some prose with ![inline](x.png) embedded mid-sentence."
        assert _parse_figure_placeholder(line) is None


_FIXTURE_PDF = Path(__file__).parent / "fixtures" / "30592559.pdf"
_AD_PREFIX_PDF = Path(__file__).parent / "fixtures" / "31051047.pdf"
_PLOS_PDF = Path(__file__).parent / "fixtures" / "32639976.pdf"
_OUTPUT_DIR = Path(__file__).parent.parent / "spike_results"


def _run_pipeline_to_file(pdf: Path, ocr: object) -> str:
    """Run the full pipeline on ``pdf`` and save the HTML for visual inspection.

    The output lands at ``spike_results/<pdf-stem>.html`` so every integration
    run leaves an on-disk copy of each fixture's rendering to open in a browser.
    """
    from pdfparser.pipeline import OcrModel, lightonocr_pdf_to_html

    assert isinstance(ocr, OcrModel)
    html = lightonocr_pdf_to_html(pdf, ocr=ocr)
    _OUTPUT_DIR.mkdir(exist_ok=True)
    (_OUTPUT_DIR / f"{pdf.stem}.html").write_text(html, encoding="utf-8")
    return html


@pytest.fixture(scope="session")
def ocr_model() -> object:
    """Load the LightOnOCR model bundle once per session; skip if unavailable."""
    try:
        from pdfparser.pipeline import load_ocr_model

        return load_ocr_model()
    except Exception as e:
        pytest.skip(f"LightOnOCR model not available: {e}")


@pytest.fixture(scope="session")
def article_html(ocr_model: object) -> str:
    """Run the full pipeline on the no-ad fixture; skip if the model is absent.

    Writes the result to spike_results/30592559.html so the file stays current
    after each integration run.
    """
    if not _FIXTURE_PDF.exists():
        pytest.skip(f"Fixture PDF not found: {_FIXTURE_PDF}")
    return _run_pipeline_to_file(_FIXTURE_PDF, ocr_model)


def _header_h1(html: str) -> str:
    """Return the text of the document's <header><h1> title element."""
    m = re.search(r"<header>.*?<h1>(.*?)</h1>", html, re.DOTALL)
    assert m, "header <h1> not found"
    return m.group(1)


@pytest.mark.integration
class TestPipeline:
    """Integration tests: run the full LightOnOCR pipeline on the fixture PDF.

    Skipped when the model is not available (no GPU, weights not downloaded).
    Each run also refreshes spike_results/30592559.html.
    """

    def test_abstract_no_column_break(self, article_html: str) -> None:
        abstract_start = article_html.find("<section class='abstract'>")
        abstract_end = article_html.find("</section>", abstract_start)
        abstract_block = article_html[abstract_start:abstract_end]
        # Both halves must be collected: if either is absent the pipeline
        # dropped a fragment it should have kept.
        assert "classical and contemporary" in abstract_block
        assert "experimental biochemistry." in abstract_block
        # And they must appear in the same paragraph — no split <p>.
        assert "classical and contemporary</p>" not in abstract_block

    def test_which_is_not_merged_with_as_a_testament(self, article_html: str) -> None:
        assert "which is As a testament to the utility" not in article_html

    def test_ternary_complex_followed_by_clearly_showed(
        self, article_html: str
    ) -> None:
        expected = (
            "The 1.8 Å ternary complex (enzyme + 2-KPC + NAD⁺)"
            " clearly showed interaction of the R152"
        )
        assert expected in article_html

    def test_first_page_footer_metadata_in_panel(self, article_html: str) -> None:
        # The first page's bottom-of-page footer — the journal-citation /
        # correspondence line and the supporting-information note — is front matter
        # the OCR drops into the body.  It is relocated to the collapsed Metadata
        # panel (which renders before the body), not left inline.
        panel, body = _metadata(article_html), _body(article_html)
        # The OCR splits the page-bottom block into one-line pieces; every one is
        # relocated, including the journal-citation / DOI / "Published online" lines
        # that carry only a single metadata token.
        for fragment in (
            "To whom correspondence should be addressed",
            "Additional Supporting Information",
            "Received 19 June 2018",
            "Volume 47",
            "DOI 10.1002/bmb.21202",
            "Published online 28 December 2018",
        ):
            assert fragment in panel
            assert fragment not in body
        # Pulling the orphan lines out before the (cross-page) paragraph-merge stops
        # them from chaining into the page-2 prose the OCR placed after them.
        assert "Herein, I propose" in body

    def test_figure_caption_not_glued_to_following_paragraph(
        self, article_html: str
    ) -> None:
        # The Fig. 1 caption is emitted as its own <figcaption>; the body
        # paragraph that follows ("Herein, I propose …") must stay a separate
        # block, not be absorbed onto the caption.
        body = _body(article_html)
        assert "carboxylase.</figcaption>" in body
        assert "<p>Herein, I propose" in body
        assert "carboxylase. Herein, I propose" not in body
        assert "carboxylase.</em> Herein, I propose" not in body

    def test_paragraph_rejoined_across_table_i(self, article_html: str) -> None:
        # The clause "…revealed that," / "with no additives present…" is split by
        # TABLE I (its caption and footnotes folded in); it must read as one
        # paragraph, not be left split or glued to the stray Fig. 3 note sentence.
        body = _body(article_html)
        assert "revealed that, with no additives present, all forms of rR-HPCDH" in body
        assert "revealed that,</p>" not in body
        assert "revealed that, Molecule structures" not in body

    def test_table_footnotes_ride_with_their_table(self, article_html: str) -> None:
        # TABLE I's footnote run — the "Molecule structures … Fig. 3." note plus
        # the <sup>-marked footnotes — is folded onto the table block, so it
        # renders under the table rather than being swept into the article
        # footnote section before the references.
        body = _body(article_html)
        note = '<p class="footnote">Molecule structures are shown in Fig. 3.</p>'
        assert note in body
        # The note sits immediately after a </table>, not adrift in the prose.
        assert '</table><p class="footnote">Molecule structures' in body

    def test_table_iv_heading_label_does_not_absorb_body_continuation(
        self, article_html: str
    ) -> None:
        # Page 5→6: "TABLE IV" is OCR'd as a heading with its title stranded below.
        # The page-5 paragraph ("…alcohol dehydrogenase was covered, the") continues
        # as "reaction stereospecificity was not. …prochirality." after the table; it
        # must rejoin that paragraph, not be glued onto the TABLE IV caption.
        body = _body(article_html)
        assert (
            "alcohol dehydrogenase was covered, the reaction stereospecificity"
            " was not. An additional prerequisite topic, covered on the same day"
            " of the case study, was prochirality." in body
        )
        # The TABLE IV title is folded into the table as a caption, not left as a
        # body paragraph that swallowed the continuation.
        assert (
            "<caption>TABLE IV Comparison of rR- and rS-HPCDH kinetic parameters"
            " and stereoselectivity</caption>" in body
        )
        assert "stereoselectivity reaction stereospecificity was not" not in body


@pytest.fixture(scope="session")
def ad_prefix_html(ocr_model: object) -> str:
    """Full pipeline output for the ad-prefixed 31051047.pdf fixture.

    Writes the result to spike_results/31051047.html for visual inspection.
    """
    if not _AD_PREFIX_PDF.exists():
        pytest.skip(f"Fixture PDF not found: {_AD_PREFIX_PDF}")
    return _run_pipeline_to_file(_AD_PREFIX_PDF, ocr_model)


@pytest.mark.integration
class TestAdPageExclusion:
    """The 31051047.pdf fixture has an advertisement as its first page; the
    pipeline must drop it and start the document at the real article title."""

    def test_title_starts_with_article_title(self, ad_prefix_html: str) -> None:
        # The title carries the PDF's intra-title line breaks; normalize runs of
        # whitespace before matching the prefix.
        title = re.sub(r"\s+", " ", _header_h1(ad_prefix_html)).strip()
        assert title.startswith(
            "Biochemical characterization reveals the functional divergence"
        )

    def test_species_name_italicized(self, ad_prefix_html: str) -> None:
        assert "<em>Przewalskia tangutica</em>" in ad_prefix_html

    def test_gene_names_not_collapsed_by_ocr(self, ad_prefix_html: str) -> None:
        # The motivating bug: the old OCR misread both PtTRI and PtTRII as the
        # single string "PITRI", collapsing two distinct genes.  LightOnOCR reads
        # them correctly; tables are excluded so the check targets the prose.
        prose = re.sub(r"<table.*?</table>", "", ad_prefix_html, flags=re.DOTALL)
        assert "PtTRI" in prose
        assert "PtTRII" in prose
        assert "PITRI" not in prose

    def test_figures_embedded_by_detector(self, ad_prefix_html: str) -> None:
        # LightOnOCR-bbox emits a crop box per figure (incl. the Fig 2 alignment
        # the old engine OCRed into a token wall); each is cropped and embedded.
        assert ad_prefix_html.count("data:image/png;base64,") >= 4

    def test_figures_not_truncated(self, ad_prefix_html: str) -> None:
        # The model emits boxes normalized to [0, 1000]; cropping them as raw
        # pixels truncated every figure.  A page-spanning figure (the Fig 2
        # alignment) only exceeds 1000 px wide once the box is denormalized to
        # the ~1190 px render width — impossible if coords are read as pixels.
        widest = max(w for w, _ in _figure_sizes(ad_prefix_html))
        assert widest > 1000, f"widest figure is only {widest}px — boxes not scaled"

    def test_cross_page_paragraph_not_split(self, ad_prefix_html: str) -> None:
        # The clause "…TRI and" / "TRII compete…" spans a page break; it must
        # be a single paragraph, not split at the page boundary.
        assert (
            "This suggests that TRI and TRII compete for the same substrate"
            in ad_prefix_html
        )
        assert "This suggests that TRI and</p>" not in ad_prefix_html


@pytest.fixture(scope="session")
def plos_run(ocr_model: object) -> object:
    """Run the PLOS ONE 32639976.pdf pipeline once, capturing the table re-OCR
    batching and gate decisions so the gate test needs no second OCR pass.

    Wraps ``_recover_dropped_tables`` to record the size of each batched
    ``ocr_regions`` call, and ``_region_fully_captured`` to count how many regions
    the coverage gate actually skipped.  Returns the HTML plus that spy data;
    ``plos_html`` derives from it, so the whole fixture costs a single pipeline run.
    Writes spike_results/32639976.html too.
    """
    if not _PLOS_PDF.exists():
        pytest.skip(f"Fixture PDF not found: {_PLOS_PDF}")
    from types import SimpleNamespace

    from pdfparser.pipeline import (
        OcrModel,
        assemble,
        lightonocr_pdf_to_html,
        tables,
    )

    assert isinstance(ocr_model, OcrModel)
    real_recover = assemble._recover_dropped_tables
    real_gate = tables._region_fully_captured
    spy = SimpleNamespace(batches=[], gate_skips=0, html="")

    def recover_with_spy(pdf_path, pages_md, ocr_regions):  # type: ignore[no-untyped-def]
        def counting(regions):  # type: ignore[no-untyped-def]
            spy.batches.append(len(regions))
            return ocr_regions(regions)

        return real_recover(pdf_path, pages_md, counting)

    def gate_with_spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        skipped = real_gate(*args, **kwargs)
        spy.gate_skips += int(skipped)
        return skipped

    mp = pytest.MonkeyPatch()
    mp.setattr(assemble, "_recover_dropped_tables", recover_with_spy)
    # Count actual gate decisions, so the gate test can't be satisfied by a region
    # that merely failed to localize (which also lowers the batch size).
    mp.setattr(tables, "_region_fully_captured", gate_with_spy)
    try:
        spy.html = lightonocr_pdf_to_html(_PLOS_PDF, ocr=ocr_model)
    finally:
        mp.undo()
    _OUTPUT_DIR.mkdir(exist_ok=True)
    (_OUTPUT_DIR / f"{_PLOS_PDF.stem}.html").write_text(spy.html, encoding="utf-8")
    return spy


@pytest.fixture(scope="session")
def plos_html(plos_run: object) -> str:
    """Full pipeline HTML for the PLOS fixture (the single run in ``plos_run``)."""
    return plos_run.html


@pytest.mark.integration
class TestPlosSidebarMetadata:
    """32639976.pdf (PLOS ONE) prints a left-column metadata sidebar beside the
    abstract; the OCR interleaves it into the body.  Every piece — the affiliation
    and the Citation/Editor/dates/Copyright/Data-Availability/Funding/
    Competing-interests run, which LightOnOCR emits as ``<h3>`` label headings —
    must land in the collapsed Metadata panel, not the body."""

    def test_sidebar_metadata_in_panel_not_body(self, plos_html: str) -> None:
        panel, body = _metadata(plos_html), _body(plos_html)
        # The labels relocate as their own headings (not bold "**Label:**" lines),
        # so match the heading form — and use it for "Funding" specifically, whose
        # bare word also appears in the body's "Funding acquisition:" CRediT role.
        for fragment in (
            "Department of Biomedical Science and Center for Bio-Nanomaterials",
            "<h3>Citation</h3>",
            "<h3>Editor</h3>",
            "<h3>Received</h3>",
            "<h3>Accepted</h3>",
            "<h3>Published</h3>",
            "<h3>Copyright</h3>",
            "<h3>Data Availability Statement</h3>",
            "<h3>Funding</h3>",
            "<h3>Competing interests</h3>",
        ):
            assert fragment in panel, f"{fragment!r} missing from metadata panel"
            assert fragment not in body, f"{fragment!r} leaked into body"

    def test_title_in_header_not_masthead(self, plos_html: str) -> None:
        # The journal masthead "PLOS ONE" must not be taken as the article title;
        # the real title (with its italicised species) belongs in the header.
        header = plos_html[: plos_html.find("</header>")]
        title = (
            "Purification and characterization of a novel medium-chain ribitol "
            "dehydrogenase from a lichen-associated bacterium "
            "<em>Sphingomonas</em> sp."
        )
        assert title in header, "article title missing from header"
        assert "<h1>PLOS ONE</h1>" not in header

    def test_body_opens_with_introduction_prose(self, plos_html: str) -> None:
        # With the sidebar relocated, the body proper opens at the Introduction —
        # its prose must remain visible, not be swept into the panel.
        body = _body(plos_html)
        assert "Lichens have traditionally been considered" in body
        assert "Polyols have a role in carbohydrate storage" in body

    def test_running_head_not_in_body(self, plos_html: str) -> None:
        # The running head ends in "Sphingomonas sp." — an abbreviation that only
        # looks like a sentence end — so it must still be stripped as furniture,
        # not leak onto every page of the body.
        body = _body(plos_html)
        assert "lichen-associated <em>Sphingomonas</em> sp.</p>" not in body

    def test_growth_paragraph_merged_across_interleaved_figure(
        self, plos_html: str
    ) -> None:
        # The Results paragraph breaks across a page, with Fig 1 (and the stripped
        # running head) interleaved between its halves; once the running head is
        # gone the merge bridges the figure, joining the two fragments into one.
        body = _body(plos_html)
        assert "or D-mannitol) at 15°C and its growth measured for 16 days." in body

    def test_repeated_section_heading_kept(self, plos_html: str) -> None:
        # "Purification of SpRDH" titles both a Methods subsection and a Results
        # section; recurring as a heading is not running furniture, so both stay.
        body = _body(plos_html)
        assert "<h3>Purification of SpRDH</h3>" in body
        assert "<h2>Purification of SpRDH</h2>" in body


@pytest.mark.integration
class TestPlosFigureCrops:
    """Figure crops on 32639976.pdf (PLOS ONE).  The model's bbox clips the right
    edge of the wide Fig 5 alignment, and for several figures sits low enough that
    the bottom-growth would bake the caption into the crop.  The right edge is
    recovered by ``_extend_right_to_content``; ``_trim_swallowed_caption`` drops a
    recovered band that reads as the caption when the OCR also emitted that caption
    as its own text block."""

    def test_fig5_alignment_right_edge_recovered(self, plos_html: str) -> None:
        # The model clips Fig 5's right edge to ≈0.88·page-width; the alignment
        # actually spans the full text column, so the recovered crop must reach
        # well past that — independent of the model's clipped x1, which growth
        # ignores in favour of the figure's real right edge on the page.
        size = _figure_size_by_caption(plos_html, "Multiple sequence alignments")
        assert size is not None, "Fig 5 alignment figure not embedded"
        assert size[0] > 920, f"Fig 5 width {size[0]}px — right edge not recovered"

    def test_fig5_baked_caption_trimmed(self, plos_html: str) -> None:
        # The alignment is itself text, so the model boxes its caption + DOI inside
        # the figure; re-OCR confirms those trailing bands reproduce the caption and
        # trims them.  Left in, the crop runs ~130 px taller (caption + DOI baked in).
        size = _figure_size_by_caption(plos_html, "Multiple sequence alignments")
        assert size is not None, "Fig 5 alignment figure not embedded"
        assert size[1] < 830, f"Fig 5 height {size[1]}px — caption baked into crop"

    def test_carbon_source_plot_caption_not_baked_in(self, plos_html: str) -> None:
        # Fig 1 is a roughly square growth-curve plot; the model's box bottom abuts
        # its 3-line caption + DOI, which the bottom-growth pulls in.  With the
        # caption trimmed the crop stays near the plot's own height (~590 px); left
        # un-trimmed it grows past 700 px.
        size = _figure_size_by_caption(plos_html, "Effect of different carbon sources")
        assert size is not None, "Fig 1 plot figure not embedded"
        assert size[1] < 660, f"Fig 1 height {size[1]}px — caption baked into crop"


class TestTableTextHelpers:
    """Pure markup helpers for table re-OCR substitution."""

    def test_cell_texts_and_count(self) -> None:
        from pdfparser.pipeline.tables import _cell_texts, _nonempty_cell_count

        table = (
            "<table><thead><tr><th>Metal ion</th><th></th></tr></thead>"
            "<tbody><tr><td>Mg<sup>2+</sup></td><td>3.0</td></tr></tbody></table>"
        )
        assert _cell_texts(table) == ["Metal ion", "Mg2+", "3.0"]
        # the empty <th> does not count
        assert _nonempty_cell_count(table) == 3

    def test_table_regions_group_consecutive_split_on_prose(self) -> None:
        from pdfparser.pipeline.tables import _table_regions

        md = (
            "intro\n\n"
            "<table><tr><td>a</td></tr></table>\n\n"
            "<table><tr><td>b</td></tr></table>\n\n"
            "some prose\n\n"
            "<table><tr><td>c</td></tr></table>"
        )
        regions = _table_regions(md)
        assert [len(tables) for _, _, tables in regions] == [2, 1]
        # the spans address only the <table> blocks, not the prose between regions
        start, end, _ = regions[1]
        assert md[start:end] == "<table><tr><td>c</td></tr></table>"

    def test_crop_trailing_returns_text_after_last_table(self) -> None:
        from pdfparser.pipeline.tables import _crop_trailing

        legend = "MW: molecular weight, NR: Not reported"
        md = f"<table><tr><td>x</td></tr></table>\n\n{legend}"
        assert _crop_trailing(md) == legend
        assert _crop_trailing("no table here") == ""

    def test_legend_footnote_preserves_sup_marker(self) -> None:
        from pdfparser.pipeline.tables import _legend_footnote_html

        # The recovered legend's <sup> marker and *emphasis* are OCR markup: render
        # them, don't HTML-escape (which would print a literal "<sup>a</sup>").
        legend = "<sup>a</sup>Each value represents the mean ± SD, *n* = 3."
        assert _legend_footnote_html(legend) == (
            '<p class="footnote"><sup>a</sup>Each value represents the mean ± SD, '
            "<em>n</em> = 3.</p>"
        )

    def test_legend_footnote_escapes_stray_markup(self) -> None:
        from pdfparser.pipeline.tables import _legend_footnote_html

        # A bare "<"/"&" in a legend ("n<5", "Tris & HCl") must be escaped, not left
        # to start a bogus tag — while a real <sup> marker still passes through.
        legend = "<sup>a</sup>Significant at n<5; Tris & HCl buffer"
        assert _legend_footnote_html(legend) == (
            '<p class="footnote"><sup>a</sup>Significant at n&lt;5; '
            "Tris &amp; HCl buffer</p>"
        )

    def test_extract_tables_strips_inner_caption(self) -> None:
        from pdfparser.pipeline.tables import _extract_tables

        # a level-1 heading is the overall caption (carried separately), so it is
        # not folded into the table
        md = (
            "# Table 2\n\n"
            "<table><caption>Table 2. X</caption><tr><td>x</td></tr></table>\n"
        )
        assert _extract_tables(md) == ["<table><tr><td>x</td></tr></table>"]

    def test_extract_tables_folds_subheading_as_spanning_row(self) -> None:
        from pdfparser.pipeline.tables import _extract_tables

        # the crop re-OCR lifts a sub-table label into a level-2 heading; it must
        # come back as a spanning header row spanning all the table's columns
        md = (
            "## A. Effect on activity\n\n"
            "<table><tbody><tr><td>None</td><td>100</td><td>99</td></tr>"
            "</tbody></table>"
        )
        [table] = _extract_tables(md)
        expected = '<thead><tr><th colspan="3">A. Effect on activity</th></tr></thead>'
        assert expected in table


class TestTableNormalization:
    """NFKD-plus-alnum folding closes the encoding gap between an OCR'd cell and
    the PDF text layer so the same content matches across both."""

    def test_superscript_and_micro_fold_to_text_layer_form(self) -> None:
        from pdfparser.pipeline.tables import _normalize

        assert _normalize("Mg²⁺") == _normalize("Mg2+") == "mg2"
        # micro sign vs Greek mu, and superscript ⁻¹ (a U+2212 minus) vs ASCII -1
        assert _normalize("µg L⁻¹") == _normalize("μg L-1")

    def test_index_map_recovers_source_range(self) -> None:
        from pdfparser.pipeline.tables import _normalize_with_map

        text = "A: Mg²⁺ ok"
        norm, idx_map = _normalize_with_map(text)
        assert len(norm) == len(idx_map)
        # the normalized "mg2" maps back onto the original "Mg²" span
        p = norm.index("mg2")
        assert text[idx_map[p] : idx_map[p + 2] + 1] == "Mg²"


class TestTableLocalization:
    """``_locate_bbox`` grows an anchor seed through the table's rows but halts at
    the wider margin to body prose, so a re-OCR crop stays tight on the table."""

    @staticmethod
    def _layout(
        lines: list[tuple[str, float | int, float | int]],
    ) -> tuple[str, list[tuple[float, float, float, float] | None]]:
        # Lay each (text, y_top, x_left) line out as fixed 6×8 pt glyph boxes,
        # spaces and the inter-line newline carrying a degenerate (None) box —
        # mirroring how pdfium reports them.
        text = ""
        boxes: list[tuple[float, float, float, float] | None] = []
        for i, (s, y_top, x_left) in enumerate(lines):
            if i:
                text += "\n"
                boxes.append(None)
            x = float(x_left)
            top = float(y_top)
            for ch in s:
                text += ch
                boxes.append(None if ch == " " else (x, top - 8, x + 6, top))
                x += 6
        return text, boxes

    def test_bbox_covers_table_rows_excludes_prose(self) -> None:
        from pdfparser.pipeline.tables import _locate_bbox, _normalize_with_map

        # A 5-row table at the top, then a wide gap, then dense prose.  Only the
        # heading row is a (unique) anchor; growth must still reach the trailing
        # rows yet stop before the prose.
        lines = [
            ("Effect of EDTA on activity", 700, 50),
            ("Relative activity percent", 688, 200),
            ("None 100 100", 676, 50),
            ("EDTA 100 99", 664, 50),
            ("12 34 56", 652, 50),
            ("Discussion text begins here now", 600, 50),
            ("and continues across the page", 588, 50),
        ]
        text, boxes = self._layout(lines)
        norm, idx_map = _normalize_with_map(text)
        bbox = _locate_bbox(
            ["effect of edta on activity"], norm, idx_map, boxes, (400.0, 800.0)
        )
        assert bbox is not None
        left, bottom, right, top = bbox
        assert top >= 700 and bottom <= 644  # spans heading down to the "12 34 56" row
        assert bottom > 600  # the Discussion prose is excluded
        # the dropped-style interior row lies within the located region
        assert bottom <= 680 and top >= 688

    def test_returns_none_when_no_anchor_matches(self) -> None:
        from pdfparser.pipeline.tables import _locate_bbox, _normalize_with_map

        text, boxes = self._layout([("Effect of EDTA on activity", 700, 50)])
        norm, idx_map = _normalize_with_map(text)
        assert _locate_bbox(["nowhere"], norm, idx_map, boxes, (400.0, 800.0)) is None

    def test_repeated_anchor_is_ambiguous_and_skipped(self) -> None:
        from pdfparser.pipeline.tables import _locate_bbox, _normalize_with_map

        # "alpha beta" occurs twice, so it cannot seed the box on its own.
        text, boxes = self._layout(
            [("alpha beta", 700, 50), ("filler here", 660, 50), ("alpha beta", 300, 50)]
        )
        norm, idx_map = _normalize_with_map(text)
        bbox = _locate_bbox(["alpha beta"], norm, idx_map, boxes, (400.0, 800.0))
        assert bbox is None


class TestCoverageGate:
    """``_region_fully_captured`` skips a table's crop re-OCR only when the page
    already captured every distinctive text-layer token inside the located bbox."""

    @staticmethod
    def _centers(
        lines: list[tuple[str, float | int, float | int]],
    ) -> tuple[str, list[tuple[float, float] | None]]:
        from pdfparser.pipeline.tables import _glyph_centers, _normalize_with_map

        text, boxes = TestTableLocalization._layout(lines)
        norm, idx_map = _normalize_with_map(text)
        return norm, _glyph_centers(norm, idx_map, boxes)

    def test_in_bbox_tokens_keeps_only_in_box_words(self) -> None:
        from pdfparser.pipeline.tables import _in_bbox_tokens

        norm, centers = self._centers(
            [("alpha beta", 700, 50), ("gamma delta", 600, 50)]
        )
        toks = _in_bbox_tokens((0.0, 690.0, 400.0, 710.0), norm, centers)
        assert toks == ["alpha", "beta"]  # top line only, bottom line excluded

    def test_in_bbox_tokens_does_not_glue_clipped_word(self) -> None:
        from pdfparser.pipeline.tables import _in_bbox_tokens

        # The right edge cuts off "beta"; "alpha" must survive as its own token —
        # the box-less space still has to break the words, not weld "alphabeta".
        norm, centers = self._centers([("alpha beta", 700, 50)])
        toks = _in_bbox_tokens((0.0, 690.0, 82.0, 710.0), norm, centers)
        assert toks == ["alpha"]

    def test_fully_captured_when_all_distinctive_tokens_present(self) -> None:
        from pdfparser.pipeline.tables import _region_fully_captured

        norm, centers = self._centers([("alpha beta", 700, 50)])
        assert _region_fully_captured(
            (0.0, 690.0, 400.0, 710.0), norm, centers, {"alpha", "beta"}
        )

    def test_not_captured_when_a_token_is_missing(self) -> None:
        from pdfparser.pipeline.tables import _region_fully_captured

        # "beta" is in the text layer but not the captured cells — content the page
        # pass dropped — so the gate must not fire and the region is re-OCR'd.
        norm, centers = self._centers([("alpha beta", 700, 50)])
        assert not _region_fully_captured(
            (0.0, 690.0, 400.0, 710.0), norm, centers, {"alpha"}
        )

    def test_numeric_drop_is_not_skipped(self) -> None:
        from pdfparser.pipeline.tables import _region_fully_captured

        # All words captured but a multi-digit data value (26621) missing — a dropped
        # data cell. Numbers count as distinctive evidence, so the gate must not fire.
        norm, centers = self._centers([("alpha 26621", 700, 50)])
        assert not _region_fully_captured(
            (0.0, 690.0, 400.0, 710.0), norm, centers, {"alpha"}
        )

    def test_empty_text_layer_forces_reocr(self) -> None:
        from pdfparser.pipeline.tables import _region_fully_captured

        # A bbox over a region with no glyphs (a scanned page) gives no evidence, so
        # the gate stays off — it never opts out on absence of evidence.
        norm, centers = self._centers([("alpha beta", 700, 50)])
        assert not _region_fully_captured(
            (0.0, 100.0, 400.0, 200.0), norm, centers, {"alpha", "beta"}
        )

    def test_short_non_numeric_tokens_are_not_evidence(self) -> None:
        from pdfparser.pipeline.tables import _region_fully_captured

        # Only short non-numeric tokens in the bbox — no distinctive token to judge
        # completeness, so the gate does not fire even with nothing captured.
        norm, centers = self._centers([("ab cd ef", 700, 50)])
        assert not _region_fully_captured(
            (0.0, 690.0, 400.0, 710.0), norm, centers, set()
        )

    def test_adjacent_para_tokens_pick_caption_and_legend(self) -> None:
        from pdfparser.pipeline.tables import _adjacent_para_tokens

        md = (
            "Table 1. Effect of metals on activity\n\n"
            "<table><tr><td>x</td></tr></table>\n\n"
            "MW molecular weight reported"
        )
        start = md.index("<table>")
        end = md.index("</table>") + len("</table>")
        toks = _adjacent_para_tokens(md, start, end)
        assert {"effect", "metals"} <= toks  # caption above the table
        assert {"molecular", "reported"} <= toks  # legend below the table

    def test_adjacent_para_ignores_long_prose_block(self) -> None:
        from pdfparser.pipeline.tables import _adjacent_para_tokens

        # A long block flanking the table (e.g. body prose the bbox overran, or the
        # whole pre-table content when no blank line separates it) is NOT folded into
        # 'captured' — otherwise its tokens could mask a genuinely dropped table cell.
        long_prose = "discussion " * 60  # well past _ADJACENT_PARA_MAX_LEN
        md = f"{long_prose}<table><tr><td>x</td></tr></table>"
        start = md.index("<table>")
        end = md.index("</table>") + len("</table>")
        assert "discussion" not in _adjacent_para_tokens(md, start, end)


@pytest.mark.integration
class TestPlosTableReocr:
    """32639976.pdf Table 2: the full-page OCR pass drops the column-spanning
    subheader "Relative activity (%)" from part A.  Re-OCRing the table region as a
    tight crop (localized via the text layer) recovers it, with the spanning markup
    the full-page pass omitted."""

    def test_relative_activity_header_recovered(self, plos_html: str) -> None:
        body = _body(plos_html)
        assert "Relative activity (%)" in body

    def test_recovered_header_spans_its_columns(self, plos_html: str) -> None:
        # the crop re-OCR emits the subheader as a colspan cell over the two data
        # columns — structure the full-page pass never produced
        body = _body(plos_html)
        m = re.search(r'colspan="2"[^>]*>\s*Relative activity', body)
        assert m is not None, "spanning subheader not recovered with colspan"


@pytest.mark.integration
class TestTableReocrGate:
    """The coverage gate skips re-OCR for a table the page pass already captured in
    full, and the crops that remain are re-OCR'd in one batched (concurrent) call."""

    def test_complete_table_skipped_and_crops_batched(self, plos_run: object) -> None:
        # The coverage gate actually skipped at least one region (measured at the
        # gate itself, so a mere localization failure can't satisfy this): PLOS
        # Table 1 (Purification summary) is fully captured.  And the crops that
        # remain are re-OCR'd in a single batched call, not one request per region.
        # Spy data comes from the shared plos_run, so there is no second OCR.
        assert plos_run.gate_skips >= 1, "coverage gate skipped no complete table"
        assert len(plos_run.batches) == 1, "table crops were not OCR'd in one batch"
        assert plos_run.batches[0] >= 1, "no table region was re-OCR'd"


# The table-content needles below are not harvested from the pipeline's own OCR
# output (that would be circular — it would only pin whatever the OCR happens to
# emit, garbage included).  Each is verified to exist in the PDF's embedded text
# layer — the publisher's ground truth, independent of our OCR — so the test
# catches a re-OCR that drops or mislocates a table into wrong-region content.


@pytest.mark.integration
class TestHpcdhTableContent:
    """30592559.pdf carries four data tables (TABLE I–IV); each must keep its real
    cell content through the table re-OCR pass.  Needles confirmed in the PDF text
    layer (ground truth), not copied from OCR output."""

    def test_all_four_table_captions_present(self, article_html: str) -> None:
        body = _body(article_html)
        for caption in ("TABLE I", "TABLE II", "TABLE III", "TABLE IV"):
            assert caption in body, f"{caption!r} missing"

    def test_distinctive_cells_live_in_tables(self, article_html: str) -> None:
        # one distinctive ground-truth cell per table, asserted inside <table>
        # markup (not merely loose in the body prose)
        cells = _tables_text(article_html)
        for needle in (
            "Enantioselectivity",  # TABLE I header
            "2-Butanone",  # TABLE II column
            "2-Butanol production",  # TABLE III header
            "rR-HPCDH",  # TABLE IV row
        ):
            assert needle in cells, f"{needle!r} not found inside any table"


@pytest.mark.integration
class TestTropinoneTableContent:
    """31051047.pdf has two kinetics tables (TABLE 1 and an uncaptioned homolog
    comparison).  Needles confirmed in the PDF text layer (ground truth)."""

    def test_kinetics_table_content(self, ad_prefix_html: str) -> None:
        cells = _tables_text(ad_prefix_html)
        # nKat is the kinetics table's activity unit; both isoforms are its rows
        assert "nKat" in cells
        assert "PtTRI" in cells and "PtTRII" in cells

    def test_homolog_table_species_and_reference(self, ad_prefix_html: str) -> None:
        # the second table lists homologs by species with a reference column; its
        # rows must not be lost
        cells = _tables_text(ad_prefix_html)
        assert "Przewalskia tangutica" in cells
        assert "In this study" in cells


class TestUnclosedTableClosing:
    """A table the OCR leaves open at a page bottom must be closed before assembly,
    or it swallows whatever follows (most visibly the next page's prose)."""

    def test_close_unclosed_tables_balances_and_is_idempotent(self) -> None:
        from pdfparser.pipeline.tables import _close_unclosed_tables

        assert (
            _close_unclosed_tables("<table><tr><td>a</td><td>0")
            == "<table><tr><td>a</td><td>0</table>"
        )
        balanced = "<table><tr><td>a</td></tr></table>"
        assert _close_unclosed_tables(balanced) == balanced

    def test_overrun_table_does_not_swallow_next_page_prose(self) -> None:
        # page 1's table runs off the bottom with no </table>; page 2 opens with
        # prose that must render as body text, not inside the table
        page1 = "<table>\n<tr><td>Organism</td><td>M.W.</td></tr>\n<tr><td>X</td><td>0"
        page2 = "Downstream prose that follows the table on the next page."
        html = _run_lighton([page1, page2])
        assert "Downstream prose that follows" in _body(html)
        assert "Downstream prose that follows" not in _tables_text(html)


@pytest.mark.integration
class TestPlosTableOverrun:
    """Table 3 of 32639976.pdf overruns its page; the OCR leaves it unclosed and the
    following page's prose ("SpRDH operon …", verified in the PDF text layer) would
    render inside the table.  It must stay body prose, and tables must be balanced."""

    def test_post_table_prose_not_in_table(self, plos_html: str) -> None:
        cells = _tables_text(plos_html)
        assert "SpRDH operon" in _body(plos_html)  # present as prose
        assert "SpRDH operon" not in cells
        assert "PAMC 26621 genome" not in cells

    def test_all_tables_balanced(self, plos_html: str) -> None:
        # the general invariant: no table is left open to absorb following content
        body = _body(plos_html)
        assert body.count("<table") == body.count("</table>")

    def test_truncated_last_row_recovered(self, plos_html: str) -> None:
        # the last row (Enterobacter aerogenes) was cut at the page bottom; once the
        # table is closed it becomes a region the crop re-OCR recovers in full,
        # restoring the row's tail — Km NAD⁺ 0.16, kcat 318, kcat/Km 30.9 (all
        # verified in the PDF text layer)
        cells = _tables_text(plos_html)
        for value in ("0.16", "318", "30.9"):
            assert value in cells, f"{value!r} not recovered into the table"

    def test_table_legend_recovered(self, plos_html: str) -> None:
        # the legend was truncated together with the overrun row; the crop re-OCR
        # (bbox extended one line to reach it) recovers it as a footnote under the
        # table — verified in the PDF text layer
        text = re.sub(r"<[^>]+>", "", _body(plos_html))
        assert "MW: molecular weight, NR: Not reported" in text

    def test_paragraph_split_across_tables_rejoined(self, plos_html: str) -> None:
        # the sentence is split at the page break ("…carbon metabolism. The" on one
        # page, "SpRDH operon …" on the next) with Tables 2 and 3 between the halves;
        # the cross-table merge must rejoin it into one paragraph
        text = re.sub(r"<[^>]+>", "", _body(plos_html))
        assert (
            "The SpRDH operon of Sphingomonas sp. PAMC 26621 genome "
            "contains a putative ABC transporter" in text
        )


@pytest.mark.integration
class TestPlosTableFormatting:
    """Inline markup the OCR leaves inside table blocks must render, not surface
    as literal source: organism names in Table 3 cells are italicised, and Table
    2's footnote keeps its real <sup> marker rather than an escaped tag."""

    def test_table3_organism_names_italicised(self, plos_html: str) -> None:
        # Table 3's Organism column carries ``*Klebsiella aerogenes*`` etc. as raw
        # markdown inside the HTML cell; it must come out as <em>, not bare "*".
        body = _body(plos_html)
        assert "<em>Klebsiella aerogenes</em>" in body
        assert "<em>Zymomonas mobilis</em>" in body
        assert "*Klebsiella" not in body

    def test_table2_footnote_marker_not_escaped(self, plos_html: str) -> None:
        # Table 2's footnote marker is recovered as ``<sup>a</sup>…``; it must render
        # as a superscript, never as an HTML-escaped "&lt;sup&gt;" literal anywhere.
        text = re.sub(r"<[^>]+>", "", _body(plos_html))
        assert (
            "Each value represents the mean ± SD of three independent experiments"
            in text
        )
        assert "&lt;sup&gt;" not in plos_html
