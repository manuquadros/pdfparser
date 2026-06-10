"""Tests for the pdfparser.pipeline package.  Unit tests load no model and render
no PDF; the integration tests (``@pytest.mark.integration``) run the real pipeline."""

import base64
import io
import re
from pathlib import Path

import pytest
from PIL import Image


def _fake_image(width: int = 800, height: int = 1000) -> Image.Image:
    return Image.new("RGB", (width, height), color="white")


def _figure_sizes(html: str) -> list[tuple[int, int]]:
    """Decode every embedded ``data:image/png`` figure and return its (w, h)."""
    uris = re.findall(r"data:image/png;base64,([A-Za-z0-9+/=]+)", html)
    return [Image.open(io.BytesIO(base64.b64decode(u))).size for u in uris]


def _run_lighton(pages_md: list[str], image: Image.Image | None = None) -> str:
    """Assemble HTML from synthetic per-page markdown (no model, no rendering)."""
    from pdfparser.pipeline.assemble import _assemble_html

    img = image or _fake_image(1190, 1540)
    return _assemble_html(pages_md, [img for _ in pages_md])


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


class TestCrossPageMerge:
    """A paragraph split across a page break is rejoined."""

    def test_cross_page_paragraph_merge(self) -> None:
        page1 = "# T\n\n## Abstract\n\nA.\n\n## Body\n\nThis suggests that TRI and"
        page2 = "TRII compete for the same substrate tropinone."
        html = _run_lighton([page1, page2])
        assert "This suggests that TRI and TRII compete for the same substrate" in html
        assert "This suggests that TRI and</p>" not in html


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
_SPIKE_HTML = Path(__file__).parent.parent / "spike_results" / "lighton_full.html"


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

    Writes the result back to spike_results/lighton_full.html so the file
    stays current after each integration run.
    """
    if not _FIXTURE_PDF.exists():
        pytest.skip(f"Fixture PDF not found: {_FIXTURE_PDF}")
    from pdfparser.pipeline import OcrModel, lightonocr_pdf_to_html

    assert isinstance(ocr_model, OcrModel)
    html = lightonocr_pdf_to_html(_FIXTURE_PDF, ocr=ocr_model)
    _SPIKE_HTML.write_text(html, encoding="utf-8")
    return html


def _header_h1(html: str) -> str:
    """Return the text of the document's <header><h1> title element."""
    m = re.search(r"<header>.*?<h1>(.*?)</h1>", html, re.DOTALL)
    assert m, "header <h1> not found"
    return m.group(1)


@pytest.mark.integration
class TestPipeline:
    """Integration tests: run the full LightOnOCR pipeline on the fixture PDF.

    Skipped when the model is not available (no GPU, weights not downloaded).
    Each run also refreshes spike_results/lighton_full.html.
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

    def test_paper_footnotes_after_body_before_references(
        self, article_html: str
    ) -> None:
        correspondence = "To whom correspondence should be addressed"
        assert correspondence in article_html
        fn_pos = article_html.find(correspondence)
        # Must follow substantial body content, not appear near the top.
        body_start = article_html.find('<div class="body">')
        assert fn_pos - body_start > 2000
        # Must precede the reference list.
        first_ref_pos = article_html.find("<p>[1]")
        if first_ref_pos != -1:
            assert fn_pos < first_ref_pos


@pytest.fixture(scope="session")
def ad_prefix_html(ocr_model: object) -> str:
    """Full pipeline output for the ad-prefixed 31051047.pdf fixture."""
    if not _AD_PREFIX_PDF.exists():
        pytest.skip(f"Fixture PDF not found: {_AD_PREFIX_PDF}")
    from pdfparser.pipeline import OcrModel, lightonocr_pdf_to_html

    assert isinstance(ocr_model, OcrModel)
    return lightonocr_pdf_to_html(_AD_PREFIX_PDF, ocr=ocr_model)


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
