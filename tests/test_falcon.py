"""Unit tests for falcon.py — no model loading, no PDF rendering."""

import re
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


def _fake_image(width: int = 800, height: int = 1000) -> Image.Image:
    return Image.new("RGB", (width, height), color="white")


def _run_falcon(regions_per_page: list[list[dict]]) -> str:
    """Run falcon_pdf_to_html with a stub model and synthetic pages.

    The stubbed pages carry no text layer (``text_page=None``), so the pipeline
    uses Falcon's region text exactly as in production for scanned input.
    """
    from pdfparser.falcon import RenderedPage, falcon_pdf_to_html

    fake_pages = [
        RenderedPage(
            image=_fake_image(),
            scale=200 / 72,
            page_height_pt=1000.0,
            text_page=None,
        )
        for _ in regions_per_page
    ]
    fake_model = MagicMock()
    fake_model.generate_with_layout.side_effect = [
        [page_regions] for page_regions in regions_per_page
    ]

    @contextmanager
    def fake_render(_pdf_path: Path):
        yield fake_pages

    with patch("pdfparser.falcon._render_pages", fake_render):
        return falcon_pdf_to_html(Path("/fake/paper.pdf"), model=fake_model)


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


class TestAbstractColumnMerge:
    """Two-column PDFs produce the abstract as two separate layout regions
    when the text wraps at the column boundary.  The rendered HTML must stitch
    those fragments into a single unbroken <p> element."""

    def test_mid_sentence_split_yields_single_paragraph(self) -> None:
        regions = [
            {"category": "doc_title", "bbox": [0, 0, 800, 40], "text": "Test Paper"},
            {
                "category": "abstract",
                "bbox": [0, 50, 380, 120],
                "text": (
                    "This is a data rich story that combines both"
                    " classical and contemporary"
                ),
            },
            {
                "category": "abstract",
                "bbox": [420, 50, 800, 120],
                "text": "experimental biochemistry.",
            },
        ]
        html = _run_falcon([regions])

        assert "<p>experimental biochemistry.</p>" not in html
        assert "classical and contemporary</p>" not in html

    def test_mid_sentence_split_joined_with_space(self) -> None:
        regions = [
            {"category": "doc_title", "bbox": [0, 0, 800, 40], "text": "Test Paper"},
            {
                "category": "abstract",
                "bbox": [0, 50, 380, 120],
                "text": (
                    "This is a data rich story that combines both"
                    " classical and contemporary"
                ),
            },
            {
                "category": "abstract",
                "bbox": [420, 50, 800, 120],
                "text": "experimental biochemistry.",
            },
        ]
        html = _run_falcon([regions])

        assert "classical and contemporary experimental biochemistry." in html

    def test_single_abstract_region_unchanged(self) -> None:
        regions = [
            {"category": "doc_title", "bbox": [0, 0, 800, 40], "text": "Test Paper"},
            {
                "category": "abstract",
                "bbox": [0, 50, 800, 120],
                "text": "A straightforward single-region abstract.",
            },
        ]
        html = _run_falcon([regions])
        assert "A straightforward single-region abstract." in html
        assert html.count("<p>A straightforward") == 1

    def test_abstract_text_in_single_paragraph(self) -> None:
        regions = [
            {"category": "doc_title", "bbox": [0, 0, 800, 40], "text": "Test Paper"},
            {
                "category": "abstract",
                "bbox": [0, 50, 380, 120],
                "text": "First fragment without terminal punctuation",
            },
            {
                "category": "abstract",
                "bbox": [420, 50, 800, 120],
                "text": "second fragment closes the sentence.",
            },
        ]
        html = _run_falcon([regions])

        start = html.find("<section class='abstract'>")
        end = html.find("</section>", start)
        assert start >= 0
        block = html[start:end]

        assert "First fragment without terminal punctuation" in block
        assert "second fragment closes the sentence." in block
        assert block.count("<p>") == 1, "expected one <p> inside abstract"


class TestSortRegions:
    """Unit tests for _sort_regions reading-order logic."""

    def test_right_col_low_y_follows_left_col_high_y(self) -> None:
        from pdfparser.falcon import _sort_regions

        # Simulate a paragraph whose left-column fragment is near the bottom
        # of the page (y=700) and whose right-column continuation is near the
        # top (y=80).  The right fragment must sort AFTER the left fragment.
        regions = [
            {"category": "text", "bbox": [0, 700, 380, 750], "text": "left bottom"},
            {"category": "text", "bbox": [420, 80, 800, 130], "text": "right top"},
        ]
        result = _sort_regions(regions, page_width=800.0)
        assert [r["text"] for r in result] == ["left bottom", "right top"]

    def test_full_width_boundary_separates_sections(self) -> None:
        from pdfparser.falcon import _sort_regions

        # A full-width heading at y=400 creates two sections.  A right-column
        # region at y=50 (above the heading, section 0) must come AFTER the
        # left-column region at y=300 (also above, section 0) but BEFORE the
        # left-column region at y=500 (below the heading, section 1).
        regions = [
            {"category": "text", "bbox": [0, 300, 380, 340], "text": "left top"},
            {"category": "text", "bbox": [420, 50, 800, 90], "text": "right top"},
            {
                "category": "paragraph_title",
                "bbox": [0, 400, 800, 430],
                "text": "Heading",
            },
            {"category": "text", "bbox": [0, 500, 380, 540], "text": "left below"},
        ]
        result = _sort_regions(regions, page_width=800.0)
        texts = [r["text"] for r in result]
        assert texts.index("left top") < texts.index("right top")
        assert texts.index("right top") < texts.index("Heading")
        assert texts.index("Heading") < texts.index("left below")


class TestBodyColumnMerge:
    """The same column-wrap merging must apply to body text regions."""

    def _regions(self, *text_pairs: tuple[str, list[int]]) -> list[dict]:
        regions: list[dict] = [
            {"category": "doc_title", "bbox": [0, 0, 800, 40], "text": "Test Paper"},
            {
                "category": "abstract",
                "bbox": [0, 50, 800, 100],
                "text": "Abstract sentence.",
            },
        ]
        for text, bbox in text_pairs:
            regions.append({"category": "text", "bbox": bbox, "text": text})
        return regions

    def test_body_mid_sentence_split_merges(self) -> None:
        regions = self._regions(
            (
                "The enzyme catalyzes the stereospecific oxidation of",
                [0, 120, 380, 170],
            ),
            ("(R)-hydroxypropyl-coenzyme M.", [420, 120, 800, 170]),
        )
        body = _body(_run_falcon([regions]))

        assert "<p>(R)-hydroxypropyl-coenzyme M.</p>" not in body
        assert "stereospecific oxidation of (R)-hydroxypropyl-coenzyme M." in body

    def test_body_complete_paragraph_not_merged(self) -> None:
        regions = self._regions(
            ("First complete sentence ends here.", [0, 120, 800, 170]),
            ("Second complete sentence, independent.", [0, 180, 800, 230]),
        )
        body = _body(_run_falcon([regions]))

        assert "First complete sentence ends here." in body
        assert "Second complete sentence, independent." in body
        assert body.count("<p>") == 2

    def test_table_between_fragments_moves_after_merged_paragraph(self) -> None:
        from pdfparser.falcon import _merge_split_paragraphs

        table = "<table><tr><td>val</td></tr></table>"
        parts = [
            "<p>The rate constants are</p>",
            table,
            "<p>consistent with the proposed mechanism.</p>",
        ]
        result = _merge_split_paragraphs(parts)

        assert len(result) == 2
        assert "rate constants are consistent with the proposed mechanism." in result[0]
        assert result[1] == table

    def test_heading_between_fragments_is_a_barrier(self) -> None:
        regions = [
            {"category": "doc_title", "bbox": [0, 0, 800, 40], "text": "Test Paper"},
            {
                "category": "abstract",
                "bbox": [0, 50, 800, 100],
                "text": "Abstract sentence.",
            },
            {
                "category": "text",
                "bbox": [0, 120, 380, 170],
                "text": "End of introduction without terminal punctuation",
            },
            {
                "category": "paragraph_title",
                "bbox": [0, 180, 800, 210],
                "text": "Methods",
            },
            {
                "category": "text",
                "bbox": [0, 220, 800, 270],
                "text": "We used mass spectrometry.",
            },
        ]
        body = _body(_run_falcon([regions]))

        assert "<h2>Methods</h2>" in body
        assert "We used mass spectrometry." in body
        heading_pos = body.find("<h2>Methods</h2>")
        intro_frag_pos = body.find("End of introduction")
        methods_pos = body.find("We used mass spectrometry.")

        assert intro_frag_pos < heading_pos < methods_pos

    def test_enum_item_not_merged_into_preceding_paragraph(self) -> None:
        regions = self._regions(
            ("Results for all conditions are presented below", [0, 120, 380, 170]),
            ("1. Condition A yielded the highest activity.", [420, 120, 800, 170]),
        )
        body = _body(_run_falcon([regions]))

        assert "presented below" in body
        assert "1. Condition A yielded" in body
        assert "presented below 1." not in body

    def test_parenthetical_enum_not_merged(self) -> None:
        regions = self._regions(
            ("Samples were processed in three steps", [0, 120, 380, 170]),
            ("(a) centrifugation at 3000 rpm.", [420, 120, 800, 170]),
        )
        body = _body(_run_falcon([regions]))

        assert "processed in three steps" in body
        assert "(a) centrifugation" in body
        assert "three steps (a)" not in body

    def test_three_fragment_chain_fully_merged(self) -> None:
        from pdfparser.falcon import _merge_split_paragraphs

        parts = [
            "<p>The enzyme catalyzes the stereospecific</p>",
            "<p>oxidation of (R)-hydroxypropyl-coenzyme</p>",
            "<p>M to the corresponding ketone.</p>",
        ]
        result = _merge_split_paragraphs(_merge_split_paragraphs(parts))

        assert len(result) == 1
        assert (
            "stereospecific oxidation of (R)-hydroxypropyl-coenzyme M to the"
            " corresponding ketone."
        ) in result[0]

    def test_function_word_end_blocks_uppercase_continuation(self) -> None:
        from pdfparser.falcon import _merge_split_paragraphs

        # Fragment ends with "is" (function word): the continuation MUST start
        # lowercase in normal prose.  An uppercase start signals the real
        # continuation was dropped and a new sentence follows — don't merge.
        parts = [
            "<p>Propylene is metabolized into epoxypropane, which is</p>",
            "<p>As a testament to the utility of enzyme kinetics.</p>",
        ]
        result = _merge_split_paragraphs(parts)
        assert len(result) == 2
        assert "which is As a testament" not in " ".join(result)

    def test_document_type_label_suppressed(self) -> None:
        regions = self._regions(
            ("Normal body text begins here.", [0, 120, 800, 170]),
        )
        # Insert a paragraph_title region with a journal document-type label.
        regions.insert(
            2,
            {
                "category": "paragraph_title",
                "bbox": [0, 110, 800, 118],
                "text": "Article",
            },
        )
        body = _body(_run_falcon([regions]))

        assert "<h2>Article</h2>" not in body
        assert "Normal body text begins here." in body

    def test_function_word_end_merges_acronym_continuation(self) -> None:
        from pdfparser.falcon import _merge_split_paragraphs

        # Fragment ends with "and" (function word) and the continuation opens
        # with an all-caps acronym ("TRII"): a real clause split across a
        # page break, not a dropped-sentence artefact, so the two must merge.
        parts = [
            "<p>This suggests that TRI and</p>",
            "<p>TRII compete for the same substrate tropinone. TRI plays an"
            " important role in TA biosynthesis.</p>",
        ]
        result = _merge_split_paragraphs(parts)
        assert len(result) == 1
        assert (
            "This suggests that TRI and TRII compete for the same substrate"
            in result[0]
        )

    def test_function_word_end_merges_lowercase_continuation(self) -> None:
        from pdfparser.falcon import _merge_split_paragraphs

        # Same function-word ending, but the continuation starts lowercase —
        # that IS a valid column-break continuation, so the merge proceeds.
        parts = [
            "<p>The rate constant depends on whether the enzyme is</p>",
            "<p>bound to the cofactor in the ternary complex.</p>",
        ]
        result = _merge_split_paragraphs(parts)
        assert len(result) == 1
        assert "enzyme is bound to the cofactor" in result[0]

    def test_bold_label_not_merged_into_preceding_paragraph(self) -> None:
        from pdfparser.falcon import _merge_split_paragraphs

        parts = [
            "<p>California State University-Chico, Chico, California, 95929</p>",
            "<p><strong>Keywords:</strong> enzyme kinetics; stereoselectivity.</p>",
        ]
        result = _merge_split_paragraphs(parts)
        assert len(result) == 2
        assert "95929 <strong>Keywords:" not in " ".join(result)

    def test_hyphenated_word_break_joined_without_space(self) -> None:
        from pdfparser.falcon import _merge_split_paragraphs

        parts = [
            "<p>enzyme mechanism is some-</p>",
            "<p>times lost on students.</p>",
        ]
        result = _merge_split_paragraphs(parts)
        assert len(result) == 1
        assert "some- times" not in result[0]
        assert "sometimes" in result[0]

    def test_running_footer_suppressed_across_pages(self) -> None:
        footer = {
            "category": "text",
            "bbox": [0, 950, 800, 980],
            "text": "Running Footer Text",
        }
        page1 = [
            {"category": "doc_title", "bbox": [0, 0, 800, 40], "text": "Test Paper"},
            {"category": "abstract", "bbox": [0, 50, 800, 100], "text": "Abstract."},
            {
                "category": "text",
                "bbox": [0, 120, 800, 170],
                "text": "First page body.",
            },
            footer,
        ]
        page2 = [
            {
                "category": "text",
                "bbox": [0, 120, 800, 170],
                "text": "Second page body.",
            },
            footer,
        ]
        body = _body(_run_falcon([page1, page2]))

        assert "Running Footer Text" not in body
        assert "First page body." in body
        assert "Second page body." in body

    def test_footnotes_placed_before_references(self) -> None:
        regions = [
            {"category": "doc_title", "bbox": [0, 0, 800, 40], "text": "Test Paper"},
            {"category": "abstract", "bbox": [0, 50, 800, 100], "text": "Abstract."},
            {
                "category": "footnote",
                "bbox": [0, 110, 800, 130],
                "text": "Abbreviations: X, xylene.",
            },
            {
                "category": "text",
                "bbox": [0, 140, 800, 200],
                "text": "Main body paragraph.",
            },
            {
                "category": "paragraph_title",
                "bbox": [0, 210, 800, 230],
                "text": "References",
            },
            {
                "category": "text",
                "bbox": [0, 240, 800, 270],
                "text": "[1] Smith et al. 2020.",
            },
        ]
        body = _body(_run_falcon([regions]))

        fn_pos = body.find("Abbreviations: X")
        ref_heading_pos = body.find("<h2>References</h2>")
        body_pos = body.find("Main body paragraph.")
        ref_pos = body.find("[1] Smith et al.")

        assert body_pos < fn_pos, "footnote must come after body text"
        assert fn_pos < ref_heading_pos, "footnote must come before References heading"
        assert ref_heading_pos < ref_pos, (
            "References heading must precede reference items"
        )

    def test_split_table_label_and_title_concatenated(self) -> None:
        """Falcon sometimes emits a table label ("TABLE I") and its descriptive
        title as two consecutive figure_title regions.  Both must appear in the
        injected <caption>."""
        regions = [
            {"category": "doc_title", "bbox": [0, 0, 800, 40], "text": "Test Paper"},
            {"category": "abstract", "bbox": [0, 50, 800, 100], "text": "Abstract."},
            {
                "category": "figure_title",
                "bbox": [0, 120, 800, 140],
                "text": "TABLE I",
            },
            {
                "category": "figure_title",
                "bbox": [0, 142, 800, 162],
                "text": "Selected substrates and inhibitors.",
            },
            {
                "category": "table",
                "bbox": [0, 170, 800, 400],
                "text": "<table><tr><td>data</td></tr></table>",
            },
        ]
        body = _body(_run_falcon([regions]))
        assert "TABLE I" in body
        assert "Selected substrates and inhibitors." in body
        assert "<caption>TABLE I Selected substrates and inhibitors.</caption>" in body

    def test_figure_region_embedded_as_img(self) -> None:
        regions = [
            {"category": "doc_title", "bbox": [0, 0, 800, 40], "text": "Test Paper"},
            {"category": "abstract", "bbox": [0, 50, 800, 100], "text": "Abstract."},
            {
                "category": "figure_title",
                "bbox": [0, 120, 800, 140],
                "text": "Figure 1. A diagram.",
            },
            {
                "category": "figure",
                "bbox": [0, 150, 800, 500],
                "text": "",
            },
        ]
        body = _body(_run_falcon([regions]))
        assert '<img src="data:image/png;base64,' in body
        assert "<figcaption>Figure 1. A diagram.</figcaption>" in body

    def test_figure_inferred_from_gap_above_caption(self) -> None:
        # Large gap (y=170–550) sits above the caption (y=550–570); the gap
        # is > _MIN_FIGURE_HEIGHT so it should be cropped and embedded.
        regions = [
            {"category": "doc_title", "bbox": [0, 0, 800, 40], "text": "Test Paper"},
            {"category": "abstract", "bbox": [0, 50, 800, 100], "text": "Abstract."},
            {
                "category": "text",
                "bbox": [0, 120, 800, 170],
                "text": "Body text above figure.",
            },
            {"category": "figure_title", "bbox": [0, 550, 800, 570], "text": "Fig. 1."},
            {
                "category": "text",
                "bbox": [0, 590, 800, 640],
                "text": "Body text below figure.",
            },
        ]
        body = _body(_run_falcon([regions]))
        assert '<img src="data:image/png;base64,' in body
        assert "<figcaption>Fig. 1.</figcaption>" in body

    def test_figure_caption_for_table_gets_no_image(self) -> None:
        # figure_title immediately adjacent to a table (small gaps) → no crop,
        # caption is absorbed by the table as <caption>.
        regions = [
            {"category": "doc_title", "bbox": [0, 0, 800, 40], "text": "Test Paper"},
            {"category": "abstract", "bbox": [0, 50, 800, 100], "text": "Abstract."},
            {
                "category": "figure_title",
                "bbox": [0, 120, 800, 140],
                "text": "Table I.",
            },
            {
                "category": "table",
                "bbox": [0, 145, 800, 400],
                "text": "<table><tr><td>x</td></tr></table>",
            },
        ]
        body = _body(_run_falcon([regions]))
        assert '<img src="data:image/png;base64,' not in body
        assert "<caption>Table I.</caption>" in body

    def test_figure_region_without_caption(self) -> None:
        regions = [
            {"category": "doc_title", "bbox": [0, 0, 800, 40], "text": "Test Paper"},
            {"category": "abstract", "bbox": [0, 50, 800, 100], "text": "Abstract."},
            {
                "category": "figure",
                "bbox": [0, 150, 800, 500],
                "text": "",
            },
        ]
        body = _body(_run_falcon([regions]))
        assert '<img src="data:image/png;base64,' in body
        assert "<figcaption>" not in body


class TestInferFigureCrop:
    """Inferred figure crops must stay inside the caption's column on
    two-column pages and span the full width on single-column pages."""

    def test_two_column_crop_excludes_other_column(self) -> None:
        from pdfparser.falcon import _infer_figure_crop

        img = _fake_image(800, 1000)
        caption = {"category": "figure_title", "bbox": [40, 600, 380, 620]}
        # Right-column text straddles the figure's vertical gap; it must not be
        # baked into the crop, and must not bound the gap either.
        regions = [
            {"category": "text", "bbox": [40, 0, 380, 100]},
            caption,
            {"category": "text", "bbox": [420, 0, 760, 900]},
        ]
        crop = _infer_figure_crop(caption, regions, img)
        assert crop is not None
        assert crop.width <= 400  # left half only, not the full 800px page

    def test_single_column_narrow_caption_uses_full_width(self) -> None:
        from pdfparser.falcon import _infer_figure_crop

        img = _fake_image(800, 1000)
        caption = {"category": "figure_title", "bbox": [300, 600, 500, 620]}
        regions = [
            {"category": "text", "bbox": [50, 0, 750, 100]},
            caption,
        ]
        crop = _infer_figure_crop(caption, regions, img)
        assert crop is not None
        assert crop.width == 800


class TestRepeatedShortParagraphs:
    """Repeated short fragments are running-header artefacts; repeated short
    sentences are legitimate prose and must survive."""

    def test_repeated_running_header_removed(self) -> None:
        from pdfparser.falcon import _remove_repeated_short_paragraphs

        header = "<p>Smith et al  Journal of Examples</p>"
        parts = [header, "<p>Real body sentence.</p>", header]
        assert _remove_repeated_short_paragraphs(parts) == [
            "<p>Real body sentence.</p>"
        ]

    def test_repeated_short_sentence_preserved(self) -> None:
        from pdfparser.falcon import _remove_repeated_short_paragraphs

        sentence = "<p>Not applicable.</p>"
        parts = [sentence, "<p>Other text.</p>", sentence]
        assert _remove_repeated_short_paragraphs(parts) == parts


class TestLeadingNonArticlePages:
    """Pages bound before the article proper (cover ads, mastheads) must be
    dropped so metadata and body come from the real first article page."""

    def test_leading_ad_page_dropped(self) -> None:
        ad_page = [
            {
                "category": "text",
                "bbox": [0, 0, 800, 200],
                "text": "Order our reagents today and save 20% on your next purchase!",
            }
        ]
        article_page = [
            {"category": "doc_title", "bbox": [0, 0, 800, 40], "text": "Real Title"},
            {
                "category": "abstract",
                "bbox": [0, 50, 800, 120],
                "text": "We characterized the enzyme.",
            },
        ]
        html = _run_falcon([ad_page, article_page])

        assert "<h1>Real Title</h1>" in html
        assert "Order our reagents today" not in html
        assert "We characterized the enzyme." in html

    def test_introduction_marker_qualifies_page(self) -> None:
        from pdfparser.falcon import _is_article_page

        regions = [
            {
                "category": "paragraph_title",
                "bbox": [0, 0, 800, 30],
                "text": "Introduction",
            }
        ]
        assert _is_article_page(regions) is True

    def test_ad_body_mentioning_introduction_still_dropped(self) -> None:
        # "introduction" buried in advertising body copy is not a heading and
        # must not qualify the ad page as the article start.
        ad_page = [
            {
                "category": "text",
                "bbox": [0, 0, 800, 200],
                "text": "An introduction to our new assay kit — order today!",
            }
        ]
        article_page = [
            {"category": "doc_title", "bbox": [0, 0, 800, 40], "text": "Real Title"},
            {"category": "abstract", "bbox": [0, 50, 800, 120], "text": "We did it."},
        ]
        html = _run_falcon([ad_page, article_page])

        assert "<h1>Real Title</h1>" in html
        assert "introduction to our new assay kit" not in html

    def test_leading_page_with_figure_title_not_dropped(self) -> None:
        # The model missed the title/abstract on the real first page but tagged
        # a figure caption there; the page is genuine content, so nothing is
        # dropped even though a later page has an "Introduction" heading.
        first_page = [
            {
                "category": "figure_title",
                "bbox": [0, 600, 800, 620],
                "text": "Figure 1. Graphical abstract.",
            },
            {"category": "text", "bbox": [0, 60, 800, 120], "text": "Title page body."},
        ]
        second_page = [
            {
                "category": "paragraph_title",
                "bbox": [0, 0, 800, 30],
                "text": "Introduction",
            },
            {
                "category": "text",
                "bbox": [0, 60, 800, 120],
                "text": "Second page body.",
            },
        ]
        html = _run_falcon([first_page, second_page])

        assert "Title page body." in html
        assert "Second page body." in html

    def test_no_article_marker_keeps_all_pages(self) -> None:
        # No page carries any marker → _first_article_page falls back to 0 and
        # nothing is dropped.
        page1 = [
            {"category": "text", "bbox": [0, 0, 800, 100], "text": "Page one content."}
        ]
        page2 = [
            {"category": "text", "bbox": [0, 0, 800, 100], "text": "Page two content."}
        ]
        html = _run_falcon([page1, page2])

        assert "Page one content." in html
        assert "Page two content." in html


class TestRegionPdfRect:
    """Image-pixel bbox → PDF-point rect mapping is pure arithmetic."""

    def test_maps_and_flips_y_with_inset(self) -> None:
        from pdfparser.falcon import _REGION_INSET_PX, _region_pdf_rect

        # scale=2 px/pt, page 500pt tall, bbox in pixels.
        left, bottom, right, top = _region_pdf_rect(
            [100.0, 200.0, 300.0, 400.0], 2.0, 500.0
        )
        inset = _REGION_INSET_PX / 2.0
        assert left == 100 / 2 + inset
        assert right == 300 / 2 - inset
        # y flips: image-top (y0=200px) maps to the higher PDF-y edge.
        assert top == 500.0 - 200 / 2 - inset
        assert bottom == 500.0 - 400 / 2 + inset
        assert top > bottom


class TestPageHasTextLayer:
    def test_threshold(self) -> None:
        import pypdfium2 as pdfium

        from pdfparser.falcon import _MIN_TEXT_LAYER_CHARS, _page_has_text_layer

        scanned = MagicMock(spec=pdfium.PdfTextPage)
        scanned.count_chars.return_value = 0
        assert _page_has_text_layer(scanned) is False

        digital = MagicMock(spec=pdfium.PdfTextPage)
        digital.count_chars.return_value = _MIN_TEXT_LAYER_CHARS + 1
        assert _page_has_text_layer(digital) is True


def _chars(spec: list[tuple], *, y: float = 50.0, height: float = 10.0) -> list[tuple]:
    """Build a page_chars list (index, cx, cy, bottom, top, ch, bold, italic) on
    a single line at the given baseline, laid out left-to-right."""
    out = []
    for i, (ch, bold, italic) in enumerate(spec):
        out.append((i, 10.0 + i, y + height / 2, y, y + height, ch, bold, italic))
    return out


def _line_chars(
    text: str, start_index: int, *, y: float = 50.0, height: float = 10.0
) -> list[tuple]:
    """Plain (non-styled) glyphs at consecutive indices from ``start_index``.

    A gap between one line's last index and the next line's ``start_index``
    stands in for the line-ending control characters and triggers a line break.
    """
    return [
        (start_index + k, 10.0 + k, y + height / 2, y, y + height, ch, False, False)
        for k, ch in enumerate(text)
    ]


_FULL_RECT = (0.0, 0.0, 1000.0, 1000.0)


class TestRegionMarkdownEmphasis:
    """Font metadata → markdown emphasis, coalesced into contiguous runs."""

    def test_italic_run_wrapped(self) -> None:
        from pdfparser.falcon import _region_markdown

        spec = [(c, False, False) for c in "see "] + [(c, False, True) for c in "Homo"]
        md = _region_markdown(_chars(spec), _FULL_RECT)
        assert md == "see *Homo*"

    def test_bold_run_wrapped(self) -> None:
        from pdfparser.falcon import _region_markdown

        spec = [(c, True, False) for c in "Key"] + [(c, False, False) for c in ": v"]
        md = _region_markdown(_chars(spec), _FULL_RECT)
        assert md == "**Key**: v"

    def test_bold_italic_run_uses_triple(self) -> None:
        from pdfparser.falcon import _region_markdown

        # A bold+italic word as a minority within otherwise plain prose.
        spec = [(c, False, False) for c in "the "] + [(c, True, True) for c in "Genus"]
        md = _region_markdown(_chars(spec), _FULL_RECT)
        assert md == "the ***Genus***"

    def test_uniformly_bold_region_not_marked(self) -> None:
        from pdfparser.falcon import _region_markdown

        # A wholly-bold heading is the region's base style, not emphasis.
        spec = [(c, True, False) for c in "Methods"]
        md = _region_markdown(_chars(spec), _FULL_RECT)
        assert md == "Methods"

    def test_trailing_space_kept_outside_markers(self) -> None:
        from pdfparser.falcon import _region_markdown

        spec = [(c, False, True) for c in "ab "] + [(c, False, False) for c in "cd"]
        md = _region_markdown(_chars(spec), _FULL_RECT)
        # The space between the italic and plain run must sit outside the *…*.
        assert md == "*ab* cd"

    def test_chars_outside_rect_excluded(self) -> None:
        from pdfparser.falcon import _region_markdown

        inside = _chars([(c, False, False) for c in "in"])
        # A char far to the right, outside a narrow rect.
        outside = [(99, 5000.0, 55.0, 50.0, 60.0, "X", False, False)]
        md = _region_markdown(inside + outside, (0.0, 0.0, 100.0, 1000.0))
        assert md == "in"


class TestRegionMarkdownSuperscript:
    def test_raised_small_glyph_becomes_unicode_superscript(self) -> None:
        from pdfparser.falcon import _region_markdown

        # "NAD" on the baseline, then a "+" raised well above it.
        line = [
            (0, 10.0, 55.0, 50.0, 60.0, "N", False, False),
            (1, 11.0, 55.0, 50.0, 60.0, "A", False, False),
            (2, 12.0, 55.0, 50.0, 60.0, "D", False, False),
            (3, 13.0, 60.0, 57.0, 63.0, "+", False, False),
        ]
        md = _region_markdown(line, _FULL_RECT)
        assert md == "NAD⁺"

    def test_baseline_plus_not_superscripted(self) -> None:
        from pdfparser.falcon import _region_markdown

        line = [
            (0, 10.0, 55.0, 50.0, 60.0, "a", False, False),
            (1, 11.0, 55.0, 50.0, 60.0, " ", False, False),
            (2, 12.0, 55.0, 50.0, 60.0, "+", False, False),
            (3, 13.0, 55.0, 50.0, 60.0, " ", False, False),
            (4, 14.0, 55.0, 50.0, 60.0, "b", False, False),
        ]
        md = _region_markdown(line, _FULL_RECT)
        assert md == "a + b"


class TestRegionMarkdownLineBreaks:
    def test_hyphenated_word_joined_without_space(self) -> None:
        from pdfparser.falcon import _region_markdown

        # "some-" then a line break (index gap) then "times": the trailing
        # hyphen is dropped and the halves join without a space.
        line1 = _line_chars("some-", 0, y=50.0)
        line2 = _line_chars("times", 7, y=30.0)
        md = _region_markdown(line1 + line2, _FULL_RECT)
        assert md == "sometimes"

    def test_no_space_before_close_paren_at_break(self) -> None:
        from pdfparser.falcon import _region_markdown

        line1 = _line_chars("NAD", 0, y=50.0)
        line2 = _line_chars(")", 7, y=30.0)
        md = _region_markdown(line1 + line2, _FULL_RECT)
        assert md == "NAD)"

    def test_word_break_inserts_space(self) -> None:
        from pdfparser.falcon import _region_markdown

        line1 = _line_chars("foo", 0, y=50.0)
        line2 = _line_chars("bar", 7, y=30.0)
        md = _region_markdown(line1 + line2, _FULL_RECT)
        assert md == "foo bar"


class TestApplyTextLayer:
    """Per-region replacement, fallback, and the falcon/pdf escape hatches."""

    def _page(self) -> object:
        import pypdfium2 as pdfium

        from pdfparser.falcon import RenderedPage

        return RenderedPage(
            image=_fake_image(),
            scale=2.0,
            page_height_pt=500.0,
            text_page=MagicMock(spec=pdfium.PdfTextPage),  # non-None ⇒ has layer
        )

    def test_short_extraction_keeps_falcon_text(self) -> None:
        from pdfparser.falcon import _apply_text_layer

        regions = [{"category": "text", "bbox": [0, 0, 100, 50], "text": "x" * 100}]
        page = self._page()
        with (
            patch("pdfparser.falcon._page_char_styles", return_value=[]),
            patch("pdfparser.falcon._region_markdown", return_value="tiny"),
        ):
            _apply_text_layer([regions], [page], force=False)
        assert regions[0]["text"] == "x" * 100  # fallback: Falcon text kept

    def test_good_extraction_replaces_text(self) -> None:
        from pdfparser.falcon import _apply_text_layer

        regions = [{"category": "text", "bbox": [0, 0, 100, 50], "text": "ocr garble"}]
        page = self._page()
        with (
            patch("pdfparser.falcon._page_char_styles", return_value=[]),
            patch("pdfparser.falcon._region_markdown", return_value="clean layer text"),
        ):
            _apply_text_layer([regions], [page], force=False)
        assert regions[0]["text"] == "clean layer text"

    def test_table_region_never_touched(self) -> None:
        from pdfparser.falcon import _apply_text_layer

        regions = [{"category": "table", "bbox": [0, 0, 100, 50], "text": "<table>"}]
        page = self._page()
        with (
            patch("pdfparser.falcon._page_char_styles", return_value=[]),
            patch(
                "pdfparser.falcon._region_markdown", return_value="should not be used"
            ),
        ):
            _apply_text_layer([regions], [page], force=False)
        assert regions[0]["text"] == "<table>"

    def test_force_replaces_even_when_short(self) -> None:
        from pdfparser.falcon import _apply_text_layer

        regions = [{"category": "text", "bbox": [0, 0, 100, 50], "text": "x" * 100}]
        page = self._page()
        with (
            patch("pdfparser.falcon._page_char_styles", return_value=[]),
            patch("pdfparser.falcon._region_markdown", return_value="tiny"),
        ):
            _apply_text_layer([regions], [page], force=True)
        assert regions[0]["text"] == "tiny"

    def test_page_without_text_layer_skipped(self) -> None:
        from pdfparser.falcon import RenderedPage, _apply_text_layer

        regions = [{"category": "text", "bbox": [0, 0, 100, 50], "text": "falcon"}]
        page = RenderedPage(
            image=_fake_image(), scale=2.0, page_height_pt=500.0, text_page=None
        )
        _apply_text_layer([regions], [page], force=False)
        assert regions[0]["text"] == "falcon"

    def test_substantial_shorter_extraction_replaces_falcon(self) -> None:
        # Falcon over-read (e.g. duplicated) the region, so its text is >2× the
        # layer's; the layer text is still substantial (≥ abs floor) so it wins,
        # rather than being discarded by the bare length-ratio.
        from pdfparser.falcon import _apply_text_layer

        regions = [{"category": "text", "bbox": [0, 0, 100, 50], "text": "y" * 100}]
        page = self._page()
        with (
            patch("pdfparser.falcon._page_char_styles", return_value=[]),
            patch("pdfparser.falcon._region_markdown", return_value="x" * 40),
        ):
            _apply_text_layer([regions], [page], force=False)
        assert regions[0]["text"] == "x" * 40

    def test_band_narrowing_extracts_only_in_band_glyphs(self) -> None:
        # The bisect y-band must include every glyph the region spans and drop
        # those outside it (here a stray glyph far below the region).
        import pypdfium2 as pdfium

        from pdfparser.falcon import RenderedPage, _apply_text_layer

        page = RenderedPage(
            image=_fake_image(),
            scale=1.0,  # points == pixels; bbox [0,0,100,50] → PDF y in [452, 498]
            page_height_pt=500.0,
            text_page=MagicMock(spec=pdfium.PdfTextPage),
        )
        chars = [
            (0, 10.0, 475.0, 470.0, 480.0, "H", False, False),
            (1, 11.0, 475.0, 470.0, 480.0, "i", False, False),
            (2, 12.0, 100.0, 95.0, 105.0, "Z", False, False),  # below the band
        ]
        regions = [{"category": "text", "bbox": [0, 0, 100, 50], "text": "falcon"}]
        with patch("pdfparser.falcon._page_char_styles", return_value=chars):
            _apply_text_layer([regions], [page], force=True)
        assert regions[0]["text"] == "Hi"


class TestDegenerateRepetition:
    """Text-generation run over a figure can emit one label repeated dozens of
    times; such OCR noise must be detected and dropped, while real prose (even
    with some repetition) is kept."""

    def test_repeated_label_detected(self) -> None:
        from pdfparser.falcon import _is_degenerate_repetition

        assert _is_degenerate_repetition("AaTRI, " * 40) is True

    def test_real_prose_not_flagged(self) -> None:
        from pdfparser.falcon import _is_degenerate_repetition

        prose = (
            "The enzyme catalyzes the stereospecific oxidation of the substrate"
            " to the corresponding ketone under physiological conditions."
        )
        assert _is_degenerate_repetition(prose) is False

    def test_short_text_never_flagged(self) -> None:
        from pdfparser.falcon import _is_degenerate_repetition

        assert _is_degenerate_repetition("yes yes yes") is False

    def test_figure_ocr_noise_region_dropped(self) -> None:
        regions = [
            {"category": "doc_title", "bbox": [0, 0, 800, 40], "text": "Test Paper"},
            {"category": "abstract", "bbox": [0, 50, 800, 100], "text": "Abstract."},
            {
                "category": "text",
                "bbox": [0, 120, 800, 600],
                "text": "AaTRI, " * 50,
            },
            {
                "category": "text",
                "bbox": [0, 620, 800, 680],
                "text": "Real body sentence after the figure.",
            },
        ]
        body = _body(_run_falcon([regions]))
        assert "AaTRI" not in body
        assert "Real body sentence after the figure." in body

    def test_figure_region_with_stray_text_still_cropped(self) -> None:
        # A `figure` region is an image crop; degenerate stray text on it must
        # not suppress the <img> (the guard targets text-bearing prose only).
        regions = [
            {"category": "doc_title", "bbox": [0, 0, 800, 40], "text": "Test Paper"},
            {"category": "abstract", "bbox": [0, 50, 800, 100], "text": "Abstract."},
            {"category": "figure", "bbox": [0, 150, 800, 500], "text": "AaTRI, " * 50},
        ]
        body = _body(_run_falcon([regions]))
        assert '<img src="data:image/png;base64,' in body


class TestExtractMetaAuthor:
    """Author detection must not pick a flowing sentence that precedes the
    byline (the old newline-count guard is a no-op once the text layer collapses
    newlines)."""

    def test_byline_chosen_over_preceding_sentence(self) -> None:
        from pdfparser.falcon import _extract_meta

        regions = [
            {"category": "doc_title", "bbox": [0, 0, 800, 40], "text": "A Great Paper"},
            {
                "category": "text",
                "bbox": [0, 50, 800, 70],
                "text": "Received 1 May 2020; accepted 3 June 2020.",
            },
            {
                "category": "text",
                "bbox": [0, 80, 800, 100],
                "text": "Jane Doe, John Smith",
            },
        ]
        meta = _extract_meta(regions, 800.0)
        assert meta["authors"] == "Jane Doe, John Smith"


class TestRegionToHtmlSuperscriptMarker:
    """A leading reconstructed superscript must only be read as a footnote
    marker when it is short; a longer leading superscript is body prose."""

    def test_short_leading_sup_is_footnote(self) -> None:
        from pdfparser.falcon import _region_to_html

        out = _region_to_html(
            {"category": "text", "text": "<sup>a</sup> Footnote text."}
        )
        assert out == '<p class="footnote"><sup>a</sup> Footnote text.</p>'

    def test_long_leading_sup_is_paragraph(self) -> None:
        from pdfparser.falcon import _region_to_html

        out = _region_to_html(
            {"category": "text", "text": "<sup>(R)-</sup>configuration was observed."}
        )
        assert out.startswith("<p>")
        assert 'class="footnote"' not in out


class TestRegionMarkdownAsterisk:
    """Literal asterisks from the text layer must not be parsed as emphasis, nor
    introduce punctuation that the sentence-boundary heuristics key on."""

    def test_literal_asterisks_substituted(self) -> None:
        from pdfparser.falcon import _inline_md_to_html, _region_markdown

        spec = [(c, False, False) for c in "a*b*c"]
        md = _region_markdown(_chars(spec), _FULL_RECT)
        assert md == "a∗b∗c"
        assert "*" not in md
        assert "<em>" not in _inline_md_to_html(md)

    def test_substitute_is_not_sentence_terminator(self) -> None:
        # A fragment ending in a substituted asterisk must still merge with its
        # continuation — the substitute must not look like terminal punctuation.
        from pdfparser.falcon import (
            _ASTERISK_SUBSTITUTE,
            _SENTENCE_END_RE,
            _merge_split_paragraphs,
        )

        assert not _SENTENCE_END_RE.search(f"see note{_ASTERISK_SUBSTITUTE}")
        parts = [
            f"<p>the rate constant{_ASTERISK_SUBSTITUTE}</p>",
            "<p>was measured precisely.</p>",
        ]
        assert len(_merge_split_paragraphs(parts)) == 1


_FIXTURE_PDF = Path(__file__).parent / "fixtures" / "30592559.pdf"
_AD_PREFIX_PDF = Path(__file__).parent / "fixtures" / "31051047.pdf"
_SPIKE_HTML = Path(__file__).parent.parent / "spike_results" / "falcon_full.html"


@pytest.fixture(scope="session")
def falcon_model() -> object:
    """Load the Falcon-OCR model once per session; skip if unavailable."""
    try:
        from pdfparser.falcon import load_model

        return load_model()
    except Exception as e:
        pytest.skip(f"Falcon model not available: {e}")


@pytest.fixture(scope="session")
def falcon_html(falcon_model: object) -> str:
    """Run the full Falcon pipeline; skip if the model is unavailable.

    Writes the result back to spike_results/falcon_full.html so the file
    stays current after each integration run.
    """
    if not _FIXTURE_PDF.exists():
        pytest.skip(f"Fixture PDF not found: {_FIXTURE_PDF}")
    from pdfparser.falcon import falcon_pdf_to_html

    html = falcon_pdf_to_html(_FIXTURE_PDF, model=falcon_model)
    _SPIKE_HTML.write_text(html, encoding="utf-8")
    return html


def _header_h1(html: str) -> str:
    """Return the text of the document's <header><h1> title element."""
    m = re.search(r"<header>.*?<h1>(.*?)</h1>", html, re.DOTALL)
    assert m, "header <h1> not found"
    return m.group(1)


@pytest.mark.integration
class TestFalconPipeline:
    """Integration tests: run the full Falcon pipeline on the fixture PDF.

    Skipped when the model is not available (no GPU, weights not downloaded).
    Each run also refreshes spike_results/falcon_full.html.
    """

    def test_abstract_no_column_break(self, falcon_html: str) -> None:
        abstract_start = falcon_html.find("<section class='abstract'>")
        abstract_end = falcon_html.find("</section>", abstract_start)
        abstract_block = falcon_html[abstract_start:abstract_end]
        # Both halves must be collected: if either is absent the pipeline
        # dropped a fragment it should have kept.
        assert "classical and contemporary" in abstract_block
        assert "experimental biochemistry." in abstract_block
        # And they must appear in the same paragraph — no split <p>.
        assert "classical and contemporary</p>" not in abstract_block

    def test_which_is_not_merged_with_as_a_testament(self, falcon_html: str) -> None:
        assert "which is As a testament to the utility" not in falcon_html

    def test_ternary_complex_followed_by_clearly_showed(self, falcon_html: str) -> None:
        expected = (
            "The 1.8 Å ternary complex (enzyme + 2-KPC + NAD⁺)"
            " clearly showed interaction of the R152"
        )
        assert expected in falcon_html

    def test_paper_footnotes_after_body_before_references(
        self, falcon_html: str
    ) -> None:
        correspondence = "To whom correspondence should be addressed"
        assert correspondence in falcon_html
        fn_pos = falcon_html.find(correspondence)
        # Must follow substantial body content, not appear near the top.
        body_start = falcon_html.find('<div class="body">')
        assert fn_pos - body_start > 2000
        # Must precede the reference list.
        first_ref_pos = falcon_html.find("<p>[1]")
        if first_ref_pos != -1:
            assert fn_pos < first_ref_pos


@pytest.fixture(scope="session")
def ad_prefix_html(falcon_model: object) -> str:
    """Full pipeline output for the ad-prefixed 31051047.pdf fixture."""
    if not _AD_PREFIX_PDF.exists():
        pytest.skip(f"Fixture PDF not found: {_AD_PREFIX_PDF}")
    from pdfparser.falcon import falcon_pdf_to_html

    return falcon_pdf_to_html(_AD_PREFIX_PDF, model=falcon_model)


@pytest.mark.integration
class TestFalconAdPageExclusion:
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
        # The motivating bug: Falcon OCR misread both PtTRI and PtTRII as the
        # single string "PITRI", collapsing two distinct genes.  Sourcing prose
        # from the text layer keeps them distinct.  Tables still come from Falcon
        # OCR (out of scope), so the "PITRI" check is restricted to the prose.
        prose = re.sub(r"<table.*?</table>", "", ad_prefix_html, flags=re.DOTALL)
        assert "PtTRI" in prose
        assert "PtTRII" in prose
        assert "PITRI" not in prose

    def test_cross_page_paragraph_not_split(self, ad_prefix_html: str) -> None:
        # The clause "…TRI and" / "TRII compete…" spans a page break; it must
        # be a single paragraph, not split at the page boundary.
        assert (
            "This suggests that TRI and TRII compete for the same substrate"
            in ad_prefix_html
        )
        assert "This suggests that TRI and</p>" not in ad_prefix_html
