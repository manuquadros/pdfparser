"""Unit tests for falcon.py — no model loading, no PDF rendering."""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


def _fake_image(width: int = 800, height: int = 1000) -> Image.Image:
    return Image.new("RGB", (width, height), color="white")


def _run_falcon(regions_per_page: list[list[dict]]) -> str:
    """Run falcon_pdf_to_html with a stub model and synthetic page images."""
    from pdfparser.falcon import falcon_pdf_to_html

    fake_images = [_fake_image() for _ in regions_per_page]
    fake_model = MagicMock()
    fake_model.generate_with_layout.side_effect = [
        [page_regions] for page_regions in regions_per_page
    ]

    with patch("pdfparser.falcon._render_pages", return_value=fake_images):
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

    def test_cross_page_paragraph_not_split(self, ad_prefix_html: str) -> None:
        # The clause "…TRI and" / "TRII compete…" spans a page break; it must
        # be a single paragraph, not split at the page boundary.
        assert (
            "This suggests that TRI and TRII compete for the same substrate"
            in ad_prefix_html
        )
        assert "This suggests that TRI and</p>" not in ad_prefix_html
