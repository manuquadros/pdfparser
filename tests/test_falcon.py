"""Unit tests for falcon.py — no model loading, no PDF rendering."""

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


_FIXTURE_PDF = Path(__file__).parent / "fixtures" / "30592559.pdf"
_SPIKE_HTML = Path(__file__).parent.parent / "spike_results" / "falcon_full.html"


@pytest.fixture(scope="session")
def falcon_html() -> str:
    """Run the full Falcon pipeline; skip if the model is unavailable.

    Writes the result back to spike_results/falcon_full.html so the file
    stays current after each integration run.
    """
    if not _FIXTURE_PDF.exists():
        pytest.skip(f"Fixture PDF not found: {_FIXTURE_PDF}")
    try:
        from pdfparser.falcon import falcon_pdf_to_html, load_model

        model = load_model()
    except Exception as e:
        pytest.skip(f"Falcon model not available: {e}")

    html = falcon_pdf_to_html(_FIXTURE_PDF, model=model)
    _SPIKE_HTML.write_text(html, encoding="utf-8")
    return html


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
