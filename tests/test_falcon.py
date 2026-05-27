"""Unit tests for falcon.py — no model loading, no PDF rendering."""

from pathlib import Path
from unittest.mock import MagicMock, patch

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

        # The two fragments must not appear as separate <p> elements.
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

        # A single space must separate the two joined fragments.
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

    def test_abstract_text_appears_in_abstract_section(self) -> None:
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

        abstract_start = html.find("<section class='abstract'>")
        abstract_end = html.find("</section>", abstract_start)
        assert abstract_start >= 0, "abstract section not found"
        abstract_block = html[abstract_start:abstract_end]

        assert "First fragment without terminal punctuation" in abstract_block
        assert "second fragment closes the sentence." in abstract_block
        assert abstract_block.count("<p>") == 1, (
            "expected exactly one <p> inside abstract, got multiple"
        )
