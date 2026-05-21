"""Integration tests for pdf_to_html — requires GROBID and a sample PDF."""

from pathlib import Path

import pytest

from pdfparser import pdf_to_html
from pdfparser.tables import extract_tables


@pytest.mark.integration
class TestPdfToHtml:
    @pytest.fixture(scope="class")
    def html_result(self, sample_pdf: Path, grobid_url: str) -> str:
        return pdf_to_html(sample_pdf, grobid_url=grobid_url)

    def test_returns_valid_html(self, html_result: str) -> None:
        assert html_result.startswith("<!DOCTYPE html>")
        assert html_result.rstrip().endswith("</html>")

    def test_all_html_tables_come_from_extract_tables(
        self, sample_pdf: Path, html_result: str
    ) -> None:
        tables = extract_tables(sample_pdf)
        # Every <table> in the output must be one of the tables gmft extracted,
        # injected verbatim — no stray or hallucinated tables.
        # Count may be zero when GROBID finds no <figure type="table"> in the TEI;
        # the invariant still holds and is non-trivially exercised when count > 0.
        injected = sum(1 for t in tables if t.html in html_result)
        assert injected == html_result.count("<table>")
        assert injected == len(tables)

    def test_no_unresolved_placeholders(self, html_result: str) -> None:
        assert "[[TABLE_" not in html_result

    def test_file_not_found_raises(self, grobid_url: str) -> None:
        with pytest.raises(FileNotFoundError):
            pdf_to_html(Path("/nonexistent/paper.pdf"), grobid_url=grobid_url)

    def test_grobid_unreachable_raises(self, sample_pdf: Path) -> None:
        with pytest.raises(RuntimeError, match="GROBID"):
            pdf_to_html(sample_pdf, grobid_url="http://localhost:9999")
