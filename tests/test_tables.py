"""Integration tests for table extraction — loads ML models, requires a PDF."""

from pathlib import Path

import pytest

from pdfparser.tables import ExtractedTable, extract_tables


@pytest.mark.integration
class TestExtractTables:
    def test_returns_list_of_extracted_tables(self, sample_pdf: Path) -> None:
        tables = extract_tables(sample_pdf)
        assert isinstance(tables, list)
        assert len(tables) > 0

    def test_all_results_are_extracted_table_instances(self, sample_pdf: Path) -> None:
        tables = extract_tables(sample_pdf)
        assert all(isinstance(t, ExtractedTable) for t in tables)

    def test_confidence_at_or_above_threshold(self, sample_pdf: Path) -> None:
        threshold = 0.99
        tables = extract_tables(sample_pdf, threshold=threshold)
        assert all(t.confidence >= threshold for t in tables)

    def test_html_is_table_element(self, sample_pdf: Path) -> None:
        tables = extract_tables(sample_pdf)
        for t in tables:
            assert t.html.startswith("<table>")
            assert t.html.endswith("</table>")

    def test_no_nan_in_html(self, sample_pdf: Path) -> None:
        tables = extract_tables(sample_pdf)
        for t in tables:
            assert "NaN" not in t.html
            assert ">nan<" not in t.html

    def test_page_numbers_are_positive(self, sample_pdf: Path) -> None:
        tables = extract_tables(sample_pdf)
        assert all(t.page >= 1 for t in tables)

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            extract_tables(Path("/nonexistent/paper.pdf"))

    def test_ffill_spanning_does_not_crash(self, sample_pdf: Path) -> None:
        tables = extract_tables(sample_pdf, ffill_spanning=True)
        assert len(tables) > 0
