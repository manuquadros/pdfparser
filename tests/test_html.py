"""Unit tests for internal HTML helpers — no models, no PDFs."""

import pandas as pd

from pdfparser._html import clean_col_name, df_to_table_html


class TestCleanColName:
    def test_plain_string_unchanged(self) -> None:
        assert clean_col_name("Enzyme") == "Enzyme"

    def test_literal_backslash_n_replaced(self) -> None:
        assert clean_col_name("2-KPC \\nkcat") == "2-KPC  kcat"

    def test_actual_newline_replaced(self) -> None:
        assert clean_col_name("col\nname") == "col name"

    def test_control_characters_replaced(self) -> None:
        assert clean_col_name("col\x01name") == "col name"

    def test_strips_leading_trailing_whitespace(self) -> None:
        assert clean_col_name("  col  ") == "col"

    def test_non_string_input_coerced(self) -> None:
        assert clean_col_name(42) == "42"


class TestDfToTableHtml:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({"A": ["x", None, "z"], "B": ["1", "2", "3"]})

    def test_nan_rendered_as_empty(self) -> None:
        df = self._make_df()
        html = df_to_table_html(df)
        assert "NaN" not in html
        assert "nan" not in html
        assert "<td></td>" in html

    def test_column_names_in_th(self) -> None:
        df = self._make_df()
        html = df_to_table_html(df)
        assert "<th>A</th>" in html
        assert "<th>B</th>" in html

    def test_ffill_spanning_propagates_values(self) -> None:
        df = pd.DataFrame({"Additive": ["X", None, None], "Enzyme": ["a", "b", "c"]})
        html = df_to_table_html(df, ffill_spanning=True)
        assert html.count("<td>X</td>") == 3

    def test_structure_is_valid_table(self) -> None:
        df = self._make_df()
        html = df_to_table_html(df)
        assert html.startswith("<table>")
        assert html.endswith("</table>")
        assert html.count("<tr>") == 4  # 1 header + 3 data rows
