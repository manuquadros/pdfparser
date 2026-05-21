"""Internal HTML rendering helpers shared across pdfparser modules."""

from __future__ import annotations

import pandas as pd  # noqa: TCH002 — beartype resolves annotations at runtime


def clean_col_name(name: object) -> str:
    """Normalise a DataFrame column name to a plain, printable string.

    gmft concatenates multi-line header text with literal ``\\n`` (two chars,
    not a newline) and occasionally inserts control characters from the PDF
    font stream.  Both are replaced with spaces here.
    """
    s = str(name)
    s = s.replace("\\n", " ").replace("\n", " ")
    s = "".join(c if c.isprintable() else " " for c in s)
    return s.strip()


def df_to_table_html(df: pd.DataFrame, ffill_spanning: bool = False) -> str:
    """Render a DataFrame as an HTML ``<table>`` string.

    Args:
        df: Table data as returned by gmft's ``FormattedTable.df()``.
        ffill_spanning: If True, forward-fill NaN runs within each column.
            Useful when the PDF uses spanning/merged cells that gmft leaves
            as NaN for the covered rows.

    Returns:
        A ``<table>…</table>`` HTML string with cleaned headers and no
        ``NaN`` literals in cell content.
    """
    if ffill_spanning:
        df = df.ffill()
    df = df.fillna("")

    cols = [clean_col_name(c) for c in df.columns]
    header = "".join(f"<th>{c}</th>" for c in cols)
    rows = [
        "<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>"
        for row in df.to_numpy()
    ]
    return "<table>\n<tr>" + header + "</tr>\n" + "\n".join(rows) + "\n</table>"
