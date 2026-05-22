"""Table extraction from PDF files using gmft (Table Transformer).

Typical usage::

    from pdfparser.tables import extract_tables

    tables = extract_tables("paper.pdf")
    for t in tables:
        print(f"page {t.page}, confidence {t.confidence:.3f}")
        print(t.html)
"""

from __future__ import annotations

import bisect
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pdfparser._html import df_to_table_html

DEFAULT_THRESHOLD: float = 0.99
DEFAULT_DEVICE: str = "cpu"


@dataclass(frozen=True)
class ExtractedTable:
    """A single table extracted from a PDF page.

    Attributes:
        page: 1-based page number in the source PDF.
        confidence: Detection confidence score from the Table Transformer
            model (0–1).  Only tables at or above the requested threshold
            are returned.
        html: Complete ``<table>…</table>`` HTML element ready for embedding.
        col_headers: Per-column header labels extracted from text spatially
            above the detected table region.  Empty when none are found.
    """

    page: int
    confidence: float
    html: str
    col_headers: tuple[str, ...] = field(default=(), compare=False)


def extract_tables(
    pdf_path: Path | str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    device: str = DEFAULT_DEVICE,
    ffill_spanning: bool = False,
) -> list[ExtractedTable]:
    """Extract tables from a PDF and return them as HTML table elements.

    Uses Microsoft Table Transformer (via gmft) for detection and structure
    recognition.  Models are downloaded on first call (~380 MB, cached by
    HuggingFace) and re-used across subsequent calls within the same process.

    Args:
        pdf_path: Path to the input PDF file.
        threshold: Minimum detection confidence to include a table.  The
            default of 0.99 is intentionally conservative — lower it only if
            you know the document has tables that are being missed.
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        ffill_spanning: Forward-fill NaN runs in each column before rendering.
            Set to True when the PDF uses merged/spanning cells that the model
            leaves blank for covered rows.

    Returns:
        Tables in document order (page ascending, top-to-bottom within page).

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
        RuntimeError: If gmft fails to open or process the PDF.
    """
    from gmft.pdf_bindings import PyPDFium2Document

    pdf_path = Path(pdf_path)
    detector, formatter = _get_models(threshold=threshold, device=device)

    try:
        doc = PyPDFium2Document(str(pdf_path))
    except Exception as exc:
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path) from exc
        raise RuntimeError(f"Could not open {pdf_path}: {exc}") from exc

    results: list[ExtractedTable] = []
    try:
        for page in doc:
            for ct in detector.extract(page):
                try:
                    ft = formatter.extract(ct)
                    df = ft.df()
                except Exception as exc:
                    warnings.warn(
                        f"Skipping table on page {ct.page.page_number}: {exc}",
                        stacklevel=2,
                    )
                    continue

                if df is None or df.empty:
                    continue

                results.append(
                    ExtractedTable(
                        page=ct.page.page_number,
                        confidence=ct.confidence_score,
                        html=df_to_table_html(df, ffill_spanning=ffill_spanning),
                        col_headers=_extract_col_headers(ct, len(df.columns)),
                    )
                )
    finally:
        doc.close()

    return results


_HEADER_BAND_PTS: float = 40.0
_ROW_GAP_PTS: float = 8.0


def _extract_col_headers(ct: object, n_cols: int) -> tuple[str, ...]:
    """Extract per-column header labels from text spatially above the table.

    Uses the N-1 largest horizontal gaps among text-start x-positions inside
    the table to infer column boundaries, then buckets text fragments found
    above the table into those column zones.  Fragments are grouped into
    visual rows by y-proximity (threshold: _ROW_GAP_PTS) and ordered
    top-to-bottom, left-to-right within each column.

    Returns an empty tuple when no header text is found above the table.
    """
    if n_cols <= 0:
        return ()

    x_starts = sorted(
        x1
        for x1, _y1, _x2, _y2, txt in ct.text_positions()  # type: ignore[attr-defined]
        if txt.strip()
    )
    if len(x_starts) < n_cols:
        return ()

    # N-1 largest consecutive gaps give the column boundary midpoints.
    gaps = sorted(
        (x_starts[i + 1] - x_starts[i], (x_starts[i] + x_starts[i + 1]) / 2)
        for i in range(len(x_starts) - 1)
    )
    boundaries = sorted(mid for _, mid in gaps[-(n_cols - 1) :])

    table_top: float = ct.bbox[1]  # type: ignore[attr-defined]
    fragments: list[tuple[float, float, str]] = []
    for x1, y1, x2, y2, txt in ct.text_positions(outside=True):  # type: ignore[attr-defined]
        if y2 > table_top or y1 < table_top - _HEADER_BAND_PTS:
            continue
        txt = txt.strip()
        # PDF font streams may produce non-printable control characters; skip them.
        if not txt or not any(c.isprintable() and not c.isspace() for c in txt):
            continue
        fragments.append(((x1 + x2) / 2, y1, txt))

    if not fragments:
        return ()

    fragments.sort(key=lambda t: t[1])
    rows: list[list[tuple[float, float, str]]] = [[fragments[0]]]
    for frag in fragments[1:]:
        if frag[1] - rows[-1][-1][1] > _ROW_GAP_PTS:
            rows.append([frag])
        else:
            rows[-1].append(frag)

    col_parts: list[list[str]] = [[] for _ in range(n_cols)]
    for row in rows:
        for x_center, _y, txt in sorted(row, key=lambda t: t[0]):
            col_idx = min(bisect.bisect_right(boundaries, x_center), n_cols - 1)
            col_parts[col_idx].append(txt)

    return tuple(" ".join(parts) for parts in col_parts)


@lru_cache(maxsize=8)
def _get_models(threshold: float, device: str) -> tuple[object, object]:
    """Load and cache (detector, formatter) for a given threshold and device."""
    from gmft import AutoTableDetector, AutoTableFormatter
    from gmft.impl.tatr.config import TATRDetectorConfig, TATRFormatConfig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        detector = AutoTableDetector(
            TATRDetectorConfig(
                torch_device=device,
                detector_base_threshold=threshold,
            )
        )
        formatter = AutoTableFormatter(TATRFormatConfig(torch_device=device))

    return detector, formatter
