"""Table extraction from PDF files using gmft (Table Transformer).

Typical usage::

    from pdfparser.tables import extract_tables

    tables = extract_tables("paper.pdf")
    for t in tables:
        print(f"page {t.page}, confidence {t.confidence:.3f}")
        print(t.html)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
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
    """

    page: int
    confidence: float
    html: str


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
                    )
                )
    finally:
        doc.close()

    return results


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
