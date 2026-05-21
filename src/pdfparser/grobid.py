"""GROBID HTTP client for PDF → TEI XML conversion.

GROBID must be running and reachable at ``url`` before calling these functions.
See https://grobid.readthedocs.io/en/latest/Grobid-docker/ for a Docker setup.
"""

from __future__ import annotations

from pathlib import Path

import httpx

DEFAULT_GROBID_URL: str = "http://localhost:8070"


def pdf_to_tei(
    pdf_path: Path | str,
    url: str = DEFAULT_GROBID_URL,
    *,
    timeout: float = 180.0,
) -> str:
    """Submit a PDF to GROBID and return the TEI XML response body.

    Args:
        pdf_path: Path to the input PDF.
        url: Base URL of the GROBID service.
        timeout: HTTP timeout in seconds.

    Returns:
        TEI XML string as returned by GROBID's ``processFulltextDocument``
        endpoint.

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
        RuntimeError: If the GROBID request fails (HTTP error or network
            issue).
    """
    pdf_path = Path(pdf_path)

    try:
        with pdf_path.open("rb") as fh:
            response = httpx.post(
                f"{url}/api/processFulltextDocument",
                files={"input": ("paper.pdf", fh, "application/pdf")},
                data={"consolidateHeader": "1"},
                timeout=timeout,
            )
        response.raise_for_status()
    except FileNotFoundError:
        raise
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"GROBID returned HTTP {exc.response.status_code} (is {url} reachable?)"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"GROBID request failed (is {url} reachable?): {exc}"
        ) from exc

    return response.text
