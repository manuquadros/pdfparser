from pathlib import Path

import pytest

from pdfparser.grobid import DEFAULT_GROBID_URL

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_PDF = FIXTURE_DIR / "30592559.pdf"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: slow tests that load ML models and require a sample PDF",
    )


@pytest.fixture(scope="class")
def sample_pdf() -> Path:
    if not SAMPLE_PDF.exists():
        pytest.skip(f"Sample PDF not found: {SAMPLE_PDF}")
    return SAMPLE_PDF


@pytest.fixture(scope="session")
def grobid_url() -> str:
    import httpx

    try:
        httpx.get(f"{DEFAULT_GROBID_URL}/api/isalive", timeout=3.0)
    except Exception:
        pytest.skip(f"GROBID is not running at {DEFAULT_GROBID_URL}")
    return DEFAULT_GROBID_URL
