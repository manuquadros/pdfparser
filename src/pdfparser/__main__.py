"""Command-line entry point: convert a PDF to a self-contained HTML file.

pdm run python -m pdfparser paper.pdf            # writes paper.html
pdm run python -m pdfparser paper.pdf out.html   # writes out.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pdfparser.pipeline import lightonocr_pdf_to_html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pdfparser",
        description="Convert a PDF to a self-contained HTML document.",
    )
    parser.add_argument("pdf", type=Path, help="Path to the input PDF file.")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output HTML path (default: input path with a .html suffix).",
    )
    parser.add_argument(
        "--vllm-url",
        default=None,
        help="vLLM endpoint root (default: $PDFPARSER_VLLM_URL or "
        "http://127.0.0.1:8000/v1).",
    )
    parser.add_argument(
        "--vllm-model",
        default=None,
        help="Served model name (default: $PDFPARSER_VLLM_MODEL or lightonocr).",
    )
    args = parser.parse_args(argv)

    if not args.pdf.is_file():
        parser.error(f"input PDF not found: {args.pdf}")

    output = args.output or args.pdf.with_suffix(".html")

    html = lightonocr_pdf_to_html(
        args.pdf, base_url=args.vllm_url, model=args.vllm_model
    )
    output.write_text(html, encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
