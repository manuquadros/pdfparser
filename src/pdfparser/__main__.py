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
        "--device",
        default=None,
        help="Torch device for the model (default: cuda if available, else cpu).",
    )
    args = parser.parse_args(argv)

    if not args.pdf.is_file():
        parser.error(f"input PDF not found: {args.pdf}")

    output = args.output or args.pdf.with_suffix(".html")

    html = lightonocr_pdf_to_html(args.pdf, device=args.device)
    output.write_text(html, encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
