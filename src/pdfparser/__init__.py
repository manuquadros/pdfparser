"""PDF parser used to convert PDFs for the D3 Annotation Hub"""

__version__ = "0.1.0"

try:
    from beartype.claw import beartype_this_package

    beartype_this_package()
except ImportError:
    pass

from pdfparser.pipeline import lightonocr_pdf_to_html, load_ocr_model

__all__ = ["lightonocr_pdf_to_html", "load_ocr_model"]
