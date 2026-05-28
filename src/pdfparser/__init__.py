"""PDF parser used to convert PDFs for the D3 Annotation Hub"""

__version__ = "0.1.0"

try:
    from beartype.claw import beartype_this_package

    beartype_this_package()
except ImportError:
    pass

from pdfparser.falcon import falcon_pdf_to_html, load_model

__all__ = ["falcon_pdf_to_html", "load_model"]
