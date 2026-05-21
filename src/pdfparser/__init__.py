"""PDF parser used to convert PDFs for the D3 Annotation Hub"""

__version__ = "0.1.0"

try:
    from beartype.claw import beartype_this_package

    beartype_this_package()
except ImportError:
    pass

from pdfparser.tables import ExtractedTable, extract_tables

__all__ = ["ExtractedTable", "extract_tables"]
