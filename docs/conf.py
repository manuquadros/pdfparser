"""Sphinx configuration for pdfparser."""

import datetime
import sys
from pathlib import Path

# Make the package importable without installation (fallback for CI without editable install)
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

project = "pdfparser"
author = "Emanuel Quadros"
copyright = f"{datetime.date.today().year}, Emanuel Quadros"  # noqa: A001

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 2,
    "navigation_depth": 3,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
