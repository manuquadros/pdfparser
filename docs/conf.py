"""Sphinx configuration for pdfparser.

The package's logic lives almost entirely in private (``_``-prefixed) functions,
and so does its design rationale — the bulk of the docstrings document a
heuristic and *why* it makes the trade-off it does.  The configuration below is
tuned for that reality: API pages are generated for the whole package with
private members included, so a contributor reading the docs sees the same
functions they will edit.
"""

import datetime
import sys
from pathlib import Path

# Make the package importable without installation.
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

project = "pdfparser"
author = "Emanuel Quadros"
copyright = f"{datetime.date.today().year}, Emanuel Quadros"  # noqa: A001

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.apidoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True

# Drive sphinx-apidoc from the build itself (Sphinx 8.2+ ``sphinx.ext.apidoc``)
# so ``docs-build`` and the live-reload ``docs-serve`` regenerate the API stubs
# identically — neither has to remember to run apidoc first, and the ``api/``
# toctree never dangles on a fresh checkout (the stubs are gitignored).
apidoc_modules = [
    {
        "path": "../src/pdfparser",
        "destination": "api",
        # ``private-members`` is the whole point: skip it and the API pages are
        # nearly empty, because the figure-crop, paragraph-merge and
        # classification logic — and the design notes that justify it — all live
        # in ``_``-prefixed functions.  ``undoc-members`` is *off* on purpose: it
        # would dump the module-level lookup tables and regexes (the country
        # list, the superscript map, every compiled pattern) as bare reprs.  With
        # it off, only members carrying a docstring are documented, so the page
        # stays a tour of the reasoning rather than a wall of constants.
        "automodule_options": {"members", "private-members", "show-inheritance"},
        "module_first": True,
    },
]

# Read top-to-bottom, a module's functions tell a story (parse → denormalize →
# crop → merge); alphabetical order would scramble it.
autodoc_member_order = "bysource"
add_module_names = False

autodoc_default_options = {
    "members": True,
    "private-members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable", None),
}

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 2,
    "navigation_depth": 3,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
