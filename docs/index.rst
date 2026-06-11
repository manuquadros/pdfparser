pdfparser
=========

PDF parser used to convert PDFs for the D3 Annotation Hub.

``pdfparser`` turns a scientific-paper PDF into a single self-contained HTML file.
Each page is rendered and handed to the **LightOnOCR-2-1B-bbox** vision-language
model, which transcribes the whole page — reading order, emphasis, tables, math,
and figure crop boxes — to markdown; the rest of the pipeline is deterministic
clean-up that crops figures, re-stitches column-split paragraphs, and sorts the
front matter from the body before emitting the document shell.

.. code-block:: python

    from pathlib import Path

    from pdfparser import lightonocr_pdf_to_html

    html = lightonocr_pdf_to_html("paper.pdf")
    Path("paper.html").write_text(html)

Or from the command line::

    python -m pdfparser paper.pdf          # writes paper.html
    python -m pdfparser paper.pdf out.html

Start with :doc:`design` for the architecture and the trade-offs the pipeline
makes; the :doc:`api/modules` documents every module, including the private
functions where most of the logic lives.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   design
   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
