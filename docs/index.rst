pdfparser
=========

PDF parser used to convert PDFs for the D3 Annotation Hub.

``pdfparser`` turns a scientific-paper PDF into HTML.  Each page is rendered and
handed to the **LightOnOCR-2-1B-bbox** vision-language model — running out of process
in a vLLM server — which transcribes the whole page (reading order, emphasis, tables,
math, and figure crop boxes) to markdown; the rest of the pipeline is deterministic
clean-up that crops figures, recovers content the page pass dropped, re-stitches
column-split paragraphs, and sorts the front matter from the body before emitting the
document shell.  The entry point returns a
:class:`~pdfparser.pipeline.assemble.ParsedDocument` — the HTML plus the plain-text
title, byline, and best-effort DOI.

.. note::

   The OCR model runs in a separate server, so a reachable vLLM endpoint is required —
   start ``deploy/vllm/run-server.sh`` first.  Point the client at it with the
   ``PDFPARSER_VLLM_URL`` / ``PDFPARSER_VLLM_MODEL`` environment variables (or the
   ``base_url`` / ``model`` arguments).

.. code-block:: python

    from pathlib import Path

    from pdfparser import lightonocr_pdf_to_document

    doc = lightonocr_pdf_to_document("paper.pdf")
    Path("paper.html").write_text(doc.html)
    print(doc.title, doc.doi)

``lightonocr_pdf_to_html`` is the thin string wrapper when only the HTML is wanted.
Figure crops inline as base64 by default; pass ``image_dir=...`` to write them as
sidecar PNGs instead, or an ``encode_image`` sink (:obj:`~pdfparser.pipeline.figures.ImageSink`)
to store the bytes elsewhere and return a served URL.  Failures surface as the typed
exceptions in :mod:`pdfparser.pipeline.errors` (``OcrUnavailableError`` /
``OcrResponseError`` / ``PdfInputError``), so a batch worker can tell a retryable
outage from a permanent bad-input failure.

Or from the command line::

    python -m pdfparser paper.pdf          # writes paper.html
    python -m pdfparser paper.pdf out.html
    python -m pdfparser paper.pdf --vllm-url http://127.0.0.1:8000/v1 --image-dir img/

Start with :doc:`design` for the architecture and the trade-offs the pipeline
makes, :doc:`embedding` for the integration contract when you import pdfparser into
a long-lived worker, and the :doc:`api/modules` for every module — including the
private functions where most of the logic lives.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   design
   embedding
   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
