Design notes
============

This page is the orientation a contributor needs before changing the pipeline:
how the stages fit together, and — more importantly — the recurring judgement
calls the code makes and *why* it makes them the way it does.  The per-function
docstrings in the :doc:`api/modules` carry the detail; this page is the shape
they hang on.

The pipeline at a glance
------------------------

``pdfparser`` converts a PDF to a single self-contained HTML file by handing each
rendered page to **LightOnOCR-2-1B-bbox**, an end-to-end vision-language model
that returns the whole page as markdown — reading order, emphasis, ``<table>``
HTML, LaTeX math, and figure crop boxes appended to ``![image]`` placeholders.
Everything after the model is deterministic clean-up.

The flow (design *B-prime*)::

    PDF
     │  render.py        rasterize each page to a PIL image
     ▼
    page images
     │  model.py         LightOnOCR → per-page markdown   ← the only GPU/IO seam
     ▼
    per-page markdown
     │  markdown.py      markdown → list of block-HTML strings
     │  latex.py         inline $…$ → <sup>/<sub>/Unicode
     │  figures.py       ![image] box → cropped <figure>
     ▼
    flat block stream
     │  merge.py         re-stitch column-split paragraphs; fold table captions
     │  classify.py      split title / byline / abstract / front matter / body
     ▼
    assemble.py          document shell  →  HTML

The single most important structural decision is the **purity seam**.  Only
:mod:`pdfparser.pipeline.model` (and :mod:`~pdfparser.pipeline.render`) touch the
GPU or the filesystem.  Everything downstream — the entire clean-up, merge and
classification surface, which is where the bugs and the design live — is a pure
function of ``(list[page_markdown], list[page_image])``.  The seam is
:func:`~pdfparser.pipeline.assemble._assemble_html`, the render-free, model-free
core that :func:`~pdfparser.pipeline.assemble.lightonocr_pdf_to_html` wires
rendering and OCR around.  That is what makes the hard parts unit-testable
without a GPU: feed synthetic markdown plus blank images straight into the core
and assert on the HTML.  When you add a stage, keep it on the pure side of the
seam unless it genuinely needs pixels.

A note on testing: because real OCR output is the thing the heuristics actually
face, the highest-value regression tests replay *recorded* model markdown through
the pure core (split a dumped transcript on its page markers, pair each page with
one blank image) rather than hand-writing idealized markdown.

The trade-offs, and which way each one leans
--------------------------------------------

OCR clean-up is a long series of "is this X or Y?" decisions made on noisy
evidence.  There is no choice that is right every time, so each decision is
deliberately *biased* toward the failure that is cheaper to live with.  The three
biases below are not local quirks — they are applied consistently, and knowing
them tells you which way to push when you hit a new edge case.

Figures: rather lose a strip of margin than a strip of figure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A clipped figure is a serious, irrecoverable error — the reader simply never sees
that part of the image.  A crop that runs a few pixels into surrounding
whitespace is cosmetic.  So when the model's crop box and the actual figure
disagree, the code leans toward **including too much rather than too little**, and
that bias is intentional.  It shows up in three places:

* **Bottom and right recovery**
  (:func:`~pdfparser.pipeline.figures._extend_bottom_to_content`,
  :func:`~pdfparser.pipeline.figures._extend_right_to_content`).
  The model's box routinely clips the bottom — and, for a wide figure, the right
  edge — of a figure.  Rather than trust the box, the crop grows over figure
  content and stops at the whitespace gap beyond it (the space before the caption
  below, or the page margin / inter-column gutter to the right).  The "is this
  line blank?" threshold (``_FIGURE_BLANK_LINE_FRAC``) is kept near-empty on
  purpose: a sparse figure line — a thin axis, content narrower than the box —
  must count as *content* so growth doesn't stop early and behead the figure.

* **Over-segmentation merge**
  (:func:`~pdfparser.pipeline.figures._cluster_figure_boxes`,
  :func:`~pdfparser.pipeline.figures._union_box`).  The model sometimes splits one
  tall figure into stacked boxes.  Boxes that share a column and sit close
  vertically are unioned back into one crop, so a panel never gets dropped on the
  floor between two boxes.

* **Render resolution** (:mod:`pdfparser.pipeline.render`).  Pages are rasterized
  at 200 DPI and only downscaled to the model's long-side budget, so the crop has
  real pixels to recover.

The bias is held in check in three ways, each keyed on a signal that growth would
swallow prose rather than recover figure.  First, *ambiguity*: if the ink past the
box runs all the way to the search cap with no clear whitespace gap, that is
usually a correct box sitting above caption or body text, and extending would
swallow prose — so the box is left untouched.  Second, a *leading gap*
(:func:`~pdfparser.pipeline.figures._ink_run_end`): growth recovers only content
*contiguous* with the box; a clean whitespace gap before the first ink means the
box already ends at the figure boundary and what follows is separated from it (a
caption below, a neighbouring column to the right), so growth is declined.  A real
clip leaves no such gap — it cuts straight through figure ink.

Third — and this is the one case where the page geometry alone is not enough — a
caption can sit close *below* a clipped figure, contiguous with the recovered
tail, so the gap rules wave it through.  Here a *second, document-level* signal
breaks the tie: the OCR already emitted the caption as its own text block, so we
know prose sits below the figure.  When it does,
:func:`~pdfparser.pipeline.figures._trim_swallowed_caption` re-examines the
recovered band and drops it if it *reads* as prose — judged by the mean horizontal
ink-run length (``_FIGURE_PROSE_RUN_FRAC``): caption text is short letter runs,
while figure content worth recovering here (shaded panels, gel lanes, sequence
alignments) is long continuous runs.  The band is judged whole and kept-or-dropped
all-or-nothing, so a band mixing a real figure tail with a little caption is kept
rather than risk clipping the figure.  This one trades a sliver of the bias the
other way: a clipped *sparse* line-art tail (thin axes, tick labels) shares prose's
short runs and may be trimmed with the caption — an accepted loss, since the
caption is always re-emitted as ``<figcaption>`` and baking a whole caption into
the image is worse.

That last check has a blind spot, though: when the figure is *itself* text — a
sequence alignment, a data table — the caption is pixel-identical to it, and the
model sometimes draws the box low enough to bake the caption *inside* it, where no
run-length test can find it.  The only thing that still separates them is meaning,
and the OCR already supplies it: the caption was emitted as its own text block, so
its words are known.  :func:`~pdfparser.pipeline.figures._trim_baked_caption`
re-OCRs the trailing text bands (cheap and rare — gated on a caption being present
and the band being text- rather than dense-figure-textured, and confined to the
bottom of the crop) and drops a band that *reproduces the caption's words*.  Two
guards keep that honest: a high word-overlap bar, so a figure row the caption
merely *names* (an alignment labels its own rows — ``BsSDH``, ``HmSDH`` …) isn't
mistaken for the caption; and a distinct-word floor, so a row the model fails to
read and OCRs into a repeated-token wall (``BMSDH BMSDH BMSDH …``) — which would
otherwise score ~1.0 against a caption that names it — is rejected as degenerate.

The rule, then, is "expand toward the figure whenever the page shows you where it
ends; don't expand blindly into text — and when the OCR tells you a caption is
down there, trust prose-shaped pixels, or the caption's own words, to find it."

Text merging: when the join is uncertain, don't join
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Re-stitching paragraphs that a two-column layout split across a column or page
break (:func:`~pdfparser.pipeline.merge._merge_split_paragraphs`) leans the
*opposite* way from figures: a wrong merge fuses two unrelated sentences into
grammatical nonsense, which is worse than leaving a paragraph visibly broken in
two.  So the merge fires only when the evidence is positive — the first fragment
lacks terminal punctuation *and* the continuation is not a new sentence, caption,
enumeration item, or heading (those act as hard barriers).  Two guards are worth
internalizing because they encode this caution:

* The **function-word guard** (``_FUNCTION_WORD_END_RE``):
  a fragment ending in "the", "of", "and"… is grammatically incomplete, so its
  continuation must be lowercase.  If the next block starts with a capital, the
  real continuation was probably dropped by OCR, and the merge is *refused*
  rather than gluing unrelated text together.

* The **acronym exception** (``_ACRONYM_HEAD_RE``):
  an all-caps head ("TRII", "DNA") is mid-sentence, not a new-sentence capital, so
  it is allowed through — without this, a clause split as "…TRI and" / "TRII
  compete…" would be wrongly left in two.

A subtlety that bites if you touch these predicates: terminal punctuation often
hides inside a closing inline tag (``…carboxylase.</em>``).  Run sentence-end and
footnote-marker checks on the **visible** text
(:func:`~pdfparser.pipeline.text._visible_text`), never the raw block HTML.

Classification: never hide the body
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Front matter (affiliations, keywords, correspondence, submission dates, the
PLOS-style citation sidebar) is relocated into a *collapsed* "Metadata" panel so
the article opens with prose.  That makes one error mode far worse than the
other: metadata left visible in the body is a blemish, but body prose
misclassified as metadata is **hidden from the reader entirely**.  So every
front-matter heuristic demands strong, positive evidence and is happy to
under-collect:

* An affiliation OCR'd without its author superscript is accepted only on a
  structural tell — a comma-separated address ending in a postal code or a
  recognised country (:func:`~pdfparser.pipeline.classify._is_affiliation_line`).
  An address with neither is left visible rather than risk hiding a sentence that
  merely names a university.

* Stray footer metadata is relocated position-independently only on its own
  strong evidence — a known publishing label, two or more metadata tokens (DOI,
  e-mail, date), or a fixed boilerplate phrase
  (:func:`~pdfparser.pipeline.classify._is_stray_metadata`) — never because of
  where it sits.

* As a final backstop, :func:`~pdfparser.pipeline.classify._extract_front_matter`
  refuses to hide the body when the run of "front matter" would be the *entire*
  document: an all-front-matter result signals a detection failure, not a
  metadata-only paper.

Other deliberate choices
------------------------

Math is reduced, not rendered.  :mod:`pdfparser.pipeline.latex` converts inline
``$…$`` to Unicode super/subscripts or HTML ``<sup>``/``<sub>``; full equation
rendering is out of scope (a later MathJax option).  The guiding rule when a
construct can't be reduced is *lossless degradation*: keep the TeX literal so a
later pass can render it, rather than crash the page or leak ``%s`` substitution
templates.  Symbol-command translation is delegated to ``pylatexenc``'s
maintained macro table instead of a hand-curated map, with one documented
override (``^\circ`` → "°", the degree idiom).  Currency is protected too — a
``$…$`` span with no TeX markup is left verbatim so "$5 … $10" survives.

OCR decoding is greedy (:func:`~pdfparser.pipeline.model._ocr_page`).  OCR wants
the single most-likely transcription, and a deterministic decode avoids
run-to-run drift — notably a figure box that occasionally over-segments into two
stacked crops on a sampled decode.

Stage order is load-bearing.  In
:func:`~pdfparser.pipeline.assemble._assemble_html` the clean-up runs in a fixed
sequence: caption-label rejoin → table-caption colocation → table-footnote
colocation → classify → paragraph merge.  Footnote colocation must precede
classify so a table's footnotes stay with the table instead of being swept into
the article's footnote section, and it must precede the merge so the table is a
single float the cross-table paragraph merge can step over.  Classify pulls
``<sup>``-marked footnotes out of the body *before* the merge, so the merge only
ever sees what classify left behind.  Reorder these at your peril.
