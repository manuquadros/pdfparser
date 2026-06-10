"""Tests for the pdfparser.pipeline package.  Unit tests load no model and render
no PDF; the integration tests (``@pytest.mark.integration``) run the real pipeline."""

import base64
import io
import re
from pathlib import Path

import pytest
from PIL import Image


def _fake_image(width: int = 800, height: int = 1000) -> Image.Image:
    return Image.new("RGB", (width, height), color="white")


def _figure_sizes(html: str) -> list[tuple[int, int]]:
    """Decode every embedded ``data:image/png`` figure and return its (w, h)."""
    uris = re.findall(r"data:image/png;base64,([A-Za-z0-9+/=]+)", html)
    return [Image.open(io.BytesIO(base64.b64decode(u))).size for u in uris]


def _run_lighton(pages_md: list[str], image: Image.Image | None = None) -> str:
    """Assemble HTML from synthetic per-page markdown (no model, no rendering)."""
    from pdfparser.pipeline.assemble import _assemble_html

    img = image or _fake_image(1190, 1540)
    return _assemble_html(pages_md, [img for _ in pages_md])


def _body(html: str) -> str:
    """Extract the content of the <div class="body"> element.

    Depth-aware so nested </div> inside model-emitted HTML doesn't truncate.
    """
    start = html.find('<div class="body">')
    assert start >= 0, "body div not found"
    pos = start
    depth = 0
    while pos < len(html):
        if html.startswith("<div", pos):
            depth += 1
            pos += 4
        elif html.startswith("</div>", pos):
            depth -= 1
            if depth == 0:
                return html[start:pos]
            pos += 6
        else:
            pos += 1
    raise AssertionError("unclosed body div")


def _metadata(html: str) -> str:
    """Content of the collapsible <details class='metadata'> panel."""
    start = html.find("<details class='metadata'>")
    assert start >= 0, "metadata panel not found"
    return html[start : html.find("</details>", start)]


class TestArticlePageDetection:
    """Cover ads / mastheads carry no Abstract or Introduction heading, so the
    article start is the first page that does."""

    def test_ad_page_is_not_article(self) -> None:
        from pdfparser.pipeline.classify import _is_article_page_md

        ad = "# Virtual Conference\n\n## Data integrity seminar\n\nRegister here."
        assert _is_article_page_md(ad) is False

    def test_abstract_page_is_article(self) -> None:
        from pdfparser.pipeline.classify import _is_article_page_md

        assert _is_article_page_md("# Title\n\n## Abstract\n\nWe did things.") is True

    def test_introduction_page_is_article(self) -> None:
        from pdfparser.pipeline.classify import _is_article_page_md

        assert _is_article_page_md("## 1. Introduction\n\nText.") is True

    def test_leading_ad_page_skipped(self) -> None:
        from pdfparser.pipeline.classify import _leading_pages_to_skip_md

        ad = "# Conference\n\nRegister here."
        article = "# Real Title\n\n## Abstract\n\nBody."
        assert _leading_pages_to_skip_md([ad, article]) == 1
        assert _leading_pages_to_skip_md([article]) == 0


class TestRunningFurniture:
    """Short header/footer lines that recur across pages — even with differing
    page numbers — are dropped; real repeated sentences are kept."""

    def test_page_numbered_footer_removed(self) -> None:
        from pdfparser.pipeline.classify import _strip_running_furniture

        parts = [
            "<p>Biotechnology and Applied Biochemistry 601</p>",
            "<p>Real body sentence one.</p>",
            "<p>Biotechnology and Applied Biochemistry 602</p>",
        ]
        out = _strip_running_furniture(parts)
        assert out == ["<p>Real body sentence one.</p>"]

    def test_repeated_real_sentence_kept(self) -> None:
        from pdfparser.pipeline.classify import _strip_running_furniture

        parts = ["<p>This is a sentence.</p>", "<p>This is a sentence.</p>"]
        assert _strip_running_furniture(parts) == parts

    def test_short_enumerated_labels_kept(self) -> None:
        # "Fig 1"/"Fig 2" share a digit-stripped key but must not be removed —
        # only substantial recurring text (a journal footer) is furniture.
        from pdfparser.pipeline.classify import _strip_running_furniture

        parts = ["<p>Fig 1</p>", "<p>body</p>", "<p>Fig 2</p>"]
        assert _strip_running_furniture(parts) == parts

    def test_short_digit_free_footer_removed(self) -> None:
        # A bare author-surname running foot ("Clark" on alternating pages) is
        # short but digit-free, so the digit-strip collision the length floor
        # guards against can't happen — it must still be recognised as furniture.
        from pdfparser.pipeline.classify import _strip_running_furniture

        parts = [
            "<p>Clark</p>",
            "<p>Real body sentence one.</p>",
            "<p>Clark</p>",
        ]
        assert _strip_running_furniture(parts) == ["<p>Real body sentence one.</p>"]

    def test_heading_form_footer_removed(self) -> None:
        # OCR transcribes the running journal line as a heading on a sparse page
        # (last page / after references); it must still count as furniture and be
        # stripped, not survive as an <h1>.
        from pdfparser.pipeline.classify import _strip_running_furniture

        parts = [
            "<p>Biotechnology and Applied Biochemistry 601</p>",
            "<p>Real body sentence one.</p>",
            "<h1>Biotechnology and Applied Biochemistry</h1>",
        ]
        out = _strip_running_furniture(parts)
        assert out == ["<p>Real body sentence one.</p>"]

    def test_standalone_page_number_removed(self) -> None:
        # OCR sometimes isolates the folio into its own block, away from the
        # journal line, so digit-stripped recurrence can't catch it; a number-only
        # block is the page number itself and must be dropped.
        from pdfparser.pipeline.classify import _strip_running_furniture

        parts = ["<p>601</p>", "<p>Real body sentence one.</p>", "<h2>602</h2>"]
        assert _strip_running_furniture(parts) == ["<p>Real body sentence one.</p>"]

    def test_section_number_kept(self) -> None:
        # A numbered section heading ("3.4 …") is not a bare folio and stays.
        from pdfparser.pipeline.classify import _strip_running_furniture

        parts = ["<h2>3.4 Enzymatic activities</h2>", "<p>4</p>"]
        assert _strip_running_furniture(parts) == ["<h2>3.4 Enzymatic activities</h2>"]


class TestByline:
    """The block after the title becomes the header byline only when it
    positively looks like authors; otherwise it stays in the body."""

    def test_marker_line_is_byline(self) -> None:
        from pdfparser.pipeline.classify import _is_byline

        assert _is_byline("Nianyang Wu¹") is True
        assert _is_byline("Daniel D. Clark <sup>*</sup>") is True

    def test_name_list_is_byline(self) -> None:
        from pdfparser.pipeline.classify import _is_byline

        assert _is_byline("Jane Doe and John Smith") is True

    def test_metadata_lines_are_not_byline(self) -> None:
        from pdfparser.pipeline.classify import _is_byline

        assert _is_byline("Received 26 March 2019") is False
        assert _is_byline("DOI: 10.1002/bab.1760") is False
        assert _is_byline("This is a complete sentence.") is False

    def test_unmarked_single_name_is_not_byline(self) -> None:
        # No marker and not a list → ambiguous → not promoted (stays in body).
        from pdfparser.pipeline.classify import _is_byline

        assert _is_byline("Jane Doe") is False

    def test_metadata_after_title_goes_to_metadata_panel(self) -> None:
        # A date line under the title is metadata: it must not be promoted into
        # the header (and must not be lost) — it belongs in the Metadata panel.
        md = (
            "# T\n\nReceived 26 March 2019\n\n## Abstract\n\nThe abstract.\n\n"
            "## Methods\n\nMethod text."
        )
        html = _run_lighton([md])
        header = html[html.find("<header>") : html.find("</header>")]
        assert "Received 26 March 2019" in _metadata(html)
        assert "Received 26 March 2019" not in header
        assert "Received 26 March 2019" not in _body(html)

    def test_marked_authors_after_title_are_promoted(self) -> None:
        md = "# T\n\nNianyang Wu¹, Xiaoqiang Liu¹*\n\n## Abstract\n\nA."
        html = _run_lighton([md])
        header = html[html.find("<header>") : html.find("</header>")]
        assert "Nianyang Wu" in header
        assert "Nianyang Wu" not in _body(html)


class TestDegenerateRepetition:
    """A figure the model fails to box can be OCRed into a repeated-token wall;
    such a paragraph is dropped from the body, real prose is kept."""

    def test_token_wall_flagged(self) -> None:
        from pdfparser.pipeline.classify import _is_degenerate_repetition

        assert _is_degenerate_repetition("AaTRI, " * 40) is True

    def test_real_prose_not_flagged(self) -> None:
        from pdfparser.pipeline.classify import _is_degenerate_repetition

        prose = (
            "The enzyme catalyzes the stereospecific oxidation of the substrate"
            " to the corresponding ketone under physiological conditions."
        )
        assert _is_degenerate_repetition(prose) is False

    def test_token_wall_dropped_from_body(self) -> None:
        wall = "AaTRI, " * 50
        md = f"# T\n\n## Abstract\n\nA.\n\n## Body\n\n{wall}\n\nReal sentence here."
        body = _body(_run_lighton([md]))
        assert "AaTRI" not in body
        assert "Real sentence here." in body


class TestLightonAssembly:
    """End-to-end markdown → HTML assembly (the new model-free seam)."""

    def test_title_skips_document_type_label(self) -> None:
        # "# Article" is a document-type label; the real title is the next heading.
        md = "# Article\n\n## The Real Title\n\nA. Author\n\n### Abstract\n\nText."
        html = _run_lighton([md])
        assert "<h1>The Real Title</h1>" in html
        assert "Article" not in _header_h1(html)

    def test_byline_extracted_and_dropped_from_body(self) -> None:
        md = "# A Study of Things\n\nJane Doe¹\n\n## Abstract\n\nThe abstract body."
        html = _run_lighton([md])
        header = html[html.find("<header>") : html.find("</header>")]
        assert "Jane Doe" in header
        assert "Jane Doe" not in _body(html)

    def test_abstract_wrapped_in_section(self) -> None:
        md = (
            "# T\n\nA. U.\n\n## Abstract\n\nThe abstract paragraph here.\n\n"
            "## Body\n\nProse."
        )
        html = _run_lighton([md])
        start = html.find("<section class='abstract'>")
        end = html.find("</section>", start)
        assert "The abstract paragraph here." in html[start:end]
        assert "Prose." not in html[start:end]

    def test_leading_superscript_routed_to_footnote_before_refs(self) -> None:
        md = (
            "# T\n\n## Abstract\n\nAbstract.\n\n## Body\n\n"
            "<sup>*</sup>To whom correspondence should be addressed.\n\n"
            "## References\n\n[1] A reference."
        )
        html = _run_lighton([md])
        fn = html.find("To whom correspondence")
        ref = html.find("[1] A reference")
        assert 0 < fn < ref
        assert 'class="footnote"' in html

    def test_frontmatter_moved_to_metadata_panel_after_abstract(self) -> None:
        # Affiliations, keywords and abbreviations are OCR'd between the abstract
        # and the body's first section; they are pulled into the collapsible
        # Metadata panel (after the abstract) so the body opens with prose.
        md = (
            "# A Study\n\nJane Doe¹\n\n"
            "¹Department of Examples, Example University\n\n"
            "## Abstract\n\nThe abstract.\n\n"
            "**Keywords:** alpha, beta, gamma\n\n"
            "## Abbreviations\n\nTRI, tropine reductase.\n\n"
            "## 1. Introduction\n\nThe study begins here.\n\n"
            "## References\n\n[1] A reference."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        # The front matter is in the Metadata panel, not the body.
        assert "<summary>Metadata</summary>" in meta
        for fragment in (
            "Department of Examples",
            "Keywords:",
            "TRI, tropine reductase.",
        ):
            assert fragment in meta
            assert fragment not in body
        # The panel sits after the abstract and before the body.
        assert (
            html.find("</section>")
            < html.find("<details class='metadata'>")
            < html.find('<div class="body">')
        )
        # The body opens with prose (the Introduction), not metadata.
        assert "<h2>1. Introduction</h2>" in body

    def test_frontmatter_boundary_is_name_agnostic(self) -> None:
        # The boundary is the first non-metadata heading, not a literal
        # "Introduction" — a body opening with "Background" works the same.
        md = (
            "# A Study\n\n"
            "**Keywords:** alpha, beta\n\n"
            "## Background\n\nThe study begins here.\n\n"
            "## References\n\n[1] A reference."
        )
        html = _run_lighton([md])
        assert "Keywords:" in _metadata(html)
        assert "Keywords:" not in _body(html)
        assert "<h2>Background</h2>" in _body(html)

    def test_frontmatter_unchanged_when_body_opens_with_section(self) -> None:
        # First block is already a body section heading → nothing precedes it →
        # order left intact.
        md = "# T\n\n## Abstract\n\nA.\n\n## Methods\n\nFirst.\n\nSecond."
        body = _body(_run_lighton([md]))
        assert body.find("First.") < body.find("Second.")

    def test_unlabeled_body_prose_not_relocated(self) -> None:
        # An article whose body opens with unlabelled prose (no "Introduction"
        # heading, first heading is "Methods") must not have that prose moved to
        # the end: only positively-recognised metadata is relocated.
        md = (
            "# T\n\nUnlabeled opening prose paragraph here.\n\n"
            "More opening prose follows.\n\n"
            "## Methods\n\nMethod text.\n\n## References\n\n[1] A reference."
        )
        body = _body(_run_lighton([md]))
        assert body.find("Unlabeled opening prose paragraph here.") < body.find(
            "<h2>Methods</h2>"
        )

    def test_prose_starting_with_metadata_keyword_not_hidden(self) -> None:
        # A leading body paragraph that merely begins with a metadata keyword
        # ("Published…") is a sentence, not a front-matter label — it must stay
        # visible in the body, not be hidden in the Metadata panel.  (A keywords
        # label closes the abstract so the prose is the next, leading body block.)
        md = (
            "# T\n\n## Abstract\n\nThe abstract.\n\n**Keywords:** alpha, beta\n\n"
            "Published studies have shown that the enzyme is active.\n\n"
            "## Methods\n\nMethod text."
        )
        html = _run_lighton([md])
        assert "Published studies have shown" in _body(html)
        assert "Published studies have shown" not in _metadata(html)

    def test_unheaded_prose_after_metadata_heading_not_hidden(self) -> None:
        # Sticky metadata-section capture must stop at a real prose paragraph, so
        # an unheaded opening section after "## Keywords" is not swallowed.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Keywords\n\nalpha, beta, gamma.\n\n"
            "The introduction begins here without its own heading and reads as"
            " ordinary prose.\n\n## Methods\n\nMethod text."
        )
        html = _run_lighton([md])
        assert "The introduction begins here" in _body(html)
        assert "The introduction begins here" not in _metadata(html)
        assert "alpha, beta, gamma." in _metadata(html)  # short keywords stay metadata

    def test_long_abbreviation_list_stays_in_panel(self) -> None:
        # A semicolon-separated abbreviation list ending in a period reads like a
        # sentence, but as the block directly under "## Abbreviations" it is the
        # heading's own content and belongs in the panel, not the body.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Abbreviations\n\n"
            "TAs, tropane alkaloids; TRI, tropine-forming reductase; "
            "GC-MS, gas chromatography-mass spectrometer.\n\n"
            "## 1. Introduction\n\nThe study begins here without abbreviation.\n\n"
            "## References\n\n[1] A reference."
        )
        html = _run_lighton([md])
        assert "TAs, tropane alkaloids" in _metadata(html)
        assert "TAs, tropane alkaloids" not in _body(html)
        assert "<h2>1. Introduction</h2>" in _body(html)

    def test_multi_block_abbreviation_list_fully_in_panel(self) -> None:
        # A long abbreviation list OCR-split across two paragraphs: both halves
        # are ";"-separated lists, so the heading owns the whole run and neither
        # half leaks into the body (the second block is not heading-adjacent).
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Abbreviations\n\n"
            "TAs, tropane alkaloids; TRI, tropine-forming reductase; "
            "TRII, pseudotropine-forming reductase; qPCR, quantitative PCR.\n\n"
            "PGK, protein kinase; SDRs, short-chain dehydrogenases; "
            "GC-MS, gas chromatography-mass spectrometer; HPLC, chromatography.\n\n"
            "## 1. Introduction\n\nThe study begins here without abbreviation."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        for fragment in ("TAs, tropane alkaloids", "PGK, protein kinase"):
            assert fragment in meta
            assert fragment not in body
        assert "<h2>1. Introduction</h2>" in body

    def test_keyword_led_metadata_after_section_pulled_into_panel(self) -> None:
        # Correspondence / received / DOI lines OCR'd after the abbreviation list
        # are keyword-led metadata; inside the headed front-matter region they are
        # pulled into the panel even though each ends like a sentence.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Abbreviations\n\n"
            "TAs, tropane alkaloids; GC-MS, gas chromatography-mass spectrometer.\n\n"
            "Address for correspondence: Jane Doe, Example University, City. "
            "Tel. 123; e-mail: jane@example.edu.\n\n"
            "Received 26 March 2019; accepted 29 April 2019 DOI: 10.1002/bab.1760.\n\n"
            "## 1. Introduction\n\nThe study begins here without abbreviation."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        for fragment in ("Address for correspondence", "Received 26 March 2019"):
            assert fragment in meta
            assert fragment not in body
        assert "<h2>1. Introduction</h2>" in body

    def test_keyword_led_body_prose_in_section_not_hidden(self) -> None:
        # A body paragraph that merely opens with a front-matter keyword
        # ("Published …"), appearing inside a headed metadata section, must stay
        # in the body — it carries no metadata token (e-mail/DOI/date), unlike a
        # genuine "Received … DOI: …" line, so it is not hidden in the panel.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Abbreviations\n\n"
            "TAs, tropane alkaloids; GC-MS, gas chromatography-mass spectrometer.\n\n"
            "Published reports indicate that the enzyme is active across many "
            "plant species and remains the focus of ongoing research.\n\n"
            "## Methods\n\nMethod text."
        )
        html = _run_lighton([md])
        assert "Published reports indicate" in _body(html)
        assert "Published reports indicate" not in _metadata(html)

    def test_midbody_abbreviations_section_pulled_into_panel(self) -> None:
        # OCR places the Abbreviations section *after* the Introduction prose on
        # the first page, out of the leading front-matter run.  Its heading and
        # ";"-separated glossary are pulled into the panel; the misfiled body
        # content that follows (a learning-objectives list, the next section) is
        # left visible.
        md = (
            "# A Study\n\n## Abstract\n\nThe abstract.\n\n"
            "## Introduction\n\nThe study begins here with ordinary prose.\n\n"
            "## Abbreviations\n\n"
            "2-HEC, 2-(2-hydroxyethylthio)ethanesulfonate; CoM, coenzyme M; "
            "EE, enantiomeric excess; HPC, hydroxypropyl thioether.\n\n"
            "1. Students improve their appreciation for kinetic data.\n\n"
            "## Methods\n\nMethod text."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        assert "<h2>Abbreviations</h2>" in meta
        assert "2-HEC, 2-(2-hydroxyethylthio)ethanesulfonate" in meta
        assert "2-HEC, 2-(2-hydroxyethylthio)ethanesulfonate" not in body
        # The Introduction prose and the misfiled learning-objective stay visible.
        assert "The study begins here" in body
        assert "Students improve their appreciation" in body
        assert "<h2>Methods</h2>" in body

    def test_midbody_named_metadata_scoped_to_first_page(self) -> None:
        # A same-named section deeper in the document (here on a later page, e.g. a
        # back-matter glossary) is left in the body — only the first article page
        # is scanned for misplaced metadata sections.
        page1 = "# A Study\n\n## Abstract\n\nThe abstract.\n\n## Introduction\n\nProse."
        page2 = (
            "## Nomenclature\n\n"
            "F, force in newtons; m, mass in kilograms; a, acceleration.\n\n"
            "## References\n\n[1] A reference."
        )
        body = _body(_run_lighton([page1, page2]))
        assert "<h2>Nomenclature</h2>" in body
        assert "F, force in newtons" in body

    def test_midbody_bare_named_heading_stays_in_body(self) -> None:
        # A "Nomenclature" heading that opens a real prose section (no glossary
        # content under it) is a section title, not front matter: leave it visible.
        md = (
            "# A Study\n\n## Abstract\n\nThe abstract.\n\n"
            "## Introduction\n\nProse opening the article body.\n\n"
            "## Nomenclature\n\n"
            "This section explains the naming conventions used throughout the "
            "paper in ordinary prose that reads as a real section.\n\n"
            "## Methods\n\nMethod text."
        )
        body = _body(_run_lighton([md]))
        assert "<h2>Nomenclature</h2>" in body
        assert "This section explains the naming conventions" in body

    def test_stray_footer_metadata_pulled_into_panel(self) -> None:
        # The first page's bottom-of-page footer (journal citation + correspondence,
        # a supporting-info note, a submission/DOI line) is OCR'd into the body
        # after the Introduction.  Each self-contained metadata line is relocated to
        # the panel; the body prose that follows stays visible.
        md = (
            "# A Study\n\n## Abstract\n\nThe abstract.\n\n"
            "## Introduction\n\nIntro paragraph one is reasonably long prose here.\n\n"
            "Volume 47, Number 2, March/April 2019, Pages 124-132 *To whom "
            "correspondence should be addressed. Daniel D. Clark, Tel.: "
            "(530)-898-5251. E-mail: ddclark@csuchico.edu.\n\n"
            "Additional Supporting Information may be found in the online version "
            "of this article.\n\n"
            "Received 19 June 2018; Revised 23 August 2018; Accepted 6 December "
            "2018 DOI 10.1002/bmb.21202 Published online 28 December 2018 in Wiley "
            "Online Library (wileyonlinelibrary.com)\n\n"
            "Herein, I propose that data from the characterization can augment the "
            "teaching of enzyme kinetics. The case study had five goals in mind.\n\n"
            "## Methods\n\nMethod text."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        for fragment in (
            "Volume 47",
            "Additional Supporting Information",
            "Received 19 June",
        ):
            assert fragment in meta
            assert fragment not in body
        # Running before the merge keeps the ")" -terminated footer line from
        # absorbing the body prose that follows it.
        assert "Herein, I propose" in body
        assert "Herein, I propose" not in meta

    def test_body_sentence_with_one_email_not_hidden(self) -> None:
        # The stray-metadata sweep must not hide a body sentence that merely embeds
        # a single address: one token is below the threshold.
        md = (
            "# A Study\n\n## Abstract\n\nThe abstract.\n\n"
            "## Introduction\n\n"
            "Raw data are available on request from the author at "
            "ddclark@csuchico.edu.\n\n"
            "## Methods\n\nMethod text."
        )
        body = _body(_run_lighton([md]))
        assert "Raw data are available on request" in body

    def test_stray_metadata_predicate(self) -> None:
        from pdfparser.pipeline.classify import _is_stray_metadata

        # Two tokens (tel + e-mail) → relocated.
        assert _is_stray_metadata(
            "<p>*To whom correspondence should be addressed. Tel.: (530)-898-5251. "
            "E-mail: ddclark@csuchico.edu.</p>"
        )
        # Boilerplate phrase, no token → relocated.
        assert _is_stray_metadata(
            "<p>Additional Supporting Information may be found in the online "
            "version of this article.</p>"
        )
        # Single-token publication lines the OCR splits off the page-bottom block,
        # each unambiguous on its own shape → relocated.
        assert _is_stray_metadata(
            "<p>Volume 47, Number 2, March/April 2019, Pages 124-132</p>"
        )
        assert _is_stray_metadata("<p>DOI 10.1002/bmb.21202</p>")
        assert _is_stray_metadata(
            "<p>Published online 28 December 2018 in Wiley Online Library "
            "(wileyonlinelibrary.com)</p>"
        )
        # A single embedded e-mail is below the two-token bar → stays in body.
        assert not _is_stray_metadata(
            "<p>Raw data are available from the author at jane@example.edu.</p>"
        )
        # Prose that merely uses the words "volume"/"pages" or "published online"
        # — not the citation/publication shapes — stays in body.
        assert not _is_stray_metadata(
            "<p>The dataset volume reached 47 GB after we processed pages from "
            "124 to 132 of the log.</p>"
        )
        assert not _is_stray_metadata(
            "<p>These results were later published online for peer review.</p>"
        )
        # A body sentence that inline-cites a volume/pages reference is prose, not
        # a citation block: the journal shape is anchored at the block start, so
        # "Volume 47 … Pages 124" matches but a mid-sentence citation does not.
        assert not _is_stray_metadata(
            "<p>See volume 3, pages 45-67, for the original derivation.</p>"
        )
        assert not _is_stray_metadata(
            "<p>In volume 12, pages 8-9 of the proceedings, the method appeared.</p>"
        )
        assert not _is_stray_metadata(
            "<p>As shown in Vol. 5 pp. 30 onward, the trend continues.</p>"
        )
        # A long prose run with two tokens is not a footer line → stays in body.
        assert not _is_stray_metadata(
            "<p>" + "Lorem ipsum dolor sit amet. " * 20 + "Contact a@b.edu or "
            "c@d.edu.</p>"
        )

    def test_bare_affiliation_line_pulled_into_panel(self) -> None:
        # An author+affiliation line OCR'd between the title and the abstract,
        # without its author's superscript marker ("Name From the Department …,
        # City, Region, postcode"), is recognised structurally and pulled into the
        # panel, so it no longer breaks the leading run and strands the keywords.
        md = (
            "# A Study\n\n"
            "Daniel D. Clark From the Department of Chemistry and Biochemistry, "
            "California State University-Chico, Chico, California, 95929\n\n"
            "## Abstract\n\nThe abstract.\n\n"
            "**Keywords:** enzymology; enzyme kinetics; dehydrogenase\n\n"
            "## Introduction\n\nThe study begins here with ordinary prose.\n\n"
            "## References\n\n[1] A reference."
        )
        html = _run_lighton([md])
        meta, body = _metadata(html), _body(html)
        for fragment in ("From the Department of Chemistry", "Keywords:"):
            assert fragment in meta
            assert fragment not in body
        assert "The study begins here" in body
        assert "<h2>Introduction</h2>" in body

    def test_body_sentence_mentioning_university_not_hidden(self) -> None:
        # The affiliation detector must not hide a body sentence that merely names
        # an institution: a terminal period marks it as prose, not an address.
        md = (
            "# A Study\n\n"
            "The work was carried out with the University of Example, the "
            "Department of Chemistry, and several partners.\n\n"
            "## Methods\n\nMethod text."
        )
        body = _body(_run_lighton([md]))
        assert "The work was carried out with the University of Example" in body

    def test_truncated_body_fragment_with_institution_not_hidden(self) -> None:
        # An OCR-truncated prose clause that names institutions and lacks terminal
        # punctuation must stay visible: without a postal-code tail it is not an
        # address.  (Guards against the false positive the no-punctuation rule
        # alone allowed.)
        md = (
            "# A Study\n\n"
            "In this work, conducted jointly with the Department of Biology, the "
            "School of Medicine, and several partner hospitals across the region\n\n"
            "## Methods\n\nMethod text."
        )
        body = _body(_run_lighton([md]))
        assert "In this work, conducted jointly" in body

    def test_affiliation_line_predicate(self) -> None:
        from pdfparser.pipeline.classify import _is_affiliation_line

        assert _is_affiliation_line(
            "Daniel D. Clark From the Department of Chemistry and Biochemistry, "
            "California State University-Chico, Chico, California, 95929"
        )
        # A terminal period marks prose, not an address.
        assert not _is_affiliation_line(
            "The work was done at the University of Example, City, Region."
        )
        # No institution keyword.
        assert not _is_affiliation_line("Jane Doe, John Smith, and Mary Major")
        # Too few comma-separated segments to be an address layout.
        assert not _is_affiliation_line("Department of Chemistry")
        # An institution-naming prose clause with no postal-code tail is not an
        # address, even without terminal punctuation.
        assert not _is_affiliation_line(
            "In this work, conducted jointly with the Department of Biology, the "
            "School of Medicine, and several partner hospitals across the region"
        )
        # A number earlier in the line cannot stand in for the address tail.
        assert not _is_affiliation_line(
            "enrolled 250 patients from the Department of Cardiology, the ICU, "
            "and two partner clinics"
        )
        # The deliberate trade: an address with no postal code is left visible.
        assert not _is_affiliation_line(
            "Department of Chemistry, University of Oxford, Oxford, United Kingdom"
        )

    def test_all_frontmatter_body_kept_visible(self) -> None:
        # If every body block looks like front matter, that signals misdetection,
        # not a metadata-only doc: keep it visible rather than emptying the body.
        md = (
            "# T\n\n## Abstract\n\nThe abstract.\n\n**Keywords:** alpha, beta\n\n"
            "¹Affiliation One, City\n\nReceived 26 March 2019"
        )
        html = _run_lighton([md])
        assert "Affiliation One" in _body(html)
        assert "<details class='metadata'>" not in html

    def test_figure_placeholder_becomes_cropped_figure(self) -> None:
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\nBefore.\n\n"
            "![image](image_1.png)100,100,900,900\n\n"
            "FIG. 1 A nice figure caption.\n\nAfter."
        )
        html = _run_lighton([md])
        assert "data:image/png;base64," in html
        assert "<figcaption>FIG. 1 A nice figure caption.</figcaption>" in html
        assert "100,100,900,900" not in html

    def test_figure_bbox_is_denormalized_to_full_extent(self) -> None:
        # A full box in [0,1000] space must crop the whole page, not a top-left
        # sliver (the truncation bug: coords are normalized, not pixels).
        img = _fake_image(1190, 1540)
        md = "# T\n\n## Abstract\n\nA.\n\n## Body\n\n![image](i.png)0,0,1000,1000"
        html = _run_lighton([md], image=img)
        assert _figure_sizes(html) == [(1190, 1540)]


class TestDenormalizeBbox:
    """[0,1000]-normalized model boxes scale to the image's pixel size."""

    def test_full_box_maps_to_full_image(self) -> None:
        from pdfparser.pipeline.figures import _denormalize_bbox

        assert _denormalize_bbox((0, 0, 1000, 1000), _fake_image(1190, 1540)) == (
            0,
            0,
            1190,
            1540,
        )

    def test_half_box(self) -> None:
        from pdfparser.pipeline.figures import _denormalize_bbox

        assert _denormalize_bbox((0, 0, 500, 500), _fake_image(1000, 2000)) == (
            0,
            0,
            500,
            1000,
        )


class TestFigureBoxMerge:
    """A figure the model over-segments into stacked boxes is unioned into one
    crop; genuinely separate figures stay separate."""

    def test_same_column_adjacent_boxes_merge(self) -> None:
        from pdfparser.pipeline.figures import _figures_same

        assert _figures_same((100, 100, 900, 500), (110, 500, 890, 560), 50.0) is True

    def test_vertically_separated_boxes_do_not_merge(self) -> None:
        from pdfparser.pipeline.figures import _figures_same

        assert _figures_same((100, 100, 900, 300), (100, 800, 900, 950), 50.0) is False

    def test_side_by_side_boxes_do_not_merge(self) -> None:
        from pdfparser.pipeline.figures import _figures_same

        assert _figures_same((0, 0, 100, 500), (200, 0, 300, 500), 50.0) is False

    def test_union_box(self) -> None:
        from pdfparser.pipeline.figures import _union_box

        assert _union_box([(100, 100, 900, 500), (120, 480, 880, 560)]) == (
            100,
            100,
            900,
            560,
        )

    def test_split_figure_emits_single_crop(self) -> None:
        img = _fake_image(1190, 1540)
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\n"
            "![image](a.png)100,100,900,500\n\n"
            "![image](b.png)100,500,900,560\n\n"
            "FIG. 1 One caption."
        )
        sizes = _figure_sizes(_run_lighton([md], image=img))
        assert len(sizes) == 1
        # The union spans both boxes (down to y≈862 px), not just the first.
        assert sizes[0][1] > 700

    def test_two_separated_figures_stay_separate(self) -> None:
        img = _fake_image(1190, 1540)
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Body\n\n"
            "![image](a.png)100,100,900,300\n\n"
            "Some intervening prose between the two figures.\n\n"
            "![image](b.png)100,800,900,950"
        )
        assert len(_figure_sizes(_run_lighton([md], image=img))) == 2


class TestFigureBottomGrowth:
    """The crop grows down over contiguous ink to the figure's true bottom and
    stops at the whitespace before the caption; a box already ending in
    whitespace grows nothing, so caption text is never pulled in."""

    @staticmethod
    def _image() -> Image.Image:
        # White page: figure block y[100,300), caption block y[360,380),
        # separated by a 60 px whitespace gap.
        img = Image.new("RGB", (400, 800), "white")
        img.paste(Image.new("RGB", (300, 200), "black"), (50, 100))
        img.paste(Image.new("RGB", (300, 20), "black"), (50, 360))
        return img

    def test_tight_box_grows_to_figure_bottom(self) -> None:
        from pdfparser.pipeline.figures import _extend_bottom_to_content

        assert _extend_bottom_to_content(self._image(), 50, 350, 250) == 300

    def test_box_at_bottom_does_not_grow(self) -> None:
        from pdfparser.pipeline.figures import _extend_bottom_to_content

        assert _extend_bottom_to_content(self._image(), 50, 350, 300) == 300

    def test_no_growth_when_ink_runs_without_gap(self) -> None:
        # Ink continues past the search window with no whitespace gap (caption /
        # body text below a correct box) → ambiguous → leave the box unchanged.
        from pdfparser.pipeline.figures import _extend_bottom_to_content

        img = Image.new("RGB", (400, 800), "white")
        img.paste(Image.new("RGB", (300, 300), "black"), (50, 100))
        assert _extend_bottom_to_content(img, 50, 350, 250) == 250

    def test_narrow_content_below_box_is_not_read_as_gap(self) -> None:
        # A figure tail narrower than the box (here 3 px of a 300 px-wide box,
        # ~1% ink) must count as content, not be mistaken for the whitespace gap
        # — otherwise the clipped bottom is dropped.
        from pdfparser.pipeline.figures import _extend_bottom_to_content

        img = Image.new("RGB", (400, 800), "white")
        img.paste(Image.new("RGB", (300, 150), "black"), (50, 100))  # y[100,250)
        img.paste(Image.new("RGB", (3, 40), "black"), (198, 250))  # narrow tail
        assert _extend_bottom_to_content(img, 50, 350, 270) == 290

    def test_growth_stops_before_caption(self) -> None:
        from pdfparser.pipeline.figures import _extend_bottom_to_content

        assert _extend_bottom_to_content(self._image(), 50, 350, 250) < 360

    def test_safe_crop_excludes_caption(self) -> None:
        from pdfparser.pipeline.figures import _safe_crop

        crop = _safe_crop(self._image(), (50, 100, 350, 250))
        assert crop is not None and crop.size == (300, 200)


class TestCrossPageMerge:
    """A paragraph split across a page break is rejoined."""

    def test_cross_page_paragraph_merge(self) -> None:
        page1 = "# T\n\n## Abstract\n\nA.\n\n## Body\n\nThis suggests that TRI and"
        page2 = "TRII compete for the same substrate tropinone."
        html = _run_lighton([page1, page2])
        assert "This suggests that TRI and TRII compete for the same substrate" in html
        assert "This suggests that TRI and</p>" not in html


class TestCaptionMergeBarrier:
    """A figure/table caption is never absorbed as a paragraph continuation,
    even across intervening floats and even when wrapped in <strong>."""

    def test_table_caption_after_floats_not_glued_to_fragment(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        parts = [
            "<p>PtTRII catalyzed the reduction of tropinone to form</p>",
            "<figure><img src='a' alt=''></figure>",
            "<figure><img src='b' alt=''></figure>",
            "<p><strong>TABLE 1</strong> Enzyme kinetics of PtTRI and PtTRII</p>",
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
        ]
        out = _merge_split_paragraphs(parts)
        # Caption stays its own block, immediately before its table; the
        # fragment and the floats keep their order.
        assert out == parts
        assert "to form <strong>TABLE 1</strong>" not in "".join(out)

    def test_real_continuation_still_merges(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        parts = [
            "<p>This suggests that TRI and</p>",
            "<p>TRII compete for the substrate.</p>",
        ]
        out = _merge_split_paragraphs(parts)
        assert out == [
            "<p>This suggests that TRI and TRII compete for the substrate.</p>"
        ]

    def test_function_word_with_trailing_comma_blocks_capital_continuation(
        self,
    ) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        # "revealed that," is grammatically incomplete; a capitalised new
        # sentence is not its continuation (here an OCR-misplaced figure caption),
        # so the trailing comma must not disarm the capital-letter guard.
        parts = [
            "<p>analyses of 2-butanol production revealed that,</p>",
            "<p>Molecule structures are shown in Fig. 3.</p>",
        ]
        assert _merge_split_paragraphs(parts) == parts

    def test_function_word_with_trailing_comma_still_merges_lowercase(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        # The genuine lowercase continuation of the same clause still joins.
        parts = [
            "<p>analyses of 2-butanol production revealed that,</p>",
            "<p>with no additives present, all forms preferred re-face addition.</p>",
        ]
        out = _merge_split_paragraphs(parts)
        assert out == [
            "<p>analyses of 2-butanol production revealed that, with no additives "
            "present, all forms preferred re-face addition.</p>"
        ]

    def test_preposition_comma_does_not_block_proper_noun_continuation(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        # The comma allowance is only for clause-introducers; after a preposition
        # a trailing comma before a capitalised proper noun is a genuine
        # continuation, so the merge must still join across the break.
        parts = [
            "<p>the epoxide-metabolising strains studied here consist of,</p>",
            "<p>Xanthobacter autotrophicus and related species.</p>",
        ]
        out = _merge_split_paragraphs(parts)
        assert out == [
            "<p>the epoxide-metabolising strains studied here consist of, "
            "Xanthobacter autotrophicus and related species.</p>"
        ]

    def test_metadata_line_not_merged_into_following_prose(self) -> None:
        # A self-contained footer-metadata line that ends without terminal
        # punctuation (here a ")") must not be treated as an incomplete paragraph
        # and glued to the body prose the OCR placed after it.
        from pdfparser.pipeline.merge import _merge_split_paragraphs_stable

        parts = [
            "<p>Published online 28 December 2018 in Wiley Online Library "
            "(wileyonlinelibrary.com)</p>",
            "<p>Herein, I propose that the method generalizes to other enzymes.</p>",
        ]
        assert _merge_split_paragraphs_stable(parts) == parts

    def test_continuation_after_two_figures_and_a_table_merges(self) -> None:
        from pdfparser.pipeline.merge import _merge_split_paragraphs

        # A column break stranded the continuation behind a figure+figure+table
        # cluster (the table's caption already folded in by colocation).
        parts = [
            "<p>PtTRII catalyzed the reduction of tropinone to form</p>",
            "<figure><img src='a' alt=''></figure>",
            "<figure><img src='b' alt=''></figure>",
            "<table><caption>TABLE 1</caption><tbody><tr><td>1</td></tr></tbody>"
            "</table>",
            "<p>pseudotropine with higher affinity to tropinone.</p>",
        ]
        out = _merge_split_paragraphs(parts)
        assert out[0] == (
            "<p>PtTRII catalyzed the reduction of tropinone to form "
            "pseudotropine with higher affinity to tropinone.</p>"
        )
        # The floats are relocated after the joined paragraph, order preserved.
        assert out[1:] == parts[1:4]


class TestTableCaptionColocation:
    """A free-standing "Table N …" caption is folded into its <table> as a
    <caption> first child so it renders with the table, not adrift."""

    def test_caption_before_table_folded(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        parts = [
            "<p><strong>TABLE 1</strong> Enzyme kinetics of PtTRI and PtTRII</p>",
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
        ]
        assert _colocate_table_captions(parts) == [
            "<table><caption><strong>TABLE 1</strong> Enzyme kinetics of PtTRI "
            "and PtTRII</caption><tbody><tr><td>1</td></tr></tbody></table>"
        ]

    def test_caption_after_table_folded(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        parts = [
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
            "<p>Table 2. Results.</p>",
        ]
        out = _colocate_table_captions(parts)
        assert out == [
            "<table><caption>Table 2. Results.</caption>"
            "<tbody><tr><td>1</td></tr></tbody></table>"
        ]

    def test_caption_separated_by_figure_folded(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        # The reported bug: a figure floats between the caption and its table.
        parts = [
            "<p>Table 3 Kinetic constants</p>",
            "<figure><img src='x' alt=''></figure>",
            "<table><tr><td>a</td></tr></table>",
        ]
        out = _colocate_table_captions(parts)
        assert out == [
            "<figure><img src='x' alt=''></figure>",
            "<table><caption>Table 3 Kinetic constants</caption>"
            "<tr><td>a</td></tr></table>",
        ]

    def test_caption_not_pulled_across_prose(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        # A real paragraph between caption and table breaks the association.
        parts = [
            "<p>Table 4 X</p>",
            "<p>Unrelated body sentence.</p>",
            "<table><tr><td>a</td></tr></table>",
        ]
        assert _colocate_table_captions(parts) == parts

    def test_orphan_caption_left_intact(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        parts = [
            "<p>Table 9 Orphan caption with no table near it.</p>",
            "<p>Prose.</p>",
        ]
        assert _colocate_table_captions(parts) == parts

    def test_two_tables_pair_with_own_captions(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        parts = [
            "<p>Table 1 A</p>",
            "<table><tr><td>1</td></tr></table>",
            "<p>Table 2 B</p>",
            "<table><tr><td>2</td></tr></table>",
        ]
        out = _colocate_table_captions(parts)
        assert "<caption>Table 1 A</caption>" in out[0]
        assert "<caption>Table 2 B</caption>" in out[1]
        assert len(out) == 2

    def test_existing_caption_not_duplicated(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        parts = [
            "<p>Table 5 Duplicate guard</p>",
            "<table><caption>existing</caption><tr><td>1</td></tr></table>",
        ]
        out = _colocate_table_captions(parts)
        # The table keeps its own caption; the stray block is left, not lost.
        assert out == parts

    def test_table_attributes_preserved(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        # The caption goes after the *whole* opening tag, not inside it.
        parts = [
            "<p>Table 1 Kinetics</p>",
            '<table class="data"><tbody><tr><td>1</td></tr></tbody></table>',
        ]
        out = _colocate_table_captions(parts)
        assert out == [
            '<table class="data"><caption>Table 1 Kinetics</caption>'
            "<tbody><tr><td>1</td></tr></tbody></table>"
        ]

    def test_caption_between_tables_pairs_with_following(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        # A caption sat between two tables precedes the second, so it belongs to
        # it — the first (captionless) table must not forward-steal it.
        parts = [
            "<table><tr><td>1</td></tr></table>",
            "<p>Table 2 Results</p>",
            "<table><tr><td>2</td></tr></table>",
        ]
        out = _colocate_table_captions(parts)
        assert out == [
            "<table><tr><td>1</td></tr></table>",
            "<table><caption>Table 2 Results</caption><tr><td>2</td></tr></table>",
        ]

    def test_reference_sentence_not_folded(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_captions

        # "Table N <lowercase verb> …" is a running reference, not a caption; it
        # must stay in the body, not be absorbed into the table.
        parts = [
            "<p>Table 1 summarizes the kinetic parameters of both enzymes.</p>",
            "<table><tr><td>1</td></tr></table>",
        ]
        assert _colocate_table_captions(parts) == parts

    def test_bare_label_rejoined_then_folded(self) -> None:
        from pdfparser.pipeline.merge import (
            _colocate_table_captions,
            _join_split_table_caption_labels,
        )

        # The reported bug: OCR split "TABLE I" from its title, stranding the
        # title between label and table so the caption never folds.
        parts = [
            "<p>TABLE I</p>",
            "<p>Selected substrates and inhibitors used to investigate.</p>",
            "<table><tbody><tr><td>a</td></tr></tbody></table>",
        ]
        out = _colocate_table_captions(_join_split_table_caption_labels(parts))
        assert out == [
            "<table><caption>TABLE I Selected substrates and inhibitors "
            "used to investigate.</caption><tbody><tr><td>a</td></tr></tbody></table>"
        ]

    def test_labelled_caption_not_rejoined(self) -> None:
        from pdfparser.pipeline.merge import _join_split_table_caption_labels

        # A label that already carries its title ("Table 4 X") is a complete
        # caption; the following block is unrelated prose and must stay separate.
        parts = [
            "<p>Table 4 X</p>",
            "<p>Unrelated body sentence.</p>",
        ]
        assert _join_split_table_caption_labels(parts) == parts

    def test_bare_label_before_table_not_rejoined_with_table(self) -> None:
        from pdfparser.pipeline.merge import _join_split_table_caption_labels

        # A bare label sitting directly on its table needs no rejoin (the next
        # block is the <table>, not a stray title paragraph).
        parts = [
            "<p>TABLE I</p>",
            "<table><tbody><tr><td>a</td></tr></tbody></table>",
        ]
        assert _join_split_table_caption_labels(parts) == parts

    def test_split_caption_lets_paragraph_merge_across_table(self) -> None:
        # End-to-end: a paragraph split across a captioned table rejoins once the
        # split caption is folded into the <table> (only the float remains between
        # the two halves).
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Results\n\n"
            "analyses of 2-butanol production revealed that,\n\n"
            "TABLE I\n\n"
            "Selected substrates and inhibitors used to investigate.\n\n"
            "<table><tbody><tr><td>Km</td></tr></tbody></table>\n\n"
            "with no additives present, all forms preferred a re-face addition."
        )
        body = _body(_run_lighton([md]))
        assert "revealed that, with no additives present, all forms preferred" in body
        assert "<caption>TABLE I Selected substrates" in body
        assert "revealed that,</p>" not in body

    def test_end_to_end_caption_heads_table(self) -> None:
        # Full assembly: the fragment must not absorb the caption, and the
        # caption must end up inside its table, not before the figures.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Results\n\n"
            "PtTRII catalyzed the reduction of tropinone to form\n\n"
            "![image](0,0,500,500)\n\n"
            "**TABLE 1** Enzyme kinetics of PtTRI and PtTRII\n\n"
            "<table><tbody><tr><td>Km</td></tr></tbody></table>"
        )
        body = _body(_run_lighton([md]))
        assert "<caption><strong>TABLE 1</strong> Enzyme kinetics" in body
        assert "to form <strong>TABLE 1</strong>" not in body
        # The caption no longer appears as a stand-alone paragraph.
        assert "<p><strong>TABLE 1</strong>" not in body


class TestTableFootnoteColocation:
    """A table's trailing footnote run — superscript-marker lines plus a note
    sentence wedged before them — is folded onto the table block, not left adrift
    or swept into the article footnote section."""

    def test_marker_footnotes_folded_onto_table(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # The table carries the a/b markers its footnotes annotate.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td><td>E<sup>b</sup></td></tr>"
            "</tbody></table>",
            "<p><sup>a</sup>Apparent K values.</p>",
            "<p><sup>b</sup>ND = not determined.</p>",
        ]
        assert _colocate_table_footnotes(parts) == [
            "<table><tbody><tr><td>K<sup>a</sup></td><td>E<sup>b</sup></td></tr>"
            "</tbody></table>"
            '<p class="footnote"><sup>a</sup>Apparent K values.</p>'
            '<p class="footnote"><sup>b</sup>ND = not determined.</p>'
        ]

    def test_note_sentence_before_markers_folded(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # The reported case: a note sentence sits between the table and its
        # superscript footnotes, so it rides along into the table block.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p>Molecule structures are shown in Fig. 3.</p>",
            "<p><sup>a</sup>Apparent K values.</p>",
        ]
        assert _colocate_table_footnotes(parts) == [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>"
            '<p class="footnote">Molecule structures are shown in Fig. 3.</p>'
            '<p class="footnote"><sup>a</sup>Apparent K values.</p>'
        ]

    def test_body_after_markers_stays_in_stream(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # The body paragraph that resumes after the footnotes is not absorbed.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p><sup>a</sup>Apparent K values.</p>",
            "<p>with no additives present, all forms preferred re-face.</p>",
        ]
        out = _colocate_table_footnotes(parts)
        assert out == [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>"
            '<p class="footnote"><sup>a</sup>Apparent K values.</p>',
            "<p>with no additives present, all forms preferred re-face.</p>",
        ]

    def test_article_footnote_marker_not_in_table_not_absorbed(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # The hardening: a superscript line whose label the table does NOT carry
        # is an article footnote that merely follows the table, not a table
        # footnote, so it is left for the classifier to route before references.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p><sup>*</sup>Corresponding author: a@b.com.</p>",
        ]
        assert _colocate_table_footnotes(parts) == parts

    def test_numeric_marker_matching_table_exponent_not_absorbed(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # A numbered article footnote whose digit collides with a table exponent
        # ("cm<sup>2</sup>") must not be folded — exponents are not footnote
        # referents, so the numeric label does not qualify as a table marker.
        parts = [
            "<table><tbody><tr><td>area cm<sup>2</sup></td></tr></tbody></table>",
            "<p><sup>2</sup>A numbered article footnote.</p>",
        ]
        assert _colocate_table_footnotes(parts) == parts

    def test_letter_marker_folded_despite_table_exponents(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # A letter footnote still folds when the table mixes exponents and an
        # 'a' referent; the exponent does not interfere.
        parts = [
            "<table><tbody><tr><td>cm<sup>2</sup></td><td>K<sup>a</sup></td></tr>"
            "</tbody></table>",
            "<p><sup>a</sup>Apparent K values.</p>",
        ]
        assert _colocate_table_footnotes(parts) == [
            "<table><tbody><tr><td>cm<sup>2</sup></td><td>K<sup>a</sup></td></tr>"
            "</tbody></table>"
            '<p class="footnote"><sup>a</sup>Apparent K values.</p>'
        ]

    def test_note_before_unmatched_marker_not_absorbed(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # A leading note rides along only when matching markers follow; an
        # unmatched marker abandons the run, so the note stays in the stream.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p>A note sentence.</p>",
            "<p><sup>*</sup>An article footnote.</p>",
        ]
        assert _colocate_table_footnotes(parts) == parts

    def test_note_without_markers_not_absorbed(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # A plain paragraph after a table with no footnote markers is body, not a
        # note, and is left untouched.
        parts = [
            "<table><tbody><tr><td>1</td></tr></tbody></table>",
            "<p>This paragraph continues the discussion.</p>",
        ]
        assert _colocate_table_footnotes(parts) == parts

    def test_runaway_leading_prose_not_swallowed(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # More than a note's worth of prose before any marker is body, so the
        # run is abandoned and nothing is folded — even though the table carries
        # the late marker's label.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p>First body paragraph.</p>",
            "<p>Second body paragraph.</p>",
            "<p>Third body paragraph.</p>",
            "<p><sup>a</sup>A late marker.</p>",
        ]
        assert _colocate_table_footnotes(parts) == parts

    def test_second_leading_line_exceeds_note_cap(self) -> None:
        from pdfparser.pipeline.merge import _colocate_table_footnotes

        # A table note is a single line; two non-marker lines before the marker
        # exceed the cap, so the run is abandoned rather than swallowing a second
        # (possibly body) line.
        parts = [
            "<table><tbody><tr><td>K<sup>a</sup></td></tr></tbody></table>",
            "<p>First note line.</p>",
            "<p>Second note line.</p>",
            "<p><sup>a</sup>Apparent K values.</p>",
        ]
        assert _colocate_table_footnotes(parts) == parts

    def test_end_to_end_table_footnotes_unblock_merge(self) -> None:
        # Full assembly: the table footnotes (a note sentence + marker lines) fold
        # into the table, so the paragraph split across the table rejoins and the
        # footnotes are not relocated to the article footnote section.
        md = (
            "# T\n\n## Abstract\n\nA.\n\n## Results\n\n"
            "analyses of 2-butanol production revealed that,\n\n"
            "<table><tbody><tr><td>Km<sup>a</sup></td><td>E<sup>b</sup></td></tr>"
            "</tbody></table>\n\n"
            "Molecule structures are shown in Fig. 3.\n\n"
            "<sup>a</sup>Apparent K values.\n\n"
            "<sup>b</sup>ND = not determined.\n\n"
            "with no additives present, all forms preferred a re-face addition."
        )
        body = _body(_run_lighton([md]))
        assert "revealed that, with no additives present, all forms preferred" in body
        # The footnotes ride with the table inside the body, not before references.
        note = '<p class="footnote">Molecule structures are shown in Fig. 3.</p>'
        assert note in body
        assert '<p class="footnote"><sup>a</sup>Apparent K values.</p>' in body


class TestLatexToHtml:
    """Inline `$…$` math is converted to deterministic sub/superscript HTML
    before markdown parsing."""

    def test_simple_subscript(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        assert _latex_to_html("$K_m$") == "K<sub>m</sub>"

    def test_braced_subscript(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        assert _latex_to_html("$V_{max}$") == "V<sub>max</sub>"

    def test_superscript_becomes_unicode(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # All-mappable superscript chars collapse to Unicode (matches "NAD⁺").
        assert _latex_to_html("NAD$^+$") == "NAD⁺"

    def test_superscript_letters_fall_back_to_tag(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        assert _latex_to_html("pH$^{S}$") == "pH<sup>S</sup>"

    def test_ratio_of_kinetic_constants(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        assert _latex_to_html("$k_{cat}/K_m$") == "k<sub>cat</sub>/K<sub>m</sub>"

    def test_degree_command_superscript(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # ``$^\circ$`` is the LaTeX degree idiom; the single-char superscript
        # rule used to capture only the backslash, leaving a lone ``\`` inside
        # <sup> that markdown then mangled into "<sup></sup>circ".
        assert _latex_to_html(r"grown at 25 $\pm$ 1$^\circ$C") == "grown at 25 ± 1°C"

    def test_braced_degree_command(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        assert _latex_to_html(r"$^{\circ}$C") == "°C"

    def test_symbol_command_translated(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        assert _latex_to_html(r"$5 \times 10^{3}$ cells") == "5 × 10³ cells"

    def test_greek_command_as_subscript(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        assert _latex_to_html(r"$T_\alpha$") == "T<sub>α</sub>"

    def test_command_matched_as_whole_token(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # Commands are matched as maximal "\name" tokens and looked up whole, so
        # a short command never eats the head of a longer one ("\to" vs "\top",
        # "\sim" vs "\simeq") — each resolves to its own glyph.
        assert _latex_to_html(r"$\to$") == "→"
        assert _latex_to_html(r"$\top$") == "⊤"
        assert _latex_to_html(r"$A \simeq B$") == "A ≃ B"

    def test_command_still_terminated_by_digit(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        assert _latex_to_html(r"$\alpha2$") == "α2"

    def test_unknown_command_left_literal(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # pylatexenc returns "" for an unknown macro; we keep the literal rather
        # than silently dropping it.
        assert _latex_to_html(r"$x\notacommand y$") == r"x\notacommand y"

    def test_extended_symbol_coverage(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # Coverage we get for free from pylatexenc that the old hand map lacked.
        assert _latex_to_html(r"$T_\beta + \nabla$") == "T<sub>β</sub> + ∇"

    def test_arg_macro_that_raises_does_not_crash(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # pylatexenc raises when fed a bare arg-taking macro like \sqrt; the
        # exception must be swallowed and the span left intact, not propagated
        # up to crash the whole page conversion.
        assert _latex_to_html(r"$\sqrt{x}$") == r"\sqrtx"

    def test_arg_macro_template_not_leaked(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # \frac's substitution template ("%s/%s") must not reach the output; the
        # command stays literal so real math survives for a later MathJax pass.
        assert "%s" not in _latex_to_html(r"$\frac{a}{b}$")
        assert _latex_to_html(r"$\frac{a}{b}$") == r"\fracab"

    def test_extended_symbol_coverage_via_command_helper(self) -> None:
        from pdfparser.pipeline.latex import _latex_command_to_unicode

        assert _latex_command_to_unicode(r"\sqrt") == r"\sqrt"
        assert _latex_command_to_unicode(r"\frac") == r"\frac"
        assert _latex_command_to_unicode(r"\alpha") == "α"

    def test_plain_text_untouched(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        assert _latex_to_html("no math here") == "no math here"

    def test_currency_dollars_left_alone(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # No TeX markup between the '$' → not math; must not be stripped/merged.
        assert _latex_to_html("costs $5 and $10 total") == "costs $5 and $10 total"

    def test_math_wrapped_bare_number_unwrapped(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # A lone number in a math span is just a value the model wrapped — drop
        # the '$' delimiters but keep the number (and the surrounding spaces).
        assert _latex_to_html("was $42.26$ Sec") == "was 42.26 Sec"

    def test_script_span_reattaches_to_preceding_token(self) -> None:
        from pdfparser.pipeline.latex import _latex_to_html

        # The model writes a unit and its exponent with a gap ("Sec $^{-1}$"); a
        # span opening with a script attaches to the previous token, no space.
        assert _latex_to_html("Sec $^{-1}$ mM $^{-1}$") == "Sec⁻¹ mM⁻¹"


class TestMdToHtmlBlocks:
    """A page's markdown becomes one HTML string per top-level block, with raw
    HTML (tables, <sup>) passed through and thematic breaks dropped."""

    def test_heading_and_paragraph_split(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        blocks = _md_to_html_blocks("## Introduction\n\nSome prose here.")
        assert blocks == ["<h2>Introduction</h2>", "<p>Some prose here.</p>"]

    def test_emphasis_rendered(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        (block,) = _md_to_html_blocks("*Przewalskia tangutica* is **rare**.")
        assert (
            block == "<p><em>Przewalskia tangutica</em> is <strong>rare</strong>.</p>"
        )

    def test_table_passthrough(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        table = "<table><tbody><tr><td>1</td></tr></tbody></table>"
        assert _md_to_html_blocks(table) == [table]

    def test_sup_passthrough(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        (block,) = _md_to_html_blocks("NAD<sup>+</sup> dependent.")
        assert block == "<p>NAD<sup>+</sup> dependent.</p>"

    def test_degree_does_not_bleed_into_superscript(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        # Regression: "$^\circ$" once produced "<sup>\</sup>", whose lone "\<"
        # markdown escaped into "&lt;/sup&gt;", swallowing the rest of the
        # sentence into a superscript.  No <sup> must survive here.
        (block,) = _md_to_html_blocks(
            r"grown at 25 $\pm$ 1$^\circ$C under 16 H of light."
        )
        assert block == "<p>grown at 25 ± 1°C under 16 H of light.</p>"

    def test_thematic_break_dropped(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        assert _md_to_html_blocks("A.\n\n---\n\nB.") == ["<p>A.</p>", "<p>B.</p>"]

    def test_list_kept_as_one_block(self) -> None:
        from pdfparser.pipeline.markdown import _md_to_html_blocks

        (block,) = _md_to_html_blocks("- one\n- two")
        assert block.startswith("<ul>")
        assert "<li>one</li>" in block and "<li>two</li>" in block


class TestParseFigurePlaceholder:
    """LightOnOCR-bbox emits figures as `![image](...)x0,y0,x1,y1`; the parser
    must recover the crop box, recognise a bbox-less placeholder, and reject
    ordinary prose."""

    def test_box_extracted(self) -> None:
        from pdfparser.pipeline.figures import _parse_figure_placeholder

        assert _parse_figure_placeholder("![image](image_1.png)122,89,877,614") == (
            122,
            89,
            877,
            614,
        )

    def test_box_with_surrounding_whitespace(self) -> None:
        from pdfparser.pipeline.figures import _parse_figure_placeholder

        assert _parse_figure_placeholder("  ![image](img.png) 10, 20, 30, 40 ") == (
            10,
            20,
            30,
            40,
        )

    def test_bboxless_placeholder_returns_true(self) -> None:
        from pdfparser.pipeline.figures import _parse_figure_placeholder

        assert _parse_figure_placeholder("![image](image_1.png)") is True

    def test_caption_line_is_not_a_placeholder(self) -> None:
        from pdfparser.pipeline.figures import _parse_figure_placeholder

        assert _parse_figure_placeholder("FIG. 2 Protein alignments of TRI.") is None

    def test_inline_image_in_prose_is_not_a_placeholder(self) -> None:
        from pdfparser.pipeline.figures import _parse_figure_placeholder

        line = "Some prose with ![inline](x.png) embedded mid-sentence."
        assert _parse_figure_placeholder(line) is None


_FIXTURE_PDF = Path(__file__).parent / "fixtures" / "30592559.pdf"
_AD_PREFIX_PDF = Path(__file__).parent / "fixtures" / "31051047.pdf"
_OUTPUT_DIR = Path(__file__).parent.parent / "spike_results"


def _run_pipeline_to_file(pdf: Path, ocr: object) -> str:
    """Run the full pipeline on ``pdf`` and save the HTML for visual inspection.

    The output lands at ``spike_results/<pdf-stem>.html`` so every integration
    run leaves an on-disk copy of each fixture's rendering to open in a browser.
    """
    from pdfparser.pipeline import OcrModel, lightonocr_pdf_to_html

    assert isinstance(ocr, OcrModel)
    html = lightonocr_pdf_to_html(pdf, ocr=ocr)
    _OUTPUT_DIR.mkdir(exist_ok=True)
    (_OUTPUT_DIR / f"{pdf.stem}.html").write_text(html, encoding="utf-8")
    return html


@pytest.fixture(scope="session")
def ocr_model() -> object:
    """Load the LightOnOCR model bundle once per session; skip if unavailable."""
    try:
        from pdfparser.pipeline import load_ocr_model

        return load_ocr_model()
    except Exception as e:
        pytest.skip(f"LightOnOCR model not available: {e}")


@pytest.fixture(scope="session")
def article_html(ocr_model: object) -> str:
    """Run the full pipeline on the no-ad fixture; skip if the model is absent.

    Writes the result to spike_results/30592559.html so the file stays current
    after each integration run.
    """
    if not _FIXTURE_PDF.exists():
        pytest.skip(f"Fixture PDF not found: {_FIXTURE_PDF}")
    return _run_pipeline_to_file(_FIXTURE_PDF, ocr_model)


def _header_h1(html: str) -> str:
    """Return the text of the document's <header><h1> title element."""
    m = re.search(r"<header>.*?<h1>(.*?)</h1>", html, re.DOTALL)
    assert m, "header <h1> not found"
    return m.group(1)


@pytest.mark.integration
class TestPipeline:
    """Integration tests: run the full LightOnOCR pipeline on the fixture PDF.

    Skipped when the model is not available (no GPU, weights not downloaded).
    Each run also refreshes spike_results/30592559.html.
    """

    def test_abstract_no_column_break(self, article_html: str) -> None:
        abstract_start = article_html.find("<section class='abstract'>")
        abstract_end = article_html.find("</section>", abstract_start)
        abstract_block = article_html[abstract_start:abstract_end]
        # Both halves must be collected: if either is absent the pipeline
        # dropped a fragment it should have kept.
        assert "classical and contemporary" in abstract_block
        assert "experimental biochemistry." in abstract_block
        # And they must appear in the same paragraph — no split <p>.
        assert "classical and contemporary</p>" not in abstract_block

    def test_which_is_not_merged_with_as_a_testament(self, article_html: str) -> None:
        assert "which is As a testament to the utility" not in article_html

    def test_ternary_complex_followed_by_clearly_showed(
        self, article_html: str
    ) -> None:
        expected = (
            "The 1.8 Å ternary complex (enzyme + 2-KPC + NAD⁺)"
            " clearly showed interaction of the R152"
        )
        assert expected in article_html

    def test_first_page_footer_metadata_in_panel(self, article_html: str) -> None:
        # The first page's bottom-of-page footer — the journal-citation /
        # correspondence line and the supporting-information note — is front matter
        # the OCR drops into the body.  It is relocated to the collapsed Metadata
        # panel (which renders before the body), not left inline.
        panel, body = _metadata(article_html), _body(article_html)
        # The OCR splits the page-bottom block into one-line pieces; every one is
        # relocated, including the journal-citation / DOI / "Published online" lines
        # that carry only a single metadata token.
        for fragment in (
            "To whom correspondence should be addressed",
            "Additional Supporting Information",
            "Received 19 June 2018",
            "Volume 47",
            "DOI 10.1002/bmb.21202",
            "Published online 28 December 2018",
        ):
            assert fragment in panel
            assert fragment not in body
        # Pulling the orphan lines out before the (cross-page) paragraph-merge stops
        # them from chaining into the page-2 prose the OCR placed after them.
        assert "Herein, I propose" in body

    def test_figure_caption_not_glued_to_following_paragraph(
        self, article_html: str
    ) -> None:
        # The Fig. 1 caption ends in a period *inside* a closing </em>; the body
        # paragraph that follows ("Herein, I propose …") must not be absorbed onto
        # the caption (the sentence-end test runs on visible text, not raw HTML).
        body = _body(article_html)
        assert "oxidoreductase/carboxylase.</em> Herein, I propose" not in body
        assert "carboxylase.</em></p>" in body

    def test_paragraph_rejoined_across_table_i(self, article_html: str) -> None:
        # The clause "…revealed that," / "with no additives present…" is split by
        # TABLE I (its caption and footnotes folded in); it must read as one
        # paragraph, not be left split or glued to the stray Fig. 3 note sentence.
        body = _body(article_html)
        assert "revealed that, with no additives present, all forms of rR-HPCDH" in body
        assert "revealed that,</p>" not in body
        assert "revealed that, Molecule structures" not in body

    def test_table_footnotes_ride_with_their_table(self, article_html: str) -> None:
        # TABLE I's footnote run — the "Molecule structures … Fig. 3." note plus
        # the <sup>-marked footnotes — is folded onto the table block, so it
        # renders under the table rather than being swept into the article
        # footnote section before the references.
        body = _body(article_html)
        note = '<p class="footnote">Molecule structures are shown in Fig. 3.</p>'
        assert note in body
        # The note sits immediately after a </table>, not adrift in the prose.
        assert '</table><p class="footnote">Molecule structures' in body


@pytest.fixture(scope="session")
def ad_prefix_html(ocr_model: object) -> str:
    """Full pipeline output for the ad-prefixed 31051047.pdf fixture.

    Writes the result to spike_results/31051047.html for visual inspection.
    """
    if not _AD_PREFIX_PDF.exists():
        pytest.skip(f"Fixture PDF not found: {_AD_PREFIX_PDF}")
    return _run_pipeline_to_file(_AD_PREFIX_PDF, ocr_model)


@pytest.mark.integration
class TestAdPageExclusion:
    """The 31051047.pdf fixture has an advertisement as its first page; the
    pipeline must drop it and start the document at the real article title."""

    def test_title_starts_with_article_title(self, ad_prefix_html: str) -> None:
        # The title carries the PDF's intra-title line breaks; normalize runs of
        # whitespace before matching the prefix.
        title = re.sub(r"\s+", " ", _header_h1(ad_prefix_html)).strip()
        assert title.startswith(
            "Biochemical characterization reveals the functional divergence"
        )

    def test_species_name_italicized(self, ad_prefix_html: str) -> None:
        assert "<em>Przewalskia tangutica</em>" in ad_prefix_html

    def test_gene_names_not_collapsed_by_ocr(self, ad_prefix_html: str) -> None:
        # The motivating bug: the old OCR misread both PtTRI and PtTRII as the
        # single string "PITRI", collapsing two distinct genes.  LightOnOCR reads
        # them correctly; tables are excluded so the check targets the prose.
        prose = re.sub(r"<table.*?</table>", "", ad_prefix_html, flags=re.DOTALL)
        assert "PtTRI" in prose
        assert "PtTRII" in prose
        assert "PITRI" not in prose

    def test_figures_embedded_by_detector(self, ad_prefix_html: str) -> None:
        # LightOnOCR-bbox emits a crop box per figure (incl. the Fig 2 alignment
        # the old engine OCRed into a token wall); each is cropped and embedded.
        assert ad_prefix_html.count("data:image/png;base64,") >= 4

    def test_figures_not_truncated(self, ad_prefix_html: str) -> None:
        # The model emits boxes normalized to [0, 1000]; cropping them as raw
        # pixels truncated every figure.  A page-spanning figure (the Fig 2
        # alignment) only exceeds 1000 px wide once the box is denormalized to
        # the ~1190 px render width — impossible if coords are read as pixels.
        widest = max(w for w, _ in _figure_sizes(ad_prefix_html))
        assert widest > 1000, f"widest figure is only {widest}px — boxes not scaled"

    def test_cross_page_paragraph_not_split(self, ad_prefix_html: str) -> None:
        # The clause "…TRI and" / "TRII compete…" spans a page break; it must
        # be a single paragraph, not split at the page boundary.
        assert (
            "This suggests that TRI and TRII compete for the same substrate"
            in ad_prefix_html
        )
        assert "This suggests that TRI and</p>" not in ad_prefix_html
