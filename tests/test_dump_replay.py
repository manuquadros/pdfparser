"""Server-free regression harness: replay the six checked-in raw-OCR dumps through
the pure ``_assemble_html`` core and assert the *deterministic* block-stream
invariants that are otherwise only guarded by the ~31-min live integration gate.

The dumps under ``tests/data/dumps/`` are captured LightOnOCR page markdown for the
six fixtures.  Feeding them (plus one blank image per page) to ``_assemble_html``
exercises the entire render-free, model-free assembly path — caption/footnote
colocation, abstract/metadata routing, cross-page paragraph merge, heading
normalization, reference folding, panel fusion and latex rendering — without a GPU
or a vLLM server, so these run in the fast suite.

Scope boundary: these assertions cover only what ``_assemble_html`` itself produces
from the *raw* dump.  Anything that needs the live recovery passes (figure-crop
re-OCR, table re-OCR / text-layer rebuild, text-layer reconciliation) stays in the
integration suite — the dumps carry no PDF text layer and no real page images.
"""

import functools
import re
from pathlib import Path

import pytest
from helpers import (
    _abstract,
    _body,
    _byline,
    _fake_image,
    _header,
    _metadata,
    _run_lighton,
)

_DUMP_DIR = Path(__file__).parent / "data" / "dumps"
_PAGE_MARKER = re.compile(r"^===== PAGE \d+ =====$", re.MULTILINE)

_STEMS = (
    "30592559",
    "31051047",
    "31123167",
    "31298526",
    "32117944",
    "32639976",
)


def _load_dump(stem: str) -> list[str]:
    """Per-page markdown for ``stem``, split on the ``===== PAGE N =====`` markers."""
    text = (_DUMP_DIR / f"{stem}_raw_markdown.md").read_text(encoding="utf-8")
    # split()[0] is the preamble before PAGE 0; the rest are the pages in order.
    return [page.strip("\n") for page in _PAGE_MARKER.split(text)[1:]]


@functools.cache
def _replay(stem: str) -> str:
    """Assemble ``stem``'s dump once (cached) at the documented blank-image size."""
    return _run_lighton(_load_dump(stem), _fake_image(1540, 1995))


@pytest.mark.parametrize("stem", _STEMS)
class TestUniversalInvariants:
    """Hold for every fixture regardless of journal/layout."""

    def test_dump_loads_with_pages(self, stem: str) -> None:
        pages = _load_dump(stem)
        assert pages, f"{stem}: no pages parsed from dump"

    def test_assembles_with_single_title(self, stem: str) -> None:
        html = _replay(stem)
        assert html.count("<h1>") == 1, f"{stem}: expected exactly one <h1>"

    def test_abstract_section_present(self, stem: str) -> None:
        assert _abstract(_replay(stem)), f"{stem}: abstract section missing"

    def test_metadata_panel_present(self, stem: str) -> None:
        html = _replay(stem)
        assert "<details class='metadata'>" in html, f"{stem}: metadata panel missing"

    def test_tables_balanced(self, stem: str) -> None:
        body = _body(_replay(stem))
        opens, closes = body.count("<table"), body.count("</table>")
        assert opens == closes, (
            f"{stem}: tables unbalanced ({opens} open, {closes} close)"
        )


class TestHpcdhDump:
    """30592559 — abstract citation split, latex spans, caption/footnote colocation."""

    STEM = "30592559"

    def test_abstract_unsplit_and_complete(self) -> None:
        abstract = _abstract(_replay(self.STEM))
        assert "classical and contemporary" in abstract
        assert "experimental biochemistry." in abstract
        assert "classical and contemporary</p>" not in abstract

    def test_citation_tail_moved_to_panel(self) -> None:
        html = _replay(self.STEM)
        assert "International Union of Biochemistry" not in _abstract(html)
        assert "International Union of Biochemistry" in _metadata(html)

    def test_nad_plus_and_stereodescriptor_rendered(self) -> None:
        body = _body(_replay(self.STEM))
        assert "NAD⁺" in body
        assert "$(R)$" not in body and "$(S)$" not in body

    def test_caption_not_glued_to_following_paragraph(self) -> None:
        body = _body(_replay(self.STEM))
        assert "carboxylase. Herein, I propose" not in body

    def test_table_footnote_rides_with_table(self) -> None:
        body = _body(_replay(self.STEM))
        assert '</table><p class="footnote">Molecule structures' in body


class TestTropinoneDump:
    """31051047 — dotted-section heading normalization, cross-page merge, latex."""

    STEM = "31051047"

    def test_species_name_italicized(self) -> None:
        assert "<em>Przewalskia tangutica</em>" in _replay(self.STEM)

    def test_temperature_unit_stays_upright(self) -> None:
        assert "1°<em>C</em>" not in _body(_replay(self.STEM))

    def test_numbered_subsections_normalized(self) -> None:
        body = _body(_replay(self.STEM))
        assert "<h2>3. Results</h2>" in body
        assert "<h3>3.1." in body
        assert "<h2>3.4." not in body

    def test_cross_page_paragraph_merged(self) -> None:
        assert "This suggests that TRI and</p>" not in _replay(self.STEM)


class TestFrontiersDump:
    """32117944 — sidebar/glossary routing, byline markers, heading demotion."""

    STEM = "32117944"

    def test_keywords_routed_to_panel(self) -> None:
        html = _replay(self.STEM)
        assert "<strong>Keywords:</strong>" in _metadata(html)
        assert "<strong>Keywords:</strong>" not in _body(html)

    def test_abbreviations_routed_to_panel(self) -> None:
        html = _replay(self.STEM)
        assert "<strong>Abbreviations:</strong>" in _metadata(html)
        assert "<strong>Abbreviations:</strong>" not in _body(html)

    def test_byline_markers_are_superscripts(self) -> None:
        byline = _byline(_replay(self.STEM))
        assert "<sup>" in byline
        assert "<em>" not in byline

    def test_conclusion_heading_not_folded_into_table(self) -> None:
        assert "CONCLUSION</th>" not in _replay(self.STEM)


class TestPlosDump:
    """32639976 — masthead-vs-title, byline, S-labels, heading, refs, panels."""

    STEM = "32639976"

    def test_title_not_masthead(self) -> None:
        assert "<h1>PLOS ONE</h1>" not in _replay(self.STEM)

    def test_byline_has_no_stray_emphasis(self) -> None:
        header = _header(_replay(self.STEM))
        assert "**" not in header

    def test_supplementary_label_renders_s_not_section_sign(self) -> None:
        assert not re.search(r"§\s?\d", _replay(self.STEM))

    def test_repeated_section_heading_kept(self) -> None:
        body = _body(_replay(self.STEM))
        assert "<h3>Purification of SpRDH</h3>" in body
        assert "<h2>Purification of SpRDH</h2>" in body

    def test_reference_tail_not_stranded_outside_list(self) -> None:
        body = _body(_replay(self.STEM))
        assert not re.search(r"</ol>\s*<p>dehydrogenase is independent", body)

    def test_panel_b_data_under_merged_caption(self) -> None:
        assert "Zn²⁺" in _replay(self.STEM)
