"""Document-structure classification of the block-HTML stream.

Pure, no GPU.  Pulls title/byline/abstract/footnotes out of the flat block list,
detects the article's first page, strips running headers/footers and degenerate
OCR repetition, and separates leading front matter (affiliations, keywords,
correspondence, dates) from the body.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from pdfparser.pipeline.text import (
    _BOLD_LABEL_RE,
    _SENTENCE_END_RE,
    _STRIP_TAGS_RE,
    _heading_inner,
    _plain_p_text,
)

# A leading footnote marker is a SHORT superscript ("<sup>a</sup>", "<sup>1</sup>").
# Bounding the marker length stops a body paragraph that merely opens with a
# reconstructed multi-character superscript from being mistaken for a footnote.
_SUP_MARKER_RE = re.compile(r"^<sup>[^<]{1,3}</sup>")
_DOCUMENT_TYPE_LABELS = frozenset(
    {
        "abstract",
        "article",
        "research article",
        "original article",
        "letter",
        "review",
        "communication",
        "report",
        "brief communication",
        "short communication",
    }
)
# re.IGNORECASE: OCR output casing is unreliable ("REFERENCES", "References").
# <h\d[^>]*> tolerates class/id attributes the model may inject.
_REF_SECTION_RE = re.compile(
    r"^(?:<h\d[^>]*>\s*References\s*</h\d>|<p>\[1\])", re.IGNORECASE
)

# The article starts at the first page carrying an "Abstract"/"Introduction"
# heading; a cover ad / masthead has neither.
_ARTICLE_HEADING_RE = re.compile(
    r"^\s*(?:\d+[.)]?\s+)?(?:abstract|introduction)\b", re.IGNORECASE
)
# Headings that may legitimately sit *inside* the front matter (between the
# abstract and the article body).  Everything up to the first heading that is
# neither one of these nor a document-type label is treated as front matter,
# so the boundary doesn't depend on the body's opening section being literally
# named "Introduction" (it may be "Background", numbered, non-English, etc.).
_FRONTMATTER_HEADING_LABELS = frozenset(
    {"abbreviations", "keywords", "key words", "nomenclature"}
)
_SECTION_NUMBER_RE = re.compile(r"^\d+(?:[.)]\d*)*[.)]?\s+")
# A plain <p> is positively front matter when it carries a metadata label
# ("Keywords:", reuses _BOLD_LABEL_RE), opens with an affiliation/footnote
# superscript marker, or is a submission/correspondence/copyright line.
_LEADING_SUP_RE = re.compile(r"^[¹²³⁰-ⁿ*†‡§]")
_FRONTMATTER_TEXT_RE = re.compile(
    r"^(?:received|accepted|published|revised|doi|https?://|©|copyright|e-?mail|"
    r"(?:address\s+for\s+)?correspond(?:ence|ing\s+author))\b",
    re.IGNORECASE,
)
# A token only metadata carries: an e-mail, a DOI/URL, a submission date, or a
# phone/fax label.  Its presence separates a genuine multi-clause metadata line
# ("Received … DOI: …", "Address for correspondence: … e-mail: …") from a body
# sentence that merely opens with a front-matter keyword ("Published reports
# indicate …"), which has none of these.
_METADATA_TOKEN_RE = re.compile(
    r"""
      \S+@\S+\.\S                                # e-mail address
    | doi:\s*10\.\d{4,}                          # DOI
    | https?://                                  # URL
    | \b\d{1,2}\s+
      (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+
      \d{4}\b                                    # a "26 March 2019" date
    | \b(?:tel|fax|phone)\b                      # phone / fax label
    """,
    re.IGNORECASE | re.VERBOSE,
)
# Front matter is hidden in a collapsed panel, so misclassifying body prose as
# front matter makes it invisible.  A real prose paragraph under a metadata
# section is recognised by length + a sentence ending, and breaks the run.
_METADATA_PROSE_MIN_LEN = 80
# A metadata list (abbreviations, multi-affiliation / correspondence lines) packs
# several ";"-separated entries; body prose rarely uses more than one semicolon.
# This keeps such a list owned by its heading even when it is long enough to read
# like a sentence, and even when OCR splits the list across paragraphs.
_LIST_LIKE_MIN_SEMICOLONS = 2

_ABSTRACT_HEADING_RE = re.compile(r"^\s*abstract\b", re.IGNORECASE)
# Running header/footer: a short, terminal-punctuation-free line that recurs
# across pages.  Page numbers vary per page, so they are stripped before the
# recurrence is counted (e.g. "Biotechnology … 601" / "… 602" share a key).
_FURNITURE_MAX_LEN = 120
_DIGITS_RE = re.compile(r"\d+")
# A footer/header is identified by its digit-stripped text recurring across
# pages ("… 601" / "… 602" share a key).  Require that text to be substantial so
# stripping the digits can't collapse short enumerated labels ("Fig 1" / "Fig 2",
# "Step 1" / "Step 2") into one key and delete them as furniture.
_MIN_FURNITURE_KEY_LEN = 12

# Even though LightOnOCR-bbox usually boxes figures (so they never reach the text
# stream), a diagram it misses can still be OCRed into one label repeated dozens
# of times ("AaTRI, AaTRI, …") — many tokens, almost no diversity.  This drops
# such a paragraph from the body; real prose (even with some repetition) stays.
_MIN_REPEAT_TOKENS = 8
_MAX_REPEAT_SHARE = 0.6
_TOKEN_RE = re.compile(r"\w+")

# Affiliation / corresponding-author markers that accompany author names:
# superscript digits (¹²³, ⁰⁴–⁹), a <sup>, or footnote symbols.
_AUTHOR_MARKER_RE = re.compile(r"<sup>|[¹²³⁰-ⁿ*†‡§]")
# A comma / "and" / ";"-separated author segment: a short, capitalized,
# digit-free name.
_NAME_SEGMENT_RE = re.compile(r"^[A-Z][^\d]*$")


# Text-normalization for the furniture key (the only consumer): drop punctuation
# and collapse whitespace so two header/footer lines compare equal.
_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def _furniture_key(inner: str) -> str:
    text = _DIGITS_RE.sub("", _STRIP_TAGS_RE.sub("", inner))
    return _WHITESPACE_RE.sub(" ", _PUNCT_RE.sub("", text)).strip().lower()


def _is_furniture_candidate(part: str) -> str | None:
    inner = _plain_p_text(part)
    if inner is None:
        return None
    plain = _STRIP_TAGS_RE.sub("", inner)
    if len(plain) > _FURNITURE_MAX_LEN or _SENTENCE_END_RE.search(plain.rstrip()):
        return None
    key = _furniture_key(inner)
    return key if len(key) >= _MIN_FURNITURE_KEY_LEN else None


def _strip_running_furniture(parts: list[str]) -> list[str]:
    """Drop short, recurring header/footer lines (page-number-insensitive)."""
    counts: Counter[str] = Counter(
        key for part in parts if (key := _is_furniture_candidate(part)) is not None
    )
    repeated = {key for key, n in counts.items() if n > 1}
    return [p for p in parts if _is_furniture_candidate(p) not in repeated]


def _is_degenerate_repetition(text: str) -> bool:
    tokens = _TOKEN_RE.findall(_STRIP_TAGS_RE.sub("", text))
    if len(tokens) < _MIN_REPEAT_TOKENS:
        return False
    top = Counter(tokens).most_common(1)[0][1]
    return top / len(tokens) >= _MAX_REPEAT_SHARE


def _is_title_heading(inner: str) -> bool:
    plain = _STRIP_TAGS_RE.sub("", inner).strip().lower()
    return plain not in _DOCUMENT_TYPE_LABELS and not _ARTICLE_HEADING_RE.match(plain)


def _looks_like_name_list(plain: str) -> bool:
    segments = [s.strip() for s in re.split(r",|\s+and\s+|;", plain) if s.strip()]
    return len(segments) >= 2 and all(
        _NAME_SEGMENT_RE.match(s) and len(s.split()) <= 5 for s in segments
    )


def _byline_text(inner: str) -> str:
    return _STRIP_TAGS_RE.sub("", re.sub(r"<br\s*/?>", "; ", inner)).strip()


def _is_byline(inner: str) -> bool:
    """A block right after the title is the byline only when it positively looks
    like authors — it carries an affiliation/footnote marker or is a list of
    names.  Anything else (a date, DOI, journal line) falls through to the body
    rather than being silently moved into the header."""
    plain = _byline_text(inner)
    if not plain or len(plain) >= 400 or _SENTENCE_END_RE.search(plain):
        return False
    if _BOLD_LABEL_RE.match(inner):
        return False
    return bool(_AUTHOR_MARKER_RE.search(inner)) or _looks_like_name_list(plain)


def _is_article_page_md(md: str) -> bool:
    """A page is the article start if it carries an Abstract/Introduction
    heading (a cover ad / masthead has neither)."""
    for line in md.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)", line.strip())
        if m and _ARTICLE_HEADING_RE.match(_STRIP_TAGS_RE.sub("", m.group(1)).strip()):
            return True
    return False


def _leading_pages_to_skip_md(pages_md: list[str]) -> int:
    return next(
        (i for i, md in enumerate(pages_md) if _is_article_page_md(md)),
        0,
    )


@dataclass
class _Meta:
    title_html: str
    byline_html: str
    abstract: list[str]
    body: list[str]
    footnotes: list[str]


def _classify_parts(parts: list[str]) -> _Meta:
    """Single pass: pull the title, byline, abstract and footnotes out of the
    flat block list; everything else is body."""
    title_html = ""
    byline_html = ""
    abstract: list[str] = []
    body: list[str] = []
    footnotes: list[str] = []
    # The byline is only the block *immediately* after the title heading; this
    # window closes at the next block so a body sentence is never mistaken for it.
    expect_byline = False
    in_abstract = False

    for part in parts:
        heading = _heading_inner(part)
        if heading is not None:
            _, inner = heading
            if not title_html and _is_title_heading(inner):
                title_html = inner
                expect_byline = True
                continue
            expect_byline = False
            if _ABSTRACT_HEADING_RE.match(_STRIP_TAGS_RE.sub("", inner)):
                in_abstract = True
                continue
            if _STRIP_TAGS_RE.sub("", inner).strip().lower() in _DOCUMENT_TYPE_LABELS:
                continue
            in_abstract = False
            body.append(part)
            continue

        inner_p = _plain_p_text(part)
        if expect_byline:
            expect_byline = False
            if inner_p is not None and _is_byline(inner_p):
                byline_html = _byline_text(inner_p)
                continue
        if in_abstract:
            if inner_p is not None and not _BOLD_LABEL_RE.match(inner_p):
                abstract.append(part)
                continue
            in_abstract = False
        if inner_p is not None and _SUP_MARKER_RE.match(inner_p):
            footnotes.append(f'<p class="footnote">{inner_p}</p>')
            continue
        if inner_p is not None and _is_degenerate_repetition(inner_p):
            continue
        body.append(part)

    return _Meta(title_html, byline_html, abstract, body, footnotes)


def _is_metadata_heading(part: str) -> bool:
    """A heading that is itself front matter ("Abbreviations", "Keywords") or a
    document-type label, as opposed to the heading that opens the body proper."""
    heading = _heading_inner(part)
    if heading is None:
        return False
    title = _SECTION_NUMBER_RE.sub(
        "", _STRIP_TAGS_RE.sub("", heading[1]).strip().lower()
    )
    return title in _FRONTMATTER_HEADING_LABELS or title in _DOCUMENT_TYPE_LABELS


def _is_frontmatter_text(part: str, *, strict: bool = True) -> bool:
    inner = _plain_p_text(part)
    if inner is None:
        return False
    if _BOLD_LABEL_RE.match(inner):
        return True
    plain = _STRIP_TAGS_RE.sub("", inner).lstrip()
    if _LEADING_SUP_RE.match(plain):
        return True
    if not _FRONTMATTER_TEXT_RE.match(plain):
        return False
    if not _SENTENCE_END_RE.search(plain):
        return True
    # The line runs on like a sentence: either a body paragraph that merely opens
    # with the keyword ("Published reports indicate ….") or a genuine multi-clause
    # metadata line ("Received … DOI: …", "Address for correspondence: … e-mail:
    # …").  At the top level (strict) refuse it; inside an explicitly-headed
    # metadata section (strict=False) trust it only when it carries a metadata
    # token, which a body sentence does not.
    return not strict and bool(_METADATA_TOKEN_RE.search(plain))


def _looks_like_body_prose(part: str) -> bool:
    """A substantial plain-paragraph sentence — used to end a metadata section's
    sticky run so unheaded body prose isn't swallowed into the hidden panel.

    Front-matter exclusion is the caller's job: the one call site already gates
    this on ``not _is_frontmatter_text(part, strict=False)``, so re-testing it
    here would just strip tags and re-run the regexes a second time."""
    inner = _plain_p_text(part)
    if inner is None:
        return False
    text = _STRIP_TAGS_RE.sub("", inner)
    return len(text) > _METADATA_PROSE_MIN_LEN and bool(_SENTENCE_END_RE.search(text))


def _is_list_like(part: str) -> bool:
    """A plain ``<p>`` of several ``;``-separated entries — an abbreviation list
    or a multi-affiliation line — as opposed to a body sentence."""
    inner = _plain_p_text(part)
    if inner is None:
        return False
    return _STRIP_TAGS_RE.sub("", inner).count(";") >= _LIST_LIKE_MIN_SEMICOLONS


def _front_matter_len(body: list[str]) -> int:
    """Length of the leading run of recognised front-matter blocks.

    Only positively-identified metadata is counted — affiliation/superscript
    paragraphs, labelled lines (keywords, correspondence, dates) and metadata
    headings with the content they introduce.  The run stops at the first block
    that is none of these, so a body opening with unlabelled prose (or one whose
    first section heading lacks a name we recognise) is never mistaken for front
    matter and relocated."""
    n = 0
    in_metadata_section = False
    for i, part in enumerate(body):
        if _heading_inner(part) is not None:
            if not _is_metadata_heading(part):
                break
            in_metadata_section = True
        elif in_metadata_section:
            # A metadata heading owns its content up to the next heading: a
            # keyword-led metadata line (correspondence, dates) or a ";"-separated
            # list (an abbreviation list, even one long enough to read like a
            # sentence or split across paragraphs).  Only a genuine unheaded body
            # paragraph — which is neither — ends the run.
            if (
                _looks_like_body_prose(part)
                and not _is_frontmatter_text(part, strict=False)
                and not _is_list_like(part)
            ):
                break
        elif not _is_frontmatter_text(part):
            break
        n = i + 1
    return n


def _extract_front_matter(body: list[str]) -> tuple[list[str], list[str]]:
    """Split the leading run of front-matter blocks off the body.

    Affiliations, keywords, abbreviations, the corresponding-author block and
    submission dates are OCR'd between the abstract and the body's first section,
    so the body opens with metadata rather than prose.  The leading run of
    recognised front-matter blocks is pulled out (the caller surfaces it in a
    collapsible "Metadata" panel after the abstract) so the body opens with
    prose.  Returns ``(front_matter, rest)``."""
    n = _front_matter_len(body)
    # Never hide the entire body: an all-front-matter classification signals a
    # detection failure, not a metadata-only document.
    if n >= len(body):
        return [], body
    return body[:n], body[n:]
