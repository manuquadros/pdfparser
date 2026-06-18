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
    _heading_inner,
    _plain_p_text,
    _visible_text,
    _visible_text_folded,
)

# A leading footnote marker is a SHORT superscript ("<sup>a</sup>", "<sup>1</sup>").
# Bounding the marker length stops a body paragraph that merely opens with a
# reconstructed multi-character superscript from being mistaken for a footnote.
_SUP_MARKER_RE = re.compile(r"^<sup>[^<]{1,3}</sup>")
# A numbered page footnote emitted as raw unicode superscripts instead of a <sup>
# tag ("¹http://…/home.htm" — a footnote the model didn't wrap).  Restricted to
# superscript *digits*: an asterisk/dagger marker (``*†‡§``) is the shape a table
# note takes ("*Each value represents the mean …"), which belongs under its table,
# not in the article footnote run.  A digit run *immediately* (no space) followed by
# an uppercase letter is an isotope/mass-number — "²H NMR", "³⁵S-labeled", "¹⁸O" —
# i.e. body prose, not a marker, so it is excluded; a marker is instead followed by
# whitespace or non-uppercase content.  It shares its leading class with affiliation
# lines ("¹ Department of …"), so it is only consulted in the body — see
# ``seen_body_heading`` in ``_classify_parts`` — never on the first-page affiliations.
_UNICODE_SUP_MARKER_RE = re.compile(r"^[¹²³⁰⁴-⁹]{1,3}(?![¹²³⁰⁴-⁹])(?:\s+|(?![A-Z]))\S")
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
# The references *heading* alone (no "<p>[1]" branch), tolerant of a leading
# section number ("7. References") and the common synonyms.  Used where a body
# block that merely opens "[1] …" (a numbered list, an inline citation) must not be
# mistaken for the start of the bibliography — e.g. the paragraph-merge's
# references guard, which only ever applies to author–year (un-numbered) entries;
# numbered "[n]" entries are already kept apart as enumeration items.
_REF_HEADING_RE = re.compile(
    r"^<h\d[^>]*>\s*(?:\d+\.?\s+)?"
    r"(?:references|bibliography|literature\s+cited|works\s+cited)\b",
    re.IGNORECASE,
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
# An affiliation names an institution: a department, university, institute,
# laboratory, college, hospital, faculty, academy or school.  Stems
# ("universit", "institut", "laborato") cover the common international spellings
# (University/Université/Universität, Institute/Institut/Instituto,
# Laboratory/Laboratoire/Laboratorio).
_AFFILIATION_RE = re.compile(
    r"\b(?:department|universit\w*|institut\w*|laborato\w*|college|hospital|"
    r"faculty|academy|polytechnic|school\s+of)\b",
    re.IGNORECASE,
)
# The same institution stems (mirroring _AFFILIATION_RE) anchored at the line
# *start*, so a no-postcode international address ("Department of …, City,
# Country") is still recognised.  Anchored, and paired with a place-like tail
# (see _has_place_like_tail): each relaxation alone would also match an
# OCR-truncated prose clause that merely names an institution.
_AFFILIATION_HEAD_RE = re.compile(
    r"^(?:the\s+)?(?:department|centers?\s+for|centres?\s+for|universit\w*|"
    r"institut\w*|laborato\w*|college|hospital|faculty|academy|polytechnic|"
    r"school\s+of|division\s+of)\b",
    re.IGNORECASE,
)
# A place tail must *open* on a capital (a lowercase connector like "of" may
# follow): that is what rejects a list/clause continuation ("… and two partner
# clinics") while accepting "South Korea" / "Republic of Korea".
_PLACE_TAIL_MAX_WORDS = 4
_PLACE_TAIL_CONNECTORS = frozenset({"of"})
# An address almost always closes on a country/region — so a country tail
# recognises the affiliation regardless of the *institution* word's language
# (German "Labor", Czech "Ústav", …), which no stem lexicon covers.  This
# language-independent path backstops the stem-based ones below; either signal
# alone marks the line an affiliation.  Bare "us"/"uk" are omitted as too
# ambiguous a closing token.
_AFFILIATION_COUNTRIES = frozenset(
    {
        "afghanistan",
        "albania",
        "algeria",
        "argentina",
        "armenia",
        "australia",
        "austria",
        "österreich",
        "azerbaijan",
        "bahrain",
        "bangladesh",
        "belarus",
        "belgium",
        "belgië",
        "belgique",
        "bolivia",
        "bosnia and herzegovina",
        "brazil",
        "brasil",
        "bulgaria",
        "cambodia",
        "cameroon",
        "canada",
        "chile",
        "china",
        "pr china",
        "p.r. china",
        "people's republic of china",
        "colombia",
        "costa rica",
        "croatia",
        "hrvatska",
        "cuba",
        "cyprus",
        "czech republic",
        "czechia",
        "česká republika",
        "česko",
        "denmark",
        "danmark",
        "ecuador",
        "egypt",
        "estonia",
        "ethiopia",
        "finland",
        "suomi",
        "france",
        "georgia",
        "germany",
        "deutschland",
        "ghana",
        "greece",
        "hungary",
        "magyarország",
        "iceland",
        "india",
        "indonesia",
        "iran",
        "iraq",
        "ireland",
        "israel",
        "italy",
        "italia",
        "japan",
        "jordan",
        "kazakhstan",
        "kenya",
        "korea",
        "south korea",
        "north korea",
        "republic of korea",
        "kuwait",
        "latvia",
        "lebanon",
        "lithuania",
        "luxembourg",
        "malaysia",
        "malta",
        "mexico",
        "méxico",
        "moldova",
        "mongolia",
        "montenegro",
        "morocco",
        "myanmar",
        "nepal",
        "netherlands",
        "the netherlands",
        "nederland",
        "new zealand",
        "nigeria",
        "norway",
        "norge",
        "oman",
        "pakistan",
        "palestine",
        "panama",
        "peru",
        "philippines",
        "poland",
        "polska",
        "portugal",
        "qatar",
        "romania",
        "russia",
        "russian federation",
        "saudi arabia",
        "serbia",
        "singapore",
        "slovakia",
        "slovensko",
        "slovenia",
        "south africa",
        "spain",
        "españa",
        "sri lanka",
        "sudan",
        "sweden",
        "sverige",
        "switzerland",
        "schweiz",
        "suisse",
        "svizzera",
        "syria",
        "taiwan",
        "tanzania",
        "thailand",
        "tunisia",
        "turkey",
        "türkiye",
        "uganda",
        "ukraine",
        "united arab emirates",
        "uae",
        "united kingdom",
        "u.k.",
        "united states",
        "united states of america",
        "usa",
        "u.s.a.",
        "uruguay",
        "uzbekistan",
        "venezuela",
        "vietnam",
        "viet nam",
        "yemen",
        "zimbabwe",
    }
)
_POSTAL_DIGITS_RE = re.compile(r"\b\d[\d-]*\b")
# An affiliation OCR'd without its author's superscript marker (J. Biol. Chem.'s
# "From the Department of …, City, Region, postcode") is recognised structurally:
# it names an institution, is laid out as a comma-separated address ending in a
# postal code, and — being a noun phrase, not a sentence — carries no terminal
# punctuation.  The postal-code tail is the load-bearing signal: "no terminal
# punctuation" on its own also matches an OCR-truncated prose clause that happens
# to name a university ("… with the Department of Biology, the School of Medicine,
# and several partner hospitals across the region"), but such a clause does not
# end in a postcode.  The trade is deliberately safe — an address without a
# postcode is left visible rather than risk hiding prose.
_AFFILIATION_MIN_COMMAS = 2
_AFFILIATION_MAX_LEN = 300
# A postal code closing the address: a 4–6 digit run (US ZIP and most national
# codes).  Searched only in the trailing comma-segments so a number earlier in a
# prose clause ("enrolled 250 patients …") cannot stand in for the address tail.
_POSTAL_TAIL_RE = re.compile(r"\b\d{4,6}\b")
_AFFILIATION_TAIL_SEGMENTS = 2
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
# Footer-metadata line shapes that carry fewer than two countable tokens yet are
# unambiguous on their own: a supporting-information note, a "DOI 10.…" line, a
# "Published online …" line, or a "Volume N, … Pages N" journal citation.  The
# OCR often splits a journal's page-bottom block into such one-line pieces, so
# none reaches the two-token bar alone.  The DOI, "Published online" and journal
# alternatives are anchored at the block *start* (and supporting-info is a fixed
# phrase), so body prose that merely mentions a volume/DOI/"published online"
# mid-sentence ("See volume 3, pages 45-67, …") does not match — only a line that
# opens as the citation/DOI/publication line does.
_STRAY_METADATA_PHRASE_RE = re.compile(
    r"(?:additional\s+)?supporting information (?:may be found|is available)"
    r"|^\s*doi\b\s*:?\s*10\.\d"
    r"|^\s*published\s+online\b"
    r"|^\s*vol(?:\.|ume)?\s*\d+.*\bp(?:p\.?|ages?)\s*\d",
    re.IGNORECASE,
)
# A self-contained footer-metadata line (journal citation, correspondence, a
# "Received … DOI …" submission line, a supporting-information note) that OCR
# dropped into the body away from the leading front-matter run.  It is relocated
# on its own evidence, so the bar is high: a short block matching one of the fixed
# publication-line shapes above, or carrying two or more metadata tokens.  The
# token count rejects a body sentence that merely embeds one address/date; the
# length bound rejects a long prose run that happens to contain two.
_STRAY_METADATA_MAX_LEN = 400
_STRAY_METADATA_MIN_TOKENS = 2
# A first-page metadata sidebar — the "Citation / Editor / Received / … / Competing
# interests" block PLOS and many open-access journals print beside the abstract —
# is OCR'd into the body away from the leading front-matter run, each entry a bold
# label-colon paragraph ("**Funding:** …").  A lone "**Word:**" line is too weak to
# relocate position-independently (a body paragraph may open "**Note:** …"), so
# these are recognised by a fixed vocabulary of publishing-process labels rather
# than by the bold-label shape alone.  The label is decisive evidence, so a matched
# line is relocated regardless of length (a Copyright/Funding statement runs long).
_PUBLICATION_METADATA_LABELS = frozenset(
    {
        "citation",
        "editor",
        "academic editor",
        "handling editor",
        "received",
        "accepted",
        "revised",
        "published",
        "copyright",
        "data availability",
        "data availability statement",
        "funding",
        "competing interests",
        "conflict of interest",
        "conflicts of interest",
        "provenance and peer review",
        # Frontiers prints an "OPEN ACCESS" banner heading atop its first-page
        # sidebar, with a "Specialty section:" routing line among the Edited
        # by / Reviewed by / Correspondence / Citation entries — sidebar
        # furniture, not body text.
        "open access",
        "specialty section",
    }
)
# Publication-metadata headings that are bare banners, not label:value pairs:
# "OPEN ACCESS" carries no value paragraph (unlike "Citation"/"Editor"/…).  The
# value-capture clause in _extract_named_metadata_sections grabs the paragraph
# *directly* under any other label heading; for a banner it would swallow whatever
# body prose happens to follow, so a banner relocates on its own and never claims a
# trailing paragraph.
_PUBLICATION_BANNER_LABELS = frozenset({"open access"})
_BOLD_LABEL_CAPTURE_RE = re.compile(r"^<strong>([^<]+):</strong>")
# A glossary the journal prints as a column-bottom footnote on the first page —
# "**Abbreviations:** ACT, …" / "**Nomenclature:** …".  OCR drops it inline,
# mid-section, where it splits the surrounding paragraph in two; the pre-classify
# stray sweep relocates it (before the paragraph merge) so the prose rejoins.
# Keywords is excluded *here*: it doubles as the abstract terminator the classifier
# reads, so removing it pre-classify leaks the following prose into the abstract;
# a stranded keyword line is instead relocated post-classify in
# ``_extract_front_matter`` (see ``_is_inline_frontmatter_label``).
_GLOSSARY_METADATA_LABELS = frozenset({"abbreviations", "nomenclature"})
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
# pages ("… 601" / "… 602" share a key).  When the line carries digits, require
# that text to be substantial so stripping them can't collapse short enumerated
# labels ("Fig 1" / "Fig 2", "Step 1" / "Step 2") into one key and delete them as
# furniture.  A digit-free line can't suffer that collision — its key is the
# verbatim text — so a recurring short one (an author-surname running foot,
# "Clark") only needs to clear a small floor that rules out a stray initial.
_MIN_FURNITURE_KEY_LEN = 12
_MIN_DIGIT_FREE_FURNITURE_KEY_LEN = 3
# A recurring line that reads like a finished sentence (ends in terminal
# punctuation) is normally real prose, not furniture — but a running head ending
# in an abbreviation ("… lichen-associated Sphingomonas sp.") only *looks* like
# one.  Genuine prose virtually never repeats verbatim across this many pages, so
# a sentence-terminated line recurring at least this often is a running head.
_SENTENCE_LIKE_FURNITURE_MIN_REPEAT = 3
# A folio is a block whose entire visible text is a bare number.  Bounded length
# keeps it to plausible page numbers and away from longer numeric data.
_PAGE_NUMBER_RE = re.compile(r"\d{1,4}")

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
# A lone-author byline ("Daniel D. Clark") has no comma to split on and no
# affiliation marker, so it is recognised as a "given-name … surname" frame: an
# alphabetic capitalised word at each end (``_NAME_ALPHA_WORD_RE``) with a *mid-name*
# initial between them (``_PERSONAL_NAME_INITIAL_RE``).  The mid-name initial is the
# positive author signal: a bare two-word name ("Jane Doe") is left ambiguous (it
# could be a subtitle) and stays in the body, and anchoring alphabetic words at both
# ends rejects an initialism-led phrase ("U.S. Army Corps").  Residual ambiguity
# remains for a Title-Case phrase that happens to share the exact frame ("Vitamin D.
# Levels") — only a lexicon could tell those apart — but such a block standing alone
# right after the title is rare.
_NAME_ALPHA_WORD_RE = re.compile(r"^[A-Z][a-z’'-]+$")
_PERSONAL_NAME_INITIAL_RE = re.compile(r"^(?:[A-Z]\.){1,3}$")
_PERSONAL_NAME_WORD_RE = re.compile(r"^(?:[A-Z]\.?|[A-Z][a-z’'-]+|(?:[A-Z]\.){2,})$")
_PERSONAL_NAME_MAX_WORDS = 5


# Text-normalization for the furniture key (the only consumer): drop punctuation
# and collapse whitespace so two header/footer lines compare equal.
_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def _furniture_key(inner: str) -> str:
    text = _DIGITS_RE.sub("", _visible_text(inner))
    return _WHITESPACE_RE.sub(" ", _PUNCT_RE.sub("", text)).strip().lower()


def _furniture_inner(part: str) -> str | None:
    """Inner text of a running-furniture candidate — a plain paragraph or a
    heading.  OCR transcribes the same marginal line as a <p> on dense body pages
    but promotes it to a heading on sparse pages (last page, after references), so
    both forms must feed the recurrence count to be stripped consistently."""
    inner = _plain_p_text(part)
    if inner is not None:
        return inner
    heading = _heading_inner(part)
    return heading[1] if heading is not None else None


def _is_furniture_candidate(part: str) -> str | None:
    inner = _furniture_inner(part)
    if inner is None:
        return None
    plain = _visible_text(inner)
    if len(plain) > _FURNITURE_MAX_LEN:
        return None
    key = _furniture_key(inner)
    min_len = (
        _MIN_FURNITURE_KEY_LEN
        if _DIGITS_RE.search(plain)
        else _MIN_DIGIT_FREE_FURNITURE_KEY_LEN
    )
    return key if len(key) >= min_len else None


def _ends_like_sentence(part: str) -> bool:
    inner = _furniture_inner(part)
    if inner is None:
        return False
    return bool(_SENTENCE_END_RE.search(_visible_text(inner).rstrip()))


def _is_standalone_page_number(part: str) -> bool:
    """A folio printed alone in the margin that OCR emitted as its own block.

    The recurrence pass can't catch it: ``_furniture_key`` strips digits before
    keying, so a number-only block has an empty key, and each page's number is
    distinct anyway.  A block whose only content is a bare number is the folio
    itself, so it is dropped directly."""
    inner = _furniture_inner(part)
    return inner is not None and bool(
        _PAGE_NUMBER_RE.fullmatch(_visible_text(inner).strip())
    )


def _strip_running_furniture(parts: list[str]) -> list[str]:
    """Drop short, recurring header/footer lines (page-number-insensitive) and
    standalone page-number blocks.

    A heading form only counts as furniture when the same line also appears as a
    plain paragraph somewhere: a running header/footer is body text the OCR
    promotes to a heading on sparse pages, so it shows up in both forms.  A line
    that recurs *only* as a heading is a genuine section heading that the article
    legitimately repeats (e.g. "Purification of X" under both Methods and
    Results), not furniture, and must be kept.

    A sentence-terminated line is treated as real prose unless it recurs often
    enough (``_SENTENCE_LIKE_FURNITURE_MIN_REPEAT``) to be a running head whose
    trailing abbreviation only mimics a sentence end ("… Sphingomonas sp.")."""
    counts: Counter[str] = Counter()
    as_paragraph: set[str] = set()
    not_sentence_like: set[str] = set()
    for part in parts:
        key = _is_furniture_candidate(part)
        if key is None:
            continue
        counts[key] += 1
        if _plain_p_text(part) is not None:
            as_paragraph.add(key)
        if not _ends_like_sentence(part):
            not_sentence_like.add(key)
    repeated = {
        key
        for key, n in counts.items()
        if n > 1
        and key in as_paragraph
        and (key in not_sentence_like or n >= _SENTENCE_LIKE_FURNITURE_MIN_REPEAT)
    }
    return [
        p
        for p in parts
        if _is_furniture_candidate(p) not in repeated
        and not _is_standalone_page_number(p)
    ]


def _is_degenerate_repetition(text: str) -> bool:
    tokens = _TOKEN_RE.findall(_visible_text(text))
    if len(tokens) < _MIN_REPEAT_TOKENS:
        return False
    top = Counter(tokens).most_common(1)[0][1]
    return top / len(tokens) >= _MAX_REPEAT_SHARE


def _is_title_heading(inner: str) -> bool:
    plain = _visible_text_folded(inner)
    return plain not in _DOCUMENT_TYPE_LABELS and not _ARTICLE_HEADING_RE.match(plain)


_MASTHEAD_MAX_WORDS = 4


def _is_masthead_heading(inner: str) -> bool:
    """A short all-caps multi-word heading — a journal masthead / running head
    ("PLOS ONE") the OCR emitted as a leading heading.  Multi-word so a minimal
    single-token title ("T") is never mistaken for one; the caller additionally
    requires a real title heading to follow before dropping it."""
    plain = _visible_text(inner).strip()
    return (
        2 <= len(plain.split()) <= _MASTHEAD_MAX_WORDS
        and plain == plain.upper()
        and plain.lower() != plain
    )


def _leading_title_follows(parts: list[str], idx: int) -> bool:
    """True when, skipping leading furniture headings (mastheads and document-type
    labels like "RESEARCH ARTICLE"), the next heading in the run is a real title —
    i.e. the heading at ``idx`` sits above the title, not at it.  A non-heading
    block before any title heading means there is none, so it returns False."""
    for k in range(idx + 1, len(parts)):
        heading = _heading_inner(parts[k])
        if heading is None:
            return False
        inner = heading[1]
        folded = _visible_text_folded(inner)
        if _is_masthead_heading(inner) or folded in _DOCUMENT_TYPE_LABELS:
            continue
        return _is_title_heading(inner)
    return False


def _looks_like_name_list(plain: str) -> bool:
    segments = [s.strip() for s in re.split(r",|\s+and\s+|;", plain) if s.strip()]
    return len(segments) >= 2 and all(
        _NAME_SEGMENT_RE.match(s) and len(s.split()) <= 5 for s in segments
    )


def _looks_like_personal_name(plain: str) -> bool:
    """A lone personal name ("Daniel D. Clark") — capitalised words framed by an
    alphabetic given name and surname with a mid-name initial between them."""
    words = plain.split()
    return (
        2 <= len(words) <= _PERSONAL_NAME_MAX_WORDS
        and all(_PERSONAL_NAME_WORD_RE.match(w) for w in words)
        and bool(_NAME_ALPHA_WORD_RE.match(words[0]))
        and bool(_NAME_ALPHA_WORD_RE.match(words[-1]))
        and any(_PERSONAL_NAME_INITIAL_RE.match(w) for w in words[1:-1])
    )


def _byline_text(inner: str) -> str:
    return _visible_text(re.sub(r"<br\s*/?>", "; ", inner)).strip()


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
    return (
        bool(_AUTHOR_MARKER_RE.search(inner))
        or _looks_like_name_list(plain)
        or _looks_like_personal_name(plain)
    )


def _is_article_page_md(md: str) -> bool:
    """A page is the article start if it carries an Abstract/Introduction
    heading (a cover ad / masthead has neither)."""
    for line in md.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)", line.strip())
        if m and _ARTICLE_HEADING_RE.match(_visible_text(m.group(1)).strip()):
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
    # The article title is the only legitimate <h1>; a body section heading the
    # model mis-levelled as <h1> ("# Molecular Mass Determination of Lxmdh"
    # mid-Methods) is demoted to <h2> once the title is claimed, so the document
    # carries a single top-level heading.  ``seen_body_heading`` also gates the
    # unicode-superscript footnote sweep below to the body proper.
    seen_body_heading = False

    for idx, part in enumerate(parts):
        heading = _heading_inner(part)
        if heading is not None:
            level, inner = heading
            if not title_html:
                # A masthead / running head ("PLOS ONE", which may sit above a
                # "RESEARCH ARTICLE" doc-type label) is dropped only when the real
                # title heading follows it in the leading heading run; a section
                # heading like "Abstract" after it means this heading IS the title.
                if _is_masthead_heading(inner) and _leading_title_follows(parts, idx):
                    continue
                if _is_title_heading(inner):
                    title_html = inner
                    expect_byline = True
                    continue
            expect_byline = False
            if _ABSTRACT_HEADING_RE.match(_visible_text(inner)):
                in_abstract = True
                continue
            if _visible_text_folded(inner) in _DOCUMENT_TYPE_LABELS:
                continue
            in_abstract = False
            if not (
                _is_named_metadata_heading(part)
                or _is_publication_metadata_heading(part)
            ):
                seen_body_heading = True
            body.append(f"<h2>{inner}</h2>" if level == 1 and title_html else part)
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
        if inner_p is not None and (
            _SUP_MARKER_RE.match(inner_p)
            or (
                seen_body_heading
                and _UNICODE_SUP_MARKER_RE.match(inner_p)
                # A "¹ Department of …, Country" affiliation shares the leading-marker
                # shape; it is front matter for the panel, never an article footnote,
                # even when OCR ordering drops it after the first body heading.
                and not _is_affiliation_line(_visible_text(inner_p).strip())
            )
        ):
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
    title = _SECTION_NUMBER_RE.sub("", _visible_text_folded(heading[1]))
    return title in _FRONTMATTER_HEADING_LABELS or title in _DOCUMENT_TYPE_LABELS


def _is_named_metadata_heading(part: str) -> bool:
    """A heading naming a front-matter section ("Abbreviations", "Nomenclature",
    "Keywords") — the subset of metadata headings that can be located by name,
    excluding the document-type labels."""
    heading = _heading_inner(part)
    if heading is None:
        return False
    title = _SECTION_NUMBER_RE.sub("", _visible_text_folded(heading[1]))
    return title in _FRONTMATTER_HEADING_LABELS


def _is_publication_metadata_heading(part: str) -> bool:
    """A heading whose text is a journal-sidebar label ("Citation", "Editor",
    "Received" …).  LightOnOCR renders the PLOS-style first-page metadata sidebar
    as a run of such label headings, each followed by its value paragraph(s),
    rather than the bold ``**Label:**`` lines other journals' OCR produces."""
    heading = _heading_inner(part)
    if heading is None:
        return False
    title = _SECTION_NUMBER_RE.sub("", _visible_text_folded(heading[1]))
    return title in _PUBLICATION_METADATA_LABELS


def _is_publication_value_heading(part: str) -> bool:
    """A publication-metadata heading that takes a value paragraph ("Citation",
    "Editor", …), as opposed to a bare banner ("OPEN ACCESS").  Only such a heading
    owns the paragraph directly beneath it; a banner does not (see
    ``_PUBLICATION_BANNER_LABELS``)."""
    heading = _heading_inner(part)
    if heading is None:
        return False
    title = _SECTION_NUMBER_RE.sub("", _visible_text_folded(heading[1]))
    return (
        title in _PUBLICATION_METADATA_LABELS
        and title not in _PUBLICATION_BANNER_LABELS
    )


def _has_place_like_tail(plain: str) -> bool:
    """True when the last comma-segment looks like a place name — a short run of
    capitalised words, opening on a capital (see ``_PLACE_TAIL_MAX_WORDS``)."""
    words = plain.rsplit(",", 1)[-1].split()
    if not 1 <= len(words) <= _PLACE_TAIL_MAX_WORDS or not words[0][:1].isupper():
        return False
    return all(w[:1].isupper() or w.lower() in _PLACE_TAIL_CONNECTORS for w in words)


def _ends_with_country(plain: str) -> bool:
    """True when the final comma-segment is a recognised country/region — the
    language-independent affiliation cue (see ``_AFFILIATION_COUNTRIES``).  Any
    postal digits sharing the segment ("South Korea 08826") are stripped first."""
    tail = _POSTAL_DIGITS_RE.sub("", plain.rsplit(",", 1)[-1]).strip()
    return tail.lower() in _AFFILIATION_COUNTRIES


def _is_affiliation_line(plain: str) -> bool:
    """A bare affiliation address that lost its author's superscript marker.

    Identified structurally (a comma-separated address layout with no terminal
    punctuation) so it is recognised without a leading ``¹``/``*``, which
    ``_LEADING_SUP_RE`` already handles.  Any one of the closing signals marks it:
    a country/region tail (language-independent), or — needing an institution
    name, so limited to the stem spellings — a place-name tail behind a
    keyword-headed line, or a postal-code tail."""
    if (
        len(plain) > _AFFILIATION_MAX_LEN
        or plain.count(",") < _AFFILIATION_MIN_COMMAS
        or _SENTENCE_END_RE.search(plain)
    ):
        return False
    if _ends_with_country(plain):
        return True
    if not _AFFILIATION_RE.search(plain):
        return False
    if _AFFILIATION_HEAD_RE.match(plain) and _has_place_like_tail(plain):
        return True
    tail = ",".join(
        plain.rsplit(",", _AFFILIATION_TAIL_SEGMENTS)[-_AFFILIATION_TAIL_SEGMENTS:]
    )
    return bool(_POSTAL_TAIL_RE.search(tail))


def _is_frontmatter_text(part: str, *, strict: bool = True) -> bool:
    inner = _plain_p_text(part)
    if inner is None:
        return False
    if _BOLD_LABEL_RE.match(inner):
        return True
    plain = _visible_text(inner).lstrip()
    if _LEADING_SUP_RE.match(plain):
        return True
    if _is_affiliation_line(plain):
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
    text = _visible_text(inner)
    return len(text) > _METADATA_PROSE_MIN_LEN and bool(_SENTENCE_END_RE.search(text))


def _is_list_like(part: str) -> bool:
    """A plain ``<p>`` of several ``;``-separated entries — an abbreviation list
    or a multi-affiliation line — as opposed to a body sentence."""
    inner = _plain_p_text(part)
    if inner is None:
        return False
    return _visible_text(inner).count(";") >= _LIST_LIKE_MIN_SEMICOLONS


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
    # Beyond the leading run, an inline front-matter label that survived into the
    # body — a keyword line stranded after a *headingless* abstract that stayed in
    # the body, so the run never started — is unambiguous front matter, so relocate
    # it too.  Pulled here, after classify, not in the pre-classify stray sweep,
    # because the keyword label doubles as the abstract terminator there.  Scoped to
    # the leading run *before the first section heading*: past that we are in the
    # body proper, where a labelled line is a back-matter glossary ("Abbreviations"
    # at the article end) that belongs with its own heading, not yanked to the panel.
    trailing: list[str] = []
    kept: list[str] = []
    in_lead = True
    for part in body[n:]:
        if in_lead and _heading_inner(part) is not None:
            in_lead = False
        inner = _plain_p_text(part)
        if in_lead and inner and _is_inline_frontmatter_label(inner):
            trailing.append(part)
        else:
            kept.append(part)
    return list(body[:n]) + trailing, kept


def _extract_named_metadata_sections(parts: list[str]) -> tuple[list[str], list[str]]:
    """Pull glossary-style metadata sections out of a single page's block stream,
    wherever they sit — returns ``(metadata, rest)``.

    OCR often places an "Abbreviations"/"Nomenclature" section *after* the
    article's opening prose rather than in the leading front matter, so
    ``_extract_front_matter`` (which only scans the leading run) never reaches it.
    The caller scopes this to the article's first page, so a same-named section
    deeper in the document (e.g. a back-matter "Nomenclature") stays in place.

    A matched heading owns only the contiguous run of positively-recognised
    metadata blocks that follow it — a ";"-separated list or a labelled metadata
    line — stopping at the first heading, figure, list, table or body paragraph,
    so real content the OCR misfiled under the heading is not hidden.  A heading
    with no such content is left in the body, as it is probably a real section
    title rather than a glossary."""
    metadata: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if _is_named_metadata_heading(part):
            j = i + 1
            while j < len(parts) and (
                _is_frontmatter_text(parts[j], strict=False) or _is_list_like(parts[j])
            ):
                j += 1
            if j > i + 1:
                metadata.extend(parts[i:j])
                i = j
                continue
        if _is_publication_metadata_heading(part):
            # A sidebar label heading owns the <p> value(s) directly under it, and
            # the sidebar is a contiguous run of such headings.  Pull label headings
            # and their values — but a <p> that neither directly follows a *value*
            # label heading nor is itself recognised front matter ends the run, so
            # trailing body prose is never swept into the hidden panel.  A bare banner
            # ("OPEN ACCESS") owns no value paragraph, so it does not claim the <p>
            # under it — only an independently-recognised front-matter <p> follows it
            # into the panel.
            j = i + 1
            while j < len(parts):
                nxt = parts[j]
                if _is_publication_metadata_heading(nxt):
                    j += 1
                    continue
                if (
                    _heading_inner(nxt) is None
                    and _plain_p_text(nxt) is not None
                    and (
                        _is_publication_value_heading(parts[j - 1])
                        or _is_frontmatter_text(nxt, strict=False)
                    )
                ):
                    j += 1
                    continue
                break
            metadata.extend(parts[i:j])
            i = j
            continue
        rest.append(part)
        i += 1
    return metadata, rest


def _bold_label_in(inner: str, labels: frozenset[str]) -> bool:
    """True when ``inner`` opens with a ``**Label:**`` bold label whose name is in
    ``labels`` — the shared shape of the metadata-label checks below."""
    m = _BOLD_LABEL_CAPTURE_RE.match(inner)
    return m is not None and m.group(1).strip().lower() in labels


def _is_publication_metadata_label(inner: str) -> bool:
    """A journal-metadata bold label ("**Citation:**", "**Competing interests:**" …
    — see ``_PUBLICATION_METADATA_LABELS``).  (When the OCR emits the sidebar as label
    *headings* instead, ``_extract_named_metadata_sections`` relocates it.)"""
    return _bold_label_in(inner, _PUBLICATION_METADATA_LABELS)


def _is_glossary_metadata_label(inner: str) -> bool:
    """An inline glossary bold label ("**Abbreviations:**", "**Nomenclature:**" —
    see ``_GLOSSARY_METADATA_LABELS``).  Decisive regardless of the entry list's
    length, so it is matched before the stray-metadata length cap."""
    return _bold_label_in(inner, _GLOSSARY_METADATA_LABELS)


def _is_inline_frontmatter_label(inner: str) -> bool:
    """An inline bold label naming any front-matter section ("**Abbreviations:**",
    "**Keywords:**", "**Nomenclature:**") — the same labels recognised as headings
    (``_FRONTMATTER_HEADING_LABELS``).

    Broader than ``_is_glossary_metadata_label`` (it also matches keywords); used by
    ``_extract_front_matter`` *after* classify to relocate a labelled front-matter
    line that survived past the leading run, so it never displaces the keyword line's
    pre-classify role as the abstract terminator."""
    return _bold_label_in(inner, _FRONTMATTER_HEADING_LABELS)


def _is_stray_metadata(part: str) -> bool:
    """A self-contained footer-metadata line OCR'd into the body (see
    ``_STRAY_METADATA_MAX_LEN``).  Unlike ``_is_frontmatter_text`` it is judged
    position-independently, so it must stand on its own strong evidence: a
    recognised publishing-process label, a short block with two or more metadata
    tokens, or the fixed boilerplate phrase."""
    inner = _plain_p_text(part)
    if inner is None:
        return False
    if _is_publication_metadata_label(inner) or _is_glossary_metadata_label(inner):
        return True
    plain = _visible_text(inner)
    # An affiliation is front matter wherever it sits — relocate it even when the
    # leading run that _extract_front_matter scans was broken (e.g. a masthead
    # heading pushed the title, byline and affiliation down into the body).
    if _is_affiliation_line(plain):
        return True
    if len(plain) > _STRAY_METADATA_MAX_LEN:
        return False
    if _STRAY_METADATA_PHRASE_RE.search(plain):
        return True
    return len(_METADATA_TOKEN_RE.findall(plain)) >= _STRAY_METADATA_MIN_TOKENS


def _extract_stray_metadata(parts: list[str]) -> tuple[list[str], list[str]]:
    """Split self-contained stray metadata blocks (see ``_is_stray_metadata``) off
    a page's block stream — returns ``(metadata, rest)``.

    The caller scopes this to the first article page and runs it *before* the
    paragraph-merge, so a footer line ending in ")" (e.g. "… Published online …
    (wileyonlinelibrary.com)") is pulled out before the merge can glue the
    following body prose onto it."""
    metadata: list[str] = []
    rest: list[str] = []
    for part in parts:
        (metadata if _is_stray_metadata(part) else rest).append(part)
    return metadata, rest
