"""LaTeX and inline-markdown → HTML conversion.

Leaf module, no GPU or IO.  Covers the deterministic text transforms applied to
LightOnOCR's markdown: bold/italic, and inline ``$…$`` spans reduced to Unicode
super/subscripts and HTML ``<sup>``/``<sub>`` (full math is out of scope — a
later MathJax option).  Symbol-command translation is delegated to pylatexenc's
maintained macro table rather than a hand-curated map.
"""

from __future__ import annotations

import functools
import re

from pylatexenc.latex2text import LatexNodes2Text  # type: ignore[import-untyped]

# Unicode superscript forms.  A LaTeX ``$^{…}$`` run is rendered with these
# glyphs when every character has one (so "NAD$^+$" → "NAD⁺"); otherwise it falls
# back to an HTML <sup> wrapper so nothing is lost.
_SUPERSCRIPT_MAP = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "−": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "°": "°",
    "a": "ᵃ",
    "b": "ᵇ",
    "c": "ᶜ",
    "d": "ᵈ",
    "e": "ᵉ",
    "f": "ᶠ",
    "g": "ᵍ",
    "h": "ʰ",
    "i": "ⁱ",
    "j": "ʲ",
    "k": "ᵏ",
    "l": "ˡ",
    "m": "ᵐ",
    "n": "ⁿ",
    "o": "ᵒ",
    "p": "ᵖ",
    "r": "ʳ",
    "s": "ˢ",
    "t": "ᵗ",
    "u": "ᵘ",
    "v": "ᵛ",
    "w": "ʷ",
    "x": "ˣ",
    "y": "ʸ",
    "z": "ᶻ",
}

# Symbol-command translation is delegated to pylatexenc's maintained macro
# table rather than a hand-curated map.  A command is matched as a maximal
# ``\name`` token and looked up whole, so a command can never eat the head of a
# longer one ("\to" vs "\top", "\sim" vs "\simeq").
_LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+")
_L2T = LatexNodes2Text()
# pylatexenc returns "" for a few common symbol macros (version-dependent), which
# would leak the raw "\ddagger"/"\S" into author/affiliation footnote markers.
# Supplied as a whole-token fallback so "\S" can never corrupt "\Section".  "\|"
# (non-alphabetic, so never matched by _LATEX_COMMAND_RE) is handled separately.
_LATEX_SYMBOL_FALLBACK = {
    r"\ddagger": "‡",
    r"\S": "§",
    r"\P": "¶",
}
_LATEX_VERT_RE = re.compile(r"\\\|")

# The degree idiom is the one place we override pylatexenc: "^\circ" means
# *degrees* ("°"), but \circ on its own is the ring operator ("∘"), which is
# what pylatexenc (correctly, for general LaTeX) returns.  \degree is unknown to
# pylatexenc, so it is handled here too.  This runs before script handling so the
# ``^`` is consumed and the result isn't wrapped in <sup>.
_LATEX_DEGREE_RE = re.compile(r"\^\s*\{?\s*\\circ\s*\}?|\\degree(?![a-zA-Z])")


@functools.cache
def _latex_command_to_unicode(command: str) -> str:
    """Translate a single no-arg ``\\name`` symbol command to its Unicode glyph.

    Only symbol macros are in scope.  Each command is matched and looked up in
    isolation, but pylatexenc parses full LaTeX *with* arguments — fed a bare
    arg-taking macro it either raises (``\\sqrt`` → KeyError) or returns its
    substitution template (``\\frac`` → ``"%s/%s"``).  Treat any such case, and
    an unknown macro (empty result), as untranslatable and keep the command
    literal so real math survives intact for a later MathJax pass rather than
    crashing the page or leaking ``%s`` garbage."""
    try:
        text = str(_L2T.latex_to_text(command)).strip()
    except Exception:
        return command
    if not text or "%" in text:
        return _LATEX_SYMBOL_FALLBACK.get(command, command)
    return text


# A literal '*'/'_' surviving a converted math span (an author footnote marker
# like "1,*", or a multiplication "a*b") would be re-read as Markdown emphasis by
# the downstream inline parser, which pairs two such characters and italicises the
# text between them.  The span's emitted HTML (<sup>/<sub> tags, Unicode command
# glyphs) carries neither character, so escaping the whole converted span to HTML
# entities neutralises every residual marker without touching the structure.
_MD_EMPHASIS_ESCAPE = str.maketrans({"*": "&#42;", "_": "&#95;"})


def _to_superscript(core: str) -> str:
    if all(ch in _SUPERSCRIPT_MAP or ch.isspace() for ch in core):
        return "".join(_SUPERSCRIPT_MAP.get(ch, ch) for ch in core)
    return f"<sup>{core}</sup>"


_BOLDITALIC_RE = re.compile(r"\*\*\*(.+?)\*\*\*", re.DOTALL)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
# re.DOTALL intentionally omitted: italic spans in academic text don't cross
# line boundaries, and DOTALL would cause two stray footnote asterisks anywhere
# in a multi-line region to wrap the entire intervening content in <em>.
_ITALIC_RE = re.compile(r"\*(.+?)\*")


def _inline_md_to_html(text: str) -> str:
    text = _BOLDITALIC_RE.sub(r"<strong><em>\1</em></strong>", text)
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = _ITALIC_RE.sub(r"<em>\1</em>", text)
    return text.strip()


# An inline math span: $…$ not preceded by a backslash, shortest match, on a
# single line (no DOTALL — a stray '$' must not swallow across paragraphs).  A
# single space *before* the span is captured so a span that opens with a script
# ("Sec $^{-1}$") can re-attach to the preceding token instead of stranding the
# superscript after a gap.
_LATEX_SPAN_RE = re.compile(r"( ?)(?<!\\)\$([^\n$]+)(?<!\\)\$")
# Only spans that actually contain TeX (a sub/superscript or a command) are
# converted; a paired '$' around plain text (e.g. currency "$5 … $10") is left
# untouched rather than stripped.
_LATEX_MATH_RE = re.compile(r"[_^\\]")
# A span whose content is a lone number ("$42.26$") is the model wrapping a plain
# value in math mode, so the delimiters are dropped and the number kept.  Currency
# is safe: the regex only pairs '$' spuriously across words ("$5 … $10"), whose
# content is never a lone number.
_BARE_NUMBER_RE = re.compile(r"[-+±]?\d[\d.,]*%?")
# A markup-free span can still be a plain inline equation/variable the model
# wrapped in math mode ("$x = 22$", "$a = 84.9$", "$P < 0.05$", "$a, b, c$"); its
# delimiters should drop like a bare number's.  See ``_is_inline_math_span``.
_MATH_RELATION_RE = re.compile(r"[=<>≤≥≈≠]")
_MATH_EXPR_RE = re.compile(r"[\w.,()\s=<>≤≥≈≠+\-−×⋅*/±°]+")
_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def _is_inline_math_span(content: str) -> bool:
    """True when a markup-free ``$…$`` span is plain inline math the model wrapped
    in math mode, so its delimiters should drop (like a bare number's).

    The whole span must be math-like (identifiers, numbers, operators) and open
    with an identifier, not a digit: currency is always digit-led ("$5", "$10"), so
    a spuriously paired span opens with a digit and is left intact — even one
    bracketing a relation ("$10 > $5") or a hyphen range ("$5 - $10").  Given an
    identifier lead, a relational operator (``=``/``<``/``>``) marks an equation
    ("x = 22", "pH = 7", "P < 0.05") and is decisive; failing that, only an
    operator/comma-separated list of single-letter variables ("a, b, c") qualifies,
    with no multi-letter English word — so a stray pairing over prose isn't
    swallowed."""
    stripped = content.strip()
    if not stripped or not _MATH_EXPR_RE.fullmatch(stripped):
        return False
    if not stripped[:1].isalpha():
        return False
    if _MATH_RELATION_RE.search(stripped):
        return True
    return not _ENGLISH_WORD_RE.search(stripped)


# Sub/superscript inside a math span: ^{multi} / ^cmd / ^x and the _ forms.
# A bare script target may be a command pylatexenc left literal (``^\dagger``);
# capture the whole command, not just the leading backslash, so a stray "\<"
# can't reach markdown and get mangled into a broken tag.
_LATEX_SUP_RE = re.compile(r"\^\{([^{}]*)\}|\^(\\[a-zA-Z]+|\S)")
_LATEX_SUB_RE = re.compile(r"_\{([^{}]*)\}|_(\\[a-zA-Z]+|\S)")
# Font/style wrappers (\text{…}, \mathrm{…}) carry no semantics here — unwrap to
# their content so the inner sub/superscript handling sees plain text.
_LATEX_WRAP_RE = re.compile(
    r"\\(?:text|mathrm|mathit|mathbf|mathsf|operatorname)\{([^{}]*)\}"
)


def _latex_span_to_html(content: str) -> str:
    """Convert the inside of a ``$…$`` span: sub/superscripts to HTML, then drop
    residual TeX syntax.  Full math is out of scope (a later MathJax option)."""
    content = _LATEX_WRAP_RE.sub(r"\1", content)
    content = _LATEX_VERT_RE.sub("‖", content)
    content = _LATEX_DEGREE_RE.sub("°", content)
    content = _LATEX_COMMAND_RE.sub(
        lambda m: _latex_command_to_unicode(m.group(0)), content
    )
    content = _LATEX_SUP_RE.sub(
        lambda m: _to_superscript(m.group(1) if m.group(1) is not None else m.group(2)),
        content,
    )
    content = _LATEX_SUB_RE.sub(
        lambda m: f"<sub>{m.group(1) if m.group(1) is not None else m.group(2)}</sub>",
        content,
    )
    content = content.replace("\\,", " ").replace("{", "").replace("}", "")
    return content.translate(_MD_EMPHASIS_ESCAPE)


def _latex_to_html(text: str) -> str:
    """Replace each inline ``$…$`` *math* span with deterministic HTML.

    Runs on the markdown *before* parsing so the emitted ``<sub>``/``<sup>`` pass
    through as raw HTML and the ``_`` inside ``V_{max}`` isn't read as emphasis.
    Spans without any TeX markup are left verbatim.
    """

    def replace(m: re.Match[str]) -> str:
        space, content = m.group(1), m.group(2)
        if not _LATEX_MATH_RE.search(content):
            stripped = content.strip()
            if _BARE_NUMBER_RE.fullmatch(stripped):
                return space + stripped
            if _is_inline_math_span(content):
                return space + _latex_span_to_html(content)
            return m.group(0)
        html = _latex_span_to_html(content)
        if space and content.lstrip()[:1] in ("^", "_"):
            return html
        return space + html

    return _LATEX_SPAN_RE.sub(replace, text)
