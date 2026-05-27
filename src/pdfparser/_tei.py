"""TEI XML → JATS XML conversion.

Converts GROBID's TEI output into the subset of JATS that
``xmlparser.transform_article()`` expects.  Table figures are replaced by
``[[TABLE_N]]`` placeholders (zero-indexed) so the caller can inject the
corresponding gmft-extracted HTML after the JATS→HTML transform.
"""

from __future__ import annotations

from lxml import etree

TEI_NS = "http://www.tei-c.org/ns/1.0"

_RENAME: dict[str, str] = {
    "div": "sec",
    "head": "title",
    "listItem": "list-item",
}

_UNWRAP_TAGS = {"ref", "note", "formula"}

# Shared with document.py so both sides of the placeholder contract stay in sync.
_TABLE_PLACEHOLDER = "[[TABLE_{}]]"
_TABLE_PLACEHOLDER_RE = r"\[\[TABLE_(\d+)\]\]"


def _tei(local: str) -> str:
    return f"{{{TEI_NS}}}{local}"


def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _convert_element(
    src: etree._Element,
    table_counter: list[int] | None,
) -> etree._Element | None:
    """Recursively convert one TEI element to its JATS equivalent.

    ``table_counter`` is a single-element list shared across recursive calls —
    a mutable container so the integer can be incremented without ``nonlocal``.
    Pass ``None`` in contexts where table figures should be dropped rather than
    converted to placeholders (e.g. inside the abstract).
    """
    local = _strip_ns(src.tag)

    if local == "figure":
        if src.get("type") == "table" and table_counter is not None:
            n = table_counter[0]
            table_counter[0] += 1
            p = etree.Element("p")
            p.text = _TABLE_PLACEHOLDER.format(n)
            return p
        return None

    jats_tag = _RENAME.get(local, local)
    dst = etree.Element(jats_tag)
    dst.text = src.text
    dst.tail = src.tail

    if local in _UNWRAP_TAGS:
        dst.tag = "span"
        dst.text = src.text or ""
        for child in src:
            child_dst = _convert_element(child, table_counter)
            if child_dst is not None:
                dst.append(child_dst)
        return dst

    for child in src:
        child_dst = _convert_element(child, table_counter)
        if child_dst is not None:
            dst.append(child_dst)

    return dst


def _extract_abstract(root: etree._Element) -> str | None:
    abstract_el = root.find(f".//{_tei('abstract')}")
    if abstract_el is None:
        return None

    jats = etree.Element("abstract")
    for div in abstract_el.findall(_tei("div")):
        if div.text and div.text.strip():
            p = etree.SubElement(jats, "p")
            p.text = div.text
        for child in div:
            converted = _convert_element(child, None)
            if converted is not None:
                jats.append(converted)

    return etree.tostring(jats, encoding="unicode")


def _extract_body(root: etree._Element) -> str | None:
    body_el = root.find(f".//{_tei('body')}")
    if body_el is None:
        return None

    counter: list[int] = [0]
    jats = etree.Element("body")
    for div in body_el.findall(_tei("div")):
        converted = _convert_element(div, counter)
        if converted is not None:
            jats.append(converted)

    return etree.tostring(jats, encoding="unicode")


def _extract_meta(root: etree._Element) -> dict[str, str]:
    title_el = root.find(f".//{_tei('titleStmt')}/{_tei('title')}[@level='a']")
    title = "".join(title_el.itertext()) if title_el is not None else ""

    analytic = root.find(f".//{_tei('sourceDesc')}//{_tei('analytic')}")
    search_root = analytic if analytic is not None else root
    authors: list[str] = []
    for person in search_root.findall(f".//{_tei('persName')}"):
        forenames = " ".join(el.text or "" for el in person.findall(_tei("forename")))
        surname_el = person.find(_tei("surname"))
        surname = surname_el.text or "" if surname_el is not None else ""
        name = f"{forenames} {surname}".strip()
        if name:
            authors.append(name)

    date_el = root.find(
        f".//{_tei('publicationStmt')}/{_tei('date')}[@type='published']"
    )
    year = ""
    if date_el is not None:
        when = date_el.get("when", "")
        year = when[:4]

    return {"title": title, "authors": "; ".join(authors), "year": year}


def tei_to_parts(
    tei_xml: str,
) -> tuple[str | None, str | None, dict[str, str]]:
    """Parse GROBID TEI XML into JATS fragments and document metadata.

    Returns:
        A ``(abstract_xml, body_xml, meta)`` triple where ``abstract_xml``
        and ``body_xml`` are JATS XML strings (or ``None`` if absent) and
        ``meta`` contains ``title``, ``authors``, and ``year``.
        Body XML contains ``[[TABLE_N]]`` placeholders where GROBID detected
        table figures.
    """
    root = etree.fromstring(tei_xml.encode())
    return _extract_abstract(root), _extract_body(root), _extract_meta(root)
