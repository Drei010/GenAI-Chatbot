"""Read a Word (.docx) upload into pages of text.

A .docx file is a ZIP package of XML parts. Only the main document part is read: its paragraphs in
order, each table row as one line with its cells separated by " | ", and text boxes after the
paragraph they're anchored to. Headers, footers, footnotes, endnotes and comments are separate parts
and aren't read; neither are deleted text in tracked changes, hidden text or field codes (a field's
displayed result is kept). Paragraphs in Word's heading styles are passed on as headings.

Word files don't store their page layout, so page numbers can only come from the breaks a file
records: page breaks the author inserted, sections that start on a new page, and the marks Word
leaves where pages broke when it last laid the document out (`w:lastRenderedPageBreak`). Those pages
are used only when they add up: no page holds more text than a page can, and their count matches the
page count Word saved with the file (or, when the file doesn't record a current one, Word's layout
marks are present). Otherwise the text is divided into parts of about a page, which stand in for pages
internally, and the document is marked as having no page numbers.
"""

from __future__ import annotations

import itertools
import posixpath
import re
import unicodedata
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from io import BytesIO
from xml.etree.ElementTree import Element

from defusedxml.ElementTree import fromstring

from genai_chatbot.documents import (
    MAX_PAGES,
    MIN_TEXT_CHARS,
    DocumentError,
    display_name,
    document_fingerprint,
    meaningful_chars,
)
from genai_chatbot.schemas import ExtractedDocument, PageText, StyledHeading

ZIP_SIGNATURE = b"PK\x03\x04"
# The compound file format holds legacy .doc documents, and .docx files saved with a password.
COMPOUND_FILE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_ENCRYPTED_PACKAGE = "EncryptedPackage".encode("utf-16-le")

# Largest XML part read, uncompressed. zipfile never inflates an entry past its declared size.
MAX_PART_BYTES = 32 * 1024 * 1024
# A printed page rarely holds more characters than this; a longer "page" means breaks are missing.
MAX_PAGE_CHARS = 6000
# Without reliable page breaks, the text is divided into parts of about this many characters.
PART_CHARS = 3000

_WORD_NAMESPACES = frozenset(
    {
        "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "http://purl.oclc.org/ooxml/wordprocessingml/main",  # Strict Open XML
    }
)
_MATH_NAMESPACES = (
    "{http://schemas.openxmlformats.org/officeDocument/2006/math}",
    "{http://purl.oclc.org/ooxml/officeDocument/math}",
)
_COMPATIBILITY = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
_RELATIONSHIP = "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"

# Elements whose content is read as if it stood in their place.
_WRAPPERS = frozenset({"sdt", "sdtContent", "customXml"})
_INLINE_WRAPPERS = _WRAPPERS | {"hyperlink", "smartTag", "ins", "moveTo", "fldSimple", "bdo", "dir"}
_NEW_PAGE_SECTIONS = frozenset({"nextPage", "evenPage", "oddPage"})
_OFF = frozenset({"0", "false", "off"})
_SPACE = re.compile(r"\s+")
_HEADING_STYLE_NAME = re.compile(r"heading ([1-9])")

# A line of text and its heading level (0 for the top level), or None for body text.
Line = tuple[str, int | None]


def _content(element: Element) -> Iterator[Element]:
    """An element's children, with each markup-compatibility block replaced by its first choice."""
    for child in element:
        if child.tag == f"{_COMPATIBILITY}AlternateContent":
            choices = [option for option in child if option.tag == f"{_COMPATIBILITY}Choice"]
            chosen = choices[0] if choices else child.find(f"{_COMPATIBILITY}Fallback")
            if chosen is not None:
                yield from _content(chosen)
        else:
            yield child


def _level_value(value: str | None) -> int | None:
    """An outline level, 0 to 8 for headings and 9 for body text."""
    try:
        level = int(value or "")
    except ValueError:
        return None
    return level if 0 <= level <= 9 else None


@dataclass
class _Page:
    lines: list[Line] = field(default_factory=list)
    has_content: bool = False  # text or a picture


class _BodyReader:
    """Walks the body in reading order, collecting lines and the page breaks between them."""

    def __init__(
        self,
        w: str,
        style_levels: dict[str, int],
        page_sections: frozenset[int] = frozenset(),
        *,
        paginate: bool = True,
    ) -> None:
        self.w = w
        self.style_levels = style_levels
        self.page_sections = page_sections
        self.paginate = paginate
        self.pages = [_Page()]
        self.rendered_breaks = 0
        self._line: list[str] = []
        self._level: int | None = None
        self._after_inserted_break = False
        self._text_boxes: list[Element] = []

    def _name(self, element: Element) -> str:
        """The element's WordprocessingML name, or "" for elements of other vocabularies."""
        tag = element.tag
        return tag[len(self.w) :] if tag.startswith(self.w) else ""

    def _is_on(self, properties: Element | None, name: str) -> bool:
        flag = properties.find(self.w + name) if properties is not None else None
        return flag is not None and flag.get(self.w + "val", "true").lower() not in _OFF

    # Lines and pages

    def _saw_content(self) -> None:
        self.pages[-1].has_content = True
        self._after_inserted_break = False

    def _add_text(self, text: str) -> None:
        self._line.append(text)
        if text and not text.isspace():
            self._saw_content()

    def _end_line(self) -> None:
        self.pages[-1].lines.append(("".join(self._line), self._level))
        self._line = []

    def _line_break(self) -> None:
        """A line break inside a paragraph. Headings stay on one line."""
        if self._level is None:
            self._end_line()
        else:
            self._line.append(" ")

    def _page_break(self, *, inserted: bool) -> None:
        """Start a new page, unless this is Word's layout mark for the inserted break just before it."""
        if not self.paginate:
            return
        if not inserted:
            self.rendered_breaks += 1
            if self._after_inserted_break:
                self._after_inserted_break = False
                return
        if len(self.pages) == 1 and not self.pages[0].has_content:
            return  # nothing comes before it
        if self._line:
            self._end_line()
        self.pages.append(_Page())
        self._after_inserted_break = inserted

    # Document structure

    def blocks(self, container: Element) -> None:
        for child in _content(container):
            name = self._name(child)
            if name == "p":
                self._paragraph(child)
            elif name == "tbl":
                self._table(child)
            elif name in _WRAPPERS:
                self.blocks(child)

    def _heading_level(self, properties: Element | None) -> int | None:
        if properties is None:
            return None
        outline = properties.find(self.w + "outlineLvl")
        if outline is not None:
            level = _level_value(outline.get(self.w + "val"))
            return level if level is not None and level < 9 else None
        style = properties.find(self.w + "pStyle")
        return self.style_levels.get(style.get(self.w + "val", "")) if style is not None else None

    def _paragraph(self, paragraph: Element) -> None:
        properties = paragraph.find(self.w + "pPr")
        if self._is_on(properties, "pageBreakBefore"):
            self._page_break(inserted=True)
        self._level = self._heading_level(properties)
        self._text_boxes = []
        self._inline(paragraph)
        self._end_line()
        self._level = None
        boxes, self._text_boxes = self._text_boxes, []
        if boxes:
            self._read_text_boxes(boxes)
        section = properties.find(self.w + "sectPr") if properties is not None else None
        if section is not None and id(section) in self.page_sections:
            self._page_break(inserted=True)

    def _inline(self, container: Element) -> None:
        for child in _content(container):
            name = self._name(child)
            if name == "r":
                self._run(child)
            elif name in _INLINE_WRAPPERS:
                self._inline(child)
            elif child.tag.startswith(_MATH_NAMESPACES):
                self._add_text("".join(child.itertext()))
            # Deleted and moved-away text (w:del, w:moveFrom) and bookmarks, comments and proofing
            # marks are skipped.

    def _run(self, run: Element) -> None:
        if self._is_on(run.find(self.w + "rPr"), "vanish"):
            return  # hidden text
        for child in _content(run):
            name = self._name(child)
            if name == "t":
                self._add_text(child.text or "")
            elif name in ("tab", "ptab"):
                self._line.append("\t")
            elif name == "br":
                if child.get(self.w + "type") == "page":
                    self._page_break(inserted=True)
                else:
                    self._line_break()
            elif name == "cr":
                self._line_break()
            elif name == "noBreakHyphen":
                self._line.append("-")
            elif name == "lastRenderedPageBreak":
                self._page_break(inserted=False)
            elif name in ("drawing", "pict", "object"):
                self._saw_content()
                self._text_boxes.extend(self._boxes_in(child))
            # Deleted text, field codes, symbols, soft hyphens and note references are skipped.

    def _boxes_in(self, element: Element) -> Iterator[Element]:
        """Text box contents inside a picture, not counting boxes nested in them."""
        for child in _content(element):
            if child.tag == self.w + "txbxContent":
                yield child
            else:
                yield from self._boxes_in(child)

    def _read_text_boxes(self, boxes: list[Element]) -> None:
        paginate, self.paginate = self.paginate, False  # breaks inside a floating box don't break pages
        for box in boxes:
            self.blocks(box)
        self.paginate = paginate

    def _named(self, element: Element, name: str) -> Iterator[Element]:
        """Children called `name`, including those inside content controls and custom XML."""
        for child in _content(element):
            child_name = self._name(child)
            if child_name == name:
                yield child
            elif child_name in _WRAPPERS:
                yield from self._named(child, name)

    def _table(self, table: Element) -> None:
        """One line per row. A row Word laid out on a new page moves there as a whole."""
        for row in self._named(table, "tr"):
            if any(element.tag == self.w + "lastRenderedPageBreak" for element in row.iter()):
                self._page_break(inserted=False)
            cells = []
            for cell in self._named(row, "tc"):
                reader = _BodyReader(self.w, self.style_levels, paginate=False)
                reader.blocks(cell)
                if reader.pages[0].has_content:
                    self._saw_content()
                text = " ".join(line for line, _ in reader.pages[0].lines if line.strip())
                if text:
                    cells.append(text)
            if cells:
                self.pages[-1].lines.append((" | ".join(cells), None))


# Package parts


@dataclass(frozen=True)
class _Package:
    document: Element
    styles: Element | None
    properties: Element | None


def _read_xml(package: zipfile.ZipFile, name: str) -> Element | None:
    try:
        info = package.getinfo(name)
    except KeyError:
        return None
    if info.file_size > MAX_PART_BYTES:
        raise DocumentError("too_large", "docx")
    return fromstring(package.read(info), forbid_dtd=True)


def _optional_xml(package: zipfile.ZipFile, name: str) -> Element | None:
    """A part the text doesn't depend on. When it's missing, too large or damaged, go on without it."""
    try:
        return _read_xml(package, name)
    except Exception:
        return None


def _relationship_targets(package: zipfile.ZipFile, part: str) -> dict[str, str]:
    """Where a part's relationships point (the package's own for ""), keyed by the end of their type."""
    folder, name = posixpath.split(part)
    relationships = _optional_xml(package, posixpath.join(folder, "_rels", f"{name}.rels"))
    targets: dict[str, str] = {}
    for relationship in relationships.iter(_RELATIONSHIP) if relationships is not None else ():
        target = relationship.get("Target", "")
        if not target or relationship.get("TargetMode") == "External":
            continue
        kind = relationship.get("Type", "").rsplit("/", 1)[-1]
        path = target[1:] if target.startswith("/") else posixpath.normpath(posixpath.join(folder, target))
        targets.setdefault(kind, path)
    return targets


def _open_package(data: bytes) -> _Package:
    try:
        with zipfile.ZipFile(BytesIO(data)) as package:
            if "[Content_Types].xml" not in package.namelist():
                raise DocumentError("unsupported")  # a ZIP file, but not an Office document
            targets = _relationship_targets(package, "")
            main = targets.get("officeDocument", "word/document.xml")
            document = _read_xml(package, main)
            if document is None:
                raise DocumentError("corrupt", "docx")
            styles_part = posixpath.join(posixpath.dirname(main), "styles.xml")
            styles = _optional_xml(package, _relationship_targets(package, main).get("styles", styles_part))
            properties = _optional_xml(package, targets.get("extended-properties", "docProps/app.xml"))
    except DocumentError:
        raise
    except Exception as exc:  # zipfile and the XML parser raise a variety of errors for damaged files
        raise DocumentError("corrupt", "docx") from exc
    return _Package(document, styles, properties)


def _style_levels(styles: Element | None, w: str) -> dict[str, int]:
    """Outline level (0 = top) of each paragraph style used for headings, inherited through basedOn."""
    if styles is None:
        return {}
    own: dict[str, int | None] = {}
    parents: dict[str, str] = {}
    for style in styles.iter(w + "style"):
        style_id = style.get(w + "styleId")
        if style.get(w + "type") != "paragraph" or not style_id:
            continue
        properties = style.find(w + "pPr")
        outline = properties.find(w + "outlineLvl") if properties is not None else None
        name = style.find(w + "name")
        style_name = (name.get(w + "val", "") if name is not None else "").strip().lower()
        heading = _HEADING_STYLE_NAME.fullmatch(style_name)
        if outline is not None:
            own[style_id] = _level_value(outline.get(w + "val"))
        elif heading:
            own[style_id] = int(heading.group(1)) - 1
        else:
            own[style_id] = 0 if style_name == "title" else None
        based_on = style.find(w + "basedOn")
        if based_on is not None and based_on.get(w + "val"):
            parents[style_id] = based_on.get(w + "val", "")

    levels: dict[str, int] = {}
    for style_id in own:
        current: str | None = style_id
        seen: set[str] = set()
        while current in own and own[current] is None and current not in seen:
            seen.add(current)
            current = parents.get(current)
        level = own.get(current) if current is not None else None
        if level is not None and level < 9:
            levels[style_id] = level
    return levels


def _page_sections(body: Element, w: str) -> frozenset[int]:
    """IDs of the section properties after which the next section starts on a new page.

    A section's properties close it, and its start type is recorded with the section that follows.
    """
    replaced = {id(old) for change in body.iter(w + "sectPrChange") for old in change.iter(w + "sectPr")}
    sections = [section for section in body.iter(w + "sectPr") if id(section) not in replaced]
    breaks = set()
    for section, following in itertools.pairwise(sections):
        start = following.find(w + "type")
        if start is None or start.get(w + "val", "nextPage") in _NEW_PAGE_SECTIONS:
            breaks.add(id(section))
    return frozenset(breaks)


def _recorded_page_count(properties: Element | None, word_count: int) -> int | None:
    """The page count Word saved with the file, when the word count saved with it shows it's current."""
    if properties is None:
        return None
    values = {child.tag.rsplit("}", 1)[-1]: (child.text or "").strip() for child in properties}
    try:
        pages, words = int(values["Pages"]), int(values["Words"])
    except (KeyError, ValueError):
        return None
    if pages < 1 or words < 1 or not 0.5 <= word_count / words <= 2:
        return None
    return pages


# Pages


def _clean_line(raw: str) -> str:
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", raw)).strip()


def _pages_add_up(pages: list[list[Line]], recorded: int | None, rendered_breaks: int) -> bool:
    if any(sum(len(text) + 1 for text, _ in page) > MAX_PAGE_CHARS for page in pages):
        return False
    return len(pages) == recorded if recorded is not None else rendered_breaks > 0


def _parts(lines: list[Line]) -> list[list[Line]]:
    """Divide the text into parts of about PART_CHARS characters. A heading starts a new part once the
    current one is half full, so sections tend to begin a part."""
    parts: list[list[Line]] = [[]]
    size = 0
    for text, level in lines:
        if parts[-1] and (size + len(text) > PART_CHARS or (level is not None and size >= PART_CHARS // 2)):
            parts.append([])
            size = 0
        parts[-1].append((text, level))
        size += len(text) + 1
    return parts


def _page_text(number: int, lines: list[Line]) -> PageText:
    headings: list[StyledHeading] = []
    offset = 0
    for text, level in lines:
        if level is not None:
            headings.append(StyledHeading(offset=offset, level=min(level + 1, 3)))
        offset += len(text) + 1
    return PageText(page_number=number, text="\n".join(text for text, _ in lines), headings=tuple(headings))


def extract_docx(data: bytes, filename: str) -> ExtractedDocument:
    """Return a Word document's cleaned, 1-based pages (see the module docstring for what a page is).
    `extraction.extract_document` checks first that the upload looks like a Word file.

    Raises DocumentError for legacy .doc, password-protected, damaged, oversized and text-less files,
    and for Office files that aren't Word documents.
    """
    if data.startswith(COMPOUND_FILE_SIGNATURE):
        raise DocumentError("encrypted" if _ENCRYPTED_PACKAGE in data else "legacy_doc", "docx")
    package = _open_package(data)
    namespace, _, name = package.document.tag.partition("}")
    if name != "document" or namespace[1:] not in _WORD_NAMESPACES:
        raise DocumentError("unsupported")  # a spreadsheet or presentation, for example
    w = namespace + "}"
    body = package.document.find(w + "body")
    if body is None:
        raise DocumentError("corrupt", "docx")

    reader = _BodyReader(w, _style_levels(package.styles, w), _page_sections(body, w))
    reader.blocks(body)
    pages = [
        [(text, level) for raw, level in page.lines if (text := _clean_line(raw))] for page in reader.pages
    ]
    lines = [line for page in pages for line in page]
    if sum(meaningful_chars(text) for text, _ in lines) < MIN_TEXT_CHARS:
        raise DocumentError("no_text", "docx")

    words = sum(1 for text, _ in lines for word in text.split() if word != "|")
    paged = _pages_add_up(pages, _recorded_page_count(package.properties, words), reader.rendered_breaks)
    groups = pages if paged else _parts(lines)
    if len(groups) > MAX_PAGES:
        kind = "too_many_pages" if paged else "too_long"
        raise DocumentError(kind, "docx", pages=len(groups), limit=MAX_PAGES)
    return ExtractedDocument(
        document_id=document_fingerprint(data),
        filename=display_name(filename, "document.docx"),
        pages=[_page_text(number, group) for number, group in enumerate(groups, start=1)],
        has_page_numbers=paged,
    )
