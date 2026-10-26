"""Small Word (.docx) files built from WordprocessingML snippets, so tests never depend on private or
downloaded documents.

The helpers return XML text for pieces of a document body; `make_docx` packages a body as a .docx
file that Word and LibreOffice open.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Mapping, Sequence
from io import BytesIO
from xml.sax.saxutils import escape

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
STRICT_W = "http://purl.oclc.org/ooxml/wordprocessingml/main"
_NAMESPACES = (
    'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
    'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
    'xmlns:v="urn:schemas-microsoft-com:vml" '
    'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
    'mc:Ignorable="wps"'
)
_RELATIONSHIP_TYPES = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

PAGE_BREAK = '<w:r><w:br w:type="page"/></w:r>'
# Where Word's last layout started a new page.
RENDERED_BREAK = "<w:r><w:lastRenderedPageBreak/></w:r>"
PICTURE = (
    '<w:r><w:drawing><wp:inline><wp:extent cx="914400" cy="914400"/><wp:docPr id="1" name="Picture 1"/>'
    '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"/></a:graphic>'
    "</wp:inline></w:drawing></w:r>"
)


def run(text: str, *, hidden: bool = False) -> str:
    properties = "<w:rPr><w:vanish/></w:rPr>" if hidden else ""
    return f'<w:r>{properties}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def _content(parts: Sequence[str]) -> str:
    """Plain strings become runs; strings that start with "<" are XML already."""
    return "".join(part if part.startswith("<") else run(part) for part in parts)


def paragraph(
    *parts: str,
    style: str | None = None,
    outline_level: int | str | None = None,
    page_break_before: bool = False,
    section: str | None = None,
) -> str:
    """A paragraph of runs. `section` closes a section here with that start type ("" for none given)."""
    properties = ""
    if style:
        properties += f'<w:pStyle w:val="{style}"/>'
    if page_break_before:
        properties += "<w:pageBreakBefore/>"
    if outline_level is not None:
        properties += f'<w:outlineLvl w:val="{outline_level}"/>'
    if section is not None:
        properties += section_properties(section)
    return f"<w:p>{f'<w:pPr>{properties}</w:pPr>' if properties else ''}{_content(parts)}</w:p>"


def section_properties(start: str = "") -> str:
    """Section properties; `start` is how the section begins (nextPage, continuous...), "" for the default."""
    kind = f'<w:type w:val="{start}"/>' if start else ""
    return f'<w:sectPr>{kind}<w:pgSz w:w="12240" w:h="15840"/></w:sectPr>'


def cell(*parts: str, merged: bool = False) -> str:
    """A table cell; `merged` continues the cell above (a vertical merge)."""
    properties = "<w:tcPr><w:vMerge/></w:tcPr>" if merged else ""
    content = "".join(part if part.startswith(("<w:p>", "<w:tbl>")) else paragraph(part) for part in parts)
    return f"<w:tc>{properties}{content or '<w:p/>'}</w:tc>"


def table(rows: Sequence[Sequence[str]]) -> str:
    """A table; each cell is text, or cell() XML."""
    xml_rows = "".join(
        "<w:tr>" + "".join(item if item.startswith("<w:tc>") else cell(item) for item in row) + "</w:tr>"
        for row in rows
    )
    return f"<w:tbl><w:tblPr/>{xml_rows}</w:tbl>"


def text_box(*paragraphs: str) -> str:
    """A floating text box, written the way Word writes it: DrawingML with a VML copy for older readers."""
    content = "".join(paragraphs)
    return (
        "<w:r><mc:AlternateContent>"
        '<mc:Choice Requires="wps"><w:drawing><wp:anchor><wp:docPr id="2" name="Text Box 2"/><a:graphic>'
        '<a:graphicData uri="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"><wps:wsp>'
        f"<wps:txbx><w:txbxContent>{content}</w:txbxContent></wps:txbx>"
        "</wps:wsp></a:graphicData></a:graphic></wp:anchor></w:drawing></mc:Choice>"
        f"<mc:Fallback><w:pict><v:shape><v:textbox><w:txbxContent>{content}</w:txbxContent></v:textbox>"
        "</v:shape></w:pict></mc:Fallback>"
        "</mc:AlternateContent></w:r>"
    )


def style(
    style_id: str, name: str, *, based_on: str | None = None, outline_level: int | str | None = None
) -> str:
    parts = f'<w:name w:val="{name}"/>'
    if based_on:
        parts += f'<w:basedOn w:val="{based_on}"/>'
    if outline_level is not None:
        parts += f'<w:pPr><w:outlineLvl w:val="{outline_level}"/></w:pPr>'
    return f'<w:style w:type="paragraph" w:styleId="{style_id}">{parts}</w:style>'


# Word's built-in names; their outline levels come from the names, as in many real files.
HEADING_STYLES = (
    style("Normal", "Normal"),
    style("Title", "Title", based_on="Normal"),
    style("Heading1", "heading 1", based_on="Normal"),
    style("Heading2", "heading 2", based_on="Normal"),
    style("Heading3", "heading 3", based_on="Normal"),
)


def words_in(body: str) -> int:
    return len(re.sub(r"<[^>]+>", " ", body).split())


def make_docx(
    body: str | Sequence[str],
    *,
    styles: Sequence[str] | None = HEADING_STYLES,
    pages: int | None = None,
    words: int | None = None,
    namespace: str = W,
    final_section: str = "",
    parts: Mapping[str, str | bytes] | None = None,
    omit: Sequence[str] = (),
) -> bytes:
    """Package body XML as a .docx file.

    `pages` records a page count in docProps/app.xml the way Word does when it saves (with a word count
    for the body, unless `words` is given). `final_section` is the start type of the last section.
    `parts` adds or replaces package parts, and `omit` leaves parts out.
    """
    content = body if isinstance(body, str) else "".join(body)
    package: dict[str, str | bytes] = {
        "_rels/.rels": _relationships(
            ("officeDocument", "word/document.xml"),
            *([("extended-properties", "docProps/app.xml")] if pages is not None else []),
        ),
        "word/document.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:document xmlns:w="{namespace}" {_NAMESPACES}><w:body>{content}'
            f"{section_properties(final_section)}</w:body></w:document>"
        ),
    }
    if styles is not None:
        package["word/_rels/document.xml.rels"] = _relationships(("styles", "styles.xml"))
        package["word/styles.xml"] = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:styles xmlns:w="{namespace}">{"".join(styles)}</w:styles>'
        )
    if pages is not None:
        package["docProps/app.xml"] = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
            f"<Application>Microsoft Office Word</Application><Pages>{pages}</Pages>"
            f"<Words>{words if words is not None else words_in(content)}</Words></Properties>"
        )
    package.update(parts or {})
    package = {name: data for name, data in package.items() if name not in omit}
    if "[Content_Types].xml" not in omit:
        package["[Content_Types].xml"] = _content_types(package)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in package.items():
            archive.writestr(name, data)
    return buffer.getvalue()


_CONTENT_TYPES = {
    "word/document.xml": "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
    "word/styles.xml": "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml",
    "docProps/app.xml": "application/vnd.openxmlformats-officedocument.extended-properties+xml",
}


def _content_types(package: Mapping[str, str | bytes]) -> str:
    overrides = "".join(
        f'<Override PartName="/{name}" ContentType="{kind}"/>'
        for name, kind in _CONTENT_TYPES.items()
        if name in package
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        f'<Default Extension="xml" ContentType="application/xml"/>{overrides}</Types>'
    )


def _relationships(*targets: tuple[str, str]) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(
            f'<Relationship Id="rId{number}" Type="{_RELATIONSHIP_TYPES}/{kind}" Target="{target}"/>'
            for number, (kind, target) in enumerate(targets, start=1)
        )
        + "</Relationships>"
    )


def compound_file(*stream_names: str) -> bytes:
    """The start of a compound file (legacy .doc, or a .docx saved with a password) naming its streams."""
    names = b"".join(name.encode("utf-16-le") + b"\x00\x00" for name in stream_names)
    return b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 504 + names + b"\x00" * 512
