from __future__ import annotations

import hashlib
import zipfile
from io import BytesIO

import pytest

from genai_chatbot import docx_processing
from genai_chatbot.documents import DocumentError
from genai_chatbot.extraction import extract_document
from genai_chatbot.schemas import ExtractedDocument
from tests.support.docx import (
    HEADING_STYLES,
    PAGE_BREAK,
    PICTURE,
    RENDERED_BREAK,
    STRICT_W,
    W,
    cell,
    compound_file,
    make_docx,
    paragraph,
    run,
    style,
    table,
    text_box,
)

INTRO = "Harbor Tea Company employed 85 people at the end of 2024, up from 72 a year earlier."
SHOPS = "Harbor Tea opened two new shops in Portland during 2024."
ONLINE = "Online orders made up 31% of sales, compared with 26% in 2023."


def read(body: str | list[str], **options) -> ExtractedDocument:
    return extract_document(make_docx(body, **options), "harbor-tea.docx")


def text_of(body: str | list[str], **options) -> str:
    return "\n".join(page.text for page in read(body, **options).pages)


def problem(data: bytes) -> DocumentError:
    with pytest.raises(DocumentError) as caught:
        extract_document(data, "upload.docx")
    return caught.value


def headings(document: ExtractedDocument) -> list[tuple[str, int]]:
    return [
        (page.text[heading.offset :].split("\n")[0], heading.level)
        for page in document.pages
        for heading in page.headings
    ]


# Text in reading order


def test_paragraphs_become_lines_of_text() -> None:
    data = make_docx([paragraph("Annual Review 2024"), paragraph(INTRO), paragraph(SHOPS)])

    document = extract_document(data, "C:\\Users\\me\\Reports\\harbor-tea.docx")

    assert document.filename == "harbor-tea.docx"
    assert document.document_id == hashlib.sha256(data).hexdigest()
    assert document.page(1).text == f"Annual Review 2024\n{INTRO}\n{SHOPS}"


def test_runs_tabs_and_line_breaks_join_as_written() -> None:
    body = [
        paragraph(
            run("Revenue"), "<w:r><w:tab/></w:r>", run("$12.4 million"), "<w:r><w:br/></w:r>", run("Up 8%")
        ),
        paragraph(
            "<w:r><w:t>Year</w:t><w:noBreakHyphen/><w:t>end review of the co</w:t><w:softHyphen/>"
            '<w:t>operative</w:t><w:sym w:font="Wingdings" w:char="F0E0"/><w:ptab w:alignment="right"/>'
            "<w:t>page</w:t><w:cr/><w:t>Next line</w:t></w:r>"
        ),
        paragraph(run("   "), run("Spaced\u00a0  out\ttext ")),
        paragraph(),
    ]

    assert text_of(body) == (
        "Revenue $12.4 million\nUp 8%\nYear-end review of the cooperative page\nNext line\nSpaced out text"
    )


def test_hidden_deleted_and_field_code_text_is_left_out() -> None:
    body = paragraph(
        run("Revenue grew "),
        run("(draft: check this figure) ", hidden=True),
        '<w:del w:id="1" w:author="Ana"><w:r><w:delText>slightly </w:delText></w:r></w:del>',
        '<w:ins w:id="2" w:author="Ana"><w:r><w:t xml:space="preserve">strongly </w:t></w:r></w:ins>',
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>',
        "<w:r><w:instrText> DOCPROPERTY Year </w:instrText></w:r>",
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>',
        run("in 2024"),
        '<w:r><w:fldChar w:fldCharType="end"/></w:r>',
        '<w:moveFrom w:id="3" w:author="Ana"><w:r><w:t> (moved away)</w:t></w:r></w:moveFrom>',
        '<w:moveTo w:id="4" w:author="Ana"><w:r><w:t>.</w:t></w:r></w:moveTo>',
        '<w:r><w:rPr><w:vanish w:val="false"/></w:rPr><w:t> Visible.</w:t></w:r>',
    )

    assert text_of(body) == "Revenue grew strongly in 2024. Visible."


def test_links_content_controls_fields_and_equations_keep_their_text() -> None:
    body = [
        paragraph(
            '<w:hyperlink w:anchor="shops"><w:r><w:t>Harbor Tea</w:t></w:r></w:hyperlink>',
            run(" sells "),
            '<w:sdt><w:sdtPr><w:alias w:val="Product"/></w:sdtPr>'
            "<w:sdtContent><w:r><w:t>green tea</w:t></w:r></w:sdtContent></w:sdt>",
            run(" in "),
            '<w:fldSimple w:instr=" DOCPROPERTY Shops "><w:r><w:t>14</w:t></w:r></w:fldSimple>',
            run(" shops"),
            '<w:smartTag w:uri="urn:places" w:element="city"><w:r><w:t> in Portland</w:t></w:r></w:smartTag>',
            '<w:bookmarkStart w:id="0" w:name="shops"/><w:bookmarkEnd w:id="0"/>',
        ),
        f"<w:sdt><w:sdtPr/><w:sdtContent>{paragraph(SHOPS)}</w:sdtContent></w:sdt>",
        f'<w:customXml w:element="summary">{paragraph(ONLINE)}</w:customXml>',
        paragraph(run("Growth rate: "), "<m:oMath><m:r><m:t>g=(85-72)/72</m:t></m:r></m:oMath>"),
        paragraph(
            "<mc:AlternateContent><mc:Fallback><w:r><w:t>Read from the fallback.</w:t></w:r></mc:Fallback>"
            "</mc:AlternateContent><mc:AlternateContent/>"
        ),
    ]

    assert text_of(body) == (
        f"Harbor Tea sells green tea in 14 shops in Portland\n{SHOPS}\n{ONLINE}\n"
        "Growth rate: g=(85-72)/72\nRead from the fallback."
    )


def test_tables_become_one_line_per_row() -> None:
    body = [
        paragraph("Sales by region"),
        table(
            [
                ["Region", "2024", "2023"],
                ["North", "$ 4.1", "$ 3.8"],
                [cell("South", "and coast"), cell("$ 2.2"), cell(PICTURE)],
                [cell("South"), cell(merged=True), cell("")],
                ["", "", ""],
                [cell(table([["Online", "31%"]])), "$ 1.9", "$ 1.4"],
            ]
        ),
        "<w:tbl><w:tblPr/><w:sdt><w:sdtContent><w:tr>"
        f'{cell("Total")}<w:customXml w:element="figure">{cell("$ 8.2")}</w:customXml>'
        "</w:tr></w:sdtContent></w:sdt></w:tbl>",
    ]

    assert text_of(body).split("\n") == [
        "Sales by region",
        "Region | 2024 | 2023",
        "North | $ 4.1 | $ 3.8",
        "South and coast | $ 2.2",
        "South",
        "Online | 31% | $ 1.9 | $ 1.4",
        "Total | $ 8.2",
    ]


def test_text_boxes_are_read_once_after_their_paragraph() -> None:
    nested = text_box(paragraph("Tins sold: 1.2 million."))
    body = [
        paragraph(run("Highlights"), text_box(paragraph("Sales rose 12% in 2024."), paragraph(nested))),
        paragraph(SHOPS),
    ]

    assert text_of(body).split("\n") == [
        "Highlights",
        "Sales rose 12% in 2024.",
        "Tins sold: 1.2 million.",
        SHOPS,
    ]


def test_strict_open_xml_documents_are_read() -> None:
    styles = [style("Heading1", "heading 1")]

    document = read(
        [paragraph("Outlook", style="Heading1"), paragraph(ONLINE)], namespace=STRICT_W, styles=styles
    )

    assert document.page(1).text == f"Outlook\n{ONLINE}"
    assert headings(document) == [("Outlook", 1)]


# Headings from Word styles


def test_heading_styles_and_outline_levels_mark_headings() -> None:
    styles = (
        *HEADING_STYLES,
        style("Chapter", "Chapter", outline_level=0),
        style("Subhead", "Subhead", based_on="Heading2"),
        style("Callout", "Callout", based_on="Heading1", outline_level=9),
        style("Deep", "heading 5"),
        style("Odd", "Odd", outline_level="x"),
        style("LoopA", "Loop A", based_on="LoopB"),
        style("LoopB", "Loop B", based_on="LoopA"),
        '<w:style w:type="character" w:styleId="Strong"><w:name w:val="heading 1"/></w:style>',
        '<w:style w:type="paragraph"><w:name w:val="heading 1"/></w:style>',
        '<w:style w:type="paragraph" w:styleId="Unnamed"/>',
    )
    body = [
        paragraph("Annual Review", style="Title"),
        paragraph("Our Year", style="Chapter"),
        paragraph(INTRO),
        paragraph("Shops", "<w:r><w:br/></w:r>", "and Growth", style="Heading2"),
        paragraph("Portland", style="Subhead"),
        paragraph("A note from the team", style="Callout"),
        paragraph("Online", style="Deep"),
        paragraph("Outlook", outline_level=2),
        paragraph("Not a heading after all", style="Heading1", outline_level=9),
        paragraph("Odd one", style="Odd"),
        paragraph("Looping style", style="LoopA"),
        paragraph("Unknown style", style="Missing"),
        paragraph("Unnamed style", style="Unnamed"),
        paragraph("Odd level", outline_level="x"),
    ]

    document = read(body, styles=styles)

    assert headings(document) == [
        ("Annual Review", 1),
        ("Our Year", 1),
        ("Shops and Growth", 2),
        ("Portland", 2),
        ("Online", 3),
        ("Outlook", 3),
    ]


def test_files_without_styles_have_no_marked_headings() -> None:
    assert headings(read([paragraph("Outlook", style="Heading1"), paragraph(ONLINE)], styles=None)) == []


# Page numbers


def test_page_breaks_that_add_up_give_page_numbers() -> None:
    body = [
        paragraph("Annual Review 2024"),
        paragraph(INTRO, PAGE_BREAK),
        paragraph(RENDERED_BREAK, "Shops and Growth"),
        paragraph(SHOPS),
    ]

    document = read(body, pages=2)

    assert document.has_page_numbers
    assert [page.text for page in document.pages] == [
        f"Annual Review 2024\n{INTRO}",
        f"Shops and Growth\n{SHOPS}",
    ]


def test_word_layout_marks_give_page_numbers_when_no_page_count_is_saved() -> None:
    body = [
        paragraph(INTRO),
        paragraph(run("Shops: "), RENDERED_BREAK, run(SHOPS)),
        paragraph(RENDERED_BREAK, ONLINE),
    ]

    document = read(body)

    assert document.has_page_numbers
    assert [page.text for page in document.pages] == [f"{INTRO}\nShops:", SHOPS, ONLINE]


def test_two_breaks_in_a_row_leave_a_blank_page() -> None:
    body = [
        paragraph(INTRO, PAGE_BREAK),
        paragraph(RENDERED_BREAK, PAGE_BREAK),
        paragraph(RENDERED_BREAK, SHOPS),
    ]

    document = read(body, pages=3)

    assert document.has_page_numbers
    assert [page.text for page in document.pages] == [INTRO, "", SHOPS]
    assert document.pages_without_text == 1


def test_breaks_before_any_content_do_not_add_a_page() -> None:
    body = [paragraph(PAGE_BREAK), paragraph(INTRO, page_break_before=True), paragraph(RENDERED_BREAK, SHOPS)]

    assert [page.text for page in read(body, pages=2).pages] == [INTRO, SHOPS]


def test_new_page_sections_and_page_break_before_start_pages() -> None:
    # A section's start type is saved with the section's own properties, at its end. This one starts
    # without a new page; it used to start on one before a tracked change.
    continuous_section = (
        '<w:sectPr><w:type w:val="continuous"/><w:sectPrChange w:id="9" w:author="Ana">'
        '<w:sectPr><w:type w:val="nextPage"/></w:sectPr></w:sectPrChange></w:sectPr>'
    )
    body = [
        paragraph(INTRO, section=""),
        f"<w:p><w:pPr>{continuous_section}</w:pPr>{run(SHOPS)}</w:p>",
        paragraph(ONLINE),
        paragraph("Outlook", page_break_before=True),
        paragraph("Steady growth is expected."),
    ]

    document = read(body, pages=3, final_section="oddPage")

    assert [page.text for page in document.pages] == [
        f"{INTRO}\n{SHOPS}",
        ONLINE,
        "Outlook\nSteady growth is expected.",
    ]


def test_a_table_row_word_moved_to_a_new_page_starts_that_page() -> None:
    body = [
        paragraph(INTRO),
        table([["Region", "2024"], [cell(paragraph(RENDERED_BREAK, "North")), cell("$ 4.1")]]),
    ]

    assert [page.text for page in read(body, pages=2).pages] == [f"{INTRO}\nRegion | 2024", "North | $ 4.1"]


def test_breaks_inside_text_boxes_and_tables_do_not_break_pages() -> None:
    body = [
        paragraph(INTRO, text_box(paragraph("Boxed note.", page_break_before=True), paragraph(PAGE_BREAK))),
        table([[cell(paragraph("Cell text.", page_break_before=True))]]),
        paragraph(SHOPS),
    ]

    document = read(body, pages=1)

    assert document.has_page_numbers
    assert document.page(1).text == f"{INTRO}\nBoxed note.\nCell text.\n{SHOPS}"


def test_breaks_that_do_not_match_the_saved_page_count_are_not_used() -> None:
    body = [paragraph(INTRO, PAGE_BREAK), paragraph(RENDERED_BREAK, SHOPS)]

    document = read(body, pages=5)

    assert not document.has_page_numbers
    assert [page.text for page in document.pages] == [f"{INTRO}\n{SHOPS}"]


def test_a_page_count_saved_for_other_text_is_ignored() -> None:
    template_counts = {"pages": 1, "words": 0}
    one_short_page = [paragraph(INTRO), paragraph(SHOPS)]
    marked_pages = [paragraph(INTRO), paragraph(RENDERED_BREAK, SHOPS)]

    assert not read(one_short_page, **template_counts).has_page_numbers
    assert not read(one_short_page, pages=1, words=500).has_page_numbers
    assert read(marked_pages, **template_counts).page_count == 2
    assert read(marked_pages, pages=0).has_page_numbers
    unreadable_counts = {"docProps/app.xml": "<Properties><Pages>many</Pages></Properties>"}
    assert read(marked_pages, parts=unreadable_counts).has_page_numbers


def test_pages_longer_than_a_page_can_hold_are_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docx_processing, "MAX_PAGE_CHARS", 100)
    body = [paragraph(INTRO), paragraph(SHOPS), paragraph(RENDERED_BREAK, ONLINE)]

    assert not read(body, pages=2).has_page_numbers


def test_without_page_breaks_the_text_is_divided_into_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docx_processing, "PART_CHARS", 110)
    body = [
        paragraph("Annual Review", style="Heading1"),
        paragraph(INTRO),
        paragraph(SHOPS),
        paragraph("Online Sales", style="Heading1"),
        paragraph(ONLINE),
        paragraph("Outlook", style="Heading1"),
    ]

    document = read(body)

    assert not document.has_page_numbers
    assert [page.text for page in document.pages] == [
        f"Annual Review\n{INTRO}",
        SHOPS,
        f"Online Sales\n{ONLINE}",
        "Outlook",
    ]
    assert headings(document) == [("Annual Review", 1), ("Online Sales", 1), ("Outlook", 1)]


# Problems the person can fix


def test_legacy_doc_files_are_explained() -> None:
    error = problem(compound_file("WordDocument", "1Table", "SummaryInformation"))

    assert error.kind == "legacy_doc"
    assert ".doc" in error.message and ".docx" in error.message


def test_password_protected_word_files() -> None:
    error = problem(compound_file("EncryptionInfo", "EncryptedPackage"))

    assert error.kind == "encrypted"
    assert (
        error.message
        == "This Word document is password-protected. Remove the password, then upload it again."
    )


ENTITY_EXPANSION = (
    '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;&lol;">]>'
    f'<w:document xmlns:w="{W}"><w:body><w:p><w:r><w:t>&lol2;</w:t></w:r></w:p></w:body></w:document>'
)
SPREADSHEET = {
    "_rels/.rels": (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Target="xl/workbook.xml" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"/>'
        "</Relationships>"
    ),
    "xl/workbook.xml": '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>',
}


@pytest.mark.parametrize(
    "data",
    [
        make_docx(paragraph(INTRO))[:200],
        make_docx(paragraph(INTRO), omit=["word/document.xml"]),
        make_docx(paragraph(INTRO), parts={"word/document.xml": "<w:document><w:body>"}),
        make_docx(paragraph(INTRO), parts={"word/document.xml": ENTITY_EXPANSION}),
        make_docx(paragraph(INTRO), parts={"word/document.xml": f'<w:document xmlns:w="{W}"/>'}),
    ],
    ids=["truncated", "no-document-part", "malformed-xml", "entity-expansion", "no-body"],
)
def test_damaged_word_files_are_reported(data: bytes) -> None:
    error = problem(data)

    assert error.kind == "corrupt"
    assert error.message.startswith("This Word document couldn't be read.")


@pytest.mark.parametrize(
    "data",
    [
        make_docx(paragraph(INTRO), parts=SPREADSHEET),
        make_docx(paragraph(INTRO), parts={"word/document.xml": "<document><body/></document>"}),
        make_docx(paragraph(INTRO), omit=["[Content_Types].xml"]),
    ],
    ids=["spreadsheet", "no-namespace", "plain-zip"],
)
def test_other_zip_files_are_not_word_documents(data: bytes) -> None:
    error = problem(data)

    assert error.kind == "unsupported"
    assert error.message.startswith("This file isn't a PDF or a Word (.docx) document.")


def test_oversized_parts_are_refused_before_they_are_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docx_processing, "MAX_PART_BYTES", 300)

    error = problem(make_docx([paragraph(INTRO)] * 10))

    assert error.kind == "too_large"
    assert error.message.startswith("This Word document is too large to read here.")


def test_word_files_with_only_pictures_have_no_text() -> None:
    error = problem(make_docx([paragraph(PICTURE), paragraph(run("  "))]))

    assert error.kind == "no_text"
    assert "Word document" in error.message and "OCR" in error.message


def test_page_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docx_processing, "MAX_PAGES", 2)
    three_pages = [paragraph(INTRO), paragraph(RENDERED_BREAK, SHOPS), paragraph(RENDERED_BREAK, ONLINE)]

    error = problem(make_docx(three_pages, pages=3))
    assert error.kind == "too_many_pages"
    assert error.message == "This Word document has 3 pages. The limit is 2 pages per document."

    monkeypatch.setattr(docx_processing, "PART_CHARS", 60)
    error = problem(make_docx(three_pages, pages=7))
    assert error.kind == "too_long"
    assert error.message == (
        "This Word document is too long: it holds about 3 pages of text. The limit is 2 pages per document."
    )


# Package structure


def test_damaged_optional_parts_are_skipped() -> None:
    damaged = {"word/styles.xml": "<w:styles", "docProps/app.xml": "<Properties", "_rels/.rels": "<Relations"}

    document = read([paragraph("Outlook", style="Heading1"), paragraph(ONLINE)], parts=damaged)

    assert document.page(1).text == f"Outlook\n{ONLINE}"
    assert headings(document) == []


def test_parts_are_found_through_their_relationships() -> None:
    relationships = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId0" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
        'officeDocument" Target="https://example.com/remote.xml" TargetMode="External"/>'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
        'officeDocument" Target="/content/main.xml"/>'
        '<Relationship Id="rId2" Type="http://example.com/no-target"/>'
        "</Relationships>"
    )
    style_relationships = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
        'styles" Target="../shared/styles.xml"/></Relationships>'
    )
    original = make_docx([paragraph("Outlook", style="Heading1"), paragraph(ONLINE)])
    moved = make_docx(
        [],
        parts={
            "_rels/.rels": relationships,
            "content/main.xml": _part(original, "word/document.xml"),
            "content/_rels/main.xml.rels": style_relationships,
            "shared/styles.xml": _part(original, "word/styles.xml"),
        },
        omit=["word/document.xml", "word/styles.xml"],
    )

    document = extract_document(moved, "moved.docx")

    assert document.page(1).text == f"Outlook\n{ONLINE}"
    assert headings(document) == [("Outlook", 1)]


def _part(package: bytes, name: str) -> bytes:
    with zipfile.ZipFile(BytesIO(package)) as archive:
        return archive.read(name)


def test_packages_without_relationships_use_the_usual_part_names() -> None:
    body = [paragraph("Outlook", style="Heading1"), paragraph(ONLINE)]

    document = read(body, omit=["_rels/.rels", "word/_rels/document.xml.rels"])

    assert headings(document) == [("Outlook", 1)]
