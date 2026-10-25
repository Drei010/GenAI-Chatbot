from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError
from pypdf import PageObject

from genai_chatbot import pdf_processing
from genai_chatbot.pdf_processing import PDFProcessingError, clean_pages, clean_text, extract_document
from genai_chatbot.schemas import ExtractedDocument, PageText
from tests.support.pdfs import encrypt_pdf, make_image_only_pdf, make_pdf, make_zero_page_pdf

THREE_PAGES = [
    ["Quarterly overview", "Revenue grew to $12.4 million in fiscal year 2024."],
    ["Operations", "The company opened 14 new stores and hired 230 employees."],
    ["Outlook", "Management expects modest growth in the next fiscal year."],
]


def problem(data: bytes, filename: str = "upload.pdf") -> str:
    with pytest.raises(PDFProcessingError) as caught:
        extract_document(data, filename)
    return caught.value.kind


# Valid documents


def test_extracts_each_page_with_one_based_numbers_and_filename() -> None:
    data = make_pdf(THREE_PAGES)

    document = extract_document(data, "quarterly-report.pdf")

    assert document.filename == "quarterly-report.pdf"
    assert document.page_count == 3
    assert [page.page_number for page in document.pages] == [1, 2, 3]
    assert "Revenue grew to $12.4 million" in document.page(1).text
    assert "14 new stores" in document.page(2).text
    assert document.page(3).text.startswith("Outlook")
    assert document.document_id == hashlib.sha256(data).hexdigest()


def test_same_bytes_give_the_same_document_id() -> None:
    data = make_pdf(THREE_PAGES)

    assert extract_document(data, "a.pdf").document_id == extract_document(data, "b.pdf").document_id


def test_filename_is_reduced_to_a_display_name() -> None:
    document = extract_document(make_pdf(THREE_PAGES), "C:\\Users\\me\\Reports\\q3\x07.pdf")

    assert document.filename == "q3.pdf"


def test_blank_pages_keep_their_numbers() -> None:
    document = extract_document(make_pdf([THREE_PAGES[0], [], THREE_PAGES[2]]), "gap.pdf")

    assert [page.page_number for page in document.pages] == [1, 2, 3]
    assert document.page(2).text == ""
    assert document.pages_without_text == 1
    assert "Outlook" in document.page(3).text


def test_password_protected_pdf_with_empty_user_password_opens() -> None:
    data = encrypt_pdf(make_pdf(THREE_PAGES), user_password="")

    assert "Operations" in extract_document(data, "locked-for-editing.pdf").page(2).text


def test_running_headers_and_page_numbers_are_removed() -> None:
    pages = [
        [
            f"Chapter {n} notes",
            f"Paragraph about topic {n} with enough words to be real content.",
            "Closing line.",
        ]
        for n in range(1, 7)
    ]

    document = extract_document(
        make_pdf(pages, header="Northwind Annual Report", footer_page_numbers=True), "nw.pdf"
    )

    for page in document.pages:
        assert "Northwind Annual Report" not in page.text
        assert not page.text.splitlines()[-1].isdigit()
        assert "with enough words to be real content" in page.text


# Cleaning


def test_clean_text_normalizes_without_changing_words() -> None:
    raw = "The ﬁscal  year\tended\u00a0June 30,\r\n2023. Finan-\ncial results\n\n\n\nNext   section "

    assert clean_text(raw) == "The fiscal year ended June 30,\n2023. Financial results\n\nNext section"


def test_hyphenated_numbers_and_ranges_are_left_alone() -> None:
    assert clean_text("fiscal 2022-\n2023 and A-\nB") == "fiscal 2022-\n2023 and A-\nB"


def test_short_documents_keep_repeated_lines() -> None:
    pages = ["Total revenue\nbody one", "Total revenue\nbody two", "Total revenue\nbody three"]

    assert clean_pages(pages) == pages


def test_page_that_is_all_furniture_keeps_its_text() -> None:
    pages = [f"Acme Report\nRevenue line {n}.\n{n}" for n in range(1, 6)]

    assert clean_pages(pages)[0] == "Acme Report\nRevenue line 1."


# Problems the person can fix


def test_empty_upload() -> None:
    assert problem(b"") == "empty_file"


@pytest.mark.parametrize(
    "data",
    [
        b"Just some notes saved with a .pdf name.",
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 64,
        b"PK\x03\x04 a zip or docx file",
    ],
)
def test_non_pdf_content_is_rejected_even_with_a_pdf_name(data: bytes) -> None:
    assert problem(data, "notes.pdf") == "not_pdf"


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda pdf: pdf[: len(pdf) // 3],
        lambda pdf: pdf[:20] + b"\x00garbage" * 40,
        lambda pdf: b"%PDF-1.7\n",
    ],
    ids=["truncated", "garbled", "header-only"],
)
def test_corrupt_pdfs_are_reported(corrupt) -> None:
    assert problem(corrupt(make_pdf(THREE_PAGES))) == "corrupt"


def test_zero_page_pdf() -> None:
    assert problem(make_zero_page_pdf()) == "no_pages"


def test_scanned_or_image_only_pdf_mentions_ocr() -> None:
    with pytest.raises(PDFProcessingError) as caught:
        extract_document(make_image_only_pdf(), "scan.pdf")

    assert caught.value.kind == "no_text"
    assert "OCR" in caught.value.message and "isn't supported" in caught.value.message


def test_password_protected_pdf() -> None:
    assert problem(encrypt_pdf(make_pdf(THREE_PAGES), user_password="s3cret")) == "encrypted"


def test_page_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pdf_processing, "MAX_PAGES", 2)

    with pytest.raises(PDFProcessingError) as caught:
        extract_document(make_pdf(THREE_PAGES), "long.pdf")

    assert caught.value.kind == "too_many_pages"
    assert caught.value.message == "This PDF has 3 pages. The limit is 2 pages per document."


def test_one_unreadable_page_does_not_sink_the_document(monkeypatch: pytest.MonkeyPatch) -> None:
    original = PageObject.extract_text
    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("bad content stream")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PageObject, "extract_text", flaky)

    document = extract_document(make_pdf(THREE_PAGES), "flaky.pdf")
    assert document.page(2).text == "" and "Outlook" in document.page(3).text


def test_every_page_unreadable_is_corrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(self, *args, **kwargs):
        raise ValueError("bad content stream")

    monkeypatch.setattr(PageObject, "extract_text", broken)

    assert problem(make_pdf(THREE_PAGES)) == "corrupt"


def test_messages_are_plain_language() -> None:
    for kind in ("empty_file", "not_pdf", "corrupt", "encrypted", "no_pages", "no_text"):
        message = PDFProcessingError(kind).message
        assert message.endswith(".") and "pypdf" not in message.lower() and "exception" not in message.lower()


# Schema validation


def test_document_schema_rejects_misnumbered_pages() -> None:
    with pytest.raises(ValidationError):
        ExtractedDocument(
            document_id="a" * 64,
            filename="x.pdf",
            pages=[PageText(page_number=2, text="out of order")],
        )


def test_page_numbers_are_one_based() -> None:
    with pytest.raises(ValidationError):
        PageText(page_number=0, text="zero")
