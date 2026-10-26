"""Uploads are recognized by their content, and every problem is explained in plain language."""

from __future__ import annotations

import pytest

from genai_chatbot.documents import DocumentError, ProblemKind
from genai_chatbot.extraction import extract_document
from tests.support.docx import make_docx, paragraph
from tests.support.pdfs import make_pdf

TEA = "Harbor Tea Company employed 85 people at the end of 2024."


def problem(data: bytes, filename: str) -> str:
    with pytest.raises(DocumentError) as caught:
        extract_document(data, filename)
    return caught.value.kind


def test_pdfs_and_word_documents_are_recognized_whatever_their_names() -> None:
    from_pdf = extract_document(make_pdf([[TEA]]), "report.docx")
    from_word = extract_document(make_docx(paragraph(TEA)), "report.pdf")

    assert from_pdf.page(1).text == from_word.page(1).text == TEA
    assert from_pdf.has_page_numbers and not from_word.has_page_numbers


def test_a_pdf_header_after_leading_bytes_is_still_a_pdf() -> None:
    assert extract_document(b"\n\n" + make_pdf([[TEA]]), "report.pdf").page_count == 1


def test_empty_upload() -> None:
    assert problem(b"", "upload.pdf") == "empty_file"


@pytest.mark.parametrize(
    "data",
    [
        b"Just some notes saved with a .pdf name.",
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 64,
        b"PK\x05\x06" + b"\x00" * 18,
        "Notes saved as UTF-16 text".encode("utf-16"),
    ],
    ids=["text", "png", "empty-zip", "utf-16-text"],
)
@pytest.mark.parametrize("filename", ["notes.pdf", "notes.docx"])
def test_other_files_are_unsupported_whatever_their_names(data: bytes, filename: str) -> None:
    assert problem(data, filename) == "unsupported"


MESSAGE_DETAILS: dict[ProblemKind, dict[str, int]] = {
    "too_many_pages": {"pages": 612, "limit": 500},
    "too_long": {"pages": 612, "limit": 500},
}


@pytest.mark.parametrize("file_format", ["pdf", "docx"])
@pytest.mark.parametrize("kind", ProblemKind.__args__)
def test_messages_are_plain_language(kind: ProblemKind, file_format) -> None:
    message = DocumentError(kind, file_format, **MESSAGE_DETAILS.get(kind, {})).message

    assert message.endswith(".") and "{" not in message
    assert not any(word in message.lower() for word in ("pypdf", "xml", "zip", "exception", "error"))


def test_messages_name_the_kind_of_document() -> None:
    assert DocumentError("corrupt").message.startswith("This PDF couldn't be read.")
    assert DocumentError("corrupt", "docx").message.startswith("This Word document couldn't be read.")
    assert DocumentError("too_many_pages", "docx", pages=3, limit=2).message == (
        "This Word document has 3 pages. The limit is 2 pages per document."
    )
