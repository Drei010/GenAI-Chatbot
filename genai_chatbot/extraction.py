"""Recognize an upload by its content rather than its name, and read it with the PDF or Word reader."""

from __future__ import annotations

from genai_chatbot.documents import DocumentError
from genai_chatbot.docx_processing import COMPOUND_FILE_SIGNATURE, ZIP_SIGNATURE, extract_docx
from genai_chatbot.pdf_processing import extract_pdf
from genai_chatbot.schemas import ExtractedDocument

PDF_SIGNATURE = b"%PDF-"


def extract_document(data: bytes, filename: str) -> ExtractedDocument:
    """Validate an upload and return its cleaned, 1-based page texts.

    A PDF has its %PDF- header within the first kilobyte. A .docx file is a ZIP package; files in the
    older compound format are legacy .doc files or password-protected .docx files, which the Word
    reader explains. Raises DocumentError for anything else and for files that can't be read.
    """
    if not data:
        raise DocumentError("empty_file")
    if PDF_SIGNATURE in data[:1024]:
        return extract_pdf(data, filename)
    if data.startswith((ZIP_SIGNATURE, COMPOUND_FILE_SIGNATURE)):
        return extract_docx(data, filename)
    raise DocumentError("unsupported")
