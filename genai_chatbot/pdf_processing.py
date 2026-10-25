"""Check that an upload is a readable PDF and extract clean text for each page."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from io import BytesIO
from pathlib import PurePath
from typing import Literal

from pypdf import PasswordType, PdfReader

from genai_chatbot.schemas import ExtractedDocument, PageText

MAX_PAGES = 500
# A whole document with fewer letters and digits than this has no usable text layer.
MIN_TEXT_CHARS = 20
# Lines repeated at the top or bottom of this share of pages are running headers or footers.
REPEATED_EDGE_SHARE = 0.6
EDGE_LINES = 2

ProblemKind = Literal[
    "empty_file", "not_pdf", "corrupt", "encrypted", "no_pages", "too_many_pages", "no_text"
]

_MESSAGES: dict[ProblemKind, str] = {
    "empty_file": "The file is empty. Choose a PDF that has content.",
    "not_pdf": "This file isn't a PDF. Upload a document saved in PDF format.",
    "corrupt": (
        "This PDF couldn't be read. It may be damaged or incomplete; try downloading or exporting it again."
    ),
    "encrypted": "This PDF is password-protected. Remove the password, then upload it again.",
    "no_pages": "This PDF has no pages.",
    "too_many_pages": "This PDF has {pages} pages. The limit is {limit} pages per document.",
    "no_text": (
        "No selectable text was found in this PDF. It looks like a scanned or image-only document, "
        "and OCR (reading text from images) isn't supported in this version."
    ),
}


class PDFProcessingError(Exception):
    """A problem with an uploaded file, worded for the person who uploaded it."""

    def __init__(self, kind: ProblemKind, **details: object) -> None:
        self.kind = kind
        self.message = _MESSAGES[kind].format(**details)
        super().__init__(self.message)


_HORIZONTAL_SPACE = re.compile(r"[^\S\n]+")
_HYPHENATED_BREAK = re.compile(r"([a-z])-\n([a-z])")
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")
_PAGE_NUMBER_LINE = re.compile(r"^(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?$", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")


def clean_text(raw: str) -> str:
    """Normalize one page of extracted text without changing its words."""
    text = unicodedata.normalize("NFKC", raw).replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HORIZONTAL_SPACE.sub(" ", text)
    text = _HYPHENATED_BREAK.sub(r"\1\2", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _EXTRA_BLANK_LINES.sub("\n\n", text).strip()


def _edge_lines(text: str) -> list[str]:
    lines = [line for line in text.split("\n") if line]
    return lines[:EDGE_LINES] + lines[-EDGE_LINES:] if len(lines) > EDGE_LINES * 2 else lines


def _repeated_edge_lines(pages: list[str]) -> set[str]:
    """Digit-insensitive forms of short lines that open or close most pages (running headers/footers)."""
    if len(pages) < 4:
        return set()
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({_DIGITS.sub("#", line) for line in _edge_lines(page) if len(line) <= 80})
    threshold = max(3, REPEATED_EDGE_SHARE * len(pages))
    return {line for line, seen in counts.items() if seen >= threshold}


def _trim_edges(lines: list[str], is_furniture: Callable[[str], bool]) -> str:
    start, end = 0, len(lines)
    for _ in range(EDGE_LINES):
        if start < end and is_furniture(lines[start]):
            start += 1
        if start < end and is_furniture(lines[end - 1]):
            end -= 1
    return "\n".join(lines[start:end]).strip()


def _strip_page_furniture(text: str, repeated: set[str]) -> str:
    """Drop page numbers and running headers/footers from the top and bottom of a page.

    A page is never stripped to nothing: if everything looks like furniture, only page numbers go.
    """
    lines = text.split("\n")
    kept = _trim_edges(
        lines, lambda line: bool(_PAGE_NUMBER_LINE.match(line)) or _DIGITS.sub("#", line) in repeated
    )
    if not _meaningful_chars(kept):
        kept = _trim_edges(lines, lambda line: bool(_PAGE_NUMBER_LINE.match(line)))
    return kept if _meaningful_chars(kept) else text


def clean_pages(raw_pages: list[str]) -> list[str]:
    pages = [clean_text(page) for page in raw_pages]
    repeated = _repeated_edge_lines(pages)
    return [_strip_page_furniture(page, repeated) for page in pages]


def _meaningful_chars(text: str) -> int:
    return sum(1 for char in text if char.isalnum())


def _display_name(filename: str) -> str:
    name = PurePath(filename.replace("\\", "/")).name
    name = "".join(char for char in name if char.isprintable()).strip()
    return name[:255] or "document.pdf"


def _open(data: bytes) -> PdfReader:
    reader = PdfReader(BytesIO(data), strict=False)
    if reader.is_encrypted and reader.decrypt("") == PasswordType.NOT_DECRYPTED:
        raise PDFProcessingError("encrypted")
    return reader


def document_fingerprint(data: bytes) -> str:
    """The document ID: SHA-256 of the uploaded bytes, so the same file always gets the same ID."""
    return hashlib.sha256(data).hexdigest()


def extract_document(data: bytes, filename: str) -> ExtractedDocument:
    """Validate an upload and return its cleaned, 1-based page texts.

    Raises PDFProcessingError for empty, non-PDF, corrupt, password-protected, oversized and
    text-less (scanned) files.
    """
    if not data:
        raise PDFProcessingError("empty_file")
    if b"%PDF-" not in data[:1024]:
        raise PDFProcessingError("not_pdf")
    try:
        reader = _open(data)
        page_count = len(reader.pages)
    except PDFProcessingError:
        raise
    except Exception as exc:  # pypdf raises a variety of errors for damaged files
        raise PDFProcessingError("corrupt") from exc

    if page_count == 0:
        raise PDFProcessingError("no_pages")
    if page_count > MAX_PAGES:
        raise PDFProcessingError("too_many_pages", pages=page_count, limit=MAX_PAGES)

    raw_pages: list[str] = []
    failures = 0
    for page in reader.pages:
        try:
            raw_pages.append(page.extract_text() or "")
        except Exception:  # one unreadable page shouldn't sink the document
            failures += 1
            raw_pages.append("")
    if failures == page_count:
        raise PDFProcessingError("corrupt")

    texts = clean_pages(raw_pages)
    if sum(_meaningful_chars(text) for text in texts) < MIN_TEXT_CHARS:
        raise PDFProcessingError("no_text")

    return ExtractedDocument(
        document_id=document_fingerprint(data),
        filename=_display_name(filename),
        pages=[PageText(page_number=number, text=text) for number, text in enumerate(texts, start=1)],
    )
