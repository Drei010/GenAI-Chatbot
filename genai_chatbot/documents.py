"""What the PDF and Word readers share: the problems an upload can have, worded for the person who
uploaded it, the document limits, and helpers for display names and document IDs."""

from __future__ import annotations

import hashlib
from pathlib import PurePath
from typing import Literal

MAX_PAGES = 500
# A whole document with fewer letters and digits than this has no usable text.
MIN_TEXT_CHARS = 20

FileFormat = Literal["pdf", "docx"]
ProblemKind = Literal[
    "empty_file",
    "unsupported",
    "legacy_doc",
    "corrupt",
    "encrypted",
    "too_large",
    "no_pages",
    "too_many_pages",
    "too_long",
    "no_text",
]

_FORMAT_NAMES: dict[FileFormat, str] = {"pdf": "PDF", "docx": "Word document"}

_MESSAGES: dict[ProblemKind, str] = {
    "empty_file": "The file is empty. Choose a PDF or Word document that has content.",
    "unsupported": (
        "This file isn't a PDF or a Word (.docx) document. Upload a file saved in one of those formats."
    ),
    "legacy_doc": (
        "This is an older Word (.doc) file, which can't be read here. Open it in Word and save it as .docx "
        "or PDF, then upload it again."
    ),
    "corrupt": (
        "This {document} couldn't be read. It may be damaged or incomplete; try downloading or exporting it "
        "again."
    ),
    "encrypted": "This {document} is password-protected. Remove the password, then upload it again.",
    "too_large": (
        "This {document} is too large to read here. Split it into smaller files and upload them one at a "
        "time."
    ),
    "no_pages": "This {document} has no pages.",
    "too_many_pages": "This {document} has {pages} pages. The limit is {limit} pages per document.",
    "too_long": (
        "This {document} is too long: it holds about {pages} pages of text. The limit is {limit} pages per "
        "document."
    ),
    "no_text": (
        "No selectable text was found in this PDF. It looks like a scanned or image-only document, "
        "and OCR (reading text from images) isn't supported in this version."
    ),
}
# Where a Word document needs different wording.
_WORD_MESSAGES: dict[ProblemKind, str] = {
    "corrupt": (
        "This Word document couldn't be read. It may be damaged or incomplete; try downloading or saving it "
        "again."
    ),
    "no_text": (
        "No text was found in this Word document. Text inside pictures isn't read, and OCR (reading text "
        "from images) isn't supported in this version."
    ),
}


class DocumentError(Exception):
    """A problem with an uploaded file, worded for the person who uploaded it."""

    def __init__(self, kind: ProblemKind, file_format: FileFormat = "pdf", **details: object) -> None:
        self.kind = kind
        self.file_format = file_format
        template = _WORD_MESSAGES.get(kind) if file_format == "docx" else None
        self.message = (template or _MESSAGES[kind]).format(document=_FORMAT_NAMES[file_format], **details)
        super().__init__(self.message)


def meaningful_chars(text: str) -> int:
    return sum(1 for char in text if char.isalnum())


def display_name(filename: str, fallback: str) -> str:
    """The file's own name without folders or control characters, as shown in the app and in sources."""
    name = PurePath(filename.replace("\\", "/")).name
    name = "".join(char for char in name if char.isprintable()).strip()
    return name[:255] or fallback


def document_fingerprint(data: bytes) -> str:
    """The document ID: SHA-256 of the uploaded bytes, so the same file always gets the same ID."""
    return hashlib.sha256(data).hexdigest()
