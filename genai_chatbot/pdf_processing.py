"""Read a PDF upload into clean text for each page."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from io import BytesIO

from pypdf import PasswordType, PdfReader

from genai_chatbot.documents import (
    MAX_PAGES,
    MIN_TEXT_CHARS,
    DocumentError,
    display_name,
    document_fingerprint,
    meaningful_chars,
)
from genai_chatbot.schemas import ExtractedDocument, PageText

# Lines repeated at the top or bottom of this share of pages are running headers or footers.
REPEATED_EDGE_SHARE = 0.6
EDGE_LINES = 2

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
    if not meaningful_chars(kept):
        kept = _trim_edges(lines, lambda line: bool(_PAGE_NUMBER_LINE.match(line)))
    return kept if meaningful_chars(kept) else text


def clean_pages(raw_pages: list[str]) -> list[str]:
    pages = [clean_text(page) for page in raw_pages]
    repeated = _repeated_edge_lines(pages)
    return [_strip_page_furniture(page, repeated) for page in pages]


def _open(data: bytes) -> PdfReader:
    reader = PdfReader(BytesIO(data), strict=False)
    if reader.is_encrypted and reader.decrypt("") == PasswordType.NOT_DECRYPTED:
        raise DocumentError("encrypted")
    return reader


def extract_pdf(data: bytes, filename: str) -> ExtractedDocument:
    """Return a PDF's cleaned, 1-based page texts. `extraction.extract_document` checks first that the
    upload is a PDF.

    Raises DocumentError for corrupt, password-protected, oversized and text-less (scanned) files.
    """
    try:
        reader = _open(data)
        page_count = len(reader.pages)
    except DocumentError:
        raise
    except Exception as exc:  # pypdf raises a variety of errors for damaged files
        raise DocumentError("corrupt") from exc

    if page_count == 0:
        raise DocumentError("no_pages")
    if page_count > MAX_PAGES:
        raise DocumentError("too_many_pages", pages=page_count, limit=MAX_PAGES)

    raw_pages: list[str] = []
    failures = 0
    for page in reader.pages:
        try:
            raw_pages.append(page.extract_text() or "")
        except Exception:  # one unreadable page shouldn't sink the document
            failures += 1
            raw_pages.append("")
    if failures == page_count:
        raise DocumentError("corrupt")

    texts = clean_pages(raw_pages)
    if sum(meaningful_chars(text) for text in texts) < MIN_TEXT_CHARS:
        raise DocumentError("no_text")

    return ExtractedDocument(
        document_id=document_fingerprint(data),
        filename=display_name(filename, "document.pdf"),
        pages=[PageText(page_number=number, text=text) for number, text in enumerate(texts, start=1)],
    )
