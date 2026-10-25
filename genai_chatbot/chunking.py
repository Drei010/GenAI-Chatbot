"""Classify document complexity and split pages into citation-ready chunks.

Every chunk stays inside one page, so each source can be cited with a single page number, and every
chunk is an exact substring of its page's cleaned text, so excerpts can be shown verbatim.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from langchain_text_splitters import RecursiveCharacterTextSplitter

from genai_chatbot.schemas import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    Chunk,
    ChunkStrategy,
    ComplexityAssessment,
    ExtractedDocument,
    SectionOutline,
)

COMPLEX_PAGE_THRESHOLD = 20
MIN_TABLE_LINES = 3
MAX_HEADING_CHARS = 100
MAX_OUTLINE_CANDIDATES = 400
MAX_OUTLINE_HEADINGS = 150
OUTLINE_TEXT_LIMIT = 100
MIN_CHUNK_CHARS = 3
MIN_PIECE_CHARS = 200

_NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(\S.*)$")
_PART = re.compile(r"^PART\s+[IVX]+\b")
_ITEM = re.compile(r"^ITEM\s+\d{1,2}[A-C]?\b", re.IGNORECASE)
_NUMERIC_TOKEN = re.compile(r"^[(\-–$€£]*\d[\d,.]*(?:%\)?|\)%?)?[,;]?$")
_SMALL_WORDS = frozenset(
    {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with", "&"}
)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    keep_separator="end",
)


class OutlineError(ValueError):
    """The model's section outline can't be applied to the document."""


@dataclass(frozen=True)
class HeadingCandidate:
    """A line that may start a section. `strong` candidates are trusted without a model."""

    candidate_id: str
    page_number: int
    offset: int
    text: str
    level: int | None
    strong: bool
    score: float = 1.0


@dataclass(frozen=True)
class Section:
    page_number: int
    start: int
    end: int
    title_path: tuple[str, ...]


# Line classification


def is_table_line(line: str) -> bool:
    """Rows of figures: at least three numeric tokens making up 40% or more of the line."""
    tokens = line.split()
    numeric = sum(1 for token in tokens if token in {"$", "%", "—", "-"} or _NUMERIC_TOKEN.match(token))
    figures = sum(1 for token in tokens if _NUMERIC_TOKEN.match(token))
    return figures >= 3 and numeric >= 0.4 * len(tokens)


def _letters(text: str) -> int:
    return sum(1 for char in text if char.isalpha())


def _strong_level(line: str) -> int | None:
    """Level implied by numbering or 10-K style markers, or None if the line isn't a strong heading."""
    if _PART.match(line):
        return 1
    if _ITEM.match(line):
        return 2
    numbered = _NUMBERED.match(line)
    if numbered:
        title = numbered.group(2)
        if title[0].isupper() and not title.endswith((".", ",", ";", ":")) and len(title.split()) <= 12:
            return numbered.group(1).count(".") + 1
        return None
    if line.isupper() and _letters(line) >= 4 and len(line.split()) <= 10 and not line.endswith((".", ",")):
        return 1
    return None


def _is_title_like(line: str) -> bool:
    """Short Title Case lines without figures: possible headings that need a model's judgment."""
    words = line.split()
    if not 1 <= len(words) <= 10 or line.endswith((".", ",", ";")) or any(char.isdigit() for char in line):
        return False
    if len(words) == 1:
        return words[0][0].isupper() and _letters(words[0]) >= 4
    content = [word for word in words if word.lower() not in _SMALL_WORDS]
    capitalized = sum(1 for word in content if word[0].isupper())
    return bool(content) and capitalized >= 0.6 * len(content)


def _lines_with_offsets(text: str) -> Iterable[tuple[int, str]]:
    offset = 0
    for line in text.split("\n"):
        yield offset, line
        offset += len(line) + 1


def detect_headings(document: ExtractedDocument, *, include_weak: bool = False) -> list[HeadingCandidate]:
    """Find heading-like lines in document order."""
    found: list[HeadingCandidate] = []
    for page in document.pages:
        lines = list(_lines_with_offsets(page.text))
        for index, (offset, line) in enumerate(lines):
            if not line or len(line) > MAX_HEADING_CHARS or is_table_line(line):
                continue
            level = _strong_level(line)
            if level is not None:
                found.append(
                    HeadingCandidate("", page.page_number, offset, line, level, strong=True, score=2.0)
                )
            elif include_weak and _is_title_like(line):
                following = lines[index + 1][1] if index + 1 < len(lines) else ""
                score = 1.5 if len(following) > 60 else 1.0
                found.append(
                    HeadingCandidate("", page.page_number, offset, line, None, strong=False, score=score)
                )
    return [replace(candidate, candidate_id=f"h{n}") for n, candidate in enumerate(found, start=1)]


# Complexity


def assess_complexity(document: ExtractedDocument) -> ComplexityAssessment:
    """Complex means more than 20 pages, table-like content, or a multi-level section hierarchy."""
    table_lines = sum(1 for page in document.pages for line in page.text.split("\n") if is_table_line(line))
    headings = detect_headings(document)
    levels = {heading.level for heading in headings if heading.level is not None}

    reasons = []
    if document.page_count > COMPLEX_PAGE_THRESHOLD:
        reasons.append(f"more than {COMPLEX_PAGE_THRESHOLD} pages")
    if table_lines >= MIN_TABLE_LINES:
        reasons.append("table-like content")
    if len(levels) >= 2 and len(headings) >= 3:
        reasons.append("section hierarchy")
    return ComplexityAssessment(
        is_complex=bool(reasons),
        reasons=reasons,
        page_count=document.page_count,
        table_line_count=table_lines,
        heading_count=len(headings),
        heading_levels=len(levels),
    )


# Sections and chunks


def build_sections(document: ExtractedDocument, boundaries: Sequence[HeadingCandidate]) -> list[Section]:
    """Split pages at heading lines. A section continues across pages until the next heading; a
    heading with no body before the next heading on its page is merged into the section that follows."""
    by_page: dict[int, list[HeadingCandidate]] = defaultdict(list)
    for boundary in boundaries:
        by_page[boundary.page_number].append(boundary)

    stack: list[tuple[int, str]] = []
    sections: list[Section] = []
    for page in document.pages:
        text = page.text
        if not text.strip():
            continue
        marks = sorted(by_page.get(page.page_number, []), key=lambda b: b.offset)
        starts: list[tuple[int, HeadingCandidate | None]] = []
        if not marks or marks[0].offset > 0:
            starts.append((0, None))
        starts += [(mark.offset, mark) for mark in marks]

        pending: int | None = None
        for index, (offset, boundary) in enumerate(starts):
            end = starts[index + 1][0] if index + 1 < len(starts) else len(text)
            if boundary is not None:
                level = boundary.level or 1
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, boundary.text.strip()))
            start = offset if pending is None else pending
            heading_only = boundary is not None and text[offset:end].strip() == boundary.text.strip()
            if heading_only and index + 1 < len(starts):
                pending = start
                continue
            pending = None
            sections.append(Section(page.page_number, start, end, tuple(title for _, title in stack)))
    return sections


def _split(text: str) -> list[str]:
    """Split a section into pieces of at most CHUNK_SIZE characters, folding tiny pieces into a neighbor.

    The recursive splitter emits a short piece (often a lone heading line) whenever the next split is
    too long to merge, so small pieces are widened to cover their neighbor when the result still fits.
    """
    stripped = text.strip()
    if len(stripped) <= CHUNK_SIZE:
        return [stripped]
    spans: list[tuple[int, int]] = []
    cursor = 0
    for piece in _splitter.split_text(stripped):
        start = stripped.find(piece, cursor)
        spans.append((start, start + len(piece)))
        cursor = start + 1
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and merged[-1][1] - merged[-1][0] < MIN_PIECE_CHARS and end - merged[-1][0] <= CHUNK_SIZE:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    if len(merged) > 1 and merged[-1][1] - merged[-1][0] < MIN_PIECE_CHARS:
        last = merged.pop()
        if last[1] - merged[-1][0] <= CHUNK_SIZE:
            merged[-1] = (merged[-1][0], last[1])
        else:
            merged.append(last)
    return [stripped[start:end].strip() for start, end in merged]


def chunk_sections(
    document: ExtractedDocument, sections: Sequence[Section], strategy: ChunkStrategy
) -> list[Chunk]:
    """Keep sections up to CHUNK_SIZE intact; split longer ones recursively with overlap."""
    chunks: list[Chunk] = []
    per_page: dict[int, int] = defaultdict(int)
    for section in sections:
        page_text = document.page(section.page_number).text
        title = " > ".join(section.title_path) or None
        for piece in _split(page_text[section.start : section.end]):
            if sum(1 for char in piece if char.isalnum()) < MIN_CHUNK_CHARS:
                continue
            per_page[section.page_number] += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{document.document_id[:16]}-p{section.page_number:04d}-c{per_page[section.page_number]:03d}",
                    document_id=document.document_id,
                    filename=document.filename,
                    page_number=section.page_number,
                    text=piece,
                    section_title=title,
                    strategy=strategy,
                )
            )
    return chunks


def recursive_chunks(document: ExtractedDocument) -> list[Chunk]:
    """Deterministic chunking: sections from strong headings only, then the recursive splitter."""
    return chunk_sections(document, build_sections(document, detect_headings(document)), "recursive")


# Model-assisted chunking for complex documents


def outline_candidates(
    document: ExtractedDocument, limit: int = MAX_OUTLINE_CANDIDATES
) -> list[HeadingCandidate]:
    """Heading candidates to show the model, bounded to the `limit` most heading-like lines."""
    candidates = detect_headings(document, include_weak=True)
    if len(candidates) > limit:
        keep = sorted(range(len(candidates)), key=lambda i: (-candidates[i].score, i))[:limit]
        candidates = [candidates[i] for i in sorted(keep)]
    return [replace(candidate, candidate_id=f"h{n}") for n, candidate in enumerate(candidates, start=1)]


def apply_outline(
    document: ExtractedDocument, candidates: Sequence[HeadingCandidate], outline: SectionOutline
) -> list[Chunk]:
    """Chunk along the headings the model picked. Unknown IDs mean the output can't be trusted."""
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    unknown = [heading.candidate_id for heading in outline.headings if heading.candidate_id not in by_id]
    if unknown:
        raise OutlineError(f"{len(unknown)} heading id(s) are not in the candidate list")
    chosen: dict[str, HeadingCandidate] = {}
    for heading in outline.headings:
        chosen.setdefault(heading.candidate_id, replace(by_id[heading.candidate_id], level=heading.level))
    boundaries = sorted(chosen.values(), key=lambda b: (b.page_number, b.offset))
    return chunk_sections(document, build_sections(document, boundaries), "model_assisted")


def validate_chunks(document: ExtractedDocument, chunks: Sequence[Chunk]) -> list[str]:
    """Problems that make chunks unsafe to index; an empty list means they're fine."""
    if not chunks:
        return ["no chunks were produced"]
    problems: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            problems.append(f"duplicate chunk id {chunk.chunk_id}")
        seen.add(chunk.chunk_id)
        if chunk.document_id != document.document_id or chunk.filename != document.filename:
            problems.append(f"{chunk.chunk_id} belongs to another document")
            continue
        if not 1 <= chunk.page_number <= document.page_count:
            problems.append(f"{chunk.chunk_id} has page {chunk.page_number} outside 1-{document.page_count}")
        elif chunk.text not in document.page(chunk.page_number).text:
            problems.append(f"{chunk.chunk_id} text does not match page {chunk.page_number}")
    return problems
