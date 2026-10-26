"""Turn the model's passage IDs into sources a reader can check.

The model only ever sees passage IDs (P1, P2, ...). Everything a source shows (filename, page or
section, and excerpt) is copied from the retrieved passage, never taken from model output.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from genai_chatbot.schemas import AnswerEvidence, Citation, RetrievedPassage

EXCERPT_CHARS = 240
SHORT_EXCERPT_CHARS = 60

# [P2], [P1, P3] or (P2); the app renumbers them as [1], [1, 3] for display.
MARKER = re.compile(r"[\[(]\s*(P\d+(?:\s*[,;]\s*P\d+)*)\s*[\])]", re.IGNORECASE)
# A sentence ends after a word, a percentage, a closing bracket or a four-digit year, so numbered
# headings ("1. Overview") and decimals stay whole.
_SENTENCE_END = re.compile(r"(?:(?<=[a-z%)\]][.!?])|(?<=\d{4}[.!?]))\s+(?=[A-Z0-9$(\"“])")
_TOKEN = re.compile(r"[a-z]+|\d[\d,.]*\d|\d")
# fmt: off
_STOPWORDS = frozenset({
    "the", "and", "for", "was", "were", "are", "with", "from", "that", "this", "its", "their", "our",
    "has", "have", "had", "per", "which", "into", "than", "also", "been", "over", "about", "during", "each",
})
# fmt: on


class CitationError(ValueError):
    """The answer cites nothing, or cites passages that weren't retrieved."""


@dataclass(frozen=True)
class ResolvedAnswer:
    text: str
    citations: list[Citation]


def passage_labels(passages: Sequence[RetrievedPassage]) -> dict[str, RetrievedPassage]:
    """Short IDs the model uses to cite passages, in retrieval order."""
    return {f"P{number}": passage for number, passage in enumerate(passages, start=1)}


def _ids_in(marker: re.Match[str]) -> list[str]:
    ids: list[str] = []
    for part in re.split(r"[,;]", marker.group(1)):
        label = part.strip().upper()
        if label not in ids:
            ids.append(label)
    return ids


def _normalize(label: str) -> str:
    return label.strip().strip("[]()").strip().upper()


def cited_labels(answer: str, citation_ids: Sequence[str]) -> list[str]:
    """Every cited label, in order of first use: inline markers first, then the listed IDs."""
    labels: list[str] = []
    for marker in MARKER.finditer(answer):
        labels += [label for label in _ids_in(marker) if label not in labels]
    for raw in citation_ids:
        label = _normalize(raw)
        if label and label not in labels:
            labels.append(label)
    return labels


def answer_claims(answer: str) -> list[str]:
    """Factual answer units that need evidence: sentences, bullets and table rows."""
    claims: list[str] = []
    for line in answer.splitlines():
        line = line.strip()
        if not line or (line.startswith("|") and re.fullmatch(r"\|[\s|:-]+\|", line)):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if line.startswith("|") and cells[-1].lower() == "source" and not MARKER.search(line):
            continue
        claims.extend([line] if line.startswith("|") else _SENTENCE_END.split(line))
    return [claim.strip() for claim in claims if claim.strip()]


def validate_answer_evidence(
    answer: str, evidence: Sequence[AnswerEvidence], passages: Mapping[str, RetrievedPassage]
) -> None:
    """Require each factual answer unit to cite passage text copied exactly from that passage."""
    required: set[tuple[str, str]] = set()
    for claim in answer_claims(answer):
        labels = [_normalize(label) for marker in MARKER.finditer(claim) for label in _ids_in(marker)]
        if not labels:
            raise CitationError("a factual claim has no inline citation")
        required.update((" ".join(claim.split()), label) for label in labels)

    supplied = {(" ".join(item.claim.split()), item.passage_id.upper()) for item in evidence}
    if supplied != required:
        raise CitationError("claim evidence is missing or does not match the answer")
    for item in evidence:
        passage = passages.get(item.passage_id.upper())
        if passage is None or item.quote.strip() not in passage.text:
            raise CitationError("an evidence quote is not copied from its cited passage")


def _segments(answer: str) -> list[str]:
    """Lines of the answer, with prose lines split into sentences: the units a citation supports."""
    pieces: list[str] = []
    for line in answer.splitlines():
        pieces += _SENTENCE_END.split(line) if not line.lstrip().startswith("|") else [line]
    return [piece for piece in pieces if piece.strip()]


def _claims_citing(answer: str, label: str) -> str:
    claims = [
        MARKER.sub(" ", segment)
        for segment in _segments(answer)
        if any(label in _ids_in(marker) for marker in MARKER.finditer(segment))
    ]
    return " ".join(claims) if claims else MARKER.sub(" ", answer)


def _tokens(text: str) -> set[str]:
    found = set()
    for token in _TOKEN.findall(text.lower()):
        if token[0].isdigit():
            found.add(token.replace(",", "").rstrip("."))
        elif len(token) >= 3 and token not in _STOPWORDS:
            found.add(token)
    return found


def _units(text: str) -> list[tuple[int, int]]:
    """Spans of sentences and table lines in the passage text.

    Lines are joined first where a sentence wraps onto the next line (the next line starts in lower
    case), so a wrapped sentence stays one unit while table rows and headings stay separate.
    """
    blocks: list[list[int]] = []
    offset = 0
    previous = ""
    for line in text.split("\n"):
        if line.strip():
            wraps = previous and not previous.rstrip().endswith((".", "!", "?", ":")) and line[0].islower()
            if blocks and wraps:
                blocks[-1][1] = offset + len(line)
            else:
                blocks.append([offset, offset + len(line)])
            previous = line
        offset += len(line) + 1
    spans: list[tuple[int, int]] = []
    for start, end in blocks:
        cursor = start
        for boundary in _SENTENCE_END.finditer(text, start, end):
            spans.append((cursor, boundary.start()))
            cursor = boundary.end()
        spans.append((cursor, end))
    return spans


def select_excerpt(passage_text: str, claims: str) -> str:
    """The passage sentence (or table line) that best matches the claims, copied from the passage.

    Matching figures count three times as much as matching words. A short fragment (a heading or a
    table row rather than a sentence) also takes the next unit, and anything longer than EXCERPT_CHARS
    is cut at a word boundary and ends with an ellipsis. Line breaks become spaces.
    """
    units = _units(passage_text) or [(0, len(passage_text))]
    wanted = _tokens(claims)

    def score(span: tuple[int, int]) -> int:
        overlap = _tokens(passage_text[span[0] : span[1]]) & wanted
        return sum(3 if token[0].isdigit() else 1 for token in overlap)

    best = max(range(len(units)), key=lambda i: (score(units[i]), -i))
    start, end = units[best]
    is_fragment = not passage_text[start:end].rstrip().endswith((".", "!", "?"))
    if is_fragment and end - start < SHORT_EXCERPT_CHARS and best + 1 < len(units):
        following_end = units[best + 1][1]
        if following_end - start <= EXCERPT_CHARS:
            end = following_end
    truncated = end - start > EXCERPT_CHARS
    if truncated:
        cut = passage_text.rfind(" ", start, start + EXCERPT_CHARS)
        end = cut if cut > start else start + EXCERPT_CHARS
    excerpt = " ".join(passage_text[start:end].split())
    return f"{excerpt}…" if truncated else excerpt


def resolve_citations(
    answer: str, citation_ids: Sequence[str], passages: Mapping[str, RetrievedPassage]
) -> ResolvedAnswer:
    """Check every cited label against the retrieved passages and build numbered sources.

    Inline markers become [1], [2]... in order of first use. Raises CitationError when nothing is cited
    or when any label doesn't belong to a retrieved passage.
    """
    labels = cited_labels(answer, citation_ids)
    if not labels:
        raise CitationError("the answer cites no passages")
    unknown = [label for label in labels if label not in passages]
    if unknown:
        raise CitationError(f"{len(unknown)} cited label(s) were not retrieved")

    numbers = {label: number for number, label in enumerate(labels, start=1)}
    text = MARKER.sub(
        lambda marker: "[" + ", ".join(str(numbers[label]) for label in _ids_in(marker)) + "]", answer
    )
    citations = [
        Citation(
            number=numbers[label],
            chunk_id=passages[label].chunk_id,
            filename=passages[label].filename,
            page_number=passages[label].page_number,
            excerpt=select_excerpt(passages[label].text, _claims_citing(answer, label)),
            section_title=passages[label].section_title,
        )
        for label in labels
    ]
    return ResolvedAnswer(text=text.strip(), citations=citations)
