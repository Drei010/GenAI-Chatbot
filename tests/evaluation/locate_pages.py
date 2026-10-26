"""Find the pages of the sample report that support each evaluation question, and check the set against it.

Every answerable question lists evidence: short phrases copied from the report. A page is an expected
source when it contains one of them, and each expected answer point must appear on one of those pages,
so the facts in the question set are checked against the report itself rather than taken on trust.

    uv run python -m tests.evaluation.locate_pages          # check and show the pages
    uv run python -m tests.evaluation.locate_pages --write  # also save the pages to the question set
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from genai_chatbot.extraction import extract_document
from genai_chatbot.schemas import PageText
from tests.evaluation.harness import EvalItem, contains, load_items, point_found, save_items
from tests.support.paths import SAMPLE_REPORT

QUESTIONS = Path(__file__).with_name("annual_report_questions.jsonl")
# More pages than this for one question usually means an evidence phrase is too common to be useful.
MANY_PAGES = 8


@dataclass(frozen=True)
class Located:
    item_id: str
    pages: list[int]
    missing_evidence: list[str]
    unverified_points: list[list[str]]


def locate(items: Sequence[EvalItem], pages: Sequence[PageText]) -> list[Located]:
    """The pages holding each answerable question's evidence, and anything that couldn't be matched."""
    located = []
    for item in items:
        if item.expect != "answer":
            continue
        found: set[int] = set()
        missing = []
        for phrase in item.evidence:
            hits = {page.page_number for page in pages if contains(page.text, phrase)}
            if not hits:
                missing.append(phrase)
            found |= hits
        source_text = "\n".join(page.text for page in pages if page.page_number in found)
        unverified = [point for point in item.answer_points if not point_found(source_text, point)]
        located.append(Located(item.id, sorted(found), missing, unverified))
    return located


def problems(located: Sequence[Located]) -> list[str]:
    found = []
    for entry in located:
        found += [f"{entry.item_id}: evidence not found: {phrase!r}" for phrase in entry.missing_evidence]
        found += [
            f"{entry.item_id}: answer point not on its pages: {' / '.join(point)}"
            for point in entry.unverified_points
        ]
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pdf", type=Path, default=SAMPLE_REPORT, help="the sample report")
    parser.add_argument("--questions", type=Path, default=QUESTIONS, help="the JSONL question set")
    parser.add_argument("--write", action="store_true", help="save the located pages to the question set")
    args = parser.parse_args(argv)

    if not args.pdf.exists():
        print(f"{args.pdf} doesn't exist. Add the sample report first (see README, 'Sample document').")
        return 2
    items = load_items(args.questions)
    document = extract_document(args.pdf.read_bytes(), args.pdf.name)
    located = locate(items, document.pages)
    for entry in located:
        note = "  <- many pages; is an evidence phrase too common?" if len(entry.pages) > MANY_PAGES else ""
        print(f"{entry.item_id}: pages {', '.join(map(str, entry.pages)) or 'none'}{note}")
    issues = problems(located)
    for issue in issues:
        print(f"PROBLEM {issue}")

    if args.write:
        pages = {entry.item_id: entry.pages for entry in located}
        updated = [
            EvalItem.model_validate(item.model_dump() | {"source_pages": pages.get(item.id, [])})
            for item in items
        ]
        save_items(args.questions, updated)
        print(f"Saved source pages to {args.questions}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
