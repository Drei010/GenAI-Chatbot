"""Citation parsing, validation, renumbering and excerpt selection."""

from __future__ import annotations

import pytest

from genai_chatbot.citations import (
    EXCERPT_CHARS,
    CitationError,
    cited_labels,
    passage_labels,
    resolve_citations,
    select_excerpt,
)
from genai_chatbot.schemas import RetrievedPassage
from tests.support.documents import REPORT_PAGES

PAGE_2 = "\n".join(REPORT_PAGES[2])
PAGE_3 = "\n".join(REPORT_PAGES[3])


def passage(chunk_id: str, page: int, text: str) -> RetrievedPassage:
    return RetrievedPassage(chunk_id=chunk_id, filename="report.pdf", page_number=page, text=text, score=0.5)


@pytest.fixture
def passages() -> dict[str, RetrievedPassage]:
    return passage_labels([passage("c-2", 2, PAGE_2), passage("c-3", 3, PAGE_3)])


def test_passages_are_labelled_in_retrieval_order():
    labels = passage_labels([passage("a", 1, "First."), passage("b", 2, "Second.")])

    assert {label: p.chunk_id for label, p in labels.items()} == {"P1": "a", "P2": "b"}


@pytest.mark.parametrize(
    ("answer", "listed", "expected"),
    [
        ("Revenue grew [P2]. Staff rose [P1].", [], ["P2", "P1"]),
        ("Both facts [P1, P2] and again [p2; P1].", [], ["P1", "P2"]),
        ("Revenue grew (P2).", ["[P1]", " p2 ", ""], ["P2", "P1"]),
        ("Twice in one marker [P1, P1].", [], ["P1"]),
        ("No markers at all.", [], []),
    ],
)
def test_cited_labels_are_collected_in_order_of_first_use(answer, listed, expected):
    assert cited_labels(answer, listed) == expected


def test_markers_are_renumbered_by_first_use(passages):
    resolved = resolve_citations(
        "Northwind employed 2,340 people [P2]. Revenue was $412.6 million [P1]. Both appear in [P1, P2].",
        ["P1", "P2"],
        passages,
    )

    assert resolved.text == (
        "Northwind employed 2,340 people [1]. Revenue was $412.6 million [2]. Both appear in [2, 1]."
    )
    assert [(c.number, c.chunk_id, c.page_number, c.filename) for c in resolved.citations] == [
        (1, "c-3", 3, "report.pdf"),
        (2, "c-2", 2, "report.pdf"),
    ]


def test_sources_carry_the_section_of_their_passage():
    sectioned = RetrievedPassage(
        chunk_id="w-1",
        filename="report.docx",
        page_number=1,
        text=PAGE_3,
        section_title="Our People",
        score=0.5,
    )

    resolved = resolve_citations("Northwind employed 2,340 people [P1].", ["P1"], passage_labels([sectioned]))

    assert resolved.citations[0].section_title == "Our People"


def test_answers_citing_nothing_or_unknown_passages_are_rejected(passages):
    with pytest.raises(CitationError, match="no passages"):
        resolve_citations("Revenue was $412.6 million.", [], passages)
    with pytest.raises(CitationError, match="1 cited label"):
        resolve_citations("Revenue was $412.6 million [P3].", [], passages)


def test_the_excerpt_is_the_sentence_that_supports_the_claim(passages):
    resolved = resolve_citations("Operating income was $58.3 million in fiscal year 2024 [P1].", [], passages)

    assert resolved.citations[0].excerpt == (
        "Operating income was $58.3 million compared with $49.9 million in fiscal year 2023."
    )


def test_a_sentence_wrapped_across_lines_is_one_excerpt():
    excerpt = select_excerpt(PAGE_2, "revenue grew 9% from $378.5 million")

    assert excerpt == (
        "Revenue for fiscal year 2024 was $412.6 million, an increase of 9% from $378.5 million in fiscal "
        "year 2023."
    )


def test_a_short_match_such_as_a_table_row_takes_the_next_line():
    assert select_excerpt(PAGE_2, "Subscriptions $96.7 million") == (
        "Subscriptions $ 96.7 $ 79.2 22% Wholesale $ 64.5 $ 63.3 2%"
    )


def test_a_short_match_stays_alone_when_the_next_unit_is_too_long():
    long_sentence = "Revenue details follow " + "and more " * 40 + "end."
    text = f"Revenue by Segment\n{long_sentence}"

    assert select_excerpt(text, "segment") == "Revenue by Segment"


def test_long_excerpts_are_cut_at_a_word_and_marked():
    sentence = "Revenue " + " ".join(f"word{n}" for n in range(80)) + "."

    excerpt = select_excerpt(sentence, "revenue")

    assert excerpt.endswith("…") and len(excerpt) <= EXCERPT_CHARS + 1
    assert sentence.startswith(excerpt[:-1])
    assert not excerpt[:-1].endswith(" ")


def test_without_any_overlap_the_excerpt_is_the_passage_opening():
    assert select_excerpt("First sentence here.\n\nSecond sentence there.", "zebra") == "First sentence here."


def test_numbered_headings_and_decimals_do_not_split_sentences():
    text = "1. Overview\nNet income was $41.2 million, or $2.06 per diluted share. Margins rose."

    assert select_excerpt(text, "diluted share 2.06") == (
        "Net income was $41.2 million, or $2.06 per diluted share."
    )


def test_every_excerpt_is_copied_from_its_passage(passages):
    answer = (
        "| Segment | 2024 | Source |\n|---|---|---|\n| Retail stores | $251.4 million | [P1] |\n"
        "Staff: 2,340 [P2]."
    )

    resolved = resolve_citations(answer, [], passages)

    for citation in resolved.citations:
        source = passages[f"P{1 if citation.chunk_id == 'c-2' else 2}"].text
        assert citation.excerpt.rstrip("…") in " ".join(source.split())
