"""Complexity routing, deterministic chunking, and applying a model's section outline."""

from __future__ import annotations

import itertools
import re

import pytest
from pydantic import ValidationError

from genai_chatbot.chunking import (
    OutlineError,
    apply_outline,
    assess_complexity,
    detect_headings,
    is_table_line,
    outline_candidates,
    recursive_chunks,
    validate_chunks,
)
from genai_chatbot.schemas import CHUNK_OVERLAP, CHUNK_SIZE, Chunk, OutlineHeading, SectionOutline
from tests.support.documents import document_from_pages
from tests.support.pdfs import lorem_paragraph

PROSE = lorem_paragraph("coffee")


def outline(*headings: tuple[str, int]) -> SectionOutline:
    return SectionOutline(
        headings=[OutlineHeading(candidate_id=candidate_id, level=level) for candidate_id, level in headings]
    )


def span(page_text: str, chunk: Chunk) -> tuple[int, int]:
    start = page_text.find(chunk.text)
    assert start >= 0, "chunk text must be an exact substring of its page"
    return start, start + len(chunk.text)


# Line classification


@pytest.mark.parametrize(
    "line",
    [
        "Retail stores $ 251.4 $ 236.0 7%",
        "Revenue 211,915 198,270 7%",
        "Net margin (1.2) (0.8) (1)%",
        "2024 2023 2022",
    ],
)
def test_rows_of_figures_are_table_lines(line):
    assert is_table_line(line)


@pytest.mark.parametrize(
    "line",
    [
        "Revenue for fiscal year 2024 was $412.6 million, an increase of 9% from $378.5 million in fiscal",
        "Our fiscal year ends on June 30, 2024.",
        "Retail stores",
        "",
    ],
)
def test_prose_with_a_few_numbers_is_not_a_table_line(line):
    assert not is_table_line(line)


def test_numbered_parts_items_and_capitalized_lines_are_strong_headings():
    document = document_from_pages(
        "\n".join(
            [
                "PART II",
                "Item 7. Management's Discussion and Analysis",
                "LETTER TO SHAREHOLDERS",
                "1. Financial Highlights",
                "1.1 Revenue by Segment",
                "2.1.3 Cloud Services",
                PROSE,
            ]
        )
    )

    found = [(heading.text, heading.level, heading.strong) for heading in detect_headings(document)]

    assert found == [
        ("PART II", 1, True),
        ("Item 7. Management's Discussion and Analysis", 2, True),
        ("LETTER TO SHAREHOLDERS", 1, True),
        ("1. Financial Highlights", 1, True),
        ("1.1 Revenue by Segment", 2, True),
        ("2.1.3 Cloud Services", 3, True),
    ]


@pytest.mark.parametrize(
    "line",
    [
        "1. we roast coffee in small batches",
        "3. The company roasted 41 million pounds of coffee across its three roasteries in many states",
        "2. Revenue grew in every segment.",
        "Revenue grew in every segment this year.",
        "1. " + "Very Long Heading " * 8,
    ],
)
def test_sentences_and_list_items_are_not_strong_headings(line):
    assert detect_headings(document_from_pages(f"{line}\n{PROSE}")) == []


def test_title_case_lines_are_weak_candidates_only_when_asked_for():
    document = document_from_pages(f"Our Approach to Sourcing\n{PROSE}")

    assert detect_headings(document) == []
    (weak,) = detect_headings(document, include_weak=True)
    assert (weak.text, weak.level, weak.strong) == ("Our Approach to Sourcing", None, False)


def test_single_words_need_four_letters_to_be_weak_candidates():
    document = document_from_pages(f"Overview\n{PROSE}\nTea\n{PROSE}")

    assert [c.text for c in detect_headings(document, include_weak=True)] == ["Overview"]


def test_heading_ids_follow_document_order_across_pages():
    document = document_from_pages(f"1. First\n{PROSE}", f"2. Second\n{PROSE}\n3. Third\n{PROSE}")

    headings = detect_headings(document)

    assert [(h.candidate_id, h.page_number, h.text) for h in headings] == [
        ("h1", 1, "1. First"),
        ("h2", 2, "2. Second"),
        ("h3", 2, "3. Third"),
    ]
    assert document.page(2).text[headings[2].offset :].startswith("3. Third")


# Complexity routing


def test_short_prose_document_is_simple():
    assessment = assess_complexity(document_from_pages(PROSE, lorem_paragraph("tea")))

    assert not assessment.is_complex
    assert assessment.reasons == []
    assert assessment.page_count == 2


def test_more_than_twenty_pages_is_complex():
    twenty = assess_complexity(document_from_pages(*[lorem_paragraph(f"page {n}") for n in range(20)]))
    twenty_one = assess_complexity(document_from_pages(*[lorem_paragraph(f"page {n}") for n in range(21)]))

    assert not twenty.is_complex
    assert twenty_one.is_complex
    assert twenty_one.reasons == ["more than 20 pages"]


def test_table_like_content_is_complex():
    rows = ["Retail $ 251.4 $ 236.0 7%", "Online $ 96.7 $ 79.2 22%", "Wholesale $ 64.5 $ 63.3 2%"]

    two_rows = assess_complexity(document_from_pages("\n".join([PROSE, *rows[:2]])))
    three_rows = assess_complexity(document_from_pages("\n".join([PROSE, *rows])))

    assert not two_rows.is_complex
    assert three_rows.is_complex
    assert three_rows.reasons == ["table-like content"]
    assert three_rows.table_line_count == 3


def test_multi_level_section_hierarchy_is_complex():
    flat = document_from_pages(f"1. Overview\n{PROSE}\n2. Results\n{PROSE}\n3. Outlook\n{PROSE}")
    nested = document_from_pages(f"1. Overview\n{PROSE}\n1.1 Scope\n{PROSE}\n2. Results\n{PROSE}")

    assert not assess_complexity(flat).is_complex
    assessment = assess_complexity(nested)
    assert assessment.is_complex
    assert assessment.reasons == ["section hierarchy"]
    assert (assessment.heading_count, assessment.heading_levels) == (3, 2)


def test_every_reason_is_reported(northwind_long):
    assessment = assess_complexity(northwind_long)

    assert assessment.reasons == ["more than 20 pages", "table-like content", "section hierarchy"]


# Deterministic (recursive) chunking


def test_sections_up_to_the_chunk_size_stay_intact():
    overview = f"1. Overview\n{PROSE}"
    details = f"2. Details\n{lorem_paragraph('tea')}"
    document = document_from_pages(f"{overview}\n{details}")

    chunks = recursive_chunks(document)

    assert [chunk.text for chunk in chunks] == [overview, details]
    assert [chunk.section_title for chunk in chunks] == ["1. Overview", "2. Details"]
    assert {chunk.strategy for chunk in chunks} == {"recursive"}


def test_long_sections_are_split_with_overlap_and_nothing_is_lost():
    sentences = [lorem_paragraph(f"topic {n}") for n in range(6)]
    page = "1. Long Section\n" + " ".join(sentences)
    document = document_from_pages(page)

    chunks = recursive_chunks(document)

    assert len(chunks) >= 2
    assert all(len(chunk.text) <= CHUNK_SIZE for chunk in chunks)
    spans = [span(page, chunk) for chunk in chunks]
    for (_, previous_end), (next_start, _) in itertools.pairwise(spans):
        assert 0 < previous_end - next_start <= CHUNK_OVERLAP
    for sentence in re.split(r"(?<=\.) ", " ".join(sentences)):
        assert any(sentence in chunk.text for chunk in chunks)
    assert {chunk.section_title for chunk in chunks} == {"1. Long Section"}


def test_a_heading_is_not_left_alone_in_its_own_chunk():
    paragraphs = "\n".join(lorem_paragraph(f"area {n}", sentences=4) for n in range(8))
    document = document_from_pages(f"2. Another Section\n{paragraphs}")

    chunks = recursive_chunks(document)

    assert len(chunks) >= 2
    assert chunks[0].text.startswith("2. Another Section\nThe area 0 program")


def test_a_short_closing_paragraph_joins_the_passage_before_it():
    body = " ".join(lorem_paragraph(f"topic {n}") for n in range(6))[:1500]
    body = body[: body.rfind(".") + 1]
    document = document_from_pages(f"1. Long Section\n{body}\nIn short, the year went well.")

    chunks = recursive_chunks(document)

    assert len(chunks) == 2
    assert chunks[-1].text.endswith("program expanded during the year.\nIn short, the year went well.")
    assert all(len(chunk.text) <= CHUNK_SIZE for chunk in chunks)


def test_a_short_closing_paragraph_stays_separate_when_joining_would_exceed_the_limit():
    def paragraph(length: int, topic: str) -> str:
        text = " ".join(lorem_paragraph(f"{topic} {n}") for n in range(10))[:length]
        return text[: text.rfind(".") + 1]

    closing = "In short, the year went well for every part of the business, and the outlook is bright."
    middle = paragraph(960, "beta")
    document = document_from_pages(f"1. Long Section\n{paragraph(600, 'alpha')}\n{middle}\n{closing}")

    chunks = recursive_chunks(document)

    assert [chunk.text for chunk in chunks[-2:]] == [middle, closing]


def test_chunks_keep_their_page_and_never_cross_pages():
    document = document_from_pages(
        f"Page one says the roastery opened in Portland.\n{PROSE}",
        f"Page two says the cafe opened in Seattle.\n{PROSE}",
        f"Page three says the warehouse opened in Boise.\n{PROSE}",
    )

    chunks = recursive_chunks(document)

    assert [chunk.page_number for chunk in chunks] == [1, 2, 3]
    for chunk in chunks:
        assert chunk.text in document.page(chunk.page_number).text
        assert chunk.filename == "sample.pdf"
        assert chunk.document_id == document.document_id


def test_a_section_continues_onto_the_next_page_until_the_next_heading():
    document = document_from_pages(
        f"1. Alpha\n{PROSE}",
        f"Alpha continues here.\n2. Beta\n{lorem_paragraph('beta')}",
    )

    chunks = recursive_chunks(document)

    assert [(chunk.page_number, chunk.section_title) for chunk in chunks] == [
        (1, "1. Alpha"),
        (2, "1. Alpha"),
        (2, "2. Beta"),
    ]
    assert chunks[1].text == "Alpha continues here."


def test_nested_headings_build_a_title_path():
    document = document_from_pages(
        f"1. Financial Highlights\n{PROSE}\n1.1 Revenue by Segment\n{PROSE}\n2. Operations\n{PROSE}"
    )

    titles = [chunk.section_title for chunk in recursive_chunks(document)]

    assert titles == [
        "1. Financial Highlights",
        "1. Financial Highlights > 1.1 Revenue by Segment",
        "2. Operations",
    ]


def test_a_heading_with_no_body_joins_the_section_that_follows():
    document = document_from_pages(f"1. Overview\n1.1 Scope\n{PROSE}")

    (chunk,) = recursive_chunks(document)

    assert chunk.text == f"1. Overview\n1.1 Scope\n{PROSE}"
    assert chunk.section_title == "1. Overview > 1.1 Scope"


def test_pages_without_text_and_fragments_without_words_produce_no_chunks():
    document = document_from_pages(PROSE, "", f". .\n1. Outlook\n{PROSE}")

    chunks = recursive_chunks(document)

    assert [chunk.page_number for chunk in chunks] == [1, 3]
    assert chunks[1].text.startswith("1. Outlook")


def test_chunk_ids_are_unique_deterministic_and_name_the_page():
    document = document_from_pages(f"1. A Section\n{PROSE}\n2. B Section\n{PROSE}", PROSE)

    first = [chunk.chunk_id for chunk in recursive_chunks(document)]
    second = [chunk.chunk_id for chunk in recursive_chunks(document)]

    assert first == second
    assert len(set(first)) == len(first)
    prefix = document.document_id[:16]
    assert first == [f"{prefix}-p0001-c001", f"{prefix}-p0001-c002", f"{prefix}-p0002-c001"]


def test_recursive_chunks_of_the_sample_report_pass_validation(northwind):
    chunks = recursive_chunks(northwind)

    assert validate_chunks(northwind, chunks) == []
    revenue = next(chunk for chunk in chunks if "$412.6 million" in chunk.text)
    assert revenue.page_number == 2
    assert revenue.section_title == "1. Financial Highlights"


# Model-assisted chunking


WEAK_HEADINGS = (
    "Our Story\n"
    "Northwind began roasting coffee in 1998 in a small garage in Portland and grew steadily.\n"
    "Coffee Sourcing\n"
    "We buy green coffee directly from farmers in Ethiopia, Colombia and Guatemala every season."
)


def test_outline_candidates_include_title_like_lines():
    document = document_from_pages(WEAK_HEADINGS)

    candidates = outline_candidates(document)

    assert [(c.candidate_id, c.text, c.strong) for c in candidates] == [
        ("h1", "Our Story", False),
        ("h2", "Coffee Sourcing", False),
    ]


def test_outline_candidates_are_bounded_to_the_most_heading_like_lines():
    page = "\n".join(
        [
            "Short Label",
            "1. Real Section",
            "Another Label",
            "Intro Line",
            "This next line is long enough that the previous one looks like a heading above it.",
            "2. Second Section",
        ]
    )

    candidates = outline_candidates(document_from_pages(page), limit=3)

    assert [(c.candidate_id, c.text) for c in candidates] == [
        ("h1", "1. Real Section"),
        ("h2", "Intro Line"),
        ("h3", "2. Second Section"),
    ]


def test_apply_outline_splits_where_the_model_found_sections():
    document = document_from_pages(WEAK_HEADINGS)
    candidates = outline_candidates(document)

    chunks = apply_outline(document, candidates, outline(("h1", 1), ("h2", 1)))

    assert [(chunk.section_title, chunk.text.split("\n")[0]) for chunk in chunks] == [
        ("Our Story", "Our Story"),
        ("Coffee Sourcing", "Coffee Sourcing"),
    ]
    assert {chunk.strategy for chunk in chunks} == {"model_assisted"}
    assert validate_chunks(document, chunks) == []


def test_apply_outline_uses_the_models_levels_and_document_order():
    document = document_from_pages(WEAK_HEADINGS)
    candidates = outline_candidates(document)

    chunks = apply_outline(document, candidates, outline(("h2", 2), ("h1", 1), ("h2", 3)))

    assert [chunk.section_title for chunk in chunks] == ["Our Story", "Our Story > Coffee Sourcing"]


def test_apply_outline_with_no_sections_keeps_whole_pages():
    document = document_from_pages(WEAK_HEADINGS, PROSE)

    chunks = apply_outline(document, outline_candidates(document), outline())

    assert [(chunk.page_number, chunk.section_title) for chunk in chunks] == [(1, None), (2, None)]
    assert chunks[0].text == WEAK_HEADINGS


def test_apply_outline_rejects_ids_that_were_never_offered():
    document = document_from_pages(WEAK_HEADINGS)

    with pytest.raises(OutlineError, match="1 heading id"):
        apply_outline(document, outline_candidates(document), outline(("h1", 1), ("h99", 2)))


def test_model_assisted_chunks_keep_page_metadata_across_a_long_report(northwind_long):
    candidates = outline_candidates(northwind_long)
    appendix = [c.candidate_id for c in candidates if c.text.startswith("Appendix")]
    chosen = [(c.candidate_id, 1) for c in candidates if c.strong] + [(cid, 2) for cid in appendix]

    chunks = apply_outline(northwind_long, candidates, outline(*chosen))

    assert validate_chunks(northwind_long, chunks) == []
    assert {chunk.page_number for chunk in chunks} == set(range(1, northwind_long.page_count + 1))
    page_22 = next(chunk for chunk in chunks if "The Oakland store served" in chunk.text)
    assert page_22.page_number == 22
    assert page_22.section_title == "4. Risks and Outlook > Appendix: Oakland Store"


# Validation


def test_validation_accepts_good_chunks_and_names_each_problem(northwind):
    chunks = recursive_chunks(northwind)
    other = document_from_pages(PROSE, filename="other.pdf")
    foreign = recursive_chunks(other)[0]
    good = chunks[0]

    problems = validate_chunks(
        northwind,
        [
            good,
            good,
            foreign,
            good.model_copy(update={"chunk_id": "x-page", "page_number": 9}),
            good.model_copy(update={"chunk_id": "x-text", "text": "Revenue was one trillion dollars."}),
        ],
    )

    assert validate_chunks(northwind, chunks) == []
    assert validate_chunks(northwind, []) == ["no chunks were produced"]
    assert problems == [
        f"duplicate chunk id {good.chunk_id}",
        f"{foreign.chunk_id} belongs to another document",
        "x-page has page 9 outside 1-5",
        "x-text text does not match page 1",
    ]


def test_chunk_schema_rejects_blank_or_oversized_text(northwind):
    good = recursive_chunks(northwind)[0]
    fields = good.model_dump()

    with pytest.raises(ValidationError):
        Chunk(**{**fields, "text": "   "})
    with pytest.raises(ValidationError):
        Chunk(**{**fields, "text": "x" * (CHUNK_SIZE + 1)})


def test_outline_schema_only_allows_three_levels():
    with pytest.raises(ValidationError):
        OutlineHeading(candidate_id="h1", level=4)
    schema = SectionOutline.model_json_schema()
    assert "description" in schema["properties"]["headings"]
