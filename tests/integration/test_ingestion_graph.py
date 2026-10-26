"""The ingestion graph end to end: real PDF bytes in, validated chunks out, with a fake model."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from langchain_core.exceptions import OutputParserException

from genai_chatbot.chunking import HeadingCandidate
from genai_chatbot.documents import DocumentError
from genai_chatbot.graphs import ingestion
from genai_chatbot.graphs.ingestion import IngestionError, build_ingestion_graph, ingest
from genai_chatbot.schemas import Chunk, OutlineHeading, SectionOutline
from tests.support.documents import REPORT_PAGES
from tests.support.docx import make_docx, paragraph
from tests.support.fake_llm import (
    TEST_CORRELATION,
    Script,
    always,
    bad_credentials,
    make_router,
    rate_limited,
)
from tests.support.pdfs import lorem_paragraph, make_pdf


class RecordingIndex:
    """Collects indexed chunks and reports progress in batches of two, like the real index."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def add_chunks(self, chunks: Sequence[Chunk], on_progress=None) -> int:
        for start in range(0, len(chunks), 2):
            self.chunks.extend(chunks[start : start + 2])
            if on_progress:
                on_progress(min(start + 2, len(chunks)), len(chunks))
        return len(chunks)


def strong_outline(messages) -> SectionOutline:
    """A well-behaved model: every numbered or capitalized candidate is a section, at its own level."""
    headings = []
    for line in messages[-1].content.split("\n"):
        parts = line.split(" | ", 2)
        if len(parts) == 3 and parts[0].startswith("h"):
            text = parts[2]
            if text.isupper():
                headings.append(OutlineHeading(candidate_id=parts[0], level=1))
            elif text[0].isdigit():
                level = text.split(" ", 1)[0].rstrip(".").count(".") + 1
                headings.append(OutlineHeading(candidate_id=parts[0], level=min(level, 3)))
    return SectionOutline(headings=headings)


def run(
    data: bytes,
    router,
    index: RecordingIndex | None = None,
    events: list | None = None,
    filename: str = "report.pdf",
):
    index = index if index is not None else RecordingIndex()
    graph = build_ingestion_graph(index, router, TEST_CORRELATION)
    return ingest(graph, filename, data, on_progress=events.append if events is not None else None)


def word_report(topics: list[str], paragraphs_per_section: int = 1) -> bytes:
    """A Word file with a numbered Heading 1 per topic and no recorded page breaks."""
    body = []
    for number, topic in enumerate(topics, start=1):
        body.append(paragraph(f"{number}. {topic.title()}", style="Heading1"))
        body += [paragraph(lorem_paragraph(topic)) for _ in range(paragraphs_per_section)]
    return make_docx(body)


@pytest.fixture
def simple_pdf() -> bytes:
    return make_pdf([lorem_paragraph("roasting"), lorem_paragraph("brewing"), lorem_paragraph("serving")])


def test_a_simple_document_is_chunked_without_calling_a_model(simple_pdf):
    router, primary, _ = make_router(always(AssertionError("no model call expected")))
    index = RecordingIndex()

    result = run(simple_pdf, router, index)

    assert (result.is_complex, result.strategy, result.fallback_reason) == (False, "recursive", None)
    assert primary.calls == []
    assert result.chunk_count == len(index.chunks) == 3
    assert [chunk.page_number for chunk in index.chunks] == [1, 2, 3]
    assert result.page_count == 3 and result.filename == "report.pdf"


def test_a_word_document_is_chunked_by_its_heading_styles():
    router, primary, _ = make_router(always(AssertionError("no model call expected")))
    index = RecordingIndex()

    result = run(word_report(["roasting", "brewing", "serving"]), router, index, filename="report.docx")

    assert (result.filename, result.has_page_numbers, result.strategy) == ("report.docx", False, "recursive")
    assert [chunk.section_title for chunk in index.chunks] == ["1. Roasting", "2. Brewing", "3. Serving"]
    assert primary.calls == []


def test_a_long_word_document_is_outlined_from_its_heading_styles():
    topics = [f"region {chr(ord('a') + n)}" for n in range(25)]
    router, primary, _ = make_router(Script({SectionOutline: strong_outline}))
    index = RecordingIndex()

    result = run(word_report(topics, paragraphs_per_section=12), router, index, filename="long.docx")

    assert (result.is_complex, result.strategy, result.has_page_numbers) == (True, "model_assisted", False)
    assert result.page_count > 20
    (prompt,) = primary.calls_for(SectionOutline)
    candidates = [line for line in prompt[-1].content.split("\n") if line.startswith("h")]
    assert len(candidates) == 25 and candidates[0].endswith("| 1. Region A")
    assert {chunk.section_title for chunk in index.chunks} == {
        f"{n}. {topic.title()}" for n, topic in enumerate(topics, 1)
    }


def test_a_complex_document_is_chunked_along_the_models_outline(northwind_pdf, northwind):
    router, primary, fallback = make_router(Script({SectionOutline: strong_outline}), always(rate_limited()))
    index = RecordingIndex()

    result = run(northwind_pdf, router, index)

    assert result.is_complex
    assert (result.strategy, result.fallback_reason, result.notices) == ("model_assisted", None, ())
    assert len(primary.calls_for(SectionOutline)) == 1
    assert fallback.calls == []
    assert {chunk.strategy for chunk in index.chunks} == {"model_assisted"}
    by_text = {chunk.text.split("\n")[0]: chunk for chunk in index.chunks}
    assert (
        by_text["1.1 Revenue by Segment"].section_title == "1. Financial Highlights > 1.1 Revenue by Segment"
    )
    assert by_text["1.1 Revenue by Segment"].page_number == 2
    assert by_text["2.1 Employees"].page_number == 3


def test_page_metadata_survives_ingestion(northwind_pdf):
    router, _, _ = make_router(Script({SectionOutline: strong_outline}))
    index = RecordingIndex()

    run(northwind_pdf, router, index)

    for page_number, lines in REPORT_PAGES.items():
        on_page = [chunk for chunk in index.chunks if chunk.page_number == page_number]
        assert on_page, f"page {page_number} has no chunks"
        assert lines[-1] in " ".join(chunk.text for chunk in on_page)
    assert {chunk.filename for chunk in index.chunks} == {"report.pdf"}


def test_a_long_document_is_complex_and_keeps_every_page(northwind_long_pdf):
    router, _, _ = make_router(Script({SectionOutline: strong_outline}))
    index = RecordingIndex()

    result = run(northwind_long_pdf, router, index)

    assert (result.is_complex, result.page_count, result.strategy) == (True, 24, "model_assisted")
    assert {chunk.page_number for chunk in index.chunks} == set(range(1, 25))


def test_a_rate_limited_primary_hands_chunking_to_the_fallback(northwind_pdf, app_logs):
    router, primary, fallback = make_router(always(rate_limited()), Script({SectionOutline: strong_outline}))

    result = run(northwind_pdf, router)

    assert (result.strategy, result.fallback_reason) == ("model_assisted", None)
    assert (len(primary.calls), len(fallback.calls)) == (1, 1)
    calls = [record for record in app_logs() if record["event"] == "llm_call"]
    assert [(r["provider_route"], r["fallback_used"], r["outcome"]) for r in calls] == [
        ("primary", False, "error"),
        ("fallback", True, "ok"),
    ]


def test_when_both_models_are_rate_limited_chunking_falls_back_to_the_splitter(northwind_pdf, northwind):
    router, primary, fallback = make_router(always(rate_limited()), always(rate_limited()))
    index = RecordingIndex()

    result = run(northwind_pdf, router, index)

    assert (result.strategy, result.fallback_reason, result.notices) == ("recursive", "rate_limit", ())
    assert (len(primary.calls), len(fallback.calls)) == (1, 1)
    assert [chunk.text for chunk in index.chunks] == [
        chunk.text for chunk in ingestion.recursive_chunks(northwind)
    ]


@pytest.mark.parametrize(
    "bad_output",
    [
        SectionOutline(headings=[OutlineHeading(candidate_id="h404", level=1)]),
        OutputParserException("the model returned prose instead of JSON"),
    ],
    ids=["unknown-heading-id", "unparseable-output"],
)
def test_malformed_model_output_falls_back_to_the_splitter(northwind_pdf, bad_output):
    router, primary, fallback = make_router(always(bad_output), always(SectionOutline(headings=[])))

    result = run(northwind_pdf, router)

    assert (result.strategy, result.fallback_reason) == ("recursive", "malformed_output")
    assert len(primary.calls) == 1
    assert fallback.calls == []


def test_bad_credentials_fall_back_and_tell_the_user(northwind_pdf):
    router, _, fallback = make_router(always(bad_credentials()), always(SectionOutline(headings=[])))

    result = run(northwind_pdf, router)

    assert (result.strategy, result.fallback_reason) == ("recursive", "auth")
    assert fallback.calls == []
    (notice,) = result.notices
    assert "API key for the primary model" in notice


def test_without_a_model_complex_documents_use_the_splitter(northwind_pdf):
    result = run(northwind_pdf, router=None)

    assert (result.is_complex, result.strategy, result.fallback_reason) == (True, "recursive", "no_model")


def test_complex_documents_without_heading_candidates_skip_the_model():
    rows = ["Retail $ 251.4 $ 236.0 7%", "Online $ 96.7 $ 79.2 22%", "Wholesale $ 64.5 $ 63.3 2%"]
    pdf = make_pdf([["figures for the year are below and they were good overall", *rows]])
    router, primary, _ = make_router(always(AssertionError("no model call expected")))

    result = run(pdf, router)

    assert (result.is_complex, result.strategy, result.fallback_reason) == (
        True,
        "recursive",
        "no_candidates",
    )
    assert primary.calls == []


def test_model_chunks_that_fail_validation_are_replaced(northwind_pdf, monkeypatch):
    def broken_outline(document, candidates: Sequence[HeadingCandidate], outline):
        chunk = ingestion.recursive_chunks(document)[0]
        return [chunk.model_copy(update={"text": "Text the document never contained."})]

    monkeypatch.setattr(ingestion, "apply_outline", broken_outline)
    router, _, _ = make_router(Script({SectionOutline: strong_outline}))
    index = RecordingIndex()

    result = run(northwind_pdf, router, index)

    assert (result.strategy, result.fallback_reason) == ("recursive", "invalid_chunks")
    assert all("never contained" not in chunk.text for chunk in index.chunks)


def test_invalid_recursive_chunks_stop_ingestion(simple_pdf, monkeypatch):
    monkeypatch.setattr(ingestion, "recursive_chunks", lambda document: [])

    with pytest.raises(IngestionError):
        run(simple_pdf, router=None)


def test_unreadable_files_raise_the_extraction_error():
    with pytest.raises(DocumentError) as raised:
        run(b"plain text, not a PDF", router=None)

    assert raised.value.kind == "unsupported"


def test_progress_is_reported_in_order_up_to_completion(northwind_pdf):
    router, _, _ = make_router(Script({SectionOutline: strong_outline}))
    events: list[dict] = []

    result = run(northwind_pdf, router, events=events)

    steps = [event["step"] for event in events]
    assert steps[:4] == ["extract", "classify", "chunk", "index"]
    assert set(steps[4:]) == {"index"}
    fractions = [event["fraction"] for event in events]
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
    assert events[-1]["message"] == f"Indexing passages ({result.chunk_count} of {result.chunk_count})…"


def test_stage_logs_carry_correlation_ids_and_no_document_text(northwind_pdf, app_logs):
    router, _, _ = make_router(always(rate_limited()), always(rate_limited()))

    run(northwind_pdf, router)

    records = app_logs()
    stages = [(r["stage"], r.get("strategy"), r["outcome"]) for r in records if r["event"] == "stage"]
    assert stages == [
        ("extract", None, "ok"),
        ("classify", None, "ok"),
        ("chunk", "model_assisted", "fallback"),
        ("chunk", "recursive", "ok"),
        ("validate", "recursive", "ok"),
        ("index", None, "ok"),
    ]
    fallback = next(r for r in records if r.get("outcome") == "fallback")
    assert fallback["fallback_reason"] == "rate_limit"
    assert {r["session_id"] for r in records} == {"test-session"}
    logged = " ".join(str(value) for record in records for value in record.values())
    assert "Northwind" not in logged and "412.6" not in logged


def test_the_graph_has_the_documented_nodes():
    graph = build_ingestion_graph(RecordingIndex(), None, TEST_CORRELATION)

    nodes = set(graph.get_graph().nodes) - {"__start__", "__end__"}

    assert nodes == {"extract", "classify", "chunk_with_model", "chunk_recursive", "validate", "index_chunks"}
