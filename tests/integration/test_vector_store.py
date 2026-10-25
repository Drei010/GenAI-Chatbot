"""The Chroma index with the real local embedding model: retrieval, metadata, cleanup and isolation."""

from __future__ import annotations

import gc
import re

import pytest

from genai_chatbot.chunking import recursive_chunks
from genai_chatbot.graphs.ingestion import build_ingestion_graph, ingest
from genai_chatbot.schemas import Chunk
from genai_chatbot.vector_store import ADD_BATCH_SIZE, PASSAGES_PER_QUERY, DocumentIndex, get_client
from tests.support.documents import document_from_pages
from tests.support.fake_llm import TEST_CORRELATION
from tests.support.pdfs import lorem_paragraph

TEA_FACTS = (
    "Harbor Tea Company employed 85 people in 2024 and sold tea in 12 countries.\n"
    "Its best-selling blend was a smoked black tea from Fujian."
)


def collection_names() -> set[str]:
    return {collection.name for collection in get_client().list_collections()}


@pytest.fixture
def make_index():
    created: list[DocumentIndex] = []

    def factory(session_id: str, document_id: str, chunks: list[Chunk] | None = None) -> DocumentIndex:
        index = DocumentIndex(session_id, document_id)
        created.append(index)
        if chunks:
            index.add_chunks(chunks)
        return index

    yield factory
    for index in created:
        index.delete()


@pytest.fixture(scope="module")
def northwind_chunks(northwind) -> list[Chunk]:
    return recursive_chunks(northwind)


@pytest.fixture
def northwind_index(make_index, northwind, northwind_chunks) -> DocumentIndex:
    return make_index("session-a", northwind.document_id, northwind_chunks)


# Retrieval


@pytest.mark.parametrize(
    ("question", "page", "fact"),
    [
        ("How many people does Northwind employ?", 3, "2,340 people"),
        ("What was revenue in fiscal year 2024?", 2, "$412.6 million"),
        ("How much of the roasteries' energy came from renewable electricity?", 4, "64%"),
        ("What are the main risks?", 5, "price volatility"),
        ("When does the fiscal year end?", 1, "June 30"),
        ("How did subscription revenue change?", 2, "Subscriptions $ 96.7"),
    ],
)
def test_the_passage_holding_the_answer_ranks_first(northwind_index, question, page, fact):
    best = northwind_index.search(question)[0]

    assert best.page_number == page
    assert fact in best.text


def test_passages_keep_their_source_metadata(northwind_index, northwind_chunks):
    by_id = {chunk.chunk_id: chunk for chunk in northwind_chunks}

    passages = northwind_index.search("Northwind stores and employees")

    assert passages
    for passage in passages:
        chunk = by_id[passage.chunk_id]
        assert (passage.filename, passage.page_number, passage.text, passage.section_title) == (
            chunk.filename,
            chunk.page_number,
            chunk.text,
            chunk.section_title,
        )
        assert -1.0 <= passage.score <= 1.0
    scores = [passage.score for passage in passages]
    assert scores == sorted(scores, reverse=True)


def test_untitled_passages_have_no_section_title(make_index):
    document = document_from_pages(TEA_FACTS, filename="harbor.pdf")
    index = make_index("session-a", document.document_id, recursive_chunks(document))

    (passage,) = index.search("How many people work at Harbor Tea?")

    assert passage.section_title is None
    assert (passage.filename, passage.page_number) == ("harbor.pdf", 1)


def test_at_most_four_passages_come_back_per_query(northwind_index, make_index):
    assert len(northwind_index.search("coffee", k=10)) == PASSAGES_PER_QUERY == 4
    assert len(northwind_index.search("coffee", k=2)) == 2

    document = document_from_pages(TEA_FACTS, lorem_paragraph("tea"))
    small = make_index("session-a", document.document_id, recursive_chunks(document))
    assert len(small.search("tea")) == 2


def test_blank_queries_and_empty_indexes_return_nothing(northwind_index, make_index, northwind):
    assert northwind_index.search("   ") == []
    assert make_index("session-b", northwind.document_id).search("revenue") == []


# Indexing


def test_chunks_are_added_in_batches_with_progress(make_index):
    pages = [lorem_paragraph(f"store {n}", sentences=2) for n in range(ADD_BATCH_SIZE * 2 + 6)]
    document = document_from_pages(*pages)
    chunks = recursive_chunks(document)
    index = make_index("session-a", document.document_id)
    progress: list[tuple[int, int]] = []

    added = index.add_chunks(chunks, on_progress=lambda done, total: progress.append((done, total)))

    assert added == index.count() == len(chunks) == 70
    assert progress == [(32, 70), (64, 70), (70, 70)]


def test_adding_the_same_chunks_again_does_not_duplicate_them(northwind_index, northwind_chunks):
    northwind_index.add_chunks(northwind_chunks)

    assert northwind_index.count() == len(northwind_chunks)


def test_chunks_from_another_document_are_refused(northwind_index):
    document = document_from_pages(TEA_FACTS)

    with pytest.raises(ValueError, match="another document"):
        northwind_index.add_chunks(recursive_chunks(document))


# Cleanup


def test_delete_drops_the_collection_and_can_be_repeated(make_index, northwind, northwind_chunks):
    index = make_index("session-a", northwind.document_id, northwind_chunks)
    assert index.name in collection_names()

    index.delete()
    index.delete()

    assert index.is_deleted
    assert index.name not in collection_names()
    assert index.count() == 0
    assert index.search("revenue") == []
    with pytest.raises(RuntimeError):
        index.add_chunks(northwind_chunks)


def test_a_forgotten_index_drops_its_collection(northwind, northwind_chunks):
    index = DocumentIndex("session-a", northwind.document_id)
    index.add_chunks(northwind_chunks)
    name = index.name

    del index
    gc.collect()

    assert name not in collection_names()


# Isolation


def test_collection_names_are_unique_and_valid(make_index, northwind):
    first = make_index("session-a", northwind.document_id)
    second = make_index("session-a", northwind.document_id)

    assert first.name != second.name
    for name in (first.name, second.name):
        assert re.fullmatch(r"pdfchat-session-a-[0-9a-f]{16}-[0-9a-f]{12}", name)


def test_sessions_never_see_each_others_passages(make_index, northwind, northwind_chunks):
    mine = make_index("session-a", northwind.document_id, northwind_chunks)
    theirs = make_index("session-b", northwind.document_id)

    assert theirs.search("How many people does Northwind employ?") == []
    theirs.add_chunks(northwind_chunks[:1])
    mine.delete()

    assert theirs.count() == 1
    assert [p.chunk_id for p in theirs.search("Northwind")] == [northwind_chunks[0].chunk_id]


def test_a_replacement_document_only_returns_its_own_passages(make_index, northwind, northwind_chunks):
    tea = document_from_pages(TEA_FACTS, filename="harbor.pdf")
    first = make_index("session-a", northwind.document_id, northwind_chunks)
    replacement = make_index("session-a", tea.document_id, recursive_chunks(tea))
    first.delete()

    passages = replacement.search("How many people does Northwind employ?")

    assert {passage.filename for passage in passages} == {"harbor.pdf"}
    assert first.name not in collection_names()


# With the ingestion graph


def test_an_ingested_pdf_is_searchable(northwind_pdf, northwind, make_index):
    index = make_index("session-a", northwind.document_id)
    graph = build_ingestion_graph(index, None, TEST_CORRELATION)

    result = ingest(graph, "northwind-annual-report.pdf", northwind_pdf)

    assert index.count() == result.chunk_count
    best = index.search("How many people does Northwind employ?")[0]
    assert (best.page_number, best.filename) == (3, "northwind-annual-report.pdf")
    assert "2,340 people" in best.text
