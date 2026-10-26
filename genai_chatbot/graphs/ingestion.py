"""LangGraph workflow that turns an uploaded PDF or Word document into indexed, citation-ready chunks.

extract -> classify -> simple:  chunk_recursive -> validate -> index_chunks
                    -> complex: chunk_with_model -> validate -> index_chunks

Complex documents fall back to chunk_recursive when the model call fails (after failover), when its
outline is malformed, or when the chunks it produced fail validation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from genai_chatbot.chunking import (
    OutlineError,
    apply_outline,
    assess_complexity,
    outline_candidates,
    recursive_chunks,
    validate_chunks,
)
from genai_chatbot.extraction import extract_document
from genai_chatbot.observability import Correlation, log_stage
from genai_chatbot.prompts import outline_messages
from genai_chatbot.providers import ErrorKind, LLMRouter, ProviderError
from genai_chatbot.schemas import Chunk, ChunkStrategy, ComplexityAssessment, ExtractedDocument, SectionOutline

ProgressCallback = Callable[[int, int], None]


class ChunkIndex(Protocol):
    """Where validated chunks go; the app uses a Chroma collection (see vector_store.py)."""

    def add_chunks(self, chunks: Sequence[Chunk], on_progress: ProgressCallback | None = None) -> int: ...


class IngestionError(RuntimeError):
    """The document was readable but couldn't be turned into valid passages."""


class IngestionState(TypedDict, total=False):
    filename: str
    file_bytes: bytes
    document: ExtractedDocument
    assessment: ComplexityAssessment
    chunks: list[Chunk]
    strategy: ChunkStrategy
    fallback_reason: str | None
    problems: list[str]
    notices: list[str]
    indexed_count: int


@dataclass(frozen=True)
class IngestionResult:
    document_id: str
    filename: str
    page_count: int
    pages_without_text: int
    has_page_numbers: bool
    chunk_count: int
    is_complex: bool
    strategy: ChunkStrategy
    fallback_reason: str | None
    notices: tuple[str, ...]

    @classmethod
    def from_state(cls, state: IngestionState) -> IngestionResult:
        document = state["document"]
        return cls(
            document_id=document.document_id,
            filename=document.filename,
            page_count=document.page_count,
            pages_without_text=document.pages_without_text,
            has_page_numbers=document.has_page_numbers,
            chunk_count=state["indexed_count"],
            is_complex=state["assessment"].is_complex,
            strategy=state["strategy"],
            fallback_reason=state.get("fallback_reason"),
            notices=tuple(state.get("notices", [])),
        )


def _progress(step: str, message: str, fraction: float) -> None:
    get_stream_writer()({"step": step, "message": message, "fraction": round(min(fraction, 1.0), 3)})


def build_ingestion_graph(
    index: ChunkIndex, router: LLMRouter | None, correlation: Correlation
) -> CompiledStateGraph:
    """Compile the ingestion workflow for one upload. `router=None` skips model-assisted chunking."""

    def extract(state: IngestionState) -> IngestionState:
        _progress("extract", "Reading the document…", 0.05)
        with log_stage("extract", correlation) as details:
            document = extract_document(state["file_bytes"], state["filename"])
            details.update(page_count=document.page_count, has_page_numbers=document.has_page_numbers)
        return {"document": document}

    def classify(state: IngestionState) -> IngestionState:
        _progress("classify", "Checking the document's structure…", 0.15)
        with log_stage("classify", correlation) as details:
            assessment = assess_complexity(state["document"])
            details["complex"] = assessment.is_complex
        return {"assessment": assessment}

    def model_chunks(document: ExtractedDocument) -> IngestionState:
        if router is None:
            return {"fallback_reason": "no_model"}
        candidates = outline_candidates(document)
        if not candidates:
            return {"fallback_reason": "no_candidates"}
        messages = outline_messages(document.filename, document.page_count, candidates)
        try:
            outline = router.invoke_structured(
                SectionOutline, messages, stage="chunking", correlation=correlation
            )
        except ProviderError as exc:
            notices = [exc.user_message] if exc.is_configuration_problem else []
            return {"fallback_reason": exc.kind.value, "notices": notices}
        try:
            chunks = apply_outline(document, candidates, outline)
        except OutlineError:
            return {"fallback_reason": ErrorKind.MALFORMED_OUTPUT.value}
        return {"chunks": chunks, "strategy": "model_assisted", "fallback_reason": None}

    def chunk_with_model(state: IngestionState) -> IngestionState:
        _progress("chunk", "Finding the document's sections…", 0.2)
        with log_stage("chunk", correlation, strategy="model_assisted") as details:
            update = model_chunks(state["document"])
            if update.get("fallback_reason"):
                details.update(outcome="fallback", fallback_reason=update["fallback_reason"])
            else:
                details["chunk_count"] = len(update["chunks"])
        return update

    def chunk_recursive(state: IngestionState) -> IngestionState:
        _progress("chunk", "Splitting the text into passages…", 0.3)
        with log_stage("chunk", correlation, strategy="recursive") as details:
            chunks = recursive_chunks(state["document"])
            details["chunk_count"] = len(chunks)
        return {"chunks": chunks, "strategy": "recursive"}

    def validate(state: IngestionState) -> IngestionState:
        with log_stage("validate", correlation, strategy=state.get("strategy")) as details:
            problems = validate_chunks(state["document"], state.get("chunks", []))
            details["outcome"] = "invalid" if problems else "ok"
        if problems and state.get("strategy") != "model_assisted":
            raise IngestionError("the document could not be split into valid passages")
        update: IngestionState = {"problems": problems}
        if problems:
            update["fallback_reason"] = "invalid_chunks"
        return update

    def index_chunks(state: IngestionState) -> IngestionState:
        writer = get_stream_writer()
        chunks = state["chunks"]

        def on_progress(done: int, total: int) -> None:
            fraction = 0.35 + 0.65 * done / max(total, 1)
            writer(
                {"step": "index", "message": f"Indexing passages ({done} of {total})…", "fraction": fraction}
            )

        _progress("index", f"Indexing {len(chunks)} passages…", 0.35)
        with log_stage("index", correlation) as details:
            count = index.add_chunks(chunks, on_progress=on_progress)
            details["chunk_count"] = count
        return {"indexed_count": count}

    def route_by_complexity(state: IngestionState) -> Literal["chunk_with_model", "chunk_recursive"]:
        return "chunk_with_model" if state["assessment"].is_complex else "chunk_recursive"

    def route_after_model(state: IngestionState) -> Literal["validate", "chunk_recursive"]:
        return "validate" if state.get("strategy") == "model_assisted" else "chunk_recursive"

    def route_after_validation(state: IngestionState) -> Literal["index_chunks", "chunk_recursive"]:
        return "chunk_recursive" if state["problems"] else "index_chunks"

    graph = StateGraph(IngestionState)
    graph.add_node("extract", extract)
    graph.add_node("classify", classify)
    graph.add_node("chunk_with_model", chunk_with_model)
    graph.add_node("chunk_recursive", chunk_recursive)
    graph.add_node("validate", validate)
    graph.add_node("index_chunks", index_chunks)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "classify")
    graph.add_conditional_edges("classify", route_by_complexity)
    graph.add_conditional_edges("chunk_with_model", route_after_model)
    graph.add_edge("chunk_recursive", "validate")
    graph.add_conditional_edges("validate", route_after_validation)
    graph.add_edge("index_chunks", END)
    return graph.compile(name="document_ingestion")


def ingest(
    graph: CompiledStateGraph,
    filename: str,
    data: bytes,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> IngestionResult:
    """Run the ingestion graph, forwarding progress events. Raises DocumentError or IngestionError."""
    final: IngestionState = {}
    for mode, payload in graph.stream(
        {"filename": filename, "file_bytes": data}, stream_mode=["custom", "values"]
    ):
        if mode == "custom" and on_progress is not None:
            on_progress(payload)
        elif mode == "values":
            final = payload
    return IngestionResult.from_state(final)
