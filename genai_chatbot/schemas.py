"""Validated data passed between the app's stages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PageText(BaseModel):
    """Cleaned text of one PDF page. `page_number` is 1-based, as printed in a PDF viewer."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    text: str


class ExtractedDocument(BaseModel):
    """An uploaded PDF reduced to its pages of text. Lives in memory only."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256 of the uploaded bytes")
    filename: str = Field(min_length=1, max_length=255)
    pages: list[PageText] = Field(min_length=1)

    @model_validator(mode="after")
    def _pages_are_numbered_in_order(self) -> ExtractedDocument:
        if [page.page_number for page in self.pages] != list(range(1, len(self.pages) + 1)):
            raise ValueError("pages must be numbered 1..N in order")
        return self

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def pages_without_text(self) -> int:
        return sum(1 for page in self.pages if not page.text.strip())

    def page(self, page_number: int) -> PageText:
        return self.pages[page_number - 1]


ChunkStrategy = Literal["recursive", "model_assisted"]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


class ComplexityAssessment(BaseModel):
    """Why a document was routed to simple or complex chunking."""

    is_complex: bool
    reasons: list[str]
    page_count: int = Field(ge=1)
    table_line_count: int = Field(ge=0)
    heading_count: int = Field(ge=0)
    heading_levels: int = Field(ge=0)


class Chunk(BaseModel):
    """A passage of one page, ready to embed. `text` is an exact substring of that page's text."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=CHUNK_SIZE)
    section_title: str | None = None
    strategy: ChunkStrategy

    @field_validator("text")
    @classmethod
    def _has_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk text is blank")
        return value


class OutlineHeading(BaseModel):
    """One candidate line the model judged to start a section."""

    candidate_id: str = Field(description="ID of a candidate line that starts a section, for example h12.")
    level: Literal[1, 2, 3] = Field(
        description="1 for a part or chapter, 2 for a section, 3 for a subsection."
    )


class SectionOutline(BaseModel):
    """Model output for complex-document chunking: which candidate lines are real headings."""

    headings: list[OutlineHeading] = Field(
        description="The candidates that start a section, in document order. Empty if none do."
    )


class RetrievedPassage(BaseModel):
    """A chunk returned by a search, carrying everything needed to cite it."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)
    section_title: str | None = None
    score: float = Field(description="Cosine similarity to the query, from -1 to 1; higher is closer.")


# Question answering

ResponseStyle = Literal["prose", "bullets", "table"]
AnswerStatus = Literal["answered", "unsupported", "error"]

MAX_SUBQUESTIONS = 4
HISTORY_MESSAGES = 8


class StandaloneQuestion(BaseModel):
    """Model output for follow-up resolution."""

    question: str = Field(
        description="The latest message rewritten so it can be understood without the conversation."
    )


class QueryPlan(BaseModel):
    """Model output for retrieval planning. Normalized so it holds one to four distinct queries."""

    decompose: bool = Field(
        description="True when the question has several parts or needs facts from different places."
    )
    queries: list[str] = Field(
        description="Search queries: one when decompose is false, otherwise up to four self-contained "
        "subquestions, one per part of the question."
    )
    response_style: ResponseStyle = Field(
        description="table to compare items or periods, bullets for summaries or lists, prose otherwise."
    )

    @model_validator(mode="after")
    def _normalize(self) -> QueryPlan:
        queries: list[str] = []
        for query in self.queries:
            text = " ".join(query.split())
            if text and text.lower() not in {q.lower() for q in queries}:
                queries.append(text)
        if not queries:
            raise ValueError("the plan has no search queries")
        self.queries = queries[:MAX_SUBQUESTIONS] if self.decompose else queries[:1]
        self.decompose = len(self.queries) > 1
        return self


class AnswerDraft(BaseModel):
    """Model output for synthesis: the answer with passage IDs (P1, P2, ...) as citations."""

    response_style: ResponseStyle = Field(description="The format used for the answer.")
    answer: str = Field(
        description="Markdown answer that cites passages inline in square brackets, for example [P2]."
    )
    citation_ids: list[str] = Field(description="Every passage ID cited in the answer, for example P2.")
    status: Literal["supported", "unsupported"] = Field(
        description="supported when the passages answer the question, at least in part; unsupported "
        "when they don't."
    )


class Citation(BaseModel):
    """A source shown under an answer. Every field comes from the retrieved passage, not the model."""

    model_config = ConfigDict(frozen=True)

    number: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    excerpt: str = Field(min_length=1)


class ChatAnswer(BaseModel):
    """What the user sees for one question."""

    model_config = ConfigDict(frozen=True)

    status: AnswerStatus
    response_style: ResponseStyle = "prose"
    text: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _only_answers_have_sources(self) -> ChatAnswer:
        if self.status == "answered" and not self.citations:
            raise ValueError("an answer needs at least one citation")
        if self.status != "answered" and self.citations:
            raise ValueError("abstentions and errors never show sources")
        return self


class ChatTurn(BaseModel):
    """One message in the conversation. Assistant turns carry the full answer with its sources."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    content: str
    answer: ChatAnswer | None = None
