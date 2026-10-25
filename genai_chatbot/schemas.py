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
