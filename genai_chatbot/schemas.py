"""Validated data passed between the app's stages."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
