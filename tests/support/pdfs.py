"""Small PDFs generated on the fly, so tests never depend on private or downloaded documents."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from io import BytesIO

from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

Page = Sequence[str] | str


def _lines(page: Page) -> list[str]:
    if isinstance(page, str):
        lines: list[str] = []
        for paragraph in page.split("\n"):
            lines.extend(textwrap.wrap(paragraph, width=95) or [""])
        return lines
    return list(page)


def make_pdf(pages: Sequence[Page], *, header: str | None = None, footer_page_numbers: bool = False) -> bytes:
    """Build a text PDF. A page is a string (wrapped into lines) or a list of lines."""
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle("test document")
    for number, page in enumerate(pages, start=1):
        y = 740
        if header:
            pdf.setFont("Helvetica", 9)
            pdf.drawString(72, 760, header)
        pdf.setFont("Helvetica", 10)
        for line in _lines(page):
            if y < 60:
                break
            pdf.drawString(72, y, line)
            y -= 13
        if footer_page_numbers:
            pdf.setFont("Helvetica", 9)
            pdf.drawString(300, 30, str(number))
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def make_image_only_pdf(page_count: int = 2) -> bytes:
    """Pages with drawn shapes and no text layer, like a scan."""
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    for _ in range(page_count):
        pdf.setFillGray(0.4)
        pdf.rect(72, 300, 400, 300, fill=1)
        pdf.circle(300, 200, 50, fill=1)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def make_zero_page_pdf() -> bytes:
    buffer = BytesIO()
    PdfWriter().write(buffer)
    return buffer.getvalue()


def encrypt_pdf(data: bytes, *, user_password: str, owner_password: str = "owner") -> bytes:
    writer = PdfWriter(clone_from=PdfReader(BytesIO(data)))
    writer.encrypt(user_password=user_password, owner_password=owner_password, algorithm="RC4-128")
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def lorem_paragraph(topic: str, sentences: int = 6) -> str:
    """Deterministic filler prose that mentions a topic, for chunking and retrieval tests."""
    base = [
        f"The {topic} program expanded during the year.",
        f"Managers reviewed {topic} results every quarter.",
        f"Customers reported that {topic} services improved.",
        f"The board approved new {topic} investments.",
        f"Teams documented lessons learned about {topic}.",
        f"Future plans for {topic} depend on market demand.",
    ]
    return " ".join(base[i % len(base)] for i in range(sentences))
