"""A small fictional annual report with known facts on known pages, for retrieval and QA tests."""

from __future__ import annotations

import hashlib

from genai_chatbot.schemas import ExtractedDocument, PageText
from tests.support.pdfs import make_pdf

COMPANY = "Northwind Coffee Company"

# page number -> lines. Facts are deliberately specific so tests can check exact excerpts.
REPORT_PAGES: dict[int, list[str]] = {
    1: [
        "LETTER TO SHAREHOLDERS",
        f"Dear shareholders, fiscal year 2024 was a strong year for {COMPANY}.",
        "Our fiscal year ends on June 30, so fiscal year 2024 ran from July 1, 2023 to June 30, 2024.",
        "We opened new roasteries, expanded our subscription service, and reduced packaging waste.",
    ],
    2: [
        "1. Financial Highlights",
        "Revenue for fiscal year 2024 was $412.6 million, an increase of 9% from $378.5 million in fiscal",
        "year 2023. Operating income was $58.3 million compared with $49.9 million in fiscal year 2023.",
        "Net income was $41.2 million, or $2.06 per diluted share.",
        "1.1 Revenue by Segment",
        "Retail stores $ 251.4 $ 236.0 7%",
        "Subscriptions $ 96.7 $ 79.2 22%",
        "Wholesale $ 64.5 $ 63.3 2%",
        "All amounts are in millions of US dollars for fiscal years 2024 and 2023.",
    ],
    3: [
        "2. Operations",
        "Northwind operated 186 retail stores at the end of fiscal year 2024, up from 171 a year earlier.",
        "The company roasted 41 million pounds of coffee at its three roasteries in Oregon, Ohio and Texas.",
        "2.1 Employees",
        "Northwind employed 2,340 people as of June 30, 2024, of whom 1,910 worked in retail stores.",
    ],
    4: [
        "3. Sustainability",
        "Packaging waste per pound of coffee fell 18% compared with fiscal year 2023.",
        "Renewable electricity supplied 64% of the energy used by roasteries during calendar year 2023.",
        "3.1 Sourcing",
        "Ninety-two percent of green coffee was purchased through direct trade relationships.",
    ],
    5: [
        "4. Risks and Outlook",
        "The main risks are green coffee price volatility, labor costs, and competition from larger chains.",
        "For fiscal year 2025 the company plans to open 20 new stores and launch a cold brew line.",
        "Management expects revenue growth between 6% and 8% in fiscal year 2025.",
    ],
}


def report_pdf() -> bytes:
    return make_pdf([REPORT_PAGES[number] for number in sorted(REPORT_PAGES)])


STORE_CITIES = (
    "Portland",
    "Seattle",
    "Boise",
    "Spokane",
    "Eugene",
    "Tacoma",
    "Salem",
    "Bend",
    "Olympia",
    "Yakima",
    "Medford",
    "Bellingham",
    "Missoula",
    "Reno",
    "Sacramento",
    "Fresno",
    "Oakland",
    "Berkeley",
    "Davis",
    "Chico",
    "Redding",
    "Ashland",
    "Corvallis",
    "Vancouver",
    "Everett",
)


def long_report_pdf(total_pages: int = 24) -> bytes:
    """More than 20 pages: the report followed by one store profile per page, each with distinct text.

    Page n (from 6) profiles the store in STORE_CITIES[n - 6]. Lines differ by more than digits from
    page to page, so none of them look like a running header.
    """
    pages = [REPORT_PAGES[number] for number in sorted(REPORT_PAGES)]
    for n in range(len(pages) + 1, total_pages + 1):
        city = STORE_CITIES[n - 6]
        pages.append(
            [
                f"Appendix: {city} Store",
                f"The {city} store served {1200 + n * 37} customers per week in fiscal year 2024.",
                f"Its {city} baristas completed {n * 3} training hours and recycled {n + 40}% of waste.",
            ]
        )
    return make_pdf(pages)


def document_from_pages(*pages: str, filename: str = "sample.pdf") -> ExtractedDocument:
    """An already-extracted document with exactly these page texts, for tests that skip PDF parsing."""
    digest = hashlib.sha256("\f".join(pages).encode()).hexdigest()
    return ExtractedDocument(
        document_id=digest,
        filename=filename,
        pages=[PageText(page_number=number, text=text) for number, text in enumerate(pages, start=1)],
    )
