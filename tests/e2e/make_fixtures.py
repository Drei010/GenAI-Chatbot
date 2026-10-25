"""Write the files the browser tests upload to tests/e2e/fixtures/generated/.

Run with `uv run python -m tests.e2e.make_fixtures`. The files are generated on every run and never
committed, so the browser tests depend on no private document.
"""

from __future__ import annotations

from pathlib import Path

from tests.support.documents import report_pdf
from tests.support.pdfs import make_pdf

OUTPUT = Path(__file__).resolve().parent / "fixtures" / "generated"

HARBOR_TEA_PAGES = [
    [
        "Harbor Tea Company",
        "Annual Review 2024",
        "Harbor Tea Company employed 85 people at the end of 2024, up from 72 a year earlier.",
        "The company sold 1.2 million tins of tea through 14 shops and its online store.",
    ],
    [
        "Shops and Growth",
        "Harbor Tea opened two new shops in Portland during 2024.",
        "Online orders made up 31% of sales, compared with 26% in 2023.",
    ],
]


def fixtures() -> dict[str, bytes]:
    return {
        "northwind-annual-report.pdf": report_pdf(),
        "harbor-tea-report.pdf": make_pdf(HARBOR_TEA_PAGES),
        # A text file with a .pdf name: the uploader accepts it and the app has to reject it.
        "notes.pdf": b"Meeting notes, saved as plain text with a .pdf name.\n",
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, data in fixtures().items():
        (OUTPUT / name).write_bytes(data)
    print(f"Wrote {len(fixtures())} browser test files to {OUTPUT}")


if __name__ == "__main__":
    main()
