from __future__ import annotations

from streamlit.testing.v1 import AppTest

from tests.support.paths import PROJECT_ROOT
from tests.support.pdfs import make_image_only_pdf, make_pdf

APP_PATH = str(PROJECT_ROOT / "app.py")


def test_app_starts_and_explains_missing_settings() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()

    assert not at.exception
    assert [e.value for e in at.error] == [
        "The chatbot needs AI provider settings before it can answer questions."
    ]
    guidance = "\n".join(m.value for m in at.markdown)
    assert "`PRIMARY_LLM_API_KEY` is missing." in guidance
    assert ".env.example" in guidance


def test_app_starts_with_valid_settings(valid_settings_env: dict[str, str]) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()

    assert not at.exception
    assert not at.error
    assert at.title[0].value == "PDF Knowledge Chatbot"


def upload(at: AppTest, filename: str, data: bytes) -> AppTest:
    at.sidebar.file_uploader[0].set_value((filename, data, "application/pdf"))
    return at.run()


def test_uploading_a_pdf_reads_its_pages(valid_settings_env: dict[str, str]) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()

    upload(at, "northwind.pdf", make_pdf(["Coffee sales rose 8 percent.", "Logistics costs fell."]))

    assert not at.exception and not at.error
    assert "northwind.pdf** is ready: 2 pages read." in at.success[0].value
    assert at.session_state.active.page_count == 2


def test_upload_problems_are_explained(valid_settings_env: dict[str, str]) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()

    upload(at, "notes.pdf", b"plain text pretending to be a PDF")
    assert [e.value for e in at.error] == ["This file isn't a PDF. Upload a document saved in PDF format."]

    upload(at, "scan.pdf", make_image_only_pdf())
    assert "OCR" in at.error[0].value and at.session_state.active is None


def test_removing_the_file_forgets_the_document(valid_settings_env: dict[str, str]) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    upload(at, "northwind.pdf", make_pdf(["Coffee sales rose 8 percent."]))

    at.sidebar.file_uploader[0].set_value(None)
    at.run()

    assert at.session_state.active is None
    assert at.info[0].value == "Upload a PDF in the sidebar to get started."


def test_reset_clears_the_document_and_the_uploader(valid_settings_env: dict[str, str]) -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    upload(at, "northwind.pdf", make_pdf(["Coffee sales rose 8 percent."]))

    at.sidebar.button[0].click()
    at.run()

    assert at.session_state.active is None and at.session_state.generation == 1
    assert at.sidebar.file_uploader[0].key == "uploader-1"
