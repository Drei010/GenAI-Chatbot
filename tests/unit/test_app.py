"""The Streamlit app, driven in-process with AppTest and a fake model router."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

import app
from genai_chatbot.graphs.ingestion import IngestionError
from genai_chatbot.providers import LLMRouter
from genai_chatbot.schemas import OutlineHeading, SectionOutline
from genai_chatbot.vector_store import get_client
from tests.support.fake_llm import always, bad_credentials, make_router
from tests.support.paths import PROJECT_ROOT
from tests.support.pdfs import make_image_only_pdf, make_pdf

APP_PATH = str(PROJECT_ROOT / "app.py")
FIRST_HEADING = SectionOutline(headings=[OutlineHeading(candidate_id="h1", level=1)])


def run_app(router) -> None:
    """AppTest runs this function's source as the page script, so it imports what it needs itself."""
    import app

    app.main(router_factory=lambda settings: router)


def start_app(router: LLMRouter | None = None) -> AppTest:
    router = router or make_router(always(FIRST_HEADING))[0]
    return AppTest.from_function(run_app, kwargs={"router": router}, default_timeout=60).run()


def upload(at: AppTest, filename: str, data: bytes) -> AppTest:
    at.sidebar.file_uploader[0].set_value((filename, data, "application/pdf"))
    return at.run()


def collections_for(at: AppTest) -> set[str]:
    prefix = f"pdfchat-{at.session_state.session_id}-"
    return {c.name for c in get_client().list_collections() if c.name.startswith(prefix)}


@pytest.fixture(autouse=True)
def configured(valid_settings_env: dict[str, str]) -> None:
    """Every test below has valid settings unless it clears them."""


def test_app_starts_and_explains_missing_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in app.AppSettings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)

    at = AppTest.from_file(APP_PATH, default_timeout=60).run()

    assert not at.exception
    assert [e.value for e in at.error] == [
        "The chatbot needs AI provider settings before it can answer questions."
    ]
    guidance = "\n".join(m.value for m in at.markdown)
    assert "`PRIMARY_LLM_API_KEY` is missing." in guidance
    assert ".env.example" in guidance


def test_app_starts_with_valid_settings() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60).run()

    assert not at.exception
    assert not at.error
    assert at.title[0].value == "PDF Knowledge Chatbot"
    assert at.info[0].value == "Upload a PDF in the sidebar to get started."


def test_uploading_a_pdf_reads_and_indexes_it(northwind_pdf: bytes) -> None:
    router, primary, _ = make_router(always(FIRST_HEADING))
    at = start_app(router)

    upload(at, "northwind.pdf", northwind_pdf)

    assert not at.exception and not at.error
    assert "northwind.pdf** is ready: 5 pages read." in at.success[0].value
    index = at.session_state.index
    assert index.count() > 0
    assert collections_for(at) == {index.name}
    assert index.search("How many people does Northwind employ?")[0].page_number == 3
    assert len(primary.calls) == 1, "a complex document gets one outline call"


def test_reruns_do_not_process_the_same_upload_again() -> None:
    at = start_app()
    upload(at, "northwind.pdf", make_pdf(["Coffee sales rose 8 percent this year.", ""]))
    index = at.session_state.index

    at.run()

    assert at.session_state.index is index and not index.is_deleted
    assert at.caption[-1].value == "1 page(s) had no selectable text and will be skipped."


def test_upload_problems_are_explained_and_leave_no_index() -> None:
    at = start_app()

    upload(at, "notes.pdf", b"plain text pretending to be a PDF")
    assert [e.value for e in at.error] == ["This file isn't a PDF. Upload a document saved in PDF format."]

    upload(at, "scan.pdf", make_image_only_pdf())
    assert "OCR" in at.error[0].value
    assert at.session_state.active is None and at.session_state.index is None
    assert collections_for(at) == set()


def test_a_bad_api_key_is_reported_but_the_document_still_loads(northwind_pdf: bytes) -> None:
    router, _, fallback = make_router(always(bad_credentials()), always(FIRST_HEADING))
    at = start_app(router)

    upload(at, "northwind.pdf", northwind_pdf)

    assert "is ready" in at.success[0].value
    assert "API key for the primary model" in at.warning[0].value
    assert fallback.calls == []


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (IngestionError("no valid chunks"), app.NO_PASSAGES),
        (RuntimeError("index unavailable"), app.PREPARE_FAILED),
    ],
)
def test_processing_failures_show_a_plain_message(monkeypatch, failure, message) -> None:
    def failing_ingest(*args, **kwargs):
        raise failure

    monkeypatch.setattr(app, "ingest", failing_ingest)
    at = start_app()

    upload(at, "northwind.pdf", make_pdf(["Coffee sales rose 8 percent this year."]))

    assert not at.exception
    assert [e.value for e in at.error] == [message]
    assert at.session_state.index is None
    assert collections_for(at) == set()


def test_replacing_the_document_drops_the_old_index(northwind_pdf: bytes) -> None:
    at = start_app()
    upload(at, "northwind.pdf", northwind_pdf)
    first = at.session_state.index

    upload(at, "tea.pdf", make_pdf(["Harbor Tea Company employed 85 people in 2024."]))

    assert first.is_deleted
    assert at.session_state.active.filename == "tea.pdf"
    assert collections_for(at) == {at.session_state.index.name}
    (passage,) = at.session_state.index.search("How many people work there?")
    assert passage.filename == "tea.pdf"


def test_removing_the_file_forgets_the_document_and_index() -> None:
    at = start_app()
    upload(at, "northwind.pdf", make_pdf(["Coffee sales rose 8 percent this year."]))
    index = at.session_state.index

    at.sidebar.file_uploader[0].set_value(None)
    at.run()

    assert at.session_state.active is None and index.is_deleted
    assert collections_for(at) == set()
    assert at.info[0].value == "Upload a PDF in the sidebar to get started."


def test_reset_clears_the_document_index_and_uploader() -> None:
    at = start_app()
    upload(at, "northwind.pdf", make_pdf(["Coffee sales rose 8 percent this year."]))
    index = at.session_state.index

    at.sidebar.button[0].click()
    at.run()

    assert at.session_state.active is None and index.is_deleted
    assert collections_for(at) == set()
    assert at.session_state.generation == 1
    assert at.sidebar.file_uploader[0].key == "uploader-1"


def test_two_browser_sessions_keep_separate_indexes(northwind_pdf: bytes) -> None:
    mine, theirs = start_app(), start_app()
    upload(mine, "northwind.pdf", northwind_pdf)
    upload(theirs, "northwind.pdf", northwind_pdf)

    mine.sidebar.button[0].click()
    mine.run()

    assert mine.session_state.session_id != theirs.session_state.session_id
    assert collections_for(mine) == set()
    assert theirs.session_state.index.count() > 0
    assert collections_for(theirs) == {theirs.session_state.index.name}
