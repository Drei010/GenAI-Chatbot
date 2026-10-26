"""The Streamlit app, driven in-process with AppTest and a fake model router."""

from __future__ import annotations

import re
from pathlib import PurePath

import pytest
from streamlit.testing.v1 import AppTest

import app
from genai_chatbot.graphs.ingestion import IngestionError
from genai_chatbot.graphs.qa import UNSUPPORTED
from genai_chatbot.providers import LLMRouter
from genai_chatbot.schemas import AnswerDraft, Citation, OutlineHeading, QueryPlan, SectionOutline
from genai_chatbot.vector_store import get_client
from tests.support import heuristic_llm
from tests.support.docx import PAGE_BREAK, RENDERED_BREAK, compound_file, make_docx, paragraph
from tests.support.fake_llm import Script, always, bad_credentials, make_router, rate_limited
from tests.support.paths import PROJECT_ROOT
from tests.support.pdfs import make_image_only_pdf, make_pdf

APP_PATH = str(PROJECT_ROOT / "app.py")
FIRST_HEADING = SectionOutline(headings=[OutlineHeading(candidate_id="h1", level=1)])
TEA_PDF = make_pdf(["Harbor Tea Company employed 85 people in 2024."])
MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~<$])")


def run_app(router) -> None:
    """AppTest runs this function's source as the page script, so it imports what it needs itself."""
    import app

    app.main(router_factory=lambda settings: router)


def start_app(router: LLMRouter | None = None) -> AppTest:
    router = router or make_router(always(FIRST_HEADING))[0]
    return AppTest.from_function(run_app, kwargs={"router": router}, default_timeout=60).run()


def upload(at: AppTest, filename: str, data: bytes) -> AppTest:
    mime_type = MIME_TYPES.get(PurePath(filename).suffix, "application/octet-stream")
    at.sidebar.file_uploader[0].set_value((filename, data, mime_type))
    return at.run()


def collections_for(at: AppTest) -> set[str]:
    prefix = f"pdfchat-{at.session_state.session_id}-"
    return {c.name for c in get_client().list_collections() if c.name.startswith(prefix)}


def ask(at: AppTest, question: str) -> AppTest:
    at.chat_input[0].set_value(question)
    return at.run()


def shown(markdown: str) -> str:
    """Markdown source as the reader sees it, without the escapes that keep it literal."""
    return _ESCAPE.sub(r"\1", markdown)


def chatting_about_northwind(northwind_pdf: bytes, router: LLMRouter | None = None) -> AppTest:
    at = start_app(router or heuristic_llm.heuristic_router())
    return upload(at, "northwind.pdf", northwind_pdf)


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
    assert at.info[0].value == "Upload a PDF or Word file in the sidebar to get started."


def test_uploading_a_pdf_reads_and_indexes_it(northwind_pdf: bytes) -> None:
    router, primary, _ = make_router(always(FIRST_HEADING))
    at = start_app(router)

    upload(at, "northwind.pdf", northwind_pdf)

    assert not at.exception and not at.error
    assert shown(at.success[0].value) == "**northwind.pdf** is ready: 5 pages read."
    assert shown(at.sidebar.markdown[0].value) == "**northwind.\u200bpdf**  \n5 pages"
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
    assert "1 page(s) had no text to read and will be skipped." in [c.value for c in at.caption]


def test_upload_problems_are_explained_and_leave_no_index() -> None:
    at = start_app()

    upload(at, "notes.pdf", b"plain text pretending to be a PDF")
    assert [e.value for e in at.error] == [
        "This file isn't a PDF or a Word (.docx) document. Upload a file saved in one of those formats."
    ]

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
    assert at.info[0].value == "Upload a PDF or Word file in the sidebar to get started."


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


def test_questions_wait_for_a_document() -> None:
    at = start_app()

    (chat,) = at.chat_input
    assert chat.proto.disabled and chat.placeholder == app.NO_DOCUMENT_PLACEHOLDER


def test_a_question_gets_an_answer_with_its_sources(northwind_pdf: bytes) -> None:
    at = chatting_about_northwind(northwind_pdf)
    assert not at.chat_input[0].proto.disabled
    assert app.EXAMPLES in [c.value for c in at.caption]

    ask(at, "How many people did Northwind employ?")

    assert not at.exception
    user, assistant = at.chat_message
    assert (user.name, shown(user.markdown[0].value)) == ("user", "How many people did Northwind employ?")
    answer, heading, source = (shown(m.value) for m in assistant.markdown)
    assert "2,340 people" in answer and answer.endswith("[1]")
    assert heading == "**Sources**"
    assert source == "**[1]** northwind.pdf, page 3"
    assert assistant.caption[0].value.startswith("“Northwind employed 2,340 people")
    assert not assistant.status, "the progress indicator gives way to the answer"
    assert app.EXAMPLES not in [c.value for c in at.caption]


def test_follow_up_questions_build_on_the_conversation(northwind_pdf: bytes) -> None:
    at = chatting_about_northwind(northwind_pdf)
    ask(at, "How many people did Northwind employ?")

    ask(at, "And how many retail stores did it operate?")

    assert [m.name for m in at.chat_message] == ["user", "assistant", "user", "assistant"]
    latest = at.chat_message[3]
    assert "186 retail stores" in latest.markdown[0].value
    assert shown(latest.markdown[2].value) == "**[1]** northwind.pdf, page 3"
    assert len(at.session_state.conversation.history()) == 4


def test_summaries_and_comparisons_keep_their_format(northwind_pdf: bytes) -> None:
    at = chatting_about_northwind(northwind_pdf)

    ask(at, "Summarize the key facts")
    summary = at.chat_message[1].markdown[0].value
    assert summary.startswith("- ") and summary.count("\n- ") >= 2

    ask(at, "Compare revenue by segment in a table")
    table = at.chat_message[3].markdown[0].value.splitlines()
    assert table[0] == "| Item | Figures | Source |"
    assert table[2].startswith("| Retail stores | \\$ 251.4"), "dollar signs are not read as math"


def test_unsupported_questions_get_a_plain_answer_without_sources(northwind_pdf: bytes) -> None:
    at = chatting_about_northwind(northwind_pdf)

    ask(at, "What is the capital of France?")

    assistant = at.chat_message[1]
    assert [i.value for i in assistant.info] == [UNSUPPORTED]
    assert not assistant.markdown


def test_a_failed_answer_explains_the_problem_without_sources(northwind_pdf: bytes) -> None:
    script = Script(
        {SectionOutline: heuristic_llm.outline, QueryPlan: heuristic_llm.plan, AnswerDraft: rate_limited()}
    )
    at = chatting_about_northwind(northwind_pdf, make_router(script)[0])

    ask(at, "How many people did Northwind employ?")

    assistant = at.chat_message[1]
    assert "too many requests" in assistant.warning[0].value
    assert not assistant.markdown
    assert not at.chat_input[0].proto.disabled, "the next question can be asked"


def test_reset_clears_the_conversation(northwind_pdf: bytes) -> None:
    at = chatting_about_northwind(northwind_pdf)
    ask(at, "How many people did Northwind employ?")
    conversation = at.session_state.conversation

    at.sidebar.button[0].click()
    at.run()

    assert at.session_state.conversation is None and conversation.history() == []
    assert not at.chat_message
    assert at.chat_input[0].proto.disabled


def test_a_new_document_starts_a_new_conversation(northwind_pdf: bytes) -> None:
    at = chatting_about_northwind(northwind_pdf)
    ask(at, "How many people did Northwind employ?")
    first = at.session_state.conversation

    upload(at, "tea.pdf", TEA_PDF)
    assert not at.chat_message and first.history() == []

    ask(at, "How many people did Harbor Tea employ?")
    assistant = at.chat_message[1]
    assert "85 people" in assistant.markdown[0].value
    assert shown(assistant.markdown[2].value) == "**[1]** tea.pdf, page 1"


def tea_docx(*, recorded_pages: bool) -> bytes:
    """Harbor Tea's review as a Word file: two pages Word recorded, or the same text with no page breaks."""
    breaks = (PAGE_BREAK, RENDERED_BREAK) if recorded_pages else ("", "")
    return make_docx(
        [
            paragraph("Harbor Tea Company", style="Title"),
            paragraph("Harbor Tea Company employed 85 people at the end of 2024.", breaks[0]),
            paragraph(breaks[1], "Shops and Growth", style="Heading2"),
            paragraph("Harbor Tea opened two new shops in Portland during 2024."),
        ],
        pages=2 if recorded_pages else None,
    )


def test_the_uploader_takes_pdf_and_word_files() -> None:
    at = start_app()

    assert list(at.sidebar.file_uploader[0].proto.type) == [".pdf", ".docx"]


def test_a_word_document_with_recorded_pages_cites_page_numbers() -> None:
    at = start_app(heuristic_llm.heuristic_router())
    upload(at, "harbor-tea.docx", tea_docx(recorded_pages=True))

    assert not at.exception and not at.error
    assert shown(at.success[0].value) == "**harbor-tea.docx** is ready: 2 pages read."
    assert shown(at.sidebar.markdown[0].value) == "**harbor-\u200btea.\u200bdocx**  \n2 pages"

    ask(at, "How many new shops did Harbor Tea open?")

    assistant = at.chat_message[1]
    assert "two new shops" in assistant.markdown[0].value
    assert shown(assistant.markdown[2].value) == "**[1]** harbor-tea.docx, page 2"


def test_a_word_document_without_page_numbers_cites_sections() -> None:
    at = start_app(heuristic_llm.heuristic_router())
    upload(at, "harbor-tea.docx", tea_docx(recorded_pages=False))

    assert not at.exception and not at.error
    assert shown(at.success[0].value) == "**harbor-tea.docx** is ready."
    assert app.NO_PAGE_NUMBERS in [c.value for c in at.caption]
    assert (
        shown(at.sidebar.markdown[0].value)
        == "**harbor-\u200btea.\u200bdocx**  \nNo page numbers in this file"
    )

    ask(at, "How many new shops did Harbor Tea open?")
    ask(at, "How many people did Harbor Tea employ?")

    first, second = at.chat_message[1], at.chat_message[3]
    assert shown(first.markdown[2].value) == (
        "**[1]** harbor-tea.docx, section “Harbor Tea Company › Shops and Growth”"
    )
    assert shown(second.markdown[2].value) == "**[1]** harbor-tea.docx, section “Harbor Tea Company”"


def test_sources_name_their_page_or_section() -> None:
    citation = Citation(
        number=1,
        chunk_id="c-1",
        filename="tea_review.docx",
        page_number=3,
        excerpt="Harbor Tea opened two new shops.",
        section_title="Our Year > Shops [2024]",
    )

    assert shown(app.source_location(citation, show_pages=True)) == "tea_review.docx, page 3"
    assert shown(app.source_location(citation, show_pages=False)) == (
        "tea_review.docx, section “Our Year › Shops [2024]”"
    )
    untitled = citation.model_copy(update={"section_title": None})
    assert shown(app.source_location(untitled, show_pages=False)) == "tea_review.docx"


def test_word_upload_problems_are_explained() -> None:
    at = start_app()

    upload(at, "minutes.docx", compound_file("WordDocument", "1Table"))

    assert at.error[0].value.startswith("This is an older Word (.doc) file")
    assert at.session_state.active is None and collections_for(at) == set()


def test_answers_keep_formatting_but_lose_links_images_and_math() -> None:
    text = (
        "Revenue was $412.6 million [1] ![chart](https://example.com/c.png) [site](https://example.com) ~5%\n"
        "![pixel][r]\n[r]: https://example.com/p.png"
    )

    assert app.answer_markdown(text) == ("Revenue was \\$412.6 million [1] chart site \\~5%\n![pixel][r]\n")


def test_long_filenames_can_wrap_after_their_separators() -> None:
    assert app.breakable("microsoft_2023_annual-report.pdf") == (
        "microsoft_\u200b2023_\u200bannual-\u200breport.\u200bpdf"
    )


def test_user_and_document_text_is_shown_literally() -> None:
    text = "Report *v2* [draft](https://example.com) $5 #1 1. ~x~ <b>"

    escaped = app.escape_markdown(text)

    assert shown(escaped) == text
    assert re.search(r"(?<!\\)[*\[\]$#~<]", escaped) is None, "every special character is escaped"
