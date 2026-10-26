"""Streamlit entry point for the PDF Knowledge Chatbot. Run with `uv run streamlit run app.py`."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import streamlit as st

from genai_chatbot.config import AppSettings, SettingsResult, SetupIssue, load_settings
from genai_chatbot.documents import DocumentError, document_fingerprint
from genai_chatbot.graphs.ingestion import IngestionError, build_ingestion_graph, ingest
from genai_chatbot.graphs.qa import MAX_QUESTION_CHARS, Conversation
from genai_chatbot.observability import Correlation, configure_logging, log_stage, new_id
from genai_chatbot.providers import LLMRouter, build_router
from genai_chatbot.schemas import ChatAnswer, ChatTurn, Citation
from genai_chatbot.vector_store import DocumentIndex, warm_up

APP_TITLE = "PDF Knowledge Chatbot"

RouterFactory = Callable[[AppSettings], LLMRouter]

NO_PASSAGES = "This document's text couldn't be prepared for questions. Try another copy of the file."
PREPARE_FAILED = "Something went wrong while preparing this document. Please try again."
ASK_PLACEHOLDER = "Ask a question about the document"
NO_DOCUMENT_PLACEHOLDER = "Upload a document to start asking questions"
NO_PAGE_NUMBERS = (
    "This Word file doesn't record where its pages break, so each source shows its section and an "
    "excerpt instead of a page number."
)
EXAMPLES = (
    "Try a figure (“What was revenue in the last fiscal year?”), a summary (“Summarize the key facts”) "
    "or a comparison (“Compare the segments in a table”). Follow-up questions work too."
)

_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|>~<$])")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_LINK_DEFINITION = re.compile(r"^ {0,3}\[[^\]]+\]:.*$", re.MULTILINE)
_NAME_SEPARATOR = re.compile(r"([_.\-])")


def escape_markdown(text: str) -> str:
    """Show document or user text literally: no formatting, links, or math from its characters."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def breakable(name: str) -> str:
    """Let a long filename wrap after its separators (_ . -) rather than mid-word in the narrow sidebar."""
    return _NAME_SEPARATOR.sub("\\1\u200b", name)


def answer_markdown(text: str) -> str:
    """Keep the answer's lists, tables and emphasis, but drop images and links (text planted in a
    document could make the browser fetch a URL) and keep dollar signs from being read as math."""
    text = _IMAGE.sub(r"\1", text)
    text = _LINK.sub(r"\1", text)
    text = _LINK_DEFINITION.sub("", text)
    return text.replace("$", r"\$").replace("~", r"\~")


SETUP_HELP = """
**How to fix it**

1. Copy `.env.example` to `.env` in the project folder, or put the same keys at the top level of
   `.streamlit/secrets.toml`.
2. Set a provider, model and API key for both the primary and the fallback model.
3. Restart the app.

For example:

```bash
PRIMARY_LLM_PROVIDER=anthropic
PRIMARY_LLM_MODEL=claude-opus-5-5
PRIMARY_LLM_API_KEY=your-anthropic-key
FALLBACK_LLM_PROVIDER=openai
FALLBACK_LLM_MODEL=your-openai-model
FALLBACK_LLM_API_KEY=your-openai-key
```

Supported providers: `anthropic`, `openai`, `google_genai`, `groq` and `mistralai`. The README's
*Provider setup* section has more examples.
"""


@dataclass(frozen=True)
class ActiveDocument:
    """The one document this browser session is working with."""

    upload_key: str
    document_id: str
    filename: str
    page_count: int
    pages_without_text: int
    notices: tuple[str, ...] = ()
    has_page_numbers: bool = True


def render_setup_message(issues: list[SetupIssue]) -> None:
    st.error("The chatbot needs AI provider settings before it can answer questions.", icon="⚙️")
    problems = "\n".join(f"- `{issue.setting}` {issue.problem}" for issue in issues)
    st.markdown(f"**Settings to check**\n\n{problems}")
    st.markdown(SETUP_HELP)


@st.cache_resource(show_spinner="Getting ready…")
def prepare_search_model() -> bool:
    """Load the local embedding model once per process, before anyone uploads a file."""
    warm_up()
    return True


def init_session(settings: AppSettings, router_factory: RouterFactory) -> None:
    state = st.session_state
    state.setdefault("session_id", new_id())
    state.setdefault("generation", 0)
    state.setdefault("active", None)
    state.setdefault("upload_error", None)
    state.setdefault("index", None)
    state.setdefault("conversation", None)
    if state.get("router") is None:
        state.router = router_factory(settings)


def correlation() -> Correlation:
    return Correlation(session_id=st.session_state.session_id)


def clear_document() -> None:
    """Forget the active document and everything derived from it: the chat thread and the index."""
    state = st.session_state
    if state.get("conversation") is not None:
        state.conversation.clear()
    if state.get("index") is not None:
        state.index.delete()
    state.conversation = None
    state.index = None
    state.active = None
    state.upload_error = None


def reset_session() -> None:
    clear_document()
    # A new uploader key empties the file widget as well.
    st.session_state.generation += 1


def _upload_failure(exc: Exception) -> tuple[str, str, str]:
    """(outcome, error kind, user message) for a failed upload. Unexpected errors stay generic."""
    if isinstance(exc, DocumentError):
        return "rejected", exc.kind, exc.message
    if isinstance(exc, IngestionError):
        return "rejected", "no_passages", NO_PASSAGES
    return "error", type(exc).__name__, PREPARE_FAILED


def process_upload(upload_key: str, filename: str, data: bytes) -> None:
    """Read, chunk and index a new upload, showing progress; replaces any previous document."""
    clear_document()
    state = st.session_state
    request = correlation()
    index = DocumentIndex(state.session_id, document_fingerprint(data))
    graph = build_ingestion_graph(index, state.router, request)

    # Progress shows while the file is prepared; the banner in the main area then reports the outcome.
    slot = st.empty()
    with slot.status(f"Preparing {escape_markdown(filename)}…", expanded=True):
        progress = st.progress(0.0, text="Reading the document…")

        def show_progress(event: dict[str, Any]) -> None:
            progress.progress(event["fraction"], text=event["message"])

        with log_stage("upload", request) as details:
            try:
                result = ingest(graph, filename, data, on_progress=show_progress)
            except Exception as exc:
                index.delete()
                outcome, kind, message = _upload_failure(exc)
                details.update(outcome=outcome, error_kind=kind)
                state.upload_error = (upload_key, message)
                result = None
            else:
                details.update(
                    page_count=result.page_count, chunk_count=result.chunk_count, strategy=result.strategy
                )
    slot.empty()
    if result is None:
        return

    state.index = index
    state.conversation = Conversation(
        index,
        state.router,
        session_id=state.session_id,
        filename=result.filename,
        page_count=result.page_count,
    )
    state.active = ActiveDocument(
        upload_key=upload_key,
        document_id=result.document_id,
        filename=result.filename,
        page_count=result.page_count,
        pages_without_text=result.pages_without_text,
        notices=result.notices,
        has_page_numbers=result.has_page_numbers,
    )


def handle_upload(uploaded: st.runtime.uploaded_file_manager.UploadedFile | None) -> None:
    """Process a new or replaced upload; forget the document when the file is removed."""
    if uploaded is None:
        if st.session_state.active or st.session_state.upload_error:
            clear_document()
        return
    active: ActiveDocument | None = st.session_state.active
    failed = st.session_state.upload_error
    if (active and active.upload_key == uploaded.file_id) or (failed and failed[0] == uploaded.file_id):
        return
    process_upload(uploaded.file_id, uploaded.name, uploaded.getvalue())


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Document")
        uploaded = st.file_uploader(
            "Upload a PDF or Word file",
            type=["pdf", "docx"],
            key=f"uploader-{st.session_state.generation}",
            help="One PDF or Word (.docx) file at a time. It stays in memory for this browser session only.",
        )
        details = st.container()
        controls = st.container()

    handle_upload(uploaded)

    active: ActiveDocument | None = st.session_state.active
    if active:
        size = f"{active.page_count} pages" if active.has_page_numbers else "No page numbers in this file"
        details.markdown(f"**{escape_markdown(breakable(active.filename))}**  \n{size}")
    controls.button(
        "Reset chat and document",
        on_click=reset_session,
        disabled=active is None and uploaded is None,
        help="Clears the conversation and removes the document from memory.",
        width="stretch",
    )


def source_location(citation: Citation, show_pages: bool) -> str:
    """The file and page of a source or, for a Word file without page numbers, its section."""
    name = escape_markdown(citation.filename)
    if show_pages:
        return f"{name}, page {citation.page_number}"
    if citation.section_title:
        section = " › ".join(title.strip() for title in citation.section_title.split(" > "))
        return f"{name}, section “{escape_markdown(section)}”"
    return name


def render_sources(citations: Sequence[Citation], show_pages: bool) -> None:
    with st.container(border=True):
        st.markdown("**Sources**")
        for citation in citations:
            st.markdown(f"**[{citation.number}]** {source_location(citation, show_pages)}")
            st.caption(f"“{escape_markdown(citation.excerpt)}”")


def render_answer(answer: ChatAnswer, show_pages: bool) -> None:
    if answer.status == "answered":
        st.markdown(answer_markdown(answer.text))
        render_sources(answer.citations, show_pages)
    elif answer.status == "unsupported":
        st.info(answer.text, icon="🔎")
    else:
        st.warning(answer.text, icon="⚠️")


def render_turn(turn: ChatTurn, show_pages: bool) -> None:
    with st.chat_message(turn.role):
        if turn.answer is not None:
            render_answer(turn.answer, show_pages)
        else:
            st.markdown(escape_markdown(turn.content))


def answer_question(conversation: Conversation, question: str, show_pages: bool) -> None:
    """Show the question, then progress while the answer is worked out, then the answer."""
    render_turn(ChatTurn(role="user", content=question), show_pages)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        with placeholder.status("Working on your question…") as status:
            answer = conversation.ask(
                question, on_progress=lambda event: status.update(label=event["message"])
            )
        placeholder.empty()
        render_answer(answer, show_pages)


def render_main_area() -> None:
    failed = st.session_state.upload_error
    active: ActiveDocument | None = st.session_state.active
    conversation: Conversation | None = st.session_state.conversation
    question = st.chat_input(
        ASK_PLACEHOLDER if conversation else NO_DOCUMENT_PLACEHOLDER,
        max_chars=MAX_QUESTION_CHARS,
        disabled=conversation is None,
        submit_mode="disable",
    )
    if failed:
        st.error(failed[1], icon="🚫")
        return
    if active is None or conversation is None:
        st.info("Upload a PDF or Word file in the sidebar to get started.", icon="📄")
        return
    name = escape_markdown(active.filename)
    if active.has_page_numbers:
        st.success(f"**{name}** is ready: {active.page_count} pages read.", icon="✅")
    else:
        st.success(f"**{name}** is ready.", icon="✅")
        st.caption(NO_PAGE_NUMBERS)
    if active.pages_without_text:
        st.caption(f"{active.pages_without_text} page(s) had no text to read and will be skipped.")
    for notice in active.notices:
        st.warning(notice, icon="⚠️")

    history = conversation.history()
    if not history and not question:
        st.caption(EXAMPLES)
    for turn in history:
        render_turn(turn, active.has_page_numbers)
    if question:
        answer_question(conversation, question, active.has_page_numbers)


def main(
    settings_loader: Callable[[], SettingsResult] = load_settings,
    router_factory: RouterFactory = build_router,
) -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="centered")
    st.title(APP_TITLE)
    st.caption(
        "Upload a PDF or Word document, ask questions in plain language, and get answers that cite "
        "their sources."
    )

    result = settings_loader()
    if result.settings is None:
        render_setup_message(result.issues)
        st.stop()
    configure_logging(result.settings.log_level)
    prepare_search_model()

    init_session(result.settings, router_factory)
    render_sidebar()
    render_main_area()


if __name__ == "__main__":
    main()
