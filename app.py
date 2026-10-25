"""Streamlit entry point for the PDF Knowledge Chatbot. Run with `uv run streamlit run app.py`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import streamlit as st

from genai_chatbot.config import SettingsResult, SetupIssue, load_settings
from genai_chatbot.observability import Correlation, configure_logging, log_stage, new_id
from genai_chatbot.pdf_processing import PDFProcessingError, extract_document

APP_TITLE = "PDF Knowledge Chatbot"

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
    """The one PDF this browser session is working with."""

    upload_key: str
    document_id: str
    filename: str
    page_count: int
    pages_without_text: int


def render_setup_message(issues: list[SetupIssue]) -> None:
    st.error("The chatbot needs AI provider settings before it can answer questions.", icon="⚙️")
    problems = "\n".join(f"- `{issue.setting}` {issue.problem}" for issue in issues)
    st.markdown(f"**Settings to check**\n\n{problems}")
    st.markdown(SETUP_HELP)


def init_session() -> None:
    state = st.session_state
    state.setdefault("session_id", new_id())
    state.setdefault("generation", 0)
    state.setdefault("active", None)
    state.setdefault("upload_error", None)


def correlation() -> Correlation:
    return Correlation(session_id=st.session_state.session_id)


def clear_document() -> None:
    """Forget the active document and everything derived from it."""
    st.session_state.active = None
    st.session_state.upload_error = None


def reset_session() -> None:
    clear_document()
    # A new uploader key empties the file widget as well.
    st.session_state.generation += 1


def process_upload(upload_key: str, filename: str, data: bytes) -> None:
    clear_document()
    with st.status(f"Reading {filename}…", expanded=False) as status:
        with log_stage("upload", correlation()) as details:
            try:
                document = extract_document(data, filename)
            except PDFProcessingError as exc:
                details.update(outcome="rejected", error_kind=exc.kind)
                status.update(label="This file can't be used", state="error")
                st.session_state.upload_error = (upload_key, exc.message)
                return
            details["page_count"] = document.page_count
        status.update(label=f"Read {document.page_count} pages", state="complete")
    st.session_state.active = ActiveDocument(
        upload_key=upload_key,
        document_id=document.document_id,
        filename=document.filename,
        page_count=document.page_count,
        pages_without_text=document.pages_without_text,
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
            "Upload a PDF",
            type=["pdf"],
            key=f"uploader-{st.session_state.generation}",
            help="One PDF at a time. It stays in memory for this browser session only.",
        )
        details = st.container()
        controls = st.container()

    handle_upload(uploaded)

    active: ActiveDocument | None = st.session_state.active
    if active:
        details.markdown(f"**{active.filename}**  \n{active.page_count} pages")
    controls.button(
        "Reset chat and document",
        on_click=reset_session,
        disabled=active is None and uploaded is None,
        help="Clears the conversation and removes the document from memory.",
        width="stretch",
    )


def render_main_area() -> None:
    failed = st.session_state.upload_error
    if failed:
        st.error(failed[1], icon="🚫")
        return
    active: ActiveDocument | None = st.session_state.active
    if active is None:
        st.info("Upload a PDF in the sidebar to get started.", icon="📄")
        return
    st.success(f"**{active.filename}** is ready: {active.page_count} pages read.", icon="✅")
    if active.pages_without_text:
        st.caption(f"{active.pages_without_text} page(s) had no selectable text and will be skipped.")


def main(settings_loader: Callable[[], SettingsResult] = load_settings) -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="centered")
    st.title(APP_TITLE)
    st.caption("Upload a PDF, ask questions in plain language, and get answers with page-level sources.")

    result = settings_loader()
    if result.settings is None:
        render_setup_message(result.issues)
        st.stop()
    configure_logging(result.settings.log_level)

    init_session()
    render_sidebar()
    render_main_area()


if __name__ == "__main__":
    main()
