"""Streamlit entry point for the PDF Knowledge Chatbot. Run with `uv run streamlit run app.py`."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from genai_chatbot.config import SettingsResult, SetupIssue, load_settings
from genai_chatbot.observability import configure_logging

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


def render_setup_message(issues: list[SetupIssue]) -> None:
    st.error("The chatbot needs AI provider settings before it can answer questions.", icon="⚙️")
    problems = "\n".join(f"- `{issue.setting}` {issue.problem}" for issue in issues)
    st.markdown(f"**Settings to check**\n\n{problems}")
    st.markdown(SETUP_HELP)


def main(settings_loader: Callable[[], SettingsResult] = load_settings) -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📄", layout="centered")
    st.title(APP_TITLE)
    st.caption("Upload a PDF, ask questions in plain language, and get answers with page-level sources.")

    result = settings_loader()
    if result.settings is None:
        render_setup_message(result.issues)
        st.stop()
    configure_logging(result.settings.log_level)

    st.info("Upload a PDF in the sidebar to get started.", icon="📄")


if __name__ == "__main__":
    main()
