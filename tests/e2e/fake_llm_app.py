"""The real app wired to a rule-based stand-in for the AI provider, for the browser tests.

Playwright starts it with `uv run streamlit run tests/e2e/fake_llm_app.py`. It needs no API keys and
never calls a provider; tests/support/heuristic_llm.py describes the stand-in.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:  # Streamlit only puts this file's folder on the path
    sys.path.insert(0, str(PROJECT_ROOT))

import app  # noqa: E402
from genai_chatbot.config import load_settings  # noqa: E402
from tests.support.heuristic_llm import heuristic_router  # noqa: E402

BROWSER_TEST_SETTINGS = {
    "primary_llm_provider": "anthropic",
    "primary_llm_model": "browser-test-primary",
    "primary_llm_api_key": "browser-test-key",
    "fallback_llm_provider": "openai",
    "fallback_llm_model": "browser-test-fallback",
    "fallback_llm_api_key": "browser-test-key",
}

app.main(
    settings_loader=lambda: load_settings(**BROWSER_TEST_SETTINGS),
    router_factory=lambda settings: heuristic_router(),
)
