from __future__ import annotations

from streamlit.testing.v1 import AppTest

from tests.support.paths import PROJECT_ROOT

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
