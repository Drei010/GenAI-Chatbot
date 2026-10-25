"""Shared fixtures. Every test runs without the developer's real provider settings."""

from __future__ import annotations

import pytest

from genai_chatbot import config

SETTINGS_ENV_VARS = [name.upper() for name in config.AppSettings.model_fields]

VALID_SETTINGS = {
    "PRIMARY_LLM_PROVIDER": "anthropic",
    "PRIMARY_LLM_MODEL": "claude-opus-5-5",
    "PRIMARY_LLM_API_KEY": "test-primary-key",
    "FALLBACK_LLM_PROVIDER": "openai",
    "FALLBACK_LLM_MODEL": "gpt-test-mini",
    "FALLBACK_LLM_API_KEY": "test-fallback-key",
}


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """No provider env vars, no .env file and no Streamlit secrets unless a test adds them."""
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_read_streamlit_secrets", lambda: {})
    monkeypatch.setitem(config.AppSettings.model_config, "env_file", None)


@pytest.fixture
def valid_settings_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    for name, value in VALID_SETTINGS.items():
        monkeypatch.setenv(name, value)
    return dict(VALID_SETTINGS)
