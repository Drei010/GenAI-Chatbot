"""Shared fixtures. Every test runs without the developer's real provider settings."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator

import pytest

from genai_chatbot import config
from genai_chatbot.observability import LOGGER_NAME
from genai_chatbot.pdf_processing import extract_document
from genai_chatbot.schemas import ExtractedDocument
from tests.support.documents import long_report_pdf, report_pdf

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
def isolated_settings(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """No provider env vars, no .env file and no Streamlit secrets unless a test adds them.

    Live tests (marked `live_llm`, never part of the default run) use the developer's real settings.
    """
    if request.node.get_closest_marker("live_llm"):
        return
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_read_streamlit_secrets", lambda: {})
    monkeypatch.setitem(config.AppSettings.model_config, "env_file", None)


@pytest.fixture(autouse=True)
def restore_app_logger() -> Iterator[logging.Logger]:
    """The app's logging setup changes a process-wide logger; undo it after every test."""
    app_logger = logging.getLogger(LOGGER_NAME)
    saved = (list(app_logger.handlers), app_logger.propagate, app_logger.level)
    yield app_logger
    app_logger.handlers[:], app_logger.propagate, app_logger.level = saved


@pytest.fixture
def app_logs(caplog: pytest.LogCaptureFixture) -> Callable[[], list[dict]]:
    """Capture the app's structured log records; call the fixture's value to get them as dicts."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    logging.getLogger(LOGGER_NAME).propagate = True
    return lambda: [json.loads(r.getMessage()) for r in caplog.records if r.name == LOGGER_NAME]


@pytest.fixture
def valid_settings_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    for name, value in VALID_SETTINGS.items():
        monkeypatch.setenv(name, value)
    return dict(VALID_SETTINGS)


@pytest.fixture(scope="session")
def northwind_pdf() -> bytes:
    return report_pdf()


@pytest.fixture(scope="session")
def northwind_long_pdf() -> bytes:
    return long_report_pdf()


@pytest.fixture(scope="session")
def northwind(northwind_pdf: bytes) -> ExtractedDocument:
    """The five-page fictional annual report, extracted."""
    return extract_document(northwind_pdf, "northwind-annual-report.pdf")


@pytest.fixture(scope="session")
def northwind_long(northwind_long_pdf: bytes) -> ExtractedDocument:
    """The same report with appendix pages, 24 pages in all."""
    return extract_document(northwind_long_pdf, "northwind-annual-report-full.pdf")
