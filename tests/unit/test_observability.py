from __future__ import annotations

import json
import logging

import pytest

from genai_chatbot.observability import (
    LOGGER_NAME,
    Correlation,
    configure_logging,
    log_event,
    log_stage,
    safe_fields,
)


def records(caplog: pytest.LogCaptureFixture) -> list[dict]:
    return [json.loads(r.getMessage()) for r in caplog.records if r.name == LOGGER_NAME]


def test_only_allowlisted_scalar_fields_are_kept() -> None:
    kept = safe_fields(
        {
            "stage": "retrieve",
            "retrieval_count": 4,
            "fallback_used": False,
            "question": "What was revenue?",
            "excerpt": "Revenue was $211.9 billion",
            "api_key": "sk-secret",
            "provider": "x" * 200,
            "outcome": ["not", "a", "scalar"],
        }
    )

    assert kept == {"stage": "retrieve", "retrieval_count": 4, "fallback_used": False, "provider": "x" * 64}


def test_log_event_carries_correlation_ids(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    correlation = Correlation(session_id="sess123")

    log_event("upload", correlation, page_count=12, filename="secret-plan.pdf")

    (record,) = records(caplog)
    assert record == {
        "event": "upload",
        "session_id": "sess123",
        "request_id": correlation.request_id,
        "page_count": 12,
    }


def test_log_stage_times_the_block_and_records_details(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    with log_stage("retrieve", Correlation("s1", "r1"), route="direct") as details:
        details["retrieval_count"] = 3

    (record,) = records(caplog)
    assert record["stage"] == "retrieve" and record["route"] == "direct"
    assert record["retrieval_count"] == 3 and record["outcome"] == "ok"
    assert record["duration_ms"] >= 0


def test_log_stage_marks_errors_without_the_message(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    with pytest.raises(ValueError), log_stage("extract", Correlation("s1", "r2")):
        raise ValueError("page text that must not be logged")

    (record,) = records(caplog)
    assert record["outcome"] == "error" and record["error_kind"] == "ValueError"
    assert "must not be logged" not in json.dumps(record)


def test_new_requests_keep_the_session_id() -> None:
    first = Correlation(session_id="abc")
    second = first.for_new_request()

    assert second.session_id == "abc" and second.request_id != first.request_id


def test_configure_logging_is_idempotent_and_quiets_document_loggers(restore_app_logger) -> None:
    configure_logging("DEBUG")
    configure_logging("INFO")

    ours = [h for h in restore_app_logger.handlers if getattr(h, "_genai_chatbot", False)]
    assert len(ours) == 1 and restore_app_logger.level == logging.INFO
    assert logging.getLogger("pypdf").level == logging.ERROR
