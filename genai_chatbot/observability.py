"""Structured logging that records how the workflows behave without recording what users sent.

Only allowlisted keys with scalar values reach the log, so document text, prompts, questions,
excerpts and keys can't leak through a stray keyword argument.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

LOGGER_NAME = "genai_chatbot"
logger = logging.getLogger(LOGGER_NAME)

SAFE_FIELDS = frozenset(
    {
        "chunk_count",
        "citation_count",
        "complex",
        "duration_ms",
        "error_kind",
        "fallback_used",
        "model",
        "outcome",
        "page_count",
        "passage_count",
        "provider",
        "provider_route",
        "query_count",
        "response_style",
        "retrieval_count",
        "route",
        "stage",
        "strategy",
    }
)
_MAX_VALUE_LENGTH = 64


def new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class Correlation:
    """Ties log lines to one browser session and one request (an upload or a question)."""

    session_id: str
    request_id: str = field(default_factory=new_id)

    def for_new_request(self) -> Correlation:
        return Correlation(session_id=self.session_id)


def safe_fields(fields: Mapping[str, object]) -> dict[str, object]:
    """Keep allowlisted keys whose values are scalars; truncate strings."""
    kept: dict[str, object] = {}
    for key, value in fields.items():
        if key not in SAFE_FIELDS:
            continue
        if value is None or isinstance(value, bool | int | float):
            kept[key] = value
        elif isinstance(value, str):
            kept[key] = value[:_MAX_VALUE_LENGTH]
    return kept


def log_event(event: str, correlation: Correlation, **fields: object) -> None:
    record: dict[str, object] = {
        "event": event,
        "session_id": correlation.session_id,
        "request_id": correlation.request_id,
    }
    record.update(safe_fields(fields))
    logger.info(json.dumps(record, sort_keys=True))


@contextmanager
def log_stage(stage: str, correlation: Correlation, **fields: object) -> Iterator[dict[str, object]]:
    """Time a workflow stage and log one line when it ends.

    The yielded dict collects details the stage learns while running (retrieval_count, strategy...).
    """
    details: dict[str, object] = dict(fields)
    start = time.perf_counter()
    try:
        yield details
    except Exception as exc:
        details.setdefault("outcome", "error")
        details.setdefault("error_kind", type(exc).__name__)
        raise
    finally:
        details.setdefault("outcome", "ok")
        details["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
        log_event("stage", correlation, stage=stage, **details)


def configure_logging(level: str = "INFO") -> None:
    """Send the app's log lines to stderr once, and quiet libraries that log document details."""
    if not any(getattr(handler, "_genai_chatbot", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        handler._genai_chatbot = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    for noisy in ("pypdf", "httpx", "httpx2", "httpcore", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
