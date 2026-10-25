"""Chat models behind a primary/fallback router, with provider-neutral error handling.

Models are built with LangChain's `init_chat_model`, so any provider it supports works the same way.
A rate-limit response from the primary model switches to the fallback model once. Credential and
request errors never fail over: retrying them elsewhere would hide a configuration problem.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    OutputParserException,
)
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError

from genai_chatbot.config import AppSettings, ProviderConfig, Route
from genai_chatbot.observability import Correlation, log_event

T = TypeVar("T", bound=BaseModel)

# Providers whose default structured-output method doesn't suit every current model. Recent Claude
# models reject forced tool choice, so Anthropic uses its native JSON-schema structured outputs.
STRUCTURED_OUTPUT_OPTIONS: dict[str, dict[str, Any]] = {"anthropic": {"method": "json_schema"}}


class ErrorKind(StrEnum):
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    INVALID_REQUEST = "invalid_request"
    MALFORMED_OUTPUT = "malformed_output"
    UNAVAILABLE = "unavailable"


_USER_MESSAGES = {
    ErrorKind.RATE_LIMIT: (
        "The AI service is getting too many requests right now, on both configured models. "
        "Please wait a minute and try again."
    ),
    ErrorKind.AUTH: (
        "The AI provider rejected the configured API key for the {route} model. Check the key in your "
        "settings, then restart the app."
    ),
    ErrorKind.INVALID_REQUEST: (
        "The AI provider rejected the request to the {route} model. Check that the configured model name "
        "is right for that provider."
    ),
    ErrorKind.MALFORMED_OUTPUT: "The AI service sent a response that couldn't be used. Please try again.",
    ErrorKind.UNAVAILABLE: "The AI service isn't responding right now. Please try again shortly.",
}


class ProviderError(Exception):
    """A model call failed. `user_message` is safe to show; it names no provider internals."""

    def __init__(self, kind: ErrorKind, route: Route | None = None) -> None:
        self.kind = kind
        self.route = route
        super().__init__(f"{kind.value} from the {route or 'unknown'} model")

    @property
    def user_message(self) -> str:
        return _USER_MESSAGES[self.kind].format(route=self.route or "configured")

    @property
    def is_configuration_problem(self) -> bool:
        return self.kind in (ErrorKind.AUTH, ErrorKind.INVALID_REQUEST)


def _causes(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(seen) < 10:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _status_code(exc: BaseException) -> int | None:
    for attribute in ("status_code", "code", "status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    value = getattr(getattr(exc, "response", None), "status_code", None)
    return value if isinstance(value, int) else None


def classify_error(exc: BaseException) -> ErrorKind:
    """Map any provider or parsing failure onto the few cases the app treats differently."""
    for cause in _causes(exc):
        if isinstance(cause, ModelRateLimitError):
            return ErrorKind.RATE_LIMIT
        if isinstance(cause, ModelAuthenticationError | ModelPermissionDeniedError):
            return ErrorKind.AUTH
        if isinstance(cause, ModelInvalidRequestError | ModelNotFoundError | ContextOverflowError):
            return ErrorKind.INVALID_REQUEST
        if isinstance(cause, OutputParserException | ValidationError | json.JSONDecodeError):
            return ErrorKind.MALFORMED_OUTPUT
        status = _status_code(cause)
        if status == 429 or type(cause).__name__ in {
            "RateLimitError",
            "ResourceExhausted",
            "TooManyRequests",
        }:
            return ErrorKind.RATE_LIMIT
        if status in (401, 403):
            return ErrorKind.AUTH
        if status in (400, 404, 413, 422):
            return ErrorKind.INVALID_REQUEST
    return ErrorKind.UNAVAILABLE


@dataclass
class ModelEndpoint:
    """A chat model plus the labels used to route and log calls to it."""

    route: Route
    provider: str
    model_name: str
    chat_model: Any
    _structured: dict[type, Any] = field(default_factory=dict, repr=False)

    def structured(self, schema: type[T]) -> Any:
        if schema not in self._structured:
            options = STRUCTURED_OUTPUT_OPTIONS.get(self.provider, {})
            self._structured[schema] = self.chat_model.with_structured_output(schema, **options)
        return self._structured[schema]


class LLMRouter:
    """Calls the primary model and, on a rate limit, the fallback model once."""

    def __init__(self, primary: ModelEndpoint, fallback: ModelEndpoint | None = None) -> None:
        self.primary = primary
        self.fallback = fallback

    def invoke_structured(
        self, schema: type[T], messages: Sequence[BaseMessage], *, stage: str, correlation: Correlation
    ) -> T:
        try:
            return self._call(self.primary, schema, messages, stage, correlation, fallback_used=False)
        except ProviderError as exc:
            if exc.kind is not ErrorKind.RATE_LIMIT or self.fallback is None:
                raise
        return self._call(self.fallback, schema, messages, stage, correlation, fallback_used=True)

    def _call(
        self,
        endpoint: ModelEndpoint,
        schema: type[T],
        messages: Sequence[BaseMessage],
        stage: str,
        correlation: Correlation,
        *,
        fallback_used: bool,
    ) -> T:
        started = time.perf_counter()
        error: ErrorKind | None = None
        try:
            result = endpoint.structured(schema).invoke(list(messages))
            if not isinstance(result, schema):
                error = ErrorKind.MALFORMED_OUTPUT
                raise ProviderError(error, endpoint.route)
            return result
        except ProviderError:
            raise
        except Exception as exc:
            error = classify_error(exc)
            raise ProviderError(error, endpoint.route) from exc
        finally:
            log_event(
                "llm_call",
                correlation,
                stage=stage,
                provider_route=endpoint.route,
                provider=endpoint.provider,
                model=endpoint.model_name,
                fallback_used=fallback_used,
                outcome="ok" if error is None else "error",
                error_kind=error.value if error else None,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )


def create_chat_model(config: ProviderConfig, *, timeout: float) -> Any:
    """Build a chat model through LangChain's provider-neutral factory.

    SDK retries are off so a 429 reaches the router immediately, and no sampling parameters are set:
    current Claude models and some OpenAI reasoning models accept only their defaults.
    """
    return init_chat_model(
        model=config.model,
        model_provider=config.provider,
        api_key=config.api_key.get_secret_value(),
        timeout=timeout,
        max_retries=0,
    )


def build_router(settings: AppSettings) -> LLMRouter:
    primary, fallback = (
        ModelEndpoint(
            route=config.route,
            provider=config.provider,
            model_name=config.model,
            chat_model=create_chat_model(config, timeout=settings.llm_timeout_seconds),
        )
        for config in (settings.primary, settings.fallback)
    )
    return LLMRouter(primary, fallback)
