"""Deterministic stand-ins for LangChain chat models, so no test ever calls a real provider."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.exceptions import ModelAuthenticationError, ModelInvalidRequestError, ModelRateLimitError
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda

from genai_chatbot.observability import Correlation
from genai_chatbot.providers import LLMRouter, ModelEndpoint

Responder = Callable[[type, list[BaseMessage]], Any]


class FakeChatModel:
    """Implements the slice of the chat-model API the app uses: `with_structured_output(schema)`.

    Each call asks the responder for a result. Returning an exception instance raises it, which is how
    tests simulate rate limits, bad credentials and malformed output.
    """

    def __init__(self, responder: Responder) -> None:
        self.responder = responder
        self.calls: list[tuple[type, list[BaseMessage]]] = []
        self.structured_options: list[dict[str, Any]] = []

    def with_structured_output(self, schema: type, **options: Any) -> RunnableLambda:
        self.structured_options.append(options)
        return RunnableLambda(lambda messages: self._invoke(schema, messages))

    def _invoke(self, schema: type, messages: list[BaseMessage]) -> Any:
        self.calls.append((schema, messages))
        result = self.responder(schema, messages)
        if isinstance(result, BaseException):
            raise result
        return result

    def calls_for(self, schema: type) -> list[list[BaseMessage]]:
        return [messages for called, messages in self.calls if called is schema]


class Script:
    """Answers each schema from a queue of prepared results (or a function of the messages)."""

    def __init__(self, responses: dict[type, list[Any] | Callable[[list[BaseMessage]], Any]] | None = None):
        self.responses = dict(responses or {})

    def __call__(self, schema: type, messages: list[BaseMessage]) -> Any:
        entry = self.responses.get(schema)
        if callable(entry):
            return entry(messages)
        if entry:
            return entry.pop(0)
        raise AssertionError(f"unexpected model call for {schema.__name__}")


def rate_limited() -> Exception:
    return ModelRateLimitError("Error code: 429 - rate limit exceeded")


def bad_credentials() -> Exception:
    return ModelAuthenticationError("Error code: 401 - invalid x-api-key")


def bad_request() -> Exception:
    return ModelInvalidRequestError("Error code: 400 - model not supported")


def always(result: Any) -> Responder:
    return lambda schema, messages: result


def make_router(
    primary: Responder, fallback: Responder | None = None
) -> tuple[LLMRouter, FakeChatModel, FakeChatModel | None]:
    primary_model = FakeChatModel(primary)
    fallback_model = FakeChatModel(fallback) if fallback else None
    router = LLMRouter(
        ModelEndpoint("primary", "anthropic", "fake-primary", primary_model),
        ModelEndpoint("fallback", "openai", "fake-fallback", fallback_model) if fallback_model else None,
    )
    return router, primary_model, fallback_model


TEST_CORRELATION = Correlation(session_id="test-session", request_id="test-request")
