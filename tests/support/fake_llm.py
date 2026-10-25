"""Deterministic stand-ins for LangChain chat models, so no test ever calls a real provider."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from langchain_core.exceptions import ModelAuthenticationError, ModelInvalidRequestError, ModelRateLimitError
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda

from genai_chatbot.observability import Correlation
from genai_chatbot.providers import LLMRouter, ModelEndpoint
from genai_chatbot.schemas import AnswerDraft, QueryPlan, ResponseStyle

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
    """Answers each schema with a fixed result, a queue of results, or a function of the messages.

    Queue items may themselves be functions of the messages. A result that is an exception is raised.
    """

    def __init__(self, responses: dict[type, list[Any] | Callable[[list[BaseMessage]], Any]] | None = None):
        self.responses = dict(responses or {})

    def __call__(self, schema: type, messages: list[BaseMessage]) -> Any:
        if schema not in self.responses:
            raise AssertionError(f"unexpected model call for {schema.__name__}")
        entry = self.responses[schema]
        if isinstance(entry, list):
            if not entry:
                raise AssertionError(f"no scripted {schema.__name__} responses left")
            entry = entry.pop(0)
        return entry(messages) if _is_responder(entry) else entry


def _is_responder(item: Any) -> bool:
    return callable(item) and not isinstance(item, BaseException | type)


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


# Question-answering helpers: read the passages a prompt offered and answer from them.

_PASSAGE = re.compile(r'<passage id="(P\d+)"[^\n]*>\n(.*?)\n</passage>', re.DOTALL)


def passages_in(messages: list[BaseMessage]) -> dict[str, str]:
    """Passage label -> text, as shown to the model in the answer prompt."""
    return dict(_PASSAGE.findall(messages[-1].content))


def label_for(messages: list[BaseMessage], fact: str) -> str:
    for label, text in passages_in(messages).items():
        if fact in text:
            return label
    raise AssertionError(f"no retrieved passage contains {fact!r}")


def plan(*queries: str, style: ResponseStyle = "prose") -> QueryPlan:
    return QueryPlan(decompose=len(queries) > 1, queries=list(queries), response_style=style)


def cite(template: str, style: ResponseStyle = "prose", **facts: str) -> Callable[[list[BaseMessage]], Any]:
    """An answer responder: `template` with each {name} replaced by a citation of the passage holding
    facts[name], so tests don't depend on retrieval order."""

    def respond(messages: list[BaseMessage]) -> AnswerDraft:
        labels = {name: label_for(messages, fact) for name, fact in facts.items()}
        return AnswerDraft(
            response_style=style,
            answer=template.format(**{name: f"[{label}]" for name, label in labels.items()}),
            citation_ids=sorted(set(labels.values())),
            status="supported",
        )

    return respond
