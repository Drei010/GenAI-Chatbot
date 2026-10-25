"""Error classification, primary/fallback routing and provider-neutral model construction."""

from __future__ import annotations

import json

import anthropic
import groq
import httpx
import openai
import pytest
from google.genai import errors as google_errors
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    OutputParserException,
)
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, SecretStr, ValidationError

from genai_chatbot import providers
from genai_chatbot.config import ProviderConfig, load_settings
from genai_chatbot.providers import (
    ErrorKind,
    LLMRouter,
    ModelEndpoint,
    ProviderError,
    build_router,
    classify_error,
    create_chat_model,
)
from tests.support.fake_llm import (
    TEST_CORRELATION,
    FakeChatModel,
    always,
    bad_credentials,
    bad_request,
    make_router,
    rate_limited,
)

REQUEST = httpx.Request("POST", "https://provider.invalid/v1/chat")
MESSAGES = [HumanMessage("What was revenue in fiscal year 2024?")]


class Answer(BaseModel):
    text: str


def http_error(error_class: type[Exception], status: int) -> Exception:
    return error_class("provider error", response=httpx.Response(status, request=REQUEST), body=None)


def google_error(error_class: type[Exception], status: int) -> Exception:
    return error_class(status, {"error": {"code": status, "message": "provider error", "status": "X"}})


def invoke(router: LLMRouter) -> Answer:
    return router.invoke_structured(Answer, MESSAGES, stage="answer", correlation=TEST_CORRELATION)


# Classification


def _validation_error() -> ValidationError:
    try:
        Answer.model_validate({})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a validation error")


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (ModelRateLimitError("slow down"), ErrorKind.RATE_LIMIT),
        (ModelAuthenticationError("bad key"), ErrorKind.AUTH),
        (ModelPermissionDeniedError("no access"), ErrorKind.AUTH),
        (ModelInvalidRequestError("bad request"), ErrorKind.INVALID_REQUEST),
        (ModelNotFoundError("no such model"), ErrorKind.INVALID_REQUEST),
        (ContextOverflowError("too long"), ErrorKind.INVALID_REQUEST),
        (OutputParserException("not json"), ErrorKind.MALFORMED_OUTPUT),
        (_validation_error(), ErrorKind.MALFORMED_OUTPUT),
        (json.JSONDecodeError("Expecting value", "{", 1), ErrorKind.MALFORMED_OUTPUT),
        (TimeoutError("timed out"), ErrorKind.UNAVAILABLE),
        (RuntimeError("something else"), ErrorKind.UNAVAILABLE),
    ],
)
def test_langchain_errors_are_classified(error, kind):
    assert classify_error(error) is kind


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (http_error(anthropic.RateLimitError, 429), ErrorKind.RATE_LIMIT),
        (http_error(anthropic.AuthenticationError, 401), ErrorKind.AUTH),
        (http_error(anthropic.BadRequestError, 400), ErrorKind.INVALID_REQUEST),
        (http_error(anthropic.InternalServerError, 529), ErrorKind.UNAVAILABLE),
        (anthropic.APIConnectionError(request=REQUEST), ErrorKind.UNAVAILABLE),
        (http_error(openai.RateLimitError, 429), ErrorKind.RATE_LIMIT),
        (http_error(openai.PermissionDeniedError, 403), ErrorKind.AUTH),
        (http_error(openai.NotFoundError, 404), ErrorKind.INVALID_REQUEST),
        (openai.APITimeoutError(request=REQUEST), ErrorKind.UNAVAILABLE),
        (http_error(groq.RateLimitError, 429), ErrorKind.RATE_LIMIT),
        (http_error(groq.AuthenticationError, 401), ErrorKind.AUTH),
        (google_error(google_errors.ClientError, 429), ErrorKind.RATE_LIMIT),
        (google_error(google_errors.ClientError, 401), ErrorKind.AUTH),
        (google_error(google_errors.ClientError, 400), ErrorKind.INVALID_REQUEST),
        (google_error(google_errors.ServerError, 503), ErrorKind.UNAVAILABLE),
    ],
)
def test_provider_sdk_errors_are_classified(error, kind):
    assert classify_error(error) is kind


def test_wrapped_errors_are_classified_by_their_cause():
    try:
        try:
            raise http_error(openai.RateLimitError, 429)
        except openai.RateLimitError as exc:
            raise RuntimeError("the chain failed") from exc
    except RuntimeError as wrapped:
        assert classify_error(wrapped) is ErrorKind.RATE_LIMIT


def test_unknown_errors_with_status_codes_or_rate_limit_names_are_classified():
    class ResourceExhausted(Exception):
        pass

    class WithResponse(Exception):
        def __init__(self, status: int) -> None:
            super().__init__("error")
            self.response = httpx.Response(status, request=REQUEST)

    assert classify_error(ResourceExhausted("quota")) is ErrorKind.RATE_LIMIT
    assert classify_error(WithResponse(429)) is ErrorKind.RATE_LIMIT
    assert classify_error(WithResponse(401)) is ErrorKind.AUTH
    assert classify_error(WithResponse(422)) is ErrorKind.INVALID_REQUEST
    assert classify_error(WithResponse(502)) is ErrorKind.UNAVAILABLE


def test_user_messages_name_the_route_but_no_provider_internals():
    auth = ProviderError(ErrorKind.AUTH, "fallback")
    invalid = ProviderError(ErrorKind.INVALID_REQUEST, "primary")

    assert "fallback model" in auth.user_message
    assert "API key" in auth.user_message
    assert "primary model" in invalid.user_message
    assert auth.is_configuration_problem and invalid.is_configuration_problem
    for kind in (ErrorKind.RATE_LIMIT, ErrorKind.MALFORMED_OUTPUT, ErrorKind.UNAVAILABLE):
        error = ProviderError(kind)
        assert not error.is_configuration_problem
        assert "{route}" not in error.user_message
        assert "429" not in error.user_message


# Routing


def test_primary_success_never_touches_the_fallback(app_logs):
    router, primary, fallback = make_router(always(Answer(text="primary")), always(Answer(text="fallback")))

    assert invoke(router) == Answer(text="primary")
    assert len(primary.calls) == 1
    assert fallback.calls == []
    (record,) = app_logs()
    assert record["provider_route"] == "primary"
    assert record["fallback_used"] is False
    assert record["outcome"] == "ok"


def test_a_rate_limit_fails_over_to_the_fallback_once(app_logs):
    router, primary, fallback = make_router(always(rate_limited()), always(Answer(text="fallback")))

    assert invoke(router) == Answer(text="fallback")
    assert (len(primary.calls), len(fallback.calls)) == (1, 1)
    assert fallback.calls[0][1] == MESSAGES
    first, second = app_logs()
    assert (first["provider_route"], first["error_kind"]) == ("primary", "rate_limit")
    assert (second["provider_route"], second["fallback_used"], second["outcome"]) == ("fallback", True, "ok")
    assert second["model"] == "fake-fallback"


def test_when_both_models_are_rate_limited_the_router_gives_up_after_one_failover():
    router, primary, fallback = make_router(always(rate_limited()), always(rate_limited()))

    with pytest.raises(ProviderError) as raised:
        invoke(router)

    assert (raised.value.kind, raised.value.route) == (ErrorKind.RATE_LIMIT, "fallback")
    assert (len(primary.calls), len(fallback.calls)) == (1, 1)


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (bad_credentials(), ErrorKind.AUTH),
        (bad_request(), ErrorKind.INVALID_REQUEST),
        (OutputParserException("not json"), ErrorKind.MALFORMED_OUTPUT),
        (TimeoutError("timed out"), ErrorKind.UNAVAILABLE),
    ],
)
def test_other_errors_never_fail_over(error, kind):
    router, _, fallback = make_router(always(error), always(Answer(text="fallback")))

    with pytest.raises(ProviderError) as raised:
        invoke(router)

    assert (raised.value.kind, raised.value.route) == (kind, "primary")
    assert fallback.calls == []
    assert isinstance(raised.value.__cause__, type(error))


def test_bad_fallback_credentials_are_reported_for_the_fallback():
    router, _, _ = make_router(always(rate_limited()), always(bad_credentials()))

    with pytest.raises(ProviderError) as raised:
        invoke(router)

    assert (raised.value.kind, raised.value.route) == (ErrorKind.AUTH, "fallback")
    assert "fallback model" in raised.value.user_message


def test_a_result_of_the_wrong_type_is_malformed_output():
    router, _, fallback = make_router(always({"text": "a dict, not an Answer"}), always(Answer(text="x")))

    with pytest.raises(ProviderError) as raised:
        invoke(router)

    assert raised.value.kind is ErrorKind.MALFORMED_OUTPUT
    assert fallback.calls == []


def test_without_a_fallback_a_rate_limit_is_raised():
    router, _, _ = make_router(always(rate_limited()))

    with pytest.raises(ProviderError) as raised:
        invoke(router)

    assert (raised.value.kind, raised.value.route) == (ErrorKind.RATE_LIMIT, "primary")


def test_call_logs_hold_labels_and_timings_but_no_message_content(app_logs):
    router, _, _ = make_router(always(rate_limited()), always(Answer(text="Revenue was $412.6 million.")))

    invoke(router)

    for record in app_logs():
        assert set(record) <= {
            "event",
            "session_id",
            "request_id",
            "stage",
            "provider_route",
            "provider",
            "model",
            "fallback_used",
            "outcome",
            "error_kind",
            "duration_ms",
        }
        assert record["session_id"] == "test-session"
        assert record["stage"] == "answer"
        assert record["duration_ms"] >= 0
        assert "revenue" not in json.dumps(record).lower()


# Structured output


def test_structured_runnables_use_provider_specific_options_and_are_cached():
    claude = FakeChatModel(always(Answer(text="x")))
    gpt = FakeChatModel(always(Answer(text="x")))
    claude_endpoint = ModelEndpoint("primary", "anthropic", "claude-test", claude)
    gpt_endpoint = ModelEndpoint("fallback", "openai", "gpt-test", gpt)

    assert claude_endpoint.structured(Answer) is claude_endpoint.structured(Answer)
    gpt_endpoint.structured(Answer)

    assert claude.structured_options == [{"method": "json_schema"}]
    assert gpt.structured_options == [{}]


# Model construction


def test_chat_models_are_built_without_sampling_parameters_or_sdk_retries(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(providers, "init_chat_model", lambda **kwargs: captured.update(kwargs) or "model")
    config = ProviderConfig(
        route="primary", provider="anthropic", model="claude-opus-5-5", api_key=SecretStr("sk-test")
    )

    assert create_chat_model(config, timeout=45.0) == "model"
    assert captured == {
        "model": "claude-opus-5-5",
        "model_provider": "anthropic",
        "api_key": "sk-test",
        "timeout": 45.0,
        "max_retries": 0,
    }


@pytest.mark.parametrize(
    ("provider", "model", "class_name"),
    [
        ("anthropic", "claude-opus-5-5", "ChatAnthropic"),
        ("openai", "gpt-5-mini", "ChatOpenAI"),
        ("google_genai", "gemini-2.5-flash", "ChatGoogleGenerativeAI"),
        ("groq", "llama-3.3-70b-versatile", "ChatGroq"),
    ],
)
def test_each_installed_provider_accepts_the_same_settings_offline(provider, model, class_name):
    config = ProviderConfig(route="primary", provider=provider, model=model, api_key=SecretStr("test-key"))

    chat_model = create_chat_model(config, timeout=30.0)

    assert type(chat_model).__name__ == class_name
    assert chat_model.max_retries == 0
    assert ModelEndpoint("primary", provider, model, chat_model).structured(Answer) is not None


def test_build_router_wires_primary_and_fallback_from_settings(valid_settings_env):
    settings = load_settings().settings

    router = build_router(settings)

    assert (router.primary.route, router.primary.provider, router.primary.model_name) == (
        "primary",
        "anthropic",
        "claude-opus-5-5",
    )
    assert (router.fallback.route, router.fallback.provider, router.fallback.model_name) == (
        "fallback",
        "openai",
        "gpt-test-mini",
    )
    assert type(router.primary.chat_model).__name__ == "ChatAnthropic"
    assert type(router.fallback.chat_model).__name__ == "ChatOpenAI"
