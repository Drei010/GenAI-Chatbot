"""Application settings, read from environment variables, a local `.env`, or Streamlit secrets.

Precedence, highest first: explicit keyword arguments, environment variables, `.env`, then the
top-level keys of `.streamlit/secrets.toml`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# Providers the app can build with LangChain's init_chat_model, mapped to the module that must be
# importable and the package that provides it.
SUPPORTED_PROVIDERS: dict[str, tuple[str, str]] = {
    "anthropic": ("langchain_anthropic", "langchain-anthropic"),
    "openai": ("langchain_openai", "langchain-openai"),
    "google_genai": ("langchain_google_genai", "langchain-google-genai"),
    "groq": ("langchain_groq", "langchain-groq"),
    "mistralai": ("langchain_mistralai", "langchain-mistralai"),
}

_PLACEHOLDER_PREFIXES = ("your-", "your_", "<", "replace", "changeme", "xxx")

Route = Literal["primary", "fallback"]


class ProviderConfig(BaseModel):
    """One model endpoint: the provider, the model on it, and the key to call it with."""

    model_config = ConfigDict(frozen=True)

    route: Route
    provider: str
    model: str
    api_key: SecretStr


def _read_streamlit_secrets() -> dict[str, Any]:
    """Return the top-level scalar entries of Streamlit's secrets, or nothing if there are none."""
    try:
        import streamlit as st

        raw = st.secrets.to_dict()
    except Exception:  # no secrets.toml, unreadable file, or Streamlit unavailable
        return {}
    return {str(key).lower(): value for key, value in raw.items() if not isinstance(value, Mapping)}


class StreamlitSecretsSource(PydanticBaseSettingsSource):
    """Settings source backed by `.streamlit/secrets.toml`."""

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return _read_streamlit_secrets().get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        secrets = _read_streamlit_secrets()
        return {name: secrets[name] for name in self.settings_cls.model_fields if name in secrets}


class AppSettings(BaseSettings):
    """Validated runtime configuration. API keys stay wrapped in SecretStr."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
    )

    primary_llm_provider: str
    primary_llm_model: str
    primary_llm_api_key: SecretStr
    fallback_llm_provider: str
    fallback_llm_model: str
    fallback_llm_api_key: SecretStr
    llm_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            StreamlitSecretsSource(settings_cls),
            file_secret_settings,
        )

    @field_validator("primary_llm_provider", "fallback_llm_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("primary_llm_provider", "fallback_llm_provider")
    @classmethod
    def _check_provider(cls, value: str) -> str:
        if value not in SUPPORTED_PROVIDERS:
            raise ValueError(f"isn't a supported provider. Use one of: {', '.join(SUPPORTED_PROVIDERS)}.")
        module, package = SUPPORTED_PROVIDERS[value]
        if find_spec(module) is None:
            raise ValueError(f"needs the {package} package. Install it with `uv add {package}`.")
        return value

    @field_validator("primary_llm_model", "fallback_llm_model")
    @classmethod
    def _check_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must name a model, for example claude-opus-5-5.")
        return value

    @field_validator("primary_llm_api_key", "fallback_llm_api_key")
    @classmethod
    def _check_api_key(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value().strip()
        if not raw:
            raise ValueError("is empty.")
        if raw.lower().startswith(_PLACEHOLDER_PREFIXES):
            raise ValueError("still has the placeholder value from .env.example.")
        return SecretStr(raw)

    @property
    def primary(self) -> ProviderConfig:
        return ProviderConfig(
            route="primary",
            provider=self.primary_llm_provider,
            model=self.primary_llm_model,
            api_key=self.primary_llm_api_key,
        )

    @property
    def fallback(self) -> ProviderConfig:
        return ProviderConfig(
            route="fallback",
            provider=self.fallback_llm_provider,
            model=self.fallback_llm_model,
            api_key=self.fallback_llm_api_key,
        )


class SetupIssue(BaseModel):
    """A user-facing description of one setting that needs attention. Never carries the value."""

    setting: str
    problem: str


@dataclass(frozen=True)
class SettingsResult:
    settings: AppSettings | None
    issues: list[SetupIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.settings is not None


def _describe(error: Mapping[str, Any]) -> SetupIssue:
    name = str(error["loc"][0]) if error.get("loc") else "settings"
    kind = error.get("type", "")
    if kind == "missing":
        problem = "is missing."
    elif kind == "value_error":
        problem = str(error.get("msg", "")).removeprefix("Value error, ")
    elif kind == "literal_error":
        problem = "must be one of DEBUG, INFO, WARNING or ERROR."
    elif name == "llm_timeout_seconds":
        problem = "must be a number of seconds between 1 and 600."
    else:
        problem = "has an invalid value."
    return SetupIssue(setting=name.upper(), problem=problem)


def load_settings(**overrides: Any) -> SettingsResult:
    """Load settings, turning validation failures into setup issues instead of raising."""
    try:
        return SettingsResult(settings=AppSettings(**overrides))
    except ValidationError as exc:
        issues = [_describe(error) for error in exc.errors(include_input=False, include_url=False)]
        return SettingsResult(settings=None, issues=issues)
