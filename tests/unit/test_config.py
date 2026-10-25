from __future__ import annotations

from pathlib import Path

import pytest

from genai_chatbot import config
from genai_chatbot.config import AppSettings, load_settings
from tests.conftest import VALID_SETTINGS

REQUIRED = [
    "PRIMARY_LLM_PROVIDER",
    "PRIMARY_LLM_MODEL",
    "PRIMARY_LLM_API_KEY",
    "FALLBACK_LLM_PROVIDER",
    "FALLBACK_LLM_MODEL",
    "FALLBACK_LLM_API_KEY",
]


def issues_by_setting(result: config.SettingsResult) -> dict[str, str]:
    return {issue.setting: issue.problem for issue in result.issues}


def test_valid_environment_builds_primary_and_fallback(valid_settings_env: dict[str, str]) -> None:
    result = load_settings()

    assert result.ok and result.issues == []
    settings = result.settings
    assert settings is not None
    assert (settings.primary.route, settings.primary.provider, settings.primary.model) == (
        "primary",
        "anthropic",
        "claude-opus-5-5",
    )
    assert settings.fallback.provider == "openai"
    assert settings.fallback.api_key.get_secret_value() == "test-fallback-key"
    assert settings.llm_timeout_seconds == 120.0


def test_every_missing_setting_is_reported_by_its_variable_name() -> None:
    result = load_settings()

    assert result.settings is None
    assert issues_by_setting(result) == {name: "is missing." for name in REQUIRED}


def test_empty_values_count_as_missing(monkeypatch: pytest.MonkeyPatch, valid_settings_env: dict) -> None:
    monkeypatch.setenv("FALLBACK_LLM_API_KEY", "")

    assert issues_by_setting(load_settings()) == {"FALLBACK_LLM_API_KEY": "is missing."}


def test_provider_names_are_normalized(monkeypatch: pytest.MonkeyPatch, valid_settings_env: dict) -> None:
    monkeypatch.setenv("PRIMARY_LLM_PROVIDER", "  Anthropic ")

    settings = load_settings().settings
    assert settings is not None and settings.primary_llm_provider == "anthropic"


def test_unsupported_provider_lists_the_supported_ones(
    monkeypatch: pytest.MonkeyPatch, valid_settings_env: dict
) -> None:
    monkeypatch.setenv("PRIMARY_LLM_PROVIDER", "acme-ai")

    problem = issues_by_setting(load_settings())["PRIMARY_LLM_PROVIDER"]
    assert "isn't a supported provider" in problem
    assert "anthropic" in problem and "google_genai" in problem


def test_provider_without_installed_integration_explains_the_install(
    monkeypatch: pytest.MonkeyPatch, valid_settings_env: dict
) -> None:
    monkeypatch.setenv("FALLBACK_LLM_PROVIDER", "mistralai")
    monkeypatch.setattr(config, "find_spec", lambda module: None)

    problem = issues_by_setting(load_settings())["FALLBACK_LLM_PROVIDER"]
    assert "uv add langchain-mistralai" in problem


def test_placeholder_key_from_env_example_is_rejected(
    monkeypatch: pytest.MonkeyPatch, valid_settings_env: dict
) -> None:
    monkeypatch.setenv("PRIMARY_LLM_API_KEY", "your-primary-api-key")

    assert "placeholder" in issues_by_setting(load_settings())["PRIMARY_LLM_API_KEY"]


def test_blank_model_name_is_rejected(monkeypatch: pytest.MonkeyPatch, valid_settings_env: dict) -> None:
    monkeypatch.setenv("PRIMARY_LLM_MODEL", "   ")

    assert "must name a model" in issues_by_setting(load_settings())["PRIMARY_LLM_MODEL"]


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("LLM_TIMEOUT_SECONDS", "soon", "must be a number of seconds"),
        ("LLM_TIMEOUT_SECONDS", "0", "must be a number of seconds"),
        ("LOG_LEVEL", "LOUD", "must be one of DEBUG"),
    ],
)
def test_optional_settings_are_validated(
    monkeypatch: pytest.MonkeyPatch, valid_settings_env: dict, name: str, value: str, expected: str
) -> None:
    monkeypatch.setenv(name, value)

    assert expected in issues_by_setting(load_settings())[name]


def test_streamlit_secrets_are_used_when_env_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    secrets = {name.lower(): value for name, value in VALID_SETTINGS.items()}
    monkeypatch.setattr(config, "_read_streamlit_secrets", lambda: secrets)

    settings = load_settings().settings
    assert settings is not None and settings.primary_llm_model == "claude-opus-5-5"


def test_environment_overrides_streamlit_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    secrets = {name.lower(): value for name, value in VALID_SETTINGS.items()}
    monkeypatch.setattr(config, "_read_streamlit_secrets", lambda: secrets)
    monkeypatch.setenv("PRIMARY_LLM_MODEL", "claude-sonnet-5-5")

    settings = load_settings().settings
    assert settings is not None and settings.primary_llm_model == "claude-sonnet-5-5"


def test_dotenv_file_in_working_directory_is_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("\n".join(f"{k}={v}" for k, v in VALID_SETTINGS.items()))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(AppSettings.model_config, "env_file", ".env")

    assert load_settings().ok


def test_streamlit_secrets_reader_tolerates_a_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()  # use the real reader; the test runner has no secrets.toml
    assert isinstance(config._read_streamlit_secrets(), dict)


def test_secret_values_never_appear_in_issues_or_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "sk-live-DO-NOT-LEAK-123456"
    monkeypatch.setenv("PRIMARY_LLM_API_KEY", secret)
    monkeypatch.setenv("PRIMARY_LLM_PROVIDER", secret)
    monkeypatch.setenv("FALLBACK_LLM_API_KEY", f"   {secret}   ")

    result = load_settings()
    assert secret not in repr(result)
    assert all(secret not in issue.problem for issue in result.issues)

    valid = AppSettings(**{k.lower(): v for k, v in VALID_SETTINGS.items()} | {"primary_llm_api_key": secret})
    assert secret not in repr(valid) and secret not in str(valid.primary)


def test_whitespace_only_key_is_reported_as_empty(
    monkeypatch: pytest.MonkeyPatch, valid_settings_env: dict
) -> None:
    monkeypatch.setenv("PRIMARY_LLM_API_KEY", "   ")

    assert issues_by_setting(load_settings())["PRIMARY_LLM_API_KEY"] == "is empty."


def test_wrong_value_types_get_a_generic_message(valid_settings_env: dict) -> None:
    assert issues_by_setting(load_settings(primary_llm_model=42)) == {
        "PRIMARY_LLM_MODEL": "has an invalid value."
    }


def test_secrets_reader_keeps_top_level_scalars_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import streamlit

    class FakeSecrets:
        def to_dict(self) -> dict:
            return {"PRIMARY_LLM_MODEL": "claude-opus-5-5", "connections": {"db": {"url": "x"}}}

    monkeypatch.undo()
    monkeypatch.setattr(streamlit, "secrets", FakeSecrets())

    assert config._read_streamlit_secrets() == {"primary_llm_model": "claude-opus-5-5"}
    source = config.StreamlitSecretsSource(AppSettings)
    field = AppSettings.model_fields["primary_llm_model"]
    assert source.get_field_value(field, "primary_llm_model") == (
        "claude-opus-5-5",
        "primary_llm_model",
        False,
    )
