"""Provider configuration is exported from the settings-backed .env safely."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

from job_scout.config import get_settings
from job_scout.llm import _export_provider_env, with_structured_output


def test_openrouter_settings_export_to_openai_compatible_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    get_settings.cache_clear()

    _export_provider_env("openai:openai/gpt-4o-mini")

    assert os.environ["OPENAI_API_KEY"] == "test-openrouter-key"
    assert os.environ["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"


def test_openrouter_uses_json_mode_for_structured_output(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    get_settings.cache_clear()
    model = MagicMock()

    with_structured_output(model, dict, "openai:example/model")

    model.with_structured_output.assert_called_once_with(dict, method="json_mode")


def test_groq_settings_export(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    get_settings.cache_clear()

    _export_provider_env("groq:llama-3.3-70b-versatile")

    assert os.environ["GROQ_API_KEY"] == "test-groq-key"


def test_ollama_settings_export(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "")
    get_settings.cache_clear()

    _export_provider_env("ollama:qwen3.5:4b")

    assert os.environ["OLLAMA_HOST"]


def test_nvidia_nim_model_uses_its_own_openai_compatible_endpoint(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia-key")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    get_settings.cache_clear()

    from job_scout.llm import get_chat_model

    model = get_chat_model("nvidia:openai/gpt-oss-20b")

    assert type(model).__name__ == "ChatOpenAI"
    assert model.model_name == "openai/gpt-oss-20b"
    assert str(model.openai_api_base).rstrip("/") == "https://integrate.api.nvidia.com/v1"
