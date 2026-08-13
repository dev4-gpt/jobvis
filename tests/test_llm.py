"""Provider configuration is exported from the settings-backed .env safely."""

from __future__ import annotations

import os

from job_scout.config import get_settings
from job_scout.llm import _export_provider_env


def test_openrouter_settings_export_to_openai_compatible_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    get_settings.cache_clear()

    _export_provider_env("openai:openai/gpt-4o-mini")

    assert os.environ["OPENAI_API_KEY"] == "test-openrouter-key"
    assert os.environ["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"


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
