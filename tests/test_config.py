"""Configuration fails early for malformed or known-retired model names."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from job_scout.config import Settings


def test_model_settings_require_provider_prefix():
    with pytest.raises(ValidationError, match="provider:model syntax"):
        Settings(SCOUT_MODEL="llama-3.3-70b-versatile")


def test_known_retired_model_has_replacement_hint():
    with pytest.raises(ValidationError, match="retired model") as error:
        Settings(SCOUT_MODEL="groq:llama-3.3-70b-versatile")
    assert "groq:openai/gpt-oss-20b" in str(error.value)


def test_ollama_tag_with_colon_in_model_id_is_valid():
    settings = Settings(SCOUT_MODEL="ollama:qwen3.5:4b")
    assert settings.scout_model == "ollama:qwen3.5:4b"


def test_explicit_fallback_chain_is_validated_and_normalized():
    settings = Settings(SCOUT_FALLBACK_MODELS=" groq:openai/gpt-oss-20b, nvidia:openai/gpt-oss-20b ")
    assert settings.scout_fallback_models == "groq:openai/gpt-oss-20b,nvidia:openai/gpt-oss-20b"
