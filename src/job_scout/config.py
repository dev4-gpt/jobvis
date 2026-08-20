"""Application configuration loaded from the environment and ``.env``.

A single ``Settings`` object holds every setting. Secrets use ``SecretStr`` so
they never appear in logs or trace metadata by accident. Each field's ``.env``
name is documented in ``.env.example``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The repo-root .env, resolved from this file so notebooks and scripts running
# from subdirectories (e.g. notebooks/) still find it. A CWD-local .env, when
# present, is read second and wins.
_REPO_ENV = Path(__file__).resolve().parents[2] / ".env"
_MODEL_PROVIDERS = {"openai", "groq", "nvidia", "ollama"}
_RETIRED_MODELS = {
    "llama-3.3-70b-versatile": "groq:openai/gpt-oss-20b",
    "qwen/qwen3.5-397b-a17b": "groq:qwen/qwen3.6-27b",
}


class Settings(BaseSettings):
    """Runtime configuration, read once from the environment or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=(_REPO_ENV, ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    scout_model: str = Field(default="openai:gpt-4o-mini", alias="SCOUT_MODEL")
    scout_tailor_model: str = Field(default="openai:gpt-4o-mini", alias="SCOUT_TAILOR_MODEL")
    # Comma-separated, explicit fallbacks used only after a provider call
    # fails. Empty by default: a fallback must be a deliberate user choice
    # because it may make an additional billable request.
    scout_fallback_models: str = Field(default="", alias="SCOUT_FALLBACK_MODELS")

    openai_api_key: SecretStr = Field(default=SecretStr(""), alias="OPENAI_API_KEY")
    # OpenRouter exposes an OpenAI-compatible endpoint. This is deliberately
    # separate from the key so switching providers never changes secret names.
    openai_base_url: str = Field(default="", alias="OPENAI_BASE_URL")
    groq_api_key: SecretStr = Field(default=SecretStr(""), alias="GROQ_API_KEY")
    nvidia_api_key: SecretStr = Field(default=SecretStr(""), alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL")
    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_BASE_URL")

    opik_api_key: SecretStr = Field(default=SecretStr(""), alias="OPIK_API_KEY")
    opik_workspace: str = Field(default="", alias="OPIK_WORKSPACE")
    opik_project_name: str = Field(default="job-scout", alias="OPIK_PROJECT_NAME")
    opik_enabled: bool = Field(default=True, alias="OPIK_ENABLED")

    jsearch_api_key: SecretStr = Field(default=SecretStr(""), alias="JSEARCH_API_KEY")
    adzuna_app_id: SecretStr = Field(default=SecretStr(""), alias="ADZUNA_APP_ID")
    adzuna_app_key: SecretStr = Field(default=SecretStr(""), alias="ADZUNA_APP_KEY")

    tavily_api_key: SecretStr = Field(default=SecretStr(""), alias="TAVILY_API_KEY")

    elevenlabs_api_key: SecretStr = Field(default=SecretStr(""), alias="ELEVENLABS_API_KEY")
    elevenlabs_agent_id: str = Field(default="", alias="ELEVENLABS_AGENT_ID")
    elevenlabs_voice_id: str = Field(default="", alias="ELEVENLABS_VOICE_ID")

    max_llm_calls_per_run: int = Field(default=25, alias="MAX_LLM_CALLS_PER_RUN")
    scout_max_jobs: int = Field(default=10, alias="SCOUT_MAX_JOBS")
    scout_max_reformulations: int = Field(default=2, alias="SCOUT_MAX_REFORMULATIONS")
    scout_fetch_model: str = Field(default="", alias="SCOUT_FETCH_MODEL")
    scout_rank_batch: int = Field(default=4, alias="SCOUT_RANK_BATCH")
    scout_rank_timeout: float = Field(default=45.0, alias="SCOUT_RANK_TIMEOUT")
    scout_profile_timeout: float = Field(default=20.0, alias="SCOUT_PROFILE_TIMEOUT")
    scout_tailor_timeout: float = Field(default=45.0, alias="SCOUT_TAILOR_TIMEOUT")
    scout_tailor_max_repairs: int = Field(default=2, alias="SCOUT_TAILOR_MAX_REPAIRS", ge=0, le=4)
    scout_max_role_queries: int = Field(default=6, alias="SCOUT_MAX_ROLE_QUERIES")
    scout_query_concurrency: int = Field(default=3, alias="SCOUT_QUERY_CONCURRENCY")
    scout_concurrent_sources: bool = Field(default=True, alias="SCOUT_CONCURRENT_SOURCES")
    # 1.0s is measured, not guessed. The deadline is paid in full on every
    # search (adzuna is already finished by ~1s, so wall clock == deadline),
    # and jsearch has never once returned under 8s across every trace we have.
    # So a longer deadline buys jsearch no real chance and bills us the
    # difference. Phase 2 in run_search is what protects the results.
    scout_source_soft_deadline: float = Field(
        default=1.0,
        alias="SCOUT_SOURCE_SOFT_DEADLINE",
        description="Seconds to wait for the first concurrent source before falling through to faster ones.",
    )

    fab_bullet_ratio: float = Field(default=0.65, alias="SCOUT_FAB_BULLET_RATIO")
    fab_skill_ratio: float = Field(default=0.85, alias="SCOUT_FAB_SKILL_RATIO")
    fab_letter_ratio: float = Field(default=0.55, alias="SCOUT_FAB_LETTER_RATIO")

    @field_validator(
        "opik_workspace",
        "opik_project_name",
        "scout_model",
        "scout_tailor_model",
        "scout_fetch_model",
        "scout_fallback_models",
        "openai_base_url",
        "nvidia_base_url",
        "ollama_base_url",
        "elevenlabs_agent_id",
        "elevenlabs_voice_id",
        mode="before",
    )
    @classmethod
    def _drop_inline_comment(cls, value: object) -> object:
        """Treat a value that is only a ``# comment`` as empty.

        Guards the common ``.env`` mistake of leaving a key blank but keeping its
        trailing comment, which some parsers read as the value.
        """
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("#"):
                return ""
        return value

    @field_validator("scout_model", "scout_tailor_model", "scout_fetch_model", mode="after")
    @classmethod
    def _validate_model_reference(cls, value: str, info) -> str:
        """Reject malformed or known-retired model references before a run starts."""
        model = value.strip()
        if not model:
            return ""
        if ":" not in model:
            raise ValueError(
                f"{info.field_name} must use provider:model syntax, for example groq:openai/gpt-oss-20b or ollama:qwen3.5:4b"
            )
        provider, model_id = model.split(":", 1)
        if provider not in _MODEL_PROVIDERS:
            raise ValueError(
                f"{info.field_name} uses unsupported provider {provider!r}; choose one of {', '.join(sorted(_MODEL_PROVIDERS))}"
            )
        retired_hint = _RETIRED_MODELS.get(model_id.lower())
        if retired_hint:
            raise ValueError(
                f"{info.field_name} references retired model {model_id!r}; use {retired_hint!r} or another current model"
            )
        if not model_id.strip():
            raise ValueError(f"{info.field_name} has an empty model id after {provider}:")
        return model

    @field_validator("scout_fallback_models", mode="after")
    @classmethod
    def _validate_fallback_references(cls, value: str) -> str:
        """Validate each explicit fallback using the same model rules."""
        models = [item.strip() for item in value.split(",") if item.strip()]
        for model in models:
            if ":" not in model:
                raise ValueError(
                    "scout_fallback_models entries must use provider:model syntax, for example groq:openai/gpt-oss-20b"
                )
            provider, model_id = model.split(":", 1)
            if provider not in _MODEL_PROVIDERS:
                raise ValueError(f"scout_fallback_models uses unsupported provider {provider!r}")
            retired_hint = _RETIRED_MODELS.get(model_id.lower())
            if retired_hint:
                raise ValueError(
                    f"scout_fallback_models references retired model {model_id!r}; use {retired_hint!r} or another current model"
                )
            if not model_id.strip():
                raise ValueError(f"scout_fallback_models has an empty model id after {provider}:")
        return ",".join(models)

    @property
    def has_jsearch(self) -> bool:
        """Whether a JSearch API key is configured."""
        return bool(self.jsearch_api_key.get_secret_value())

    @property
    def has_adzuna(self) -> bool:
        """Whether both Adzuna credentials are configured."""
        return bool(self.adzuna_app_id.get_secret_value() and self.adzuna_app_key.get_secret_value())

    @property
    def has_tavily(self) -> bool:
        """Whether a Tavily API key is configured (optional company research)."""
        return bool(self.tavily_api_key.get_secret_value())

    @property
    def has_opik(self) -> bool:
        """Whether Opik tracing is enabled and has an API key."""
        return self.opik_enabled and bool(self.opik_api_key.get_secret_value())

    @property
    def has_voice(self) -> bool:
        """Whether Jobvis has both an ElevenLabs API key and an agent id."""
        return bool(self.elevenlabs_api_key.get_secret_value() and self.elevenlabs_agent_id)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
