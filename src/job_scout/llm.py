"""Chat-model factory and the per-run LLM call budget.

The call budget is a simple circuit breaker: every node reads the running
``llm_calls`` counter from state, checks it against ``MAX_LLM_CALLS_PER_RUN``
before calling the model, and returns the incremented total. The graph runs
sequentially, so returning the cumulative total (not a delta) keeps the counter
correct.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from job_scout.config import get_settings


class LLMBudgetExceededError(RuntimeError):
    """Raised when a run would exceed ``MAX_LLM_CALLS_PER_RUN``."""


def _export_provider_env(model: str) -> None:
    """Copy provider settings from ``.env`` into environment variables.

    ``pydantic-settings`` reads ``.env`` into the ``Settings`` object but does not
    export to ``os.environ``, which is where LangChain provider clients look for
    credentials and endpoints.
    """
    settings = get_settings()
    if model.startswith("openai:"):
        if not os.environ.get("OPENAI_API_KEY"):
            key = settings.openai_api_key.get_secret_value()
            if key:
                os.environ["OPENAI_API_KEY"] = key
        if not os.environ.get("OPENAI_BASE_URL") and settings.openai_base_url:
            os.environ["OPENAI_BASE_URL"] = settings.openai_base_url
    elif model.startswith("groq:"):
        if not os.environ.get("GROQ_API_KEY"):
            key = settings.groq_api_key.get_secret_value()
            if key:
                os.environ["GROQ_API_KEY"] = key
    elif model.startswith("nvidia:"):
        if not os.environ.get("NVIDIA_API_KEY"):
            key = settings.nvidia_api_key.get_secret_value()
            if key:
                os.environ["NVIDIA_API_KEY"] = key
    elif model.startswith("ollama:"):
        if not os.environ.get("OLLAMA_HOST") and settings.ollama_base_url:
            os.environ["OLLAMA_HOST"] = settings.ollama_base_url


@lru_cache(maxsize=8)
def get_chat_model(model: str, temperature: float = 0.0) -> BaseChatModel:
    """Return a cached chat model for a LangChain provider string (e.g. ``openai:gpt-4o-mini``)."""
    _export_provider_env(model)
    if model.startswith("nvidia:"):
        from langchain_openai import ChatOpenAI

        settings = get_settings()
        return ChatOpenAI(
            model=model.removeprefix("nvidia:"),
            api_key=settings.nvidia_api_key.get_secret_value(),
            base_url=settings.nvidia_base_url,
            temperature=temperature,
        )
    return init_chat_model(model, temperature=temperature)


def ensure_budget(current_calls: int, planned: int, max_calls: int) -> None:
    """Raise ``LLMBudgetExceededError`` if ``planned`` more calls would exceed ``max_calls``."""
    if current_calls + planned > max_calls:
        raise LLMBudgetExceededError(
            f"Run would make {current_calls + planned} LLM calls, exceeding MAX_LLM_CALLS_PER_RUN={max_calls}."
        )
