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
def get_chat_model(model: str, temperature: float = 0.0, **kwargs: object) -> BaseChatModel:
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
            **kwargs,
        )
    return init_chat_model(model, temperature=temperature, **kwargs)


def model_chain(primary: str, fallback_models: str = "") -> tuple[str, ...]:
    """Return a de-duplicated, explicitly configured provider chain."""
    return tuple(dict.fromkeys([item.strip() for item in (primary, *fallback_models.split(",")) if item.strip()]))


def reasoning_kwargs(provider_model: str) -> dict[str, str]:
    """Return only reasoning parameters supported by the selected model family.

    Groq's Qwen models reject ``reasoning_effort=none`` while GPT-OSS models
    require one of ``low``, ``medium``, or ``high``. Omitting the parameter is
    the portable default for every other provider/model combination.
    """
    model_id = provider_model.split(":", 1)[-1].lower()
    if "gpt-oss" in model_id:
        return {"reasoning_effort": "low"}
    return {}


def with_structured_output(model: BaseChatModel, schema: type, provider_model: str) -> object:
    """Bind a typed response using the provider's most reliable protocol.

    LangChain defaults to function calling for Pydantic structured output.
    Groq's Qwen endpoint supports JSON-schema output, but can reject the
    generated function schema with ``tool_use_failed``. JSON-schema mode keeps
    the response typed at our boundary without sending that fragile function
    definition.
    """
    if provider_model.startswith("groq:"):
        return model.with_structured_output(schema, method="json_schema")
    # OpenRouter fronts models with different levels of support for OpenAI's
    # ``parsed`` Structured Outputs envelope.  Some otherwise capable models
    # return an empty message with ``parsed=None`` when that envelope is used.
    # JSON-object mode is supported by a much wider set of OpenRouter models;
    # the prompts still require the exact Pydantic shape and callers validate
    # the result at the boundary.
    if provider_model.startswith("openai:") and "openrouter.ai" in get_settings().openai_base_url.lower():
        return model.with_structured_output(schema, method="json_mode")
    return model.with_structured_output(schema)


def ensure_budget(current_calls: int, planned: int, max_calls: int) -> None:
    """Raise ``LLMBudgetExceededError`` if ``planned`` more calls would exceed ``max_calls``."""
    if current_calls + planned > max_calls:
        raise LLMBudgetExceededError(
            f"Run would make {current_calls + planned} LLM calls, exceeding MAX_LLM_CALLS_PER_RUN={max_calls}."
        )
