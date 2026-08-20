"""Broaden the search query when too few good matches came back.

Increments the reformulation counter (the loop guard) and writes a new
``search_query`` that fetch_jobs will use as guidance on the next pass.
"""

from __future__ import annotations

from job_scout.config import get_settings
from job_scout.graph.prompts.reformulate import REFORMULATE_PROMPT
from job_scout.graph.state import AgentState
from job_scout.llm import ensure_budget, get_chat_model, model_chain


def reformulate_query(state: AgentState) -> dict:
    """Ask the LLM for a broader search query and bump the loop counter."""
    settings = get_settings()
    calls = state.get("llm_calls", 0)
    profile = state["profile"]

    prompt = REFORMULATE_PROMPT.format(
        profile=", ".join(profile.primary_roles + profile.skills[:10]),
        previous_query=state.get("search_query") or "",
    )
    errors = list(state.get("errors") or [])
    models = model_chain(settings.scout_model, settings.scout_fallback_models)
    new_query = " ".join(profile.primary_roles[:2]).strip() or "data scientist"
    for model_name in models:
        try:
            ensure_budget(calls, 1, settings.max_llm_calls_per_run)
            response = get_chat_model(model_name, temperature=0.0, timeout=30, max_retries=0).invoke(prompt)
            candidate = str(getattr(response, "content", "") or "").strip()
            if candidate:
                new_query = candidate
                calls += 1
                break
            raise ValueError("reformulation provider returned an empty query")
        except Exception as exc:  # noqa: BLE001 - reformulation is optional
            errors.append(f"reformulate_query: provider {model_name} failed: {type(exc).__name__}: {exc}")
            if calls >= settings.max_llm_calls_per_run:
                break
    else:
        errors.append("reformulate_query: all configured models failed; used a profile-derived query")

    return {
        "search_query": new_query,
        "reformulation_count": state.get("reformulation_count", 0) + 1,
        "llm_calls": calls,
        "errors": errors,
    }
