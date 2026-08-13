"""Keyless health/import check used by local development and container CI."""

from __future__ import annotations

import json

from job_scout.config import get_settings
from job_scout.graph import get_compiled_graph


def health() -> dict[str, object]:
    """Return non-secret runtime readiness information without network calls."""
    settings = get_settings()
    graph = get_compiled_graph()
    model = settings.scout_model
    if model.startswith("groq:"):
        has_llm_credentials = bool(settings.groq_api_key.get_secret_value())
    elif model.startswith("nvidia:"):
        has_llm_credentials = bool(settings.nvidia_api_key.get_secret_value())
    elif model.startswith("ollama:"):
        # Ollama is local and does not use an API key. Connectivity is checked
        # only when the first model call is made, never by this keyless probe.
        has_llm_credentials = True
    else:
        has_llm_credentials = bool(settings.openai_api_key.get_secret_value())
    return {
        "status": "ok",
        "graph": type(graph).__name__,
        "model": model,
        "opik_enabled": settings.opik_enabled,
        "has_llm_key": has_llm_credentials,
    }


if __name__ == "__main__":
    print(json.dumps(health(), sort_keys=True))
