"""Keyless health/import check used by local development and container CI."""

from __future__ import annotations

import json

from job_scout.config import get_settings
from job_scout.graph import get_compiled_graph


def health() -> dict[str, object]:
    """Return non-secret runtime readiness information without network calls."""
    settings = get_settings()
    graph = get_compiled_graph()
    return {
        "status": "ok",
        "graph": type(graph).__name__,
        "model": settings.scout_model,
        "opik_enabled": settings.opik_enabled,
        "has_llm_key": bool(settings.openai_api_key.get_secret_value()),
    }


if __name__ == "__main__":
    print(json.dumps(health(), sort_keys=True))
