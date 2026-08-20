"""Keyless health/import check used by local development and container CI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from job_scout.config import get_settings
from job_scout.graph import get_compiled_graph


def _model_readiness(model: str, settings) -> dict[str, object]:
    """Describe one effective model without contacting its provider."""
    provider, _, _model_id = model.partition(":")
    if provider == "groq":
        ready = bool(settings.groq_api_key.get_secret_value())
        credential = "GROQ_API_KEY"
    elif provider == "nvidia":
        ready = bool(settings.nvidia_api_key.get_secret_value())
        credential = "NVIDIA_API_KEY"
    elif provider == "ollama":
        # Ollama connectivity is deliberately not probed here: health must be
        # keyless and side-effect-free. The first model call reports a clear
        # connection error if the local daemon is not running.
        ready = True
        credential = "local Ollama daemon"
    else:
        ready = bool(settings.openai_api_key.get_secret_value())
        credential = "OPENAI_API_KEY"
    return {"model": model, "provider": provider, "ready": ready, "credential": credential}


def _git_provenance(root: Path) -> dict[str, str]:
    """Read local branch/commit labels without invoking Git or the network."""
    try:
        git_dir = root / ".git"
        if git_dir.is_file():
            git_dir = root / git_dir.read_text(encoding="utf-8").strip().removeprefix("gitdir: ")
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = head.removeprefix("ref: ")
            branch = ref.removeprefix("refs/heads/")
            commit = (git_dir / ref).read_text(encoding="utf-8").strip()
            return {"branch": branch, "commit": commit[:12]}
        return {"branch": "detached", "commit": head[:12]}
    except (OSError, ValueError):
        return {"branch": "unknown", "commit": "unknown"}


def health() -> dict[str, object]:
    """Return non-secret runtime readiness information without network calls."""
    settings = get_settings()
    graph = get_compiled_graph()
    model = settings.scout_model
    models = {
        "scout": model,
        "fetch": settings.scout_fetch_model or model,
        "tailor": settings.scout_tailor_model or model,
    }
    readiness = {role: _model_readiness(model_name, settings) for role, model_name in models.items()}
    root = Path(__file__).resolve().parents[2]
    provenance = _git_provenance(root)
    return {
        "status": "ok",
        "graph": type(graph).__name__,
        "python": sys.executable,
        "source_root": str(root),
        "source_branch": provenance["branch"],
        "source_commit": provenance["commit"],
        "model": model,
        "opik_enabled": settings.opik_enabled,
        "has_llm_key": bool(readiness["scout"]["ready"]),
        "models": readiness,
    }


if __name__ == "__main__":
    print(json.dumps(health(), sort_keys=True))
