import os
import subprocess
import sys

from job_scout.health import health


def test_health_is_keyless_and_does_not_call_external_services(monkeypatch):
    monkeypatch.setenv("OPIK_ENABLED", "false")
    monkeypatch.setenv("SCOUT_MODEL", "openai:gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    from job_scout.config import get_settings

    get_settings.cache_clear()
    result = health()
    assert result["status"] == "ok"
    assert result["graph"]
    assert result["has_llm_key"] is False


def test_app_import_is_safe_in_a_fresh_process():
    """The CLI entrypoint must not depend on import order to avoid cycles."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "-c", "import job_scout.app; print('ok')"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip() == "ok"
