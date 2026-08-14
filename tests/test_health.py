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
