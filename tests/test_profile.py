"""Profile extraction (the preprocessing step before the graph)."""

from __future__ import annotations

import job_scout.profile as profile_mod
from job_scout.profile import _augment_resume_facts, extract_profile
from tests.conftest import plain_llm, structured_llm


def test_extract_profile(monkeypatch, sample_profile):
    monkeypatch.setattr(profile_mod, "get_chat_model", lambda *a, **k: structured_llm(sample_profile))
    result = extract_profile("some cv text")
    assert result is sample_profile


def test_extract_profile_recovers_from_empty_structured_response(monkeypatch, sample_profile):
    typed_model = structured_llm(sample_profile)
    typed_model.with_structured_output.return_value.invoke.side_effect = ValueError(
        "Structured Output response does not have a 'parsed' field nor a 'refusal' field"
    )
    recovery_model = plain_llm(sample_profile.model_dump_json())
    models = iter([typed_model, recovery_model])
    monkeypatch.setattr(profile_mod, "get_chat_model", lambda *args, **kwargs: next(models))

    result = extract_profile("some cv text")

    assert result == sample_profile


def test_extract_profile_uses_source_grounded_fallback_when_provider_fails(monkeypatch):
    def fail_model(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(profile_mod, "get_chat_model", fail_model)
    result = extract_profile(
        "Aryaman Singh Dev\n"
        "Data Scientist Intern\n"
        "Built predictive maintenance models with Python.\n"
        "Built a GenAI RAG pipeline with LangChain.\n"
        "TECHNICAL SKILLS & SOFTWARES PROFICIENCY\n"
        "Python, LangChain, FAISS\n"
        "Penn State M.S. Artificial Intelligence August 2025-December 2026\n"
        "(484) 735-7279"
    )

    assert result.name == "Aryaman Singh Dev"
    assert "Data Scientist" in result.primary_roles
    assert "python" in result.skills
    assert result.expected_graduation_date.isoformat() == "2026-12-01"
    assert result.phone == "(484) 735-7279"
    assert result.resume_evidence_refs


def test_resume_facts_augment_education_timeline(sample_profile):
    result = _augment_resume_facts(
        sample_profile,
        "Penn State M.S. Artificial Intelligence Aug 2025 - Dec 2026\n"
        "NYU M.S. Computer Engineering Sep 2024 - Aug 2025\n"
        "Manipal B.Tech. Mechatronics Engineering Jul 2024\n"
        "(484) 735-7279",
    )
    assert result.expected_graduation_date.isoformat() == "2026-12-01"
    assert [entry.institution for entry in result.education_history] == ["Penn State", "NYU", "Manipal"]
    assert result.phone == "(484) 735-7279"
