"""Profile extraction (the preprocessing step before the graph)."""

from __future__ import annotations

import job_scout.profile as profile_mod
from job_scout.profile import _augment_resume_facts, extract_profile
from tests.conftest import structured_llm


def test_extract_profile(monkeypatch, sample_profile):
    monkeypatch.setattr(profile_mod, "get_chat_model", lambda *a, **k: structured_llm(sample_profile))
    result = extract_profile("some cv text")
    assert result is sample_profile


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
