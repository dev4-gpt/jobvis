"""Candidate-specific policy and eligibility tests; no network or LLM calls."""

from datetime import date

from job_scout.candidate_fit import assess_eligibility, normalize_job, preferences_from_dict, resume_persona, role_bucket
from job_scout.graph.schemas import CandidatePreferences, EducationEntry, JobPosting, Profile


def _profile() -> Profile:
    return Profile(
        name="Aryaman Singh Dev",
        primary_roles=["Data Scientist", "ML Engineer", "GenAI Engineer"],
        skills=["python", "pytorch", "rag", "sql"],
        education_history=[
            EducationEntry(
                institution="Penn State",
                degree="M.S.",
                field="Artificial Intelligence",
                start_date=date(2025, 8, 1),
                end_date=date(2026, 12, 1),
                in_progress=True,
            ),
            EducationEntry(
                institution="NYU",
                degree="M.S.",
                field="Computer Engineering",
                start_date=date(2024, 9, 1),
                end_date=date(2025, 8, 1),
            ),
        ],
        expected_graduation_date=date(2026, 12, 1),
        current_program="M.S. Artificial Intelligence",
    )


def _job(title: str, description: str) -> JobPosting:
    return JobPosting(
        job_id=title.lower().replace(" ", "-"),
        title=title,
        company="Example",
        location="United States",
        description=description,
        url="https://example.com/job",
        source="cache",
    )


def test_profile_carries_graduation_timeline():
    profile = _profile()
    assert profile.expected_graduation_date == date(2026, 12, 1)
    assert profile.education_history[1].institution == "NYU"
    assert profile.education_history[0].in_progress is True


def test_internship_is_blocked_from_primary_results():
    job = normalize_job(_job("AI/ML Intern", "Internship for summer 2027; Python and PyTorch."))
    assessment = assess_eligibility(job, _profile(), CandidatePreferences(), role_fit_score=95, evidence_fit_score=90)
    assert assessment.status == "blocked"
    assert any("internship" in reason for reason in assessment.hard_blockers)


def test_clearance_is_a_hard_blocker_when_unknown():
    job = normalize_job(_job("Data Scientist", "Full-time role requiring active Secret security clearance."))
    assessment = assess_eligibility(job, _profile(), CandidatePreferences(), role_fit_score=90, evidence_fit_score=85)
    assert assessment.status == "blocked"
    assert "explicit security clearance requirement" in assessment.hard_blockers


def test_anywhere_us_accepts_all_work_modes_without_location_penalty():
    prefs = CandidatePreferences(country_scope="us", accepted_work_modes=["remote", "hybrid", "onsite"])
    job = normalize_job(_job("GenAI Engineer", "Full-time onsite role starting January 2027; Python and RAG."))
    assessment = assess_eligibility(job, _profile(), prefs, role_fit_score=88, evidence_fit_score=80)
    assert assessment.status == "eligible"
    assert assessment.role_bucket == "primary"
    assert assessment.start_timing_fit == "compatible"


def test_adjacent_roles_are_separate_and_capped():
    job = normalize_job(_job("BI Analyst", "Full-time hybrid role starting December 2026."))
    assert role_bucket(job, CandidatePreferences()) == "adjacent"
    assessment = assess_eligibility(job, _profile(), CandidatePreferences(), role_fit_score=95, evidence_fit_score=95)
    assert assessment.role_bucket == "adjacent"
    assert assessment.final_priority_score <= 69


def test_company_history_year_does_not_become_start_date_blocker():
    job = normalize_job(_job("Data Scientist", "Full-time role. The company was founded in 2014 and uses Python and SQL."))
    assessment = assess_eligibility(job, _profile(), CandidatePreferences(), role_fit_score=90, evidence_fit_score=85)
    assert assessment.start_timing_fit == "unknown"
    assert assessment.status == "borderline"
    assert "start date outside target window" not in assessment.hard_blockers


def test_custom_employment_allowlist_is_authoritative():
    job = normalize_job(_job("Data Scientist", "Full-time role starting January 2027."))
    preferences = CandidatePreferences(employment_types=["contract"])
    assessment = assess_eligibility(job, _profile(), preferences, role_fit_score=90, evidence_fit_score=85)
    assert assessment.status == "blocked"
    assert "employment type full_time is not selected" in assessment.hard_blockers


def test_legacy_preferences_get_safe_new_defaults():
    prefs = preferences_from_dict({"locations": ["Philadelphia"], "remote": False})
    assert prefs.locations == ["Philadelphia"]
    assert prefs.exclude_internships is True
    assert prefs.authorization_status == "unknown"


def test_forward_deployed_is_a_primary_portfolio_fit():
    job = normalize_job(_job("Forward Deployed Engineer", "Full-time role deploying AI systems with customers."))
    assert role_bucket(job, CandidatePreferences()) == "primary"
    assert "Veloce AgenticOS" in resume_persona(job)


def test_generic_solutions_engineer_is_not_automatically_an_ai_primary():
    generic = normalize_job(_job("Solutions Engineer", "Full-time role supporting enterprise software customers."))
    ai = normalize_job(_job("Solutions Engineer", "Full-time role deploying LLM and RAG systems with customers."))
    assert role_bucket(generic, CandidatePreferences()) != "primary"
    assert role_bucket(ai, CandidatePreferences()) == "primary"


def test_known_full_time_metadata_survives_internship_context():
    job = _job("Data Scientist", "Full-time role; mentorship includes interns and student researchers.").model_copy(
        update={"employment_type": "full_time"}
    )
    assert normalize_job(job).employment_type == "full_time"


def test_clearance_not_required_is_not_a_blocker():
    job = normalize_job(_job("AI Engineer", "Full-time role; security clearance not required; Python and RAG."))
    assessment = assess_eligibility(job, _profile(), CandidatePreferences(), role_fit_score=90, evidence_fit_score=85)
    assert job.clearance_required is False
    assert "explicit security clearance requirement" not in assessment.hard_blockers


def test_clearance_obtainable_is_reviewable_not_blocked():
    job = normalize_job(_job("AI Engineer", "Full-time role; ability to obtain security clearance preferred."))
    assessment = assess_eligibility(job, _profile(), CandidatePreferences(), role_fit_score=90, evidence_fit_score=85)
    assert job.clearance_required is False
    assert assessment.status == "borderline"
    assert any("obtainable" in reason for reason in assessment.reasons)
