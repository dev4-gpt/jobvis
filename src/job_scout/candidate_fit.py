"""Deterministic candidate intent, job metadata, and eligibility policy.

The LLM explains fit; this module decides whether a posting is admissible. That
boundary keeps graduation, employment, location, authorization, and clearance
requirements testable and prevents a persuasive ranking response from silently
overriding the candidate's choices.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from job_scout.graph.schemas import CandidatePreferences, JobPosting, Profile

_INTERNSHIP = re.compile(r"\b(intern(ship)?|co[- ]?op|student|summer analyst|research intern)\b", re.I)
_PART_TIME = re.compile(r"\b(part[- ]?time|temporary|seasonal|volunteer)\b", re.I)
_FULL_TIME = re.compile(r"\b(full[- ]?time|permanent)\b", re.I)
_CLEARANCE = re.compile(r"\b(clearance|security clearance|secret|top secret|ts/sci|polygraph|public trust)\b", re.I)
_SPONSOR = re.compile(r"\b(sponsorship|sponsor|visa|h[- ]?1b|work authorization)\b", re.I)
_NO_SPONSOR = re.compile(r"\b(no sponsorship|without sponsorship|must be authorized|us citizen|citizenship required)\b", re.I)
_REMOTE = re.compile(r"\b(remote|work from home|distributed)\b", re.I)
_HYBRID = re.compile(r"\b(hybrid)\b", re.I)
_ONSITE = re.compile(r"\b(on[- ]?site|in[- ]office|in office)\b", re.I)

PRIMARY_FAMILY_TERMS = {
    "ai_ml": ("machine learning", "ml engineer", "ai engineer", "ai/ml", "applied ml", "deep learning"),
    "data_science": ("data scientist", "data science", "applied scientist", "decision scientist"),
    "genai": ("genai", "generative ai", "llm", "rag", "language model", "prompt engineer"),
    "forward_deployed": (
        "forward deployed",
        "forward-deployed",
        "solutions engineer",
        "customer engineer",
        "deployment engineer",
    ),
    "computer_vision": ("computer vision", "vision engineer", "perception engineer"),
    "mlops": ("mlops", "ml platform", "machine learning platform"),
}
ADJACENT_TERMS = (
    "data analyst",
    "business intelligence",
    "bi analyst",
    "java developer",
    "software engineer",
    "financial analyst",
    "automation analyst",
)


class EligibilityAssessment(NamedTuple):
    status: str
    reasons: list[str]
    hard_blockers: list[str]
    role_bucket: str
    start_timing_fit: str
    role_fit_score: int
    evidence_fit_score: int
    final_priority_score: int


def default_preferences() -> CandidatePreferences:
    """Return the product's explicit default policy for a new candidate."""
    from job_scout.graph.schemas import CandidatePreferences

    return CandidatePreferences()


def preferences_from_dict(value: dict | CandidatePreferences | None) -> CandidatePreferences:
    """Validate persisted preferences while retaining safe defaults for v1 data."""
    from job_scout.graph.schemas import CandidatePreferences

    if isinstance(value, CandidatePreferences):
        return value
    if not value:
        return default_preferences()
    legacy = dict(value)
    if "remote" in legacy:
        remote = bool(legacy.pop("remote"))
        modes = ["remote", "hybrid", "onsite"] if remote else ["hybrid", "onsite"]
        legacy.setdefault("accepted_work_modes", modes)
    return CandidatePreferences.model_validate(legacy)


def normalize_job(job: JobPosting) -> JobPosting:
    """Fill source-neutral job metadata from title, tags, and description."""
    text = " ".join((job.title, job.description, *job.tags))
    title_text = job.title.lower()
    if _INTERNSHIP.search(text):
        employment = "internship" if "intern" in text.lower() else "co_op"
    elif _PART_TIME.search(text):
        employment = "part_time"
    elif _FULL_TIME.search(text):
        employment = "full_time"
    else:
        employment = job.employment_type

    if job.work_mode != "unknown":
        work_mode = job.work_mode
    elif job.remote or _REMOTE.search(text):
        work_mode = "remote"
    elif _HYBRID.search(text):
        work_mode = "hybrid"
    elif _ONSITE.search(text):
        work_mode = "onsite"
    else:
        work_mode = "unknown"

    if re.search(r"\b(intern|internship|co[- ]?op|student)\b", title_text):
        level = "intern"
    elif re.search(r"\b(junior|jr\.?|entry[- ]?level|associate|new grad|graduate)\b", title_text):
        level = "entry"
    elif re.search(r"\b(senior|sr\.?|staff|principal|lead)\b", title_text):
        level = "senior"
    else:
        level = job.experience_level

    clearance = job.clearance_required or bool(_CLEARANCE.search(text))
    auth = job.authorization_requirement
    sponsorship = job.sponsorship_signal
    if auth == "unknown" and _NO_SPONSOR.search(text):
        auth = "restricted"
    elif auth == "unknown" and _SPONSOR.search(text):
        auth = "mentioned"
    if sponsorship == "unknown":
        sponsorship = "not_available" if _NO_SPONSOR.search(text) else "mentioned" if _SPONSOR.search(text) else "unknown"

    confidence = job.metadata_confidence
    if confidence == 0.0:
        populated = sum([employment != "unknown", work_mode != "unknown", level != "unknown", clearance, auth != "unknown"])
        confidence = min(1.0, populated / 5)
    return job.model_copy(
        update={
            "employment_type": employment,
            "work_mode": work_mode,
            "experience_level": level,
            "clearance_required": clearance,
            "authorization_requirement": auth,
            "sponsorship_signal": sponsorship,
            "source_url": job.source_url or job.url,
            "metadata_confidence": confidence,
        }
    )


def role_bucket(job: JobPosting, preferences: CandidatePreferences) -> str:
    """Classify a title into primary, adjacent, or review without an LLM."""
    haystack = f"{job.title} {job.description[:900]}".lower()
    primary_terms = tuple(term for family in preferences.primary_role_families for term in PRIMARY_FAMILY_TERMS.get(family, ()))
    if any(term in haystack for term in primary_terms):
        return "primary"
    if any(term in haystack for term in ADJACENT_TERMS):
        return "adjacent"
    return "review"


def resume_persona(job: JobPosting) -> str:
    """Choose which evidence should lead the tailored resume."""
    text = f"{job.title} {job.description}".lower()
    if any(term in text for term in ("forward deployed", "forward-deployed", "solutions engineer", "customer engineer")):
        return (
            "Forward-Deployed AI: lead with Veloce AgenticOS, agentic orchestration, research systems, "
            "and evidence of turning technical systems into usable products for stakeholders."
        )
    if any(term in text for term in ("rag", "llm", "generative ai", "genai", "language model")):
        return "GenAI/RAG and AI Engineering: lead with the legal document analyzer and agentic orchestration evidence."
    if any(term in text for term in ("computer vision", "vision", "perception", "yolo")):
        return "Computer Vision: lead with Bioqube and NUS vision/research evidence."
    if any(term in text for term in ("mlops", "platform", "infrastructure", "deployment")):
        return "MLOps and ML Platform: lead with Veloce AgenticOS, Docker/FastAPI, and production pipeline evidence."
    return "AI/ML and Data Science: lead with SymphonyAI, predictive maintenance, forecasting, and measurable model outcomes."


def _start_fit(job: JobPosting, preferences: CandidatePreferences) -> tuple[str, list[str]]:
    text = job.start_date_text.strip()
    if not text:
        # Only inspect description text when a year is close to an explicit
        # start-date cue. Company history and old project dates are not start
        # dates and must not silently block a candidate.
        start_match = re.search(
            r"(?:start(?:ing)?|begin(?:ning)?|join(?:ing)?|available from|anticipated start)" r"[^.\n]{0,100}",
            job.description,
            re.I,
        )
        text = start_match.group(0) if start_match else ""
    if not text.strip() or not (preferences.target_start_min or preferences.target_start_max):
        return "unknown", ["Start timing is not stated; confirm it with the employer."]
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]
    if not years:
        return "unknown", ["Start timing is ambiguous; confirm it with the employer."]
    min_year = preferences.target_start_min.year if preferences.target_start_min else None
    max_year = preferences.target_start_max.year if preferences.target_start_max else None
    if any((min_year is None or year >= min_year) and (max_year is None or year <= max_year) for year in years):
        return "compatible", []
    if any(year == (min_year or max_year) for year in years):
        return "borderline", ["The posting mentions a nearby start year; verify the exact start date."]
    return "outside_window", ["The stated start timing falls outside the selected window."]


def assess_eligibility(
    job: JobPosting,
    profile: Profile,
    preferences: CandidatePreferences,
    *,
    role_fit_score: int = 0,
    evidence_fit_score: int = 0,
) -> EligibilityAssessment:
    """Apply hard candidate policy and calculate the displayed priority score."""
    normalized = normalize_job(job)
    reasons: list[str] = []
    blockers: list[str] = []
    start_fit, start_reasons = _start_fit(normalized, preferences)
    reasons.extend(start_reasons)
    if start_fit == "outside_window":
        blockers.append("start date outside target window")

    allowed_types = {value.lower() for value in preferences.employment_types}
    if normalized.employment_type in {"internship", "co_op"} and preferences.exclude_internships:
        blockers.append("internship or co-op is excluded from the primary search")
    elif allowed_types and normalized.employment_type != "unknown" and normalized.employment_type not in allowed_types:
        blockers.append(f"employment type {normalized.employment_type} is not selected")
    elif normalized.employment_type in {"part_time", "temporary"}:
        blockers.append("posting is not full-time")
    elif normalized.employment_type == "unknown":
        reasons.append("employment type is not explicit; verify that it is full-time")

    if normalized.clearance_required:
        if preferences.clearance_status in {"unknown", "no_clearance"}:
            blockers.append("explicit security clearance requirement")
        else:
            reasons.append("security clearance is required")
    auth_signal = normalized.authorization_requirement in {"restricted", "mentioned"}
    sponsorship_signal = normalized.sponsorship_signal in {"not_available", "mentioned"}
    if auth_signal or sponsorship_signal:
        reasons.append("work authorization or sponsorship language needs human confirmation")

    if normalized.work_mode != "unknown" and normalized.work_mode not in preferences.accepted_work_modes:
        blockers.append(f"work mode {normalized.work_mode} is not accepted")
    if normalized.work_mode == "unknown":
        reasons.append("work mode is not explicit")

    bucket = role_bucket(normalized, preferences)
    if bucket == "adjacent":
        reasons.append("adjacent role shown separately from the primary AI/ML search")
    elif bucket == "review":
        reasons.append("role family is ambiguous; review before prioritizing")

    if blockers:
        status = "blocked"
    elif reasons or start_fit in {"unknown", "borderline"} or normalized.metadata_confidence < 0.4:
        status = "borderline"
    else:
        status = "eligible"

    base = max(0, min(100, role_fit_score or evidence_fit_score))
    if role_fit_score and evidence_fit_score:
        base = round(role_fit_score * 0.55 + evidence_fit_score * 0.45)
    if bucket == "adjacent":
        base = min(base, 69)
    if bucket == "review":
        base = min(base, 59)
    if status == "blocked":
        base = min(base, 25)
    elif status == "borderline":
        base = min(base, 79)

    return EligibilityAssessment(
        status=status,
        reasons=list(dict.fromkeys(reasons)),
        hard_blockers=list(dict.fromkeys(blockers)),
        role_bucket=bucket,
        start_timing_fit=start_fit,
        role_fit_score=role_fit_score or base,
        evidence_fit_score=evidence_fit_score or base,
        final_priority_score=base,
    )
