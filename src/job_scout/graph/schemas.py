"""Pydantic models used across the agent graph.

These are the structured-output targets for the LLM/tool calls and the shared
data contracts the nodes read and write.

Phase 2 note: the Phase 1 ``TailoringPack`` stub (a flat "emphasis brief") was
replaced by the corpus-grounded v2 models below. That is a breaking change to
the checkpoint format, and it is safe only because the sole checkpointer is an
in-process ``MemorySaver``: no checkpoint survives a restart and no Phase 1
code path ever wrote ``tailoring`` into a thread. With a persistent
checkpointer (e.g. Postgres) this would have required a real migration.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Seniority = Literal["junior", "mid", "senior", "lead", "unknown"]
JobSourceName = Literal["jsearch", "adzuna", "remotive", "cache"]
EligibilityStatus = Literal["eligible", "borderline", "blocked"]
RoleBucket = Literal["primary", "adjacent", "review"]


class Profile(BaseModel):
    """Structured candidate profile extracted from a CV."""

    name: str | None = None
    seniority: Seniority = "unknown"
    primary_roles: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    years_experience: float | None = None
    locations: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    remote_ok: bool = False
    raw_summary: str = ""
    education_history: list[EducationEntry] = Field(default_factory=list)
    expected_graduation_date: date | None = None
    current_program: str | None = None
    degree_fields: list[str] = Field(default_factory=list)
    professional_experience_months: int | None = None
    phone: str | None = None
    resume_evidence_refs: list[str] = Field(default_factory=list)


class EducationEntry(BaseModel):
    """One education record extracted from the resume."""

    institution: str
    degree: str = ""
    field: str = ""
    start_date: date | None = None
    end_date: date | None = None
    in_progress: bool = False
    source_ref: str | None = None


class CandidatePreferences(BaseModel):
    """Human-authored search policy; model output cannot override it."""

    employment_types: list[str] = Field(default_factory=lambda: ["full_time"])
    target_start_min: date | None = date(2026, 12, 1)
    target_start_max: date | None = date(2027, 3, 31)
    country_scope: str = "us"
    locations: list[str] = Field(default_factory=list)
    accepted_work_modes: list[str] = Field(default_factory=lambda: ["remote", "hybrid", "onsite"])
    primary_role_families: list[str] = Field(default_factory=lambda: ["ai_ml", "data_science", "genai", "forward_deployed"])
    adjacent_role_policy: str = "show_separately"
    authorization_status: str = "unknown"
    sponsorship_policy: str = "unknown"
    clearance_status: str = "unknown"
    exclude_internships: bool = True


class JobPosting(BaseModel):
    """A single job opening, normalized across all sources."""

    job_id: str
    title: str
    company: str
    location: str
    remote: bool = False
    description: str = ""
    url: str = ""
    tags: list[str] = Field(default_factory=list)
    source: JobSourceName
    employment_type: str = "unknown"
    work_mode: str = "unknown"
    experience_level: str = "unknown"
    posted_at: str | None = None
    start_date_text: str = ""
    clearance_required: bool = False
    authorization_requirement: str = "unknown"
    sponsorship_signal: str = "unknown"
    salary_text: str = ""
    source_url: str = ""
    metadata_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchRequest(BaseModel):
    """Typed search arguments returned by providers with fragile tool calling.

    Groq's Qwen endpoint supports structured JSON output, but some LangChain
    tool schemas can still produce a ``tool_use_failed`` response. Keeping the
    fetch contract as a small typed object gives that provider the same
    constrained query-selection behaviour without asking it to emit a function
    call.
    """

    query: str = Field(description="A two-to-four-word job title search query")
    country: str = Field(default="", description="A two-letter country code, or an empty string")
    remote: bool = Field(default=False, description="Whether to prioritize remote-friendly roles")
    limit: int = Field(default=25, ge=1, le=50, description="Maximum number of postings")


class SourceDiagnostic(BaseModel):
    """Observable outcome of one attempted job source."""

    source: str
    requested: bool = True
    completed: bool = False
    timed_out: bool = False
    latency_ms: float = 0.0
    returned: int = 0
    contributed: bool = False
    error: str | None = None


class JobScore(BaseModel):
    """The ranking LLM's score for one job, keyed back to a posting by id."""

    job_id: str
    fit_score: int = Field(ge=0, le=100)
    fit_explanation: str
    matched_skills: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    role_fit_score: int | None = Field(default=None, ge=0, le=100)
    evidence_fit_score: int | None = Field(default=None, ge=0, le=100)


class JobScores(BaseModel):
    """Structured-output container for a batch of ``JobScore``."""

    scores: list[JobScore]


class RankedJob(BaseModel):
    """A job scored against the candidate profile."""

    job: JobPosting
    fit_score: int = Field(ge=0, le=100)
    fit_explanation: str
    matched_skills: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    role_fit_score: int = Field(default=0, ge=0, le=100)
    evidence_fit_score: int = Field(default=0, ge=0, le=100)
    eligibility_status: EligibilityStatus = "borderline"
    eligibility_reasons: list[str] = Field(default_factory=list)
    hard_blockers: list[str] = Field(default_factory=list)
    primary_or_adjacent: RoleBucket = "review"
    start_timing_fit: str = "unknown"
    final_priority_score: int = Field(default=0, ge=0, le=100)


class TailoredBullet(BaseModel):
    """One CV bullet reworded for the target job.

    ``corpus_ref`` is required: every bullet must point at the ``CorpusItem``
    it derives from, so the fabrication validator can check the rewrite against
    the candidate's real experience.
    """

    text: str
    corpus_ref: str


class ExperienceEntry(BaseModel):
    """One role in the tailored CV's experience section."""

    role: str
    company: str
    dates: str = ""
    bullets: list[TailoredBullet] = Field(default_factory=list)


class CVLink(BaseModel):
    """A clickable link recovered from the source resume PDF."""

    label: str
    url: str
    page: int = Field(ge=1)
    source: str = "pdf_annotation"


class CVContent(BaseModel):
    """The tailored CV: selected and reworded content, never invented."""

    headline: str
    summary: str
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[ExperienceEntry] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    links: list[CVLink] = Field(default_factory=list)


class TailoringPack(BaseModel):
    """Application material generated for a selected job (Phase 2).

    The cover letter should be 250–350 words and reference at least two
    specific job requirements; the honesty note names real gaps the candidate
    should not paper over. The deterministic quality report is stored beside
    the pack, while fabrication validation remains a separate zero-LLM gate.
    """

    cv: CVContent
    cover_letter: str
    honesty_note: str = ""


class CoverLetterQualityReport(BaseModel):
    """Deterministic quality gate for a generated cover letter."""

    word_count: int = 0
    evidence_matches: int = 0
    requirement_matches: int = 0
    generic_phrases: list[str] = Field(default_factory=list)
    passed: bool = False
    reasons: list[str] = Field(default_factory=list)


ClaimClassification = Literal["near_miss", "unsupported"]


class FlaggedClaim(BaseModel):
    """One statement the fabrication validator could not ground in the corpus."""

    where: str  # "cv_bullet:<corpus_ref>" | "skill:<name>" | "cover_letter:sentence:<n>"
    text: str
    reason: str
    best_match_ratio: float = 0.0
    classification: ClaimClassification = "unsupported"


class FabricationReport(BaseModel):
    """Deterministic validator output: flagged claims, never a retry signal.

    ``claims_checked`` counts every claim the validator examined (bullets,
    skills, factual cover-letter sentences) so a fabrication *rate* is
    well-defined: ``flags / claims_checked``.
    """

    flags: int = 0
    claims_checked: int = 0
    flagged: list[FlaggedClaim] = Field(default_factory=list)
    confirmed_claims: int = 0
    near_miss_claims: int = 0
    unsupported_claims: int = 0
    # The knob values this report ran with — recorded so every trace states
    # what produced the flags, making threshold tuning measurable in Opik.
    thresholds: dict[str, float] = Field(default_factory=dict)
