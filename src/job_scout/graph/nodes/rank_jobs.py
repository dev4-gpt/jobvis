"""Score each fetched job against the profile, one LLM call per batch.

The LLM returns lean ``JobScore`` objects keyed by ``job_id``; we pair each back
to its ``JobPosting`` to build a ``RankedJob``.
"""

from __future__ import annotations

import contextvars
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

from job_scout.candidate_fit import assess_eligibility, normalize_job, preferences_from_dict
from job_scout.config import get_settings
from job_scout.graph.prompts.rank_jobs import RANK_JOBS_PROMPT
from job_scout.graph.schemas import JobPosting, JobScore, JobScores, Profile, RankedJob
from job_scout.graph.state import AgentState
from job_scout.llm import ensure_budget, get_chat_model, model_chain, reasoning_kwargs, with_structured_output

# Batch size is a latency knob (SCOUT_RANK_BATCH): output tokens — and so batch
# latency — scale with jobs per batch, and batches run in parallel, so smaller
# batches shave the slowest-batch time at the cost of a few more LLM calls.
MAX_PARALLEL_BATCHES = 4


def _render_profile(profile: Profile) -> str:
    """Format the profile as plain text for the ranking prompt."""
    education = (
        ", ".join(entry.degree + " " + entry.field + " at " + entry.institution for entry in profile.education_history)
        or "unknown"
    )
    return (
        f"Name: {profile.name}\n"
        f"Seniority: {profile.seniority}\n"
        f"Roles: {', '.join(profile.primary_roles)}\n"
        f"Skills: {', '.join(profile.skills)}\n"
        f"Years experience: {profile.years_experience}\n"
        f"Locations: {', '.join(profile.locations)}\n"
        f"Remote ok: {profile.remote_ok}"
        f"\nEducation: {education}"
        f"\nExpected graduation: {profile.expected_graduation_date or 'unknown'}"
        f"\nCurrent program: {profile.current_program or 'unknown'}"
    )


def _render_jobs(jobs: list[JobPosting]) -> str:
    """Format a batch of jobs as plain text for the ranking prompt."""
    rendered = []
    for job in jobs:
        rendered.append(
            f"job_id: {job.job_id}\n"
            f"title: {job.title}\n"
            f"company: {job.company}\n"
            f"location: {job.location} (remote: {job.remote})\n"
            f"employment: {job.employment_type}; work mode: {job.work_mode}; level: {job.experience_level}\n"
            f"clearance required: {job.clearance_required}; authorization: {job.authorization_requirement}; "
            f"start: {job.start_date_text or 'unknown'}\n"
            f"description: {job.description[:1500]}"
        )
    return "\n\n---\n\n".join(rendered)


def _batches(items: list[JobPosting], size: int) -> Iterator[list[JobPosting]]:
    """Yield ``items`` in chunks of ``size``."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _deterministic_score(job: JobPosting, profile: Profile) -> JobScore:
    """Produce a conservative score when every ranking provider is unavailable.

    This is deliberately not presented as an LLM judgment. It uses only the
    normalized title/description, profile roles, and exact skill mentions, so
    a provider outage still leaves the user with inspectable postings while
    avoiding invented fit explanations.
    """
    normalized = normalize_job(job)
    haystack = f"{normalized.title} {normalized.description} {' '.join(normalized.tags)}".lower()
    matched = [
        skill
        for skill in profile.skills
        if skill.strip() and re.search(rf"(?<![a-z0-9]){re.escape(skill.strip().lower())}(?![a-z0-9])", haystack)
    ]
    role_match = any(role.strip().lower() in haystack for role in profile.primary_roles if role.strip())
    # The policy layer will cap, downgrade, or block this score below. Keep the
    # fallback intentionally middle-of-the-road: exact evidence can lift it,
    # but it can never masquerade as confident model reasoning.
    role_fit = min(82, 52 + (14 if role_match else 0) + min(4, len(matched)) * 4)
    evidence_fit = min(78, 42 + min(6, len(matched)) * 6)
    gaps = [] if matched else ["No exact profile-skill match was found; inspect the requirements manually."]
    explanation = (
        "Deterministic review score: the ranking provider was unavailable. "
        "Eligibility and matched evidence were computed locally; confirm fit manually."
    )
    return JobScore(
        job_id=normalized.job_id,
        fit_score=round(role_fit * 0.55 + evidence_fit * 0.45),
        fit_explanation=explanation,
        matched_skills=matched[:12],
        gaps=gaps,
        role_fit_score=role_fit,
        evidence_fit_score=evidence_fit,
    )


def _deterministic_scores(jobs: list[JobPosting], profile: Profile) -> JobScores:
    """Score a batch locally without making another network request."""
    return JobScores(scores=[_deterministic_score(job, profile) for job in jobs])


def _provider_error(exc: Exception | None) -> str:
    """Keep provider diagnostics useful without dumping a huge raw payload."""
    if exc is None:
        return "no configured ranking provider"
    detail = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {detail[:240]}"


def rank_jobs(state: AgentState) -> dict:
    """Score each fetched job against the profile and return them sorted by fit.

    Jobs already scored in a previous pass (reformulation loops merge new
    fetches into ``state["jobs"]``) keep their scores — the profile hasn't
    changed, so re-scoring them would only re-spend the same LLM calls. Only
    genuinely new postings go to the model.
    """
    settings = get_settings()
    profile = state["profile"]
    jobs = state.get("jobs", [])
    if not jobs:
        return {"ranked_jobs": []}

    ranked: list[RankedJob] = list(state.get("ranked_jobs") or [])
    already_scored = {r.job.job_id for r in ranked}
    to_score = [job for job in jobs if job.job_id not in already_scored]
    if not to_score:
        ranked.sort(key=lambda r: r.fit_score, reverse=True)
        return {"ranked_jobs": ranked}

    by_id = {job.job_id: job for job in to_score}
    calls = state.get("llm_calls", 0)
    batch_size = settings.scout_rank_batch
    n_batches = (len(to_score) + batch_size - 1) // batch_size
    models = model_chain(settings.scout_model, settings.scout_fallback_models)
    # A fallback is opt-in and bounded. Reserve the worst-case budget before
    # starting concurrent batches so a rate-limit retry cannot exceed the run
    # circuit breaker halfway through the search.
    ensure_budget(calls, n_batches * len(models), settings.max_llm_calls_per_run)

    def structured_model(model_name: str):
        model_kwargs = {"timeout": settings.scout_rank_timeout, "max_retries": 0}
        model_kwargs.update(reasoning_kwargs(model_name))
        return with_structured_output(get_chat_model(model_name, temperature=0.0, **model_kwargs), JobScores, model_name)

    def score_batch(batch: list[JobPosting]) -> tuple[JobScores, int, str | None]:
        prompt = RANK_JOBS_PROMPT.format(profile=_render_profile(profile), jobs=_render_jobs(batch))
        last_error: Exception | None = None
        for attempts, model_name in enumerate(models, start=1):
            try:
                return structured_model(model_name).invoke(prompt), attempts, None
            except Exception as exc:  # noqa: BLE001 - try only explicit fallback models
                last_error = exc
        return (
            _deterministic_scores(batch, profile),
            len(models),
            f"ranking providers unavailable; used deterministic review scores ({_provider_error(last_error)})",
        )

    # Batches are independent, so they run concurrently — ranking latency is the
    # slowest batch, not the sum. copy_context() carries LangChain's callback
    # contextvars into the worker threads, so Opik spans and token/cost tracking
    # still attach to the run (the whole point of this repo).
    batches = list(_batches(to_score, batch_size))
    if len(batches) == 1:
        results = [score_batch(batches[0])]
    else:
        pool = ThreadPoolExecutor(max_workers=min(len(batches), MAX_PARALLEL_BATCHES))
        futures = [pool.submit(contextvars.copy_context().run, score_batch, batch) for batch in batches]
        try:
            results = [future.result(timeout=settings.scout_rank_timeout) for future in futures]
        except TimeoutError:
            for future in futures:
                future.cancel()
            # Do not wait for a provider that ignored its client timeout; the
            # graph must return visible reviewable jobs instead of hanging the
            # UI or discarding the postings that the sources already returned.
            pool.shutdown(wait=False, cancel_futures=True)
            results = [
                (
                    _deterministic_scores(batch, profile),
                    len(models),
                    f"ranking timed out after SCOUT_RANK_TIMEOUT={settings.scout_rank_timeout:.0f}s; "
                    "used deterministic review scores",
                )
                for batch in batches
            ]
        else:
            pool.shutdown(wait=True)
    calls += sum(used_calls for _scores, used_calls, _diagnostic in results)
    diagnostics = [diagnostic for _scores, _used_calls, diagnostic in results if diagnostic]

    for result, _used_calls, _diagnostic in results:
        for score in result.scores:
            job = by_id.get(score.job_id)
            if job is None:
                continue
            normalized = normalize_job(job)
            if state.get("candidate_preferences") is not None:
                preferences = preferences_from_dict(state.get("candidate_preferences"))
                assessment = assess_eligibility(
                    normalized,
                    profile,
                    preferences,
                    role_fit_score=score.role_fit_score or score.fit_score,
                    evidence_fit_score=score.evidence_fit_score or _evidence_score(score),
                )
                ranked.append(
                    RankedJob(
                        job=normalized,
                        fit_score=assessment.final_priority_score,
                        final_priority_score=assessment.final_priority_score,
                        role_fit_score=assessment.role_fit_score,
                        evidence_fit_score=assessment.evidence_fit_score,
                        eligibility_status=assessment.status,
                        eligibility_reasons=assessment.reasons,
                        hard_blockers=assessment.hard_blockers,
                        primary_or_adjacent=assessment.role_bucket,
                        start_timing_fit=assessment.start_timing_fit,
                        fit_explanation=score.fit_explanation,
                        matched_skills=score.matched_skills,
                        gaps=score.gaps,
                    )
                )
            else:
                ranked.append(
                    RankedJob(
                        job=normalized,
                        fit_score=score.fit_score,
                        final_priority_score=score.fit_score,
                        role_fit_score=score.fit_score,
                        evidence_fit_score=_evidence_score(score),
                        fit_explanation=score.fit_explanation,
                        matched_skills=score.matched_skills,
                        gaps=score.gaps,
                    )
                )

    ranked.sort(key=lambda r: (r.eligibility_status == "blocked", r.primary_or_adjacent != "primary", -r.fit_score))
    errors = list(state.get("errors") or [])
    errors.extend(f"rank_jobs: {diagnostic}" for diagnostic in diagnostics)
    return {"ranked_jobs": ranked, "llm_calls": calls, "errors": errors}


def _evidence_score(score) -> int:
    """Approximate evidence fit from explicit matches/gaps when the model emits no breakdown."""
    matches = len(score.matched_skills)
    gaps = len(score.gaps)
    return max(0, min(100, 55 + matches * 8 - gaps * 4))
