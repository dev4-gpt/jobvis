"""Tailor an application for one selected, already-ranked job (Phase 2).

This node runs as a SECOND invocation on the same checkpointer thread as the
job search: the caller passes only ``selected_job_id`` (plus an optional
LinkedIn export path) and everything else — ``profile``, ``ranked_jobs``,
``cv_text`` — is read from the thread's checkpoint. Nothing re-runs.

The candidate corpus is recomputed here from the checkpointed ``cv_text`` and
``linkedin_zip_path`` rather than stored in state: it is derived, deterministic
and LLM-free, so recomputing avoids checkpoint bloat and stale-derivation bugs
(``validate_tailoring`` recomputes it identically).

Guards never raise: a missing search state or an unknown job id is recorded in
``errors`` (visible in the trace span) and the node returns ``tailoring: None``.
"""

from __future__ import annotations

import json
import re

from job_scout.candidate_fit import preferences_from_dict, resume_persona
from job_scout.config import get_settings
from job_scout.corpus import build_corpus
from job_scout.cover_letter_quality import (
    evaluate_cover_letter,
    grounded_fallback_letter,
    remove_unconfirmed_policy_sentences,
    requirement_targets,
)
from job_scout.graph.nodes.rank_jobs import _render_profile
from job_scout.graph.prompts.tailor import RESEARCH_RULE, TAILOR_PROMPT
from job_scout.graph.schemas import CandidatePreferences, RankedJob, TailoringPack
from job_scout.graph.state import AgentState
from job_scout.llm import ensure_budget, get_chat_model, with_structured_output
from job_scout.tools.research import research_company
from job_scout.validation import validate_pack

_DESCRIPTION_LIMIT = 3000


def _message_text(message: object) -> str:
    """Extract text from a LangChain message without assuming one provider shape."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def _parse_json_pack(message: object) -> TailoringPack:
    """Parse and validate a plain JSON recovery response."""
    text = _message_text(message)
    if not text:
        raise ValueError("the model returned no JSON content")
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("the model returned invalid JSON") from exc
    return TailoringPack.model_validate(payload)


def _is_empty_structured_output_error(exc: Exception) -> bool:
    """Identify the OpenRouter empty structured-output failure without hiding API errors."""
    message = str(exc).lower()
    return isinstance(exc, ValueError) and ("parsed" in message or "refusal" in message or "structured output" in message)


def _invoke_tailoring_pack(
    model: object,
    prompt: str,
    tailor_model: str,
    *,
    current_calls: int,
    max_calls: int,
) -> tuple[TailoringPack, int]:
    """Invoke typed output, then use one bounded plain-JSON recovery call.

    OpenRouter models are not uniform: some accept ``response_format`` but do
    not populate LangChain's ``parsed`` field. The recovery deliberately uses
    the same provider/model and validates the JSON locally, so it cannot bypass
    the TailoringPack schema or the run budget.
    """
    try:
        result = model.invoke(prompt)
        if isinstance(result, TailoringPack):
            return result, 1
        return TailoringPack.model_validate(result), 1
    except Exception as exc:
        if not _is_empty_structured_output_error(exc):
            raise
        ensure_budget(current_calls + 1, 1, max_calls)

    recovery_prompt = (
        f"{prompt}\n\nThe typed response transport was unavailable for {tailor_model}. "
        "Return the complete TailoringPack as one valid JSON object only. "
        "Do not use Markdown fences or any text outside the JSON object."
    )
    try:
        recovery_model = get_chat_model(tailor_model, temperature=0.3)
        return _parse_json_pack(recovery_model.invoke(recovery_prompt)), 2
    except Exception as exc:
        raise RuntimeError(
            "Tailoring model did not return a usable TailoringPack. "
            "Try a model with JSON output support or switch SCOUT_TAILOR_MODEL; no draft was created."
        ) from exc


def _render_preferences(value: dict | CandidatePreferences | None) -> str:
    """Render human-authored policy as explicit tailoring constraints."""
    preferences = preferences_from_dict(value if value is not None else None)
    start_min = preferences.target_start_min.isoformat() if preferences.target_start_min else "unknown"
    start_max = preferences.target_start_max.isoformat() if preferences.target_start_max else "unknown"
    return (
        f"employment types: {', '.join(preferences.employment_types)}\n"
        f"target start window: {start_min} to {start_max}\n"
        f"country scope: {preferences.country_scope}; locations: {', '.join(preferences.locations) or 'anywhere in scope'}\n"
        f"accepted work modes: {', '.join(preferences.accepted_work_modes)}\n"
        f"primary role families: {', '.join(preferences.primary_role_families)}\n"
        f"exclude internships: {preferences.exclude_internships}\n"
        f"authorization: {preferences.authorization_status}; sponsorship: {preferences.sponsorship_policy}; "
        f"clearance: {preferences.clearance_status}"
    )


def _render_job(ranked: RankedJob) -> str:
    """Format the target job (including its ranking context) for the prompt."""
    job = ranked.job
    return (
        f"title: {job.title}\n"
        f"company: {job.company}\n"
        f"location: {job.location} (remote: {job.remote})\n"
        f"fit_score: {ranked.fit_score} — {ranked.fit_explanation}\n"
        f"description: {job.description[:_DESCRIPTION_LIMIT]}"
    )


def _clean_unconfirmed_policy_claims(pack: TailoringPack, preferences: CandidatePreferences) -> TailoringPack:
    """Remove unconfirmed authorization/visa/clearance claims before display."""
    cleaned = pack.model_copy(deep=True)
    kwargs = {
        "authorization_status": preferences.authorization_status,
        "sponsorship_policy": preferences.sponsorship_policy,
        "clearance_status": preferences.clearance_status,
    }
    cleaned.cv.headline = remove_unconfirmed_policy_sentences(cleaned.cv.headline, **kwargs)
    cleaned.cv.summary = remove_unconfirmed_policy_sentences(cleaned.cv.summary, **kwargs)
    cleaned.cover_letter = remove_unconfirmed_policy_sentences(cleaned.cover_letter, **kwargs)
    return cleaned


def _letter_has_grounding_flags(pack: TailoringPack, corpus, ranked: RankedJob, research: str | None, preferences) -> bool:
    """Return whether the cover letter still contains unsupported facts."""
    report = validate_pack(
        pack,
        corpus,
        research_notes=research,
        job_context=[
            f"{ranked.job.title} at {ranked.job.company}",
            ranked.job.company,
            ranked.job.description[:3000],
        ],
        candidate_preferences=preferences,
    )
    return any(flag.where.startswith("cover_letter:") or flag.where.startswith("policy:") for flag in report.flagged)


def _fallback_evidence(corpus, ranked: RankedJob) -> list[str]:
    """Select three corpus items that best match the target role vocabulary."""
    target = set(re.findall(r"[a-z0-9+#-]{3,}", f"{ranked.job.title} {ranked.job.description}".lower()))
    items = [item for item in corpus.items if item.kind in {"bullet", "summary"}]
    ranked_items = sorted(
        items,
        key=lambda item: len(target & set(re.findall(r"[a-z0-9+#-]{3,}", item.text.lower()))),
        reverse=True,
    )
    return [item.text for item in ranked_items[:3]]


def tailor(state: AgentState) -> dict:
    """Generate a ``TailoringPack`` for ``selected_job_id`` from checkpointed state."""
    settings = get_settings()
    errors = list(state.get("errors", []))
    job_id = state.get("selected_job_id")
    profile = state.get("profile")
    ranked_jobs = state.get("ranked_jobs", [])

    if profile is None or not ranked_jobs:
        errors.append("tailor: no search state on this thread — run a job search first")
        return {"tailoring": None, "errors": errors}

    ranked = next((r for r in ranked_jobs if r.job.job_id == job_id), None)
    if ranked is None:
        errors.append(f"tailor: selected job id {job_id!r} is not among the {len(ranked_jobs)} ranked jobs on this thread")
        return {"tailoring": None, "errors": errors}

    preferences = preferences_from_dict(state.get("candidate_preferences"))

    try:
        corpus = build_corpus(state.get("cv_text", ""), state.get("linkedin_zip_path"))
    except ValueError as exc:  # bad LinkedIn upload — degrade to CV-only
        errors.append(f"tailor: {exc}; continuing with the CV only")
        corpus = build_corpus(state.get("cv_text", ""))
    errors.extend(f"tailor: {warning}" for warning in corpus.warnings)
    if not corpus.items:
        errors.append("tailor: empty candidate corpus (no cv_text on this thread) — cannot ground an application")
        return {"tailoring": None, "errors": errors}

    research = research_company(ranked.job.company) if settings.has_tavily else None

    # llm_calls is checkpoint-cumulative, so on a shared thread the budget
    # effectively spans search + tailor invocations. Documented, not redesigned.
    calls = state.get("llm_calls", 0)
    ensure_budget(calls, 1, settings.max_llm_calls_per_run)

    # A dedicated tailoring model is optional; an empty setting intentionally
    # falls back to the primary provider/model used by the rest of the run.
    tailor_model = settings.scout_tailor_model or settings.scout_model
    model = with_structured_output(get_chat_model(tailor_model, temperature=0.3), TailoringPack, tailor_model)
    prompt = TAILOR_PROMPT.format(
        research_rule=RESEARCH_RULE if research else "",
        profile=_render_profile(profile),
        candidate_preferences=_render_preferences(state.get("candidate_preferences")),
        corpus=corpus.render_for_prompt(),
        job=_render_job(ranked),
        persona=resume_persona(ranked.job),
        research=research or "none",
    )
    try:
        pack, calls_used = _invoke_tailoring_pack(
            model,
            prompt,
            tailor_model,
            current_calls=calls,
            max_calls=settings.max_llm_calls_per_run,
        )
    except Exception as exc:
        errors.append(f"tailor: {exc}")
        return {"tailoring": None, "llm_calls": calls + 1, "errors": errors}
    # Links are source metadata, not LLM content. Re-attach them after every
    # response so a model can never silently discard a clickable resume link.
    pack.cv.links = list(state.get("cv_links", []))
    pack = _clean_unconfirmed_policy_claims(pack, preferences)
    quality = evaluate_cover_letter(
        pack.cover_letter,
        ranked.job.description,
        "\n".join(item.text for item in corpus.items),
        authorization_status=preferences.authorization_status,
        sponsorship_policy=preferences.sponsorship_policy,
        clearance_status=preferences.clearance_status,
    )
    total_calls = calls + calls_used
    if not quality.passed and total_calls < settings.max_llm_calls_per_run:
        targets = requirement_targets(ranked.job.description)
        target_text = "\n".join(f"- {target}" for target in targets[:4]) or "- No explicit requirement text was extracted."
        repair_prompt = (
            f"{prompt}\n\nYour first draft failed this deterministic quality gate: "
            f"{'; '.join(quality.reasons)}. Rewrite only the cover_letter field as a complete 250–350 word "
            "evidence-first letter. Address at least two distinct requirements below in separate sentences, "
            "using their concrete wording. Keep the CV and honesty_note grounded in the corpus.\n\n"
            f"Requirement targets used by the gate:\n{target_text}"
        )
        try:
            repaired, repair_calls = _invoke_tailoring_pack(
                model,
                repair_prompt,
                tailor_model,
                current_calls=total_calls,
                max_calls=settings.max_llm_calls_per_run,
            )
        except Exception as exc:
            errors.append(f"tailor: quality repair could not be completed — {exc}")
        else:
            repaired.cv.links = list(state.get("cv_links", []))
            pack = _clean_unconfirmed_policy_claims(repaired, preferences)
            total_calls += repair_calls
            quality = evaluate_cover_letter(
                pack.cover_letter,
                ranked.job.description,
                "\n".join(item.text for item in corpus.items),
                authorization_status=preferences.authorization_status,
                sponsorship_policy=preferences.sponsorship_policy,
                clearance_status=preferences.clearance_status,
            )
    if not quality.passed or _letter_has_grounding_flags(pack, corpus, ranked, research, preferences):
        # Do not present a persuasive but weak model draft as if it were ready
        # to send. The deterministic fallback is deliberately conservative: it
        # quotes the selected job requirements, copies real corpus evidence, and
        # states uncertainty instead of inventing authorization, domain study,
        # company statistics, or experience.
        pack.cover_letter = grounded_fallback_letter(
            candidate_name=profile.name or "Candidate",
            company=ranked.job.company,
            job_title=ranked.job.title,
            job_description=ranked.job.description,
            corpus_items=_fallback_evidence(corpus, ranked),
        )
        quality = evaluate_cover_letter(
            pack.cover_letter,
            ranked.job.description,
            "\n".join(item.text for item in corpus.items),
            authorization_status=preferences.authorization_status,
            sponsorship_policy=preferences.sponsorship_policy,
            clearance_status=preferences.clearance_status,
        )
        if quality.passed:
            errors.append("tailor: replaced the model letter with a grounded fallback after deterministic review")
        else:
            errors.append("tailor: cover-letter quality gate failed — review or regenerate before sending")
    return {
        "tailoring": pack if pack.cover_letter.strip() else None,
        "research_notes": research,
        "llm_calls": total_calls,
        "cover_letter_quality": quality,
        "errors": errors,
    }
