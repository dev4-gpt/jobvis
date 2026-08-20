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
from typing import Any, cast

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
from job_scout.graph.schemas import (
    CandidatePreferences,
    CVContent,
    CVLink,
    ExperienceEntry,
    RankedJob,
    TailoredBullet,
    TailoringPack,
)
from job_scout.graph.state import AgentState
from job_scout.llm import ensure_budget, get_chat_model, model_chain, with_structured_output
from job_scout.tools.research import research_company
from job_scout.validation import validate_pack

_DESCRIPTION_LIMIT = 3000


class TailoringInvocationError(RuntimeError):
    """A provider attempt failed after a known number of model calls."""

    def __init__(self, message: str, *, calls_used: int) -> None:
        super().__init__(message)
        self.calls_used = calls_used


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
    model: Any,
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
            raise TailoringInvocationError(str(exc), calls_used=1) from exc
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
        raise TailoringInvocationError(
            "Tailoring model did not return a usable TailoringPack. The provider returned an empty or invalid response.",
            calls_used=2,
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
    for entry in (*cleaned.cv.experience, *cleaned.cv.projects):
        for bullet in entry.bullets:
            bullet.text = remove_unconfirmed_policy_sentences(bullet.text, **kwargs)
    cleaned.cover_letter = remove_unconfirmed_policy_sentences(cleaned.cover_letter, **kwargs)
    return cleaned


_FHIR_CLAIM = re.compile(
    r"\b(?:self[- ]study|self[- ]studied|independent study|familiarity with|knowledge of|experience with)"
    r"[^.!?\n]{0,80}\bfhir\b[^.!?\n]*[.!?]?",
    re.IGNORECASE,
)


def _clean_unsupported_domain_claims(pack: TailoringPack, corpus) -> TailoringPack:
    """Remove known domain-study claims when the source resume has no evidence."""
    if "fhir" in " ".join(item.text for item in corpus.items).lower():
        return pack
    cleaned = pack.model_copy(deep=True)
    for attr in ("headline", "summary"):
        setattr(cleaned.cv, attr, _FHIR_CLAIM.sub("", getattr(cleaned.cv, attr)).strip())
    for entry in (*cleaned.cv.experience, *cleaned.cv.projects):
        for bullet in entry.bullets:
            bullet.text = _FHIR_CLAIM.sub("", bullet.text).strip()
    cleaned.cv.skills = [skill for skill in cleaned.cv.skills if "fhir" not in skill.lower()]
    cleaned.cover_letter = _FHIR_CLAIM.sub("", cleaned.cover_letter).strip()
    cleaned.honesty_note = _FHIR_CLAIM.sub("", cleaned.honesty_note).strip()
    return cleaned


def _resume_links(value: object) -> list[CVLink]:
    """Normalize checkpoint link values before they reach the renderer."""
    links: list[CVLink] = []
    if isinstance(value, list):
        for item in value:
            try:
                links.append(item if isinstance(item, CVLink) else CVLink.model_validate(item))
            except (TypeError, ValueError):
                continue
    return links


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


def _deterministic_pack(profile, corpus, ranked: RankedJob, links, preferences: CandidatePreferences) -> TailoringPack:
    """Build a safe, usable draft when every configured provider is unavailable.

    This is intentionally modest rather than pretending to be an LLM rewrite:
    bullets are copied verbatim from the candidate corpus, skills come only
    from the parsed skills section, and the letter is generated by the
    deterministic grounded fallback. A provider outage therefore cannot
    produce an empty pack or invent a claim.
    """
    evidence_items = [item for item in corpus.items if item.kind in {"bullet", "summary"}]
    selected = _fallback_evidence(corpus, ranked)
    selected_items = []
    for text in selected:
        item = next((candidate for candidate in evidence_items if candidate.text == text), None)
        if item is not None and item not in selected_items:
            selected_items.append(item)
    if not selected_items:
        selected_items = evidence_items[:3]

    bullets = [TailoredBullet(text=item.text, corpus_ref=item.id) for item in selected_items[:4]]
    role = (profile.primary_roles[0] if profile.primary_roles else ranked.job.title) or "Relevant experience"
    summary = profile.raw_summary.strip() or (
        f"Candidate preparing for a full-time {ranked.job.title} opportunity with evidence documented in the source resume."
    )
    summary = remove_unconfirmed_policy_sentences(
        summary,
        authorization_status=preferences.authorization_status,
        sponsorship_policy=preferences.sponsorship_policy,
        clearance_status=preferences.clearance_status,
    )
    education = []
    for entry in profile.education_history:
        line = " — ".join(part for part in (entry.institution, entry.degree, entry.field) if part)
        if entry.end_date:
            line += f" ({entry.end_date.isoformat()})"
        if line:
            education.append(line)
    pack = TailoringPack(
        cv=CVContent(
            headline=role,
            summary=summary,
            experience=[ExperienceEntry(role="Relevant resume evidence", company="", bullets=bullets)],
            skills=corpus.skills()[:18],
            education=education,
            links=list(links),
        ),
        cover_letter=grounded_fallback_letter(
            candidate_name=profile.name or "Candidate",
            company=ranked.job.company,
            job_title=ranked.job.title,
            job_description=ranked.job.description,
            corpus_items=[item.text for item in selected_items[:3]],
        ),
        honesty_note=(
            "Provider fallback used. The CV bullets and links are copied from the source resume; "
            "review job-specific wording, authorization, sponsorship, clearance, and any missing requirements before sending."
        ),
    )
    return pack


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
    prompt = TAILOR_PROMPT.format(
        research_rule=RESEARCH_RULE if research else "",
        profile=_render_profile(profile),
        candidate_preferences=_render_preferences(state.get("candidate_preferences")),
        corpus=corpus.render_for_prompt(),
        job=_render_job(ranked),
        persona=resume_persona(ranked.job),
        research=research or "none",
    )
    # The dedicated tailor model may be a free OpenRouter model with weaker
    # structured-output support. Try it first, then explicit fallbacks and the
    # primary search model, all within the normal run budget. The final
    # deterministic pack is the last safety net and is still useful to the
    # human reviewer when every provider is unavailable.
    model_names = model_chain(tailor_model, settings.scout_fallback_models)
    if settings.scout_model and settings.scout_model not in model_names:
        model_names = (*model_names, settings.scout_model)
    total_calls = calls
    pack: TailoringPack | None = None
    successful_model = None
    successful_model_name = tailor_model
    provider_errors: list[str] = []
    for model_name in model_names:
        try:
            ensure_budget(total_calls, 1, settings.max_llm_calls_per_run)
            model = cast(Any, with_structured_output(get_chat_model(model_name, temperature=0.3), TailoringPack, model_name))
            pack, calls_used = _invoke_tailoring_pack(
                model,
                prompt,
                model_name,
                current_calls=total_calls,
                max_calls=settings.max_llm_calls_per_run,
            )
            total_calls += calls_used
            successful_model = model
            successful_model_name = model_name
            break
        except TailoringInvocationError as exc:
            total_calls += exc.calls_used
            provider_errors.append(f"{model_name}: {exc}")
        except Exception as exc:
            # Budget exhaustion remains a hard circuit breaker. It should not
            # be converted into a provider failure, but the deterministic pack
            # below still gives the user a reviewable result.
            if exc.__class__.__name__ == "LLMBudgetExceededError":
                provider_errors.append(str(exc))
                break
            provider_errors.append(f"{model_name}: {exc}")

    if pack is None:
        pack = _deterministic_pack(profile, corpus, ranked, _resume_links(state.get("cv_links", [])), preferences)
        errors.append("tailor: all configured tailoring providers failed; used a deterministic CV/letter draft for review")
        errors.extend(f"tailor provider: {message}" for message in provider_errors[:3])
    # Links are source metadata, not LLM content. Re-attach them after every
    # response so a model can never silently discard a clickable resume link.
    pack.cv.links = _resume_links(state.get("cv_links", []))
    pack = _clean_unconfirmed_policy_claims(pack, preferences)
    pack = _clean_unsupported_domain_claims(pack, corpus)
    quality = evaluate_cover_letter(
        pack.cover_letter,
        ranked.job.description,
        "\n".join(item.text for item in corpus.items),
        authorization_status=preferences.authorization_status,
        sponsorship_policy=preferences.sponsorship_policy,
        clearance_status=preferences.clearance_status,
    )
    if successful_model is not None and not quality.passed and total_calls < settings.max_llm_calls_per_run:
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
                successful_model,
                repair_prompt,
                successful_model_name,
                current_calls=total_calls,
                max_calls=settings.max_llm_calls_per_run,
            )
        except Exception as exc:
            errors.append(f"tailor: quality repair could not be completed — {exc}")
        else:
            repaired.cv.links = _resume_links(state.get("cv_links", []))
            pack = _clean_unconfirmed_policy_claims(repaired, preferences)
            pack = _clean_unsupported_domain_claims(pack, corpus)
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
