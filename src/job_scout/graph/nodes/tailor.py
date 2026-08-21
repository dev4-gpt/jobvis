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
from job_scout.evals.backtest import backtest_pack, improve_pack
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

# A tailored CV is a usable document, not a three-bullet model summary. These
# are content gates rather than page-count promises: the renderer decides the
# exact pagination, while this contract keeps a normal 1.5–2 page CV from
# collapsing when a provider returns a technically valid but sparse object.
_CV_MIN_WORDS = 600
_CV_MIN_BULLETS = 10
_CV_MIN_EXPERIENCE_ENTRIES = 2
_CV_MIN_PROJECT_ENTRIES = 2
_CV_MIN_SKILLS = 12
_DATE_RANGE = re.compile(
    r"\b(?:19|20)\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\b",
    re.IGNORECASE,
)
_PROJECT_HEADER_TERMS = (
    "analyzer",
    "attacks",
    "agentic",
    "automation",
    "research assistantship",
    "veloce",
    "financial",
)
_ACTION_STARTS = (
    "built ",
    "developed ",
    "designed ",
    "implemented ",
    "created ",
    "integrated ",
    "deployed ",
    "improved ",
    "reduced ",
    "performed ",
    "investigated ",
    "analyzed ",
    "utilizes ",
    "the pipeline ",
)
_INTERNAL_SUMMARY_MARKERS = re.compile(
    r"\b(?:candidate targeting|source[- ]documented|preserved from the original|selected for this job|"
    r"relevant resume evidence|the experience and project bullets below)\b",
    re.IGNORECASE,
)


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
        recovery_model = get_chat_model(tailor_model, temperature=0.3, max_retries=0)
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
        f"deterministic eligibility: {ranked.eligibility_status}\n"
        f"deterministic role bucket: {ranked.primary_or_adjacent}\n"
        f"deterministic hard blockers: {', '.join(ranked.hard_blockers) or 'none'}\n"
        f"deterministic review reasons: {', '.join(ranked.eligibility_reasons) or 'none'}\n"
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


def _resume_email(links: list[CVLink]) -> str:
    """Recover the visible email address from the source mailto annotation."""
    for link in links:
        if link.url.lower().startswith("mailto:"):
            return link.url[7:].split("?", 1)[0]
    return ""


def _source_entry_by_refs(entries: list[ExperienceEntry]) -> dict[str, ExperienceEntry]:
    """Index source entries so model headers can be replaced deterministically."""
    return {bullet.corpus_ref: entry for entry in entries for bullet in entry.bullets}


def _restore_source_headers(pack: TailoringPack, corpus) -> TailoringPack:
    """Keep employer/project headers from the resume while retaining model emphasis."""
    cleaned = pack.model_copy(deep=True)
    source_experience = _source_entries(corpus, project=False)
    source_projects = _source_entries(corpus, project=True)
    for generated, source_entries in (
        (cleaned.cv.experience, source_experience),
        (cleaned.cv.projects, source_projects),
    ):
        by_ref = _source_entry_by_refs(source_entries)
        for index, entry in enumerate(generated):
            matches = [by_ref[bullet.corpus_ref] for bullet in entry.bullets if bullet.corpus_ref in by_ref]
            if not matches:
                continue
            source = matches[0]
            cleaned_entry = entry.model_copy(update={"role": source.role, "company": source.company, "dates": source.dates})
            generated[index] = cleaned_entry
    return cleaned


def _preserve_cv_metadata(pack: TailoringPack, profile, links: list[CVLink]) -> TailoringPack:
    """Overlay contact/link metadata after every model response."""
    cleaned = pack.model_copy(deep=True)
    cleaned.cv.links = list(links)
    cleaned.cv.email = _resume_email(links)
    cleaned.cv.phone = profile.phone or cleaned.cv.phone
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
    job_text = f"{ranked.job.title} {ranked.job.description}".lower()
    target = set(re.findall(r"[a-z0-9+#-]{3,}", job_text))
    items = [item for item in corpus.items if item.kind in {"bullet", "summary"}]
    creative_terms = (
        "creative production",
        "creative agents",
        "video, music",
        "openmontage",
        "thumbnails",
        "image generation",
        "content creation",
    )
    finance_terms = ("financial", "finance automation", "transaction classification", "financial data")
    evidence_terms = (
        "data",
        "analysis",
        "model",
        "python",
        "tensorflow",
        "scikit-learn",
        "fastapi",
        "predictive maintenance",
        "time-series",
        "rag",
        "langchain",
        "faiss",
        "computer vision",
    )

    def score(item) -> tuple[int, int, int]:
        text = item.text.lower()
        overlap = len(target & set(re.findall(r"[a-z0-9+#-]{3,}", text)))
        role_evidence = sum(term in text for term in evidence_terms)
        creative_irrelevant = bool(creative_terms and any(term in text for term in creative_terms)) and not any(
            term in job_text for term in ("creative", "media", "video", "content")
        )
        finance_irrelevant = any(term in text for term in finance_terms) and not any(
            term in job_text for term in ("finance", "financial", "fintech", "banking", "transaction")
        )
        penalty = 100 if creative_irrelevant or finance_irrelevant else 0
        return (overlap - penalty, role_evidence, len(text))

    ranked_items = sorted(enumerate(items), key=lambda pair: (*score(pair[1]), -pair[0]), reverse=True)
    selected = []
    sections: set[str] = set()
    for _, item in ranked_items:
        item_score = score(item)
        # A negative score is an explicit domain mismatch. A zero-score item
        # is only a last resort when the corpus has no better evidence; it
        # should not displace a measurable, role-relevant example.
        if item_score[0] < 0 or (item_score[0] == 0 and selected):
            continue
        section = str(getattr(item, "section", ""))
        if section and section in sections:
            continue
        selected.append(item.text)
        if section:
            sections.add(section)
        if len(selected) == 3:
            break
    return selected


def _cv_word_count(pack: TailoringPack) -> int:
    """Count words in all visible CV sections."""
    parts = [pack.cv.headline, pack.cv.summary, *pack.cv.skills, *pack.cv.education]
    for entry in (*pack.cv.experience, *pack.cv.projects):
        parts.extend((entry.role, entry.company, entry.dates))
        parts.extend(bullet.text for bullet in entry.bullets)
    return len(" ".join(parts).split())


def _cv_bullet_count(pack: TailoringPack) -> int:
    """Count tailored CV bullets across experience and projects."""
    return sum(len(entry.bullets) for entry in (*pack.cv.experience, *pack.cv.projects))


def _normalize_source_text(text: str) -> str:
    """Make PDF-extracted source text readable without changing its claims."""
    cleaned = re.sub(r"\s+", " ", text.lstrip("".join("-•*–◦"))).strip()
    return re.sub(r"\s+([,.;:)])", r"\1", cleaned)


def _clean_source_header(text: str) -> str:
    """Remove PDF-only link/location columns from a source entry heading."""
    cleaned = _normalize_source_text(text)
    cleaned = re.sub(r"\s+(?:Landing Page|Product Page).*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+(?:GitHub|Github)\s+.*$", "", cleaned)
    cleaned = re.sub(
        r"\s+(?:[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*,\s*(?:[A-Z]{2}|India|Singapore|Germany|Canada|Australia))\s*$",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\s+(?:India|Singapore|Germany|Canada|Australia)\s*$", "", cleaned)
    # The source PDF's ResearchOS heading is extracted as a known artifact.
    if re.search(r"research assistantship\s+and\s+operating system", cleaned, re.IGNORECASE):
        return "ResearchOS"
    return cleaned.strip(" -–") or "Source resume evidence"


def _group_source_items(corpus) -> list[tuple[str, list[Any]]]:
    """Group contiguous CV bullets by the metadata heading they followed."""
    groups: list[tuple[str, list[Any]]] = []
    for item in corpus.items:
        if item.source != "cv" or item.kind != "bullet":
            continue
        if groups and groups[-1][0] == item.section:
            groups[-1][1].append(item)
        else:
            groups.append((item.section, [item]))
    return groups


def _is_project_group(section: str) -> bool:
    """Identify project headings conservatively from source-only metadata."""
    lowered = section.lower()
    return any(term in lowered for term in _PROJECT_HEADER_TERMS)


def _source_entry(section: str, items: list[Any], *, project: bool) -> ExperienceEntry | None:
    """Convert one source heading group into a grounded CV entry."""
    source_items = [_normalize_source_text(item.text) for item in items]
    dated = next((text for text in source_items if _DATE_RANGE.search(text)), "")
    dates = " — ".join(re.findall(r"(?:[A-Z][a-z]+\s+)?(?:19|20)\d{2}[^,;]*", dated)).strip(" ()")
    bullets = [
        TailoredBullet(text=_normalize_source_text(item.text), corpus_ref=item.id)
        for item in items
        if len(item.text.split()) >= 8 and not (_DATE_RANGE.search(item.text) and len(item.text.split()) < 10)
    ]
    # Role/date lines are usually the only short item in an experience group;
    # remove obvious metadata that slipped through the PDF extractor.
    bullets = [
        bullet
        for bullet in bullets
        if not (_DATE_RANGE.search(bullet.text) and not bullet.text.lower().startswith(_ACTION_STARTS))
    ]
    if not bullets:
        return None
    role = _clean_source_header(section)
    company = ""
    if not project:
        role_line = next((text for text in source_items if _DATE_RANGE.search(text)), "")
        role_line = re.sub(
            r"\s*(?:\([^)]*(?:19|20)\d{2}[^)]*\)|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+)?(?:19|20)\d{2}.*)$",
            "",
            role_line,
            flags=re.IGNORECASE,
        ).strip(" ,–-")
        if section.strip().lower() in {"experience", "professional experience", "work experience"} and "," in role_line:
            role, company = (part.strip() for part in role_line.split(",", 1))
        else:
            role = role_line or role
            company = _clean_source_header(section)
    return ExperienceEntry(role=role, company=company, dates=dates, bullets=bullets)


def _source_entries(corpus, *, project: bool) -> list[ExperienceEntry]:
    """Build grounded source entries for experience or selected projects."""
    entries: list[ExperienceEntry] = []
    for section, items in _group_source_items(corpus):
        if _is_project_group(section) is not project:
            continue
        entry = _source_entry(section, items, project=project)
        if entry is not None:
            entries.append(entry)
    return entries


def _entry_relevance(entry: ExperienceEntry, ranked: RankedJob) -> int:
    """Score source projects for ordering without inventing evidence.

    Generic token overlap is not enough for a personal resume: ``data`` can
    make a finance automation project outrank stronger ML evidence for a
    general Data Scientist role. Persona bonuses and explicit domain penalties
    make that failure deterministic and regression-testable.
    """
    job_text = f"{ranked.job.title} {ranked.job.description}".lower()
    target = set(re.findall(r"[a-z0-9+#-]{3,}", job_text))
    entry_text = " ".join([entry.role, entry.company, *(bullet.text for bullet in entry.bullets)]).lower()
    score = len(target & set(re.findall(r"[a-z0-9+#-]{3,}", entry_text)))
    finance_job = any(term in job_text for term in ("finance", "financial", "fintech", "banking", "transaction"))
    finance_project = any(
        term in entry_text for term in ("financial automation", "financial analysis", "transaction classification")
    )
    if finance_project and not finance_job:
        score -= 100
    if any(term in job_text for term in ("forward deployed", "forward-deployed", "solutions engineer", "customer engineer")):
        score += 20 if "veloce" in entry_text or "research assistantship" in entry_text else 0
    elif any(term in job_text for term in ("rag", "llm", "generative ai", "genai", "language model")):
        score += 20 if "legal document analyzer" in entry_text or "research assistantship" in entry_text else 0
    elif any(term in job_text for term in ("computer vision", "vision", "perception", "yolo")):
        score += 20 if "adversarial" in entry_text or "vision" in entry_text else 0
    elif any(term in job_text for term in ("mlops", "platform", "infrastructure", "deployment")):
        score += 20 if "veloce" in entry_text or "legal document analyzer" in entry_text else 0
    else:
        score += 10 if any(term in entry_text for term in ("research assistantship", "legal document analyzer")) else 0
    return score


def _source_education(profile, corpus) -> list[str]:
    """Prefer structured education, with a source-text fallback."""
    education: list[str] = []
    for entry in profile.education_history:
        line = " — ".join(part for part in (entry.institution, entry.degree, entry.field) if part)
        if entry.end_date:
            line += f" ({entry.end_date.isoformat()})"
        if line:
            education.append(line)
    if education:
        return education
    return [_normalize_source_text(item.text) for item in corpus.items if item.kind == "education"]


def _source_summary(profile, corpus, ranked: RankedJob, preferences: CandidatePreferences) -> str:
    """Create an employer-facing summary from verified source evidence."""
    text = " ".join(item.text for item in corpus.items).lower()
    focus_terms = [
        ("predictive maintenance and time-series analysis", "predictive maintenance"),
        ("computer vision", "computer vision"),
        ("retrieval-augmented generation", "rag"),
        ("LLM agent systems", "llm agent"),
        ("deployed data services", "fastapi"),
    ]
    focus = [label for label, marker in focus_terms if marker in text]
    if not focus:
        focus = ["machine-learning and data systems"]
    outcomes = []
    for phrase in (
        "increasing product throughput by 4%",
        "reducing unplanned downtime by 25%",
        "improving asset classification accuracy by 30%",
        "reducing false positives by 30%",
    ):
        if phrase in text:
            outcomes.append(phrase)
    outcome_text = "; ".join(outcomes[:3]) if outcomes else "measured model and service outcomes"
    role = ranked.job.title.strip() or (profile.primary_roles[0] if profile.primary_roles else "AI/ML")
    tool_terms = (
        ("Python", "python"),
        ("SQL", "sql"),
        ("TensorFlow", "tensorflow"),
        ("scikit-learn", "scikit-learn"),
        ("YOLOv7", "yolov7"),
        ("FastAPI", "fastapi"),
        ("LangChain", "langchain"),
    )
    tools = ", ".join(label for label, marker in tool_terms if marker in text)
    technical_sentence = (
        f"Technical foundation includes {tools}. "
        if tools
        else "Technical foundation includes reproducible analysis, model evaluation, and deployment. "
    )
    timeline_sentence = ""
    if profile.current_program:
        graduation = (
            f" with expected graduation in {profile.expected_graduation_date.strftime('%B %Y')}"
            if profile.expected_graduation_date
            else ""
        )
        timeline_sentence = (
            f"Currently completing {profile.current_program}{graduation}, and targeting full-time work after graduation. "
        )
    return (
        f"Early-career AI/ML and data science professional pursuing full-time {role} opportunities. "
        f"Experience spans {', '.join(focus[:4])}, with practical work in model evaluation, deployment, "
        "and technical communication. "
        f"Measured outcomes include {outcome_text}. "
        f"{timeline_sentence}{technical_sentence}"
        "I focus on reproducible analysis, measurable model performance, and deployable services while communicating "
        "results clearly to technical and nontechnical collaborators across applied machine-learning projects."
    )


def _clean_cv_summary(
    pack: TailoringPack, profile, corpus, ranked: RankedJob, preferences: CandidatePreferences
) -> TailoringPack:
    """Replace internal/meta summaries before a pack reaches the renderer."""
    cleaned = pack.model_copy(deep=True)
    # The selected posting is authoritative for the visible target heading.
    # This prevents a dense model response from retaining a generic headline
    # such as "Data Scientist" when the application is for Applied ML Engineer.
    target_headline = (ranked.job.title or "").strip()
    if target_headline:
        cleaned.cv.headline = target_headline[:120]
    summary = re.sub(r"\s+", " ", cleaned.cv.summary).strip()
    if not summary or _INTERNAL_SUMMARY_MARKERS.search(summary):
        cleaned.cv.summary = _source_summary(profile, corpus, ranked, preferences)
    else:
        cleaned.cv.summary = summary
    return cleaned


def _limit_project_entries(pack: TailoringPack, ranked: RankedJob, maximum: int = 3) -> TailoringPack:
    """Keep the most relevant projects so a tailored CV is not a portfolio dump."""
    if len(pack.cv.projects) <= maximum:
        return pack
    cleaned = pack.model_copy(deep=True)
    indexed = list(enumerate(cleaned.cv.projects))
    indexed.sort(key=lambda pair: (-_entry_relevance(pair[1], ranked), pair[0]))
    selected = {index for index, _ in indexed[:maximum]}
    cleaned.cv.projects = [entry for index, entry in enumerate(cleaned.cv.projects) if index in selected]
    return cleaned


def _source_cv_content(profile, corpus, ranked: RankedJob, links, preferences: CandidatePreferences) -> CVContent:
    """Build a dense CV from source evidence without asking an LLM to fill gaps."""
    experience = _source_entries(corpus, project=False)
    projects = _source_entries(corpus, project=True)
    if not experience:
        bullets = [
            TailoredBullet(text=_normalize_source_text(item.text), corpus_ref=item.id)
            for item in corpus.items
            if item.kind == "bullet" and len(item.text.split()) >= 8
        ]
        experience = [ExperienceEntry(role="Professional and research evidence", company="", bullets=bullets[:8])]
    projects = sorted(projects, key=lambda entry: _entry_relevance(entry, ranked), reverse=True)
    # Keep the strongest three source projects. This gives the PDF
    # substance without reproducing every project in every application.
    projects = [entry.model_copy(update={"bullets": entry.bullets[:4]}) for entry in projects[:3]]
    skills = corpus.skills() or list(profile.skills)
    return CVContent(
        headline=(profile.primary_roles[0] if profile.primary_roles else ranked.job.title) or "AI/ML professional",
        summary=_source_summary(profile, corpus, ranked, preferences),
        email=_resume_email(list(links)),
        phone=profile.phone or "",
        experience=experience,
        projects=projects,
        skills=skills[:35],
        education=_source_education(profile, corpus),
        links=list(links),
    )


def _enforce_cv_contract(
    pack: TailoringPack,
    profile,
    corpus,
    ranked: RankedJob,
    links,
    preferences: CandidatePreferences,
) -> tuple[TailoringPack, bool]:
    """Reject sparse model CVs and restore missing source sections deterministically."""
    cleaned = _preserve_cv_metadata(pack, profile, links)
    cleaned = _clean_cv_summary(cleaned, profile, corpus, ranked, preferences)
    cleaned = _limit_project_entries(cleaned, ranked)
    dense_enough = (
        _cv_word_count(cleaned) >= _CV_MIN_WORDS
        and _cv_bullet_count(cleaned) >= _CV_MIN_BULLETS
        and len(cleaned.cv.experience) >= _CV_MIN_EXPERIENCE_ENTRIES
        and len(cleaned.cv.projects) >= _CV_MIN_PROJECT_ENTRIES
        and len(cleaned.cv.skills) >= _CV_MIN_SKILLS
    )
    if dense_enough:
        return cleaned, False

    source_cv = _source_cv_content(profile, corpus, ranked, links, preferences)
    valid_refs = {item.id for item in corpus.items}
    model_bullets = [
        bullet
        for entry in (*cleaned.cv.experience, *cleaned.cv.projects)
        for bullet in entry.bullets
        if bullet.corpus_ref in valid_refs and bullet.text.strip()
    ]
    # Preserve valid model emphasis by prepending it to the first source role;
    # all remaining material still comes verbatim from the original corpus.
    existing_refs = {bullet.corpus_ref for entry in (*source_cv.experience, *source_cv.projects) for bullet in entry.bullets}
    preserved = [bullet for bullet in model_bullets if bullet.corpus_ref not in existing_refs]
    if preserved and source_cv.experience:
        first = source_cv.experience[0]
        source_cv.experience[0] = first.model_copy(update={"bullets": [*preserved, *first.bullets]})
    if len(cleaned.cv.summary.split()) >= 35:
        source_cv.summary = cleaned.cv.summary
    if cleaned.cv.headline.strip() and cleaned.cv.headline.lower() not in {"data scientist", "relevant resume evidence"}:
        source_cv.headline = cleaned.cv.headline
    source_cv.skills = list(dict.fromkeys([*cleaned.cv.skills, *source_cv.skills]))[:35]
    source_cv.links = list(links)
    return cleaned.model_copy(update={"cv": source_cv}), True


def _deterministic_pack(profile, corpus, ranked: RankedJob, links, preferences: CandidatePreferences) -> TailoringPack:
    """Build a safe, usable draft when every configured provider is unavailable.

    This is intentionally modest rather than pretending to be an LLM rewrite:
    bullets are copied verbatim from the candidate corpus, skills come only
    from the parsed skills section, and the letter is generated by the
    deterministic grounded fallback. A provider outage therefore cannot
    produce an empty pack or invent a claim.
    """
    source_cv = _source_cv_content(profile, corpus, ranked, links, preferences)
    pack = TailoringPack(
        cv=source_cv,
        cover_letter=grounded_fallback_letter(
            candidate_name=profile.name or "Candidate",
            company=ranked.job.company,
            job_title=ranked.job.title,
            job_description=ranked.job.description,
            corpus_items=_fallback_evidence(corpus, ranked),
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
        return {"tailoring": None, "errors": errors, "tailor_issue_codes": [], "tailor_backtest_score": None}

    ranked = next((r for r in ranked_jobs if r.job.job_id == job_id), None)
    if ranked is None:
        errors.append(f"tailor: selected job id {job_id!r} is not among the {len(ranked_jobs)} ranked jobs on this thread")
        return {"tailoring": None, "errors": errors, "tailor_issue_codes": [], "tailor_backtest_score": None}

    # A blocked posting is already a deterministic policy decision. Do not
    # spend provider calls creating an application for a role the candidate's
    # own search policy has excluded (internship, non-full-time, clearance,
    # start-window mismatch, or rejected work mode). Adjacent/review roles are
    # still tailorable when the user explicitly chooses them, but the prompt
    # receives their bucket and reasons so it cannot present them as primary.
    if ranked.eligibility_status == "blocked":
        blockers = "; ".join(ranked.hard_blockers or ranked.eligibility_reasons) or "policy constraint"
        errors.append(f"tailor: selected job is blocked by the candidate policy ({blockers}); no application was generated")
        return {
            "tailoring": None,
            "errors": errors,
            "cover_letter_quality": None,
            "tailor_issue_codes": [],
            "tailor_backtest_score": None,
        }

    preferences = preferences_from_dict(state.get("candidate_preferences"))

    try:
        corpus = build_corpus(state.get("cv_text", ""), state.get("linkedin_zip_path"))
    except ValueError as exc:  # bad LinkedIn upload — degrade to CV-only
        errors.append(f"tailor: {exc}; continuing with the CV only")
        corpus = build_corpus(state.get("cv_text", ""))
    errors.extend(f"tailor: {warning}" for warning in corpus.warnings)
    if not corpus.items:
        errors.append("tailor: empty candidate corpus (no cv_text on this thread) — cannot ground an application")
        return {"tailoring": None, "errors": errors, "tailor_issue_codes": [], "tailor_backtest_score": None}

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
            model = cast(
                Any,
                with_structured_output(
                    get_chat_model(
                        model_name,
                        temperature=0.3,
                        timeout=settings.scout_tailor_timeout,
                        max_retries=0,
                    ),
                    TailoringPack,
                    model_name,
                ),
            )
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
    # Links and contact fields are source metadata, not LLM content. Re-attach
    # them after every response so a model can never silently discard them.
    source_links = _resume_links(state.get("cv_links", []))
    pack = _preserve_cv_metadata(pack, profile, source_links)
    pack = _clean_unconfirmed_policy_claims(pack, preferences)
    pack = _clean_unsupported_domain_claims(pack, corpus)
    pack = _restore_source_headers(pack, corpus)
    pack, cv_rebuilt = _enforce_cv_contract(
        pack,
        profile,
        corpus,
        ranked,
        source_links,
        preferences,
    )
    if cv_rebuilt:
        errors.append(
            "tailor: model CV failed the density contract; restored source experience, projects, skills, education, and links"
        )

    def evaluate_candidate(candidate: TailoringPack):
        return backtest_pack(
            candidate,
            state.get("cv_text", ""),
            ranked.job.description,
            source_links=source_links,
            candidate_preferences=preferences,
            research_notes=research,
            job_context=(
                f"{ranked.job.title} at {ranked.job.company}",
                ranked.job.company,
                ranked.job.description[:_DESCRIPTION_LIMIT],
            ),
        )

    def repair_candidate(current: TailoringPack, report, attempt: int):
        nonlocal total_calls
        if successful_model is None or total_calls >= settings.max_llm_calls_per_run:
            return None
        targets = requirement_targets(ranked.job.description)
        target_text = "\n".join(f"- {target}" for target in targets[:4]) or "- No explicit requirement text was extracted."
        failure_text = "; ".join(report.failures[:6]) or "deterministic backtest did not pass"
        repair_prompt = (
            f"{prompt}\n\nThis is bounded self-improvement attempt {attempt}. The current pack failed its "
            f"deterministic backtest: {failure_text}. Stable issue codes: "
            f"{', '.join(report.failure_codes) or 'unclassified'}. Return a complete TailoringPack JSON object that improves "
            "the failing dimensions while preserving every passing dimension. The cover letter must be a complete "
            "250–350 word evidence-first letter, address at least two distinct requirements below in separate "
            "sentences, and use only corpus-grounded candidate evidence. Do not add authorization, sponsorship, "
            "visa, clearance, domain-study, employer, metric, or company claims not in the source.\n\n"
            f"Requirement targets used by the backtest:\n{target_text}"
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
            errors.append(f"tailor: self-improvement attempt {attempt} failed — {exc}")
            return None
        total_calls += repair_calls
        repaired = _preserve_cv_metadata(repaired, profile, source_links)
        repaired = _clean_unconfirmed_policy_claims(repaired, preferences)
        repaired = _clean_unsupported_domain_claims(repaired, corpus)
        repaired = _restore_source_headers(repaired, corpus)
        repaired, rebuilt = _enforce_cv_contract(
            repaired,
            profile,
            corpus,
            ranked,
            source_links,
            preferences,
        )
        if rebuilt:
            errors.append(f"tailor: self-improvement attempt {attempt} restored sparse CV sections")
        return repaired

    improvement = improve_pack(
        pack,
        evaluate_candidate,
        repair_candidate,
        max_attempts=settings.scout_tailor_max_repairs,
    )
    pack = improvement.pack
    quality = improvement.report.cover_letter_quality
    issue_codes = list(improvement.report.failure_codes)
    if improvement.attempts:
        errors.append(
            f"tailor: deterministic backtest ran {len(improvement.history)} version(s); best score {improvement.report.score:.3f}"
        )
    if issue_codes:
        errors.append(f"tailor: quality issue codes — {', '.join(issue_codes)}")
    if not improvement.report.passed:
        if improvement.report.failures:
            errors.append(f"tailor: backtest review — {'; '.join(improvement.report.failures[:5])}")
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
        final_report = evaluate_candidate(pack)
        issue_codes = list(final_report.failure_codes)
        if not final_report.passed:
            errors.append("tailor: final deterministic backtest still failed — review or regenerate before sending")
        if quality.passed:
            errors.append("tailor: replaced the model letter with a grounded fallback after deterministic review")
        else:
            errors.append("tailor: cover-letter quality gate failed — review or regenerate before sending")
    return {
        "tailoring": pack if pack.cover_letter.strip() else None,
        "research_notes": research,
        "llm_calls": total_calls,
        "cover_letter_quality": quality,
        "tailor_issue_codes": issue_codes,
        "tailor_backtest_score": improvement.report.score,
        "errors": errors,
    }
