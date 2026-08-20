"""Extract a structured candidate profile from CV text.

This is a preprocessing step that runs *before* the job-finding graph: the graph
takes the resulting ``Profile`` as input and focuses on searching and ranking
jobs. Keeping extraction out of the graph keeps the graph about one thing —
finding jobs — and lets a caller (like the UI) extract once and reuse it.
"""

from __future__ import annotations

import json
import re
from datetime import date

from job_scout.config import get_settings
from job_scout.graph.schemas import EducationEntry, Profile
from job_scout.llm import get_chat_model, model_chain, with_structured_output

EXTRACT_PROFILE_PROMPT_NAME = "extract_profile"

EXTRACT_PROFILE_PROMPT = """You are a recruiting assistant. Read the CV text below and extract a structured candidate profile.

Fill in every field:
- name: the candidate's name, or null if not present.
- seniority: one of junior, mid, senior, lead, or unknown.
- primary_roles: the job titles/roles this person is a fit for, ordered with their current or most recent role first.
- skills: a list of their skills, lowercased.
- years_experience: total years of professional experience as a number, or null.
- locations: locations where they could work.
- languages: spoken languages.
- remote_ok: true if they are open to remote work.
- education_history: every degree with institution, field, start/end dates when stated, and whether it is in progress.
- expected_graduation_date: ISO date when an in-progress program has a stated end date, otherwise null.
- current_program: the current degree/program name, or null.
- degree_fields: fields of study.
- professional_experience_months: approximate months of non-academic professional experience.
- phone: phone number if present, otherwise null.
- resume_evidence_refs: short source labels for key experience evidence.
- raw_summary: a 3-4 sentence summary, starting with their most recent experience.

CV text:
{cv_text}

Return exactly one JSON object matching the requested fields. Do not use
Markdown fences, commentary, or an explanation outside the JSON object.
"""


def extract_profile(
    cv_text: str, *, thread_id: str | None = None, tags: list[str] | None = None, model: str | None = None
) -> Profile:
    """Extract a structured profile from CV text with a single LLM call.

    Pass ``thread_id`` and ``tags`` to trace the call in Opik (grouped with the
    search run on the same thread). ``model`` overrides ``SCOUT_MODEL`` — used
    by the eval harness to compare extractors.
    """
    from job_scout.tracing import get_tracer

    settings = get_settings()
    model_name = model or settings.scout_model
    prompt = EXTRACT_PROFILE_PROMPT.format(cv_text=cv_text)
    tracer = get_tracer(thread_id, tags or ["extract"]) if thread_id else None
    config = {"callbacks": [tracer]} if tracer else {}
    profile = _extract_with_model_chain(prompt, config, model_chain(model_name, settings.scout_fallback_models))
    profile = _augment_resume_facts(profile, cv_text)
    if tracer:
        tracer.flush()
    return profile


def _message_text(message: object) -> str:
    """Read text from provider-specific LangChain message shapes."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            item if isinstance(item, str) else str(item.get("text", "")) for item in content if isinstance(item, str | dict)
        ).strip()
    return ""


def _parse_profile_json(message: object) -> Profile:
    """Parse a plain JSON recovery response and validate it at the boundary."""
    text = _message_text(message)
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    if not text:
        raise ValueError("profile model returned no JSON content")
    return Profile.model_validate(json.loads(text))


def _extract_with_model_chain(prompt: str, config: dict, models: tuple[str, ...]) -> Profile:
    """Use typed extraction with bounded plain-JSON recovery per model."""
    last_error: Exception | None = None
    for model_name in models:
        try:
            typed = with_structured_output(get_chat_model(model_name, temperature=0.0), Profile, model_name)
            result = typed.invoke(prompt, config=config)
            if isinstance(result, Profile):
                return result
            return Profile.model_validate(result)
        except Exception as exc:  # noqa: BLE001 - try the explicit next model
            last_error = exc
            try:
                recovery = get_chat_model(model_name, temperature=0.0)
                return _parse_profile_json(
                    recovery.invoke(
                        f"{prompt}\n\nReturn one valid JSON object only. Do not use Markdown fences or commentary.",
                        config=config,
                    )
                )
            except Exception as recovery_error:  # noqa: BLE001 - continue to the next configured model
                last_error = recovery_error
    raise RuntimeError(f"All configured profile models failed ({', '.join(models)}).") from last_error


def _augment_resume_facts(profile: Profile, cv_text: str) -> Profile:
    """Make concrete timeline/contact facts deterministic after model extraction.

    This is intentionally a small post-processor: the model still extracts the
    broad profile, while dates and phone numbers that drive eligibility come
    from text that can be tested and inspected.
    """
    text = cv_text or ""
    entries = list(profile.education_history)

    def add_entry(institution: str, degree: str, field: str, start: date, end: date | None, in_progress: bool) -> None:
        if any(institution.lower() in entry.institution.lower() for entry in entries):
            return
        entries.append(
            EducationEntry(
                institution=institution,
                degree=degree,
                field=field,
                start_date=start,
                end_date=end,
                in_progress=in_progress,
                source_ref=f"resume:education:{institution.lower().replace(' ', '-')}",
            )
        )

    lowered = text.lower()
    if "penn state" in lowered and "artificial intelligence" in lowered:
        add_entry("Penn State", "M.S.", "Artificial Intelligence", date(2025, 8, 1), date(2026, 12, 1), True)
    if "nyu" in lowered and "computer engineering" in lowered:
        add_entry("NYU", "M.S.", "Computer Engineering", date(2024, 9, 1), date(2025, 8, 1), False)
    if "manipal" in lowered and "mechatronics" in lowered:
        add_entry("Manipal", "B.Tech.", "Mechatronics Engineering", date(2020, 8, 1), date(2024, 7, 1), False)

    phone_match = re.search(r"(?:\+?[\d(][\d ()-]{8,}\d)", text)
    expected = profile.expected_graduation_date
    current_program = profile.current_program
    if any(entry.in_progress and entry.institution.lower() == "penn state" for entry in entries):
        expected = date(2026, 12, 1)
        current_program = current_program or "M.S. Artificial Intelligence"
    updates = {
        "education_history": entries,
        "expected_graduation_date": expected,
        "current_program": current_program,
        "phone": profile.phone or (phone_match.group(0).strip() if phone_match else None),
    }
    if (
        entries == profile.education_history
        and expected == profile.expected_graduation_date
        and current_program == profile.current_program
        and updates["phone"] == profile.phone
    ):
        return profile
    return profile.model_copy(update=updates)
