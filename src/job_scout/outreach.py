"""Local, review-gated contact research and candidate outreach drafts.

This module only extracts contact details that the user supplies from a public
page or explicitly approves. It never guesses an email pattern, scrapes
LinkedIn, sends mail, or sends a video. Every draft is a proposal for manual
review and personalization.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from job_scout.cover_letter_quality import requirement_targets
from job_scout.graph.schemas import Profile, RankedJob, TailoringPack

_EMAIL = re.compile(r"(?<![\w.+-])([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})(?![\w.-])", re.I)
_CONTACT_TITLE = re.compile(
    r"\b(?:recruiter|recruiting|talent|people|hiring|engineering manager|director|head of|chief|founder|cto|vp)\b",
    re.I,
)
_NAME = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")


class PublicContactSource(BaseModel):
    """One user-selected public source and its approved extracted text."""

    url: str
    text: str = ""
    source_type: Literal["company_page", "careers_page", "employer_listing", "user_note"] = "user_note"


class ContactCandidate(BaseModel):
    """A public contact candidate; email values are never inferred."""

    name: str = ""
    title: str = ""
    email: str = ""
    source_url: str
    evidence: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    requires_manual_verification: bool = True


class OutreachDraft(BaseModel):
    """A manual email/video outreach draft for one selected job."""

    draft_id: str = Field(default_factory=lambda: str(uuid4()))
    job_id: str
    company: str
    role: str
    contact: ContactCandidate | None = None
    subject: str
    email_body: str
    video_script: str
    why_me: list[str] = Field(default_factory=list)
    requirement_targets: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    review_notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def _clean_email(value: str) -> str:
    return value.strip().rstrip(".,;:)]}").lower()


def discover_public_contacts(sources: list[PublicContactSource]) -> list[ContactCandidate]:
    """Extract explicit public emails from user-approved page text.

    A title/name is only attached when it appears in nearby text. The function
    intentionally returns no result for a guessed address such as
    ``firstname@company.com``.
    """
    candidates: list[ContactCandidate] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        if not source.url.startswith(("https://", "http://")):
            continue
        text = re.sub(r"\s+", " ", source.text).strip()
        for match in _EMAIL.finditer(text):
            email = _clean_email(match.group(1))
            start = max(0, match.start() - 180)
            end = min(len(text), match.end() + 180)
            context = text[start:end]
            title_match = _CONTACT_TITLE.search(context)
            title = title_match.group(0) if title_match else ""
            names = [name for name in _NAME.findall(context) if name.lower() not in {"contact us", "email us"}]
            name = names[-1] if names else ""
            key = (email, source.url)
            if key in seen:
                continue
            seen.add(key)
            confidence = 0.85 if title else 0.65
            candidates.append(
                ContactCandidate(
                    name=name,
                    title=title,
                    email=email,
                    source_url=source.url,
                    evidence=context,
                    confidence=confidence,
                )
            )
    return sorted(candidates, key=lambda item: (-item.confidence, item.email))


def _clip(text: str, limit: int = 230) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rsplit(" ", 1)[0] + "…"


def _evidence(pack: TailoringPack, limit: int = 3) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for entry in (*pack.cv.experience, *pack.cv.projects):
        for bullet in entry.bullets:
            if bullet.text.strip():
                values.append((bullet.corpus_ref, _clip(bullet.text)))
    return values[:limit]


def build_outreach_draft(
    job: RankedJob, profile: Profile, pack: TailoringPack, contact: ContactCandidate | None = None
) -> OutreachDraft:
    """Create a grounded email and short video script without an LLM."""
    targets = requirement_targets(job.job.description)[:2]
    target_phrases = [_clip(target, 145) for target in targets]
    evidence = _evidence(pack)
    candidate_name = profile.name or "Aryaman"
    target_text = " and ".join(target_phrases) if target_phrases else "the applied machine-learning work described in the posting"
    evidence_lines = [f"- {text}" for _, text in evidence]
    why_me = [text for _, text in evidence]
    refs = [ref for ref, _ in evidence]
    greeting = f"Hi {contact.name}," if contact and contact.name else "Hello,"
    subject = f"Applied ML Engineer — {candidate_name} — a concise introduction"
    email_body = (
        f"{greeting}\n\n"
        f"I’m reaching out about the {job.job.title} role at {job.job.company}. "
        f"The posting emphasizes {target_text}. My relevant evidence includes:\n"
        + "\n".join(evidence_lines)
        + f"\n\nI’m completing an M.S. in Artificial Intelligence and targeting full-time work after December 2026. "
        "I have not inferred work authorization, sponsorship, clearance, or any requirement not documented in my resume, "
        "so I would be glad to confirm those points directly. I recorded a short optional introduction to make the "
        "technical fit easier to evaluate.\n\n"
        f"Would a brief conversation be useful?\n\nBest,\n{candidate_name}"
    )
    video_script = (
        f"Hi, I’m {candidate_name}. I’m reaching out about the {job.job.title} role at {job.job.company}. "
        f"What caught my attention is the need to {target_text}. "
        f"My strongest evidence is {evidence[0][1] if evidence else 'the applied ML work in my attached resume'}. "
        f"I also {
            evidence[1][1].lower()
            if len(evidence) > 1
            else 'have additional experience building and evaluating applied ML systems'
        }. "
        "I’m finishing my M.S. in Artificial Intelligence and looking for a full-time role after December 2026. "
        "I would be excited to learn how your team measures success and discuss one concrete way I could contribute. "
        "Thanks for your time."
    )
    return OutreachDraft(
        job_id=job.job.job_id,
        company=job.job.company,
        role=job.job.title,
        contact=contact,
        subject=subject,
        email_body=email_body,
        video_script=video_script,
        why_me=why_me,
        requirement_targets=target_phrases,
        evidence_refs=refs,
        review_notes=[
            "Verify the contact and role on the employer’s public page before using this draft.",
            "Personalize the first sentence and record the video yourself; Jobvis does not send email or video.",
            "Do not infer authorization, sponsorship, clearance, or a hiring decision from this draft.",
            "Respect employer contact preferences and avoid bulk or repeated outreach.",
        ],
    )
