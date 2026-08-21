"""Bounded render, audit, repair loop for downloadable application packs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from job_scout.corpus import build_corpus
from job_scout.cover_letter_quality import grounded_fallback_letter
from job_scout.evals.artifacts import ArtifactManifest, ArtifactReport, audit_generated_pack
from job_scout.graph.schemas import CVLink, TailoringPack
from job_scout.renderer import RenderResult, render_cover_letter_pdf, render_pdf


@dataclass(frozen=True)
class VerifiedRenderedPack:
    """The latest render and every audit result from the bounded loop."""

    pack: TailoringPack
    cv: RenderResult
    cover_letter: RenderResult
    report: ArtifactReport
    attempts: int
    history: tuple[ArtifactReport, ...]
    manifest: ArtifactManifest


_RESEARCHOS_TITLE = re.compile(r"research assistantship\s+and\s+operating system(?:\s*\([^)]*\))?", re.IGNORECASE)
_MALFORMED_REQUIREMENT = re.compile(
    r"because it calls for\s+(?:about\s+)?we are looking for someone(?: who['’]s| who is)?\s+",
    re.IGNORECASE,
)
_MALFORMED_JOIN = re.compile(r"has contributed\s+and\s+to\s+AI research", re.IGNORECASE)


def _repair_cv_headers(pack: TailoringPack) -> TailoringPack:
    """Repair only the known source heading artifact; never rewrite bullets."""
    repaired = pack.model_copy(deep=True)
    for entry in (*repaired.cv.experience, *repaired.cv.projects):
        if _RESEARCHOS_TITLE.search(entry.role):
            entry.role = _RESEARCHOS_TITLE.sub("ResearchOS", entry.role)
    return repaired


def _repair_letter_surface(text: str) -> str:
    """Remove known copied-job-description joins without adding facts."""
    repaired = str(text)
    repaired = _MALFORMED_REQUIREMENT.sub("because the role seeks ", repaired)
    repaired = _MALFORMED_JOIN.sub("has contributed to AI research", repaired)
    repaired = re.sub(r"\b([A-Za-z][A-Za-z'-]{2,})\s+\1\b", r"\1", repaired, flags=re.IGNORECASE)
    return repaired.strip()


def _fallback_evidence(source_text: str, job_description: str) -> list[str]:
    """Select three distinct source-only evidence items for a safe fallback."""
    corpus = build_corpus(source_text)
    target = set(re.findall(r"[a-z0-9+#-]{3,}", job_description.lower()))
    items = [item for item in corpus.items if item.kind in {"bullet", "summary"}]

    def score(item) -> tuple[int, int]:
        tokens = set(re.findall(r"[a-z0-9+#-]{3,}", item.text.lower()))
        return (len(target & tokens), len(item.text))

    selected: list[str] = []
    sections: set[str] = set()
    for item in sorted(items, key=score, reverse=True):
        if item.section in sections:
            continue
        selected.append(item.text)
        sections.add(item.section)
        if len(selected) == 3:
            break
    return selected or [item.text for item in items[:3]]


def _repair_pack(
    pack: TailoringPack,
    report: ArtifactReport,
    *,
    source_text: str,
    job_description: str,
    candidate_name: str,
    company: str,
    job_title: str,
) -> TailoringPack | None:
    """Make one conservative repair based on deterministic issue codes."""
    repaired = _repair_cv_headers(pack)
    repaired.cover_letter = _repair_letter_surface(repaired.cover_letter)
    before = pack.model_dump(mode="json")

    quality_failed = any(issue.code == "cover_letter_quality" for issue in report.issues)
    # A rendered letter is a user-facing artifact. If its word count or
    # quality gate fails, always replace it with the bounded grounded fallback
    # rather than leaving a 366-word draft available only because it was
    # otherwise non-empty. The fallback is re-audited on the next attempt.
    if quality_failed:
        repaired.cover_letter = grounded_fallback_letter(
            candidate_name=candidate_name,
            company=company,
            job_title=job_title,
            job_description=job_description,
            corpus_items=_fallback_evidence(source_text, job_description),
        )

    if repaired.model_dump(mode="json") == before:
        return None
    return repaired


def render_verified_pack(
    pack: TailoringPack,
    *,
    candidate_name: str,
    source_text: str,
    source_links: list[CVLink],
    job_description: str,
    company: str = "your team",
    job_title: str = "this role",
    out_dir: Path,
    max_attempts: int = 3,
    selected_job_id: str = "",
    profile_version: str = "",
    generation_id: str = "",
    backtest_score: float | None = None,
) -> VerifiedRenderedPack:
    """Render and audit until pass or a finite repair budget is exhausted."""
    current = pack.model_copy(deep=True)
    history: list[ArtifactReport] = []
    last_cv = RenderResult(out_dir / "tailored_cv.tex")
    last_letter = RenderResult(out_dir / "cover_letter.tex")
    last_report: ArtifactReport | None = None

    for attempt in range(1, max(0, max_attempts) + 1):
        attempt_dir = out_dir / f"attempt-{attempt}"
        last_cv = render_pdf(current.cv, candidate_name, attempt_dir)
        last_letter = render_cover_letter_pdf(current.cover_letter, candidate_name, attempt_dir)
        last_report = audit_generated_pack(
            source_text,
            source_links,
            last_cv.pdf_path or "",
            cover_letter_pdf=last_letter.pdf_path,
            cover_letter_text=current.cover_letter,
            job_description=job_description,
            cv_tex=last_cv.tex_path,
            cover_letter_tex=last_letter.tex_path,
        )
        history.append(last_report)
        if last_report.passed:
            return VerifiedRenderedPack(
                current,
                last_cv,
                last_letter,
                last_report,
                attempt,
                tuple(history),
                _manifest(
                    last_report,
                    last_cv,
                    last_letter,
                    selected_job_id=selected_job_id,
                    profile_version=profile_version,
                    generation_id=generation_id,
                    backtest_score=backtest_score,
                ),
            )
        repaired = _repair_pack(
            current,
            last_report,
            source_text=source_text,
            job_description=job_description,
            candidate_name=candidate_name,
            company=company,
            job_title=job_title,
        )
        if repaired is None:
            break
        current = repaired

    assert last_report is not None
    return VerifiedRenderedPack(
        current,
        last_cv,
        last_letter,
        last_report,
        len(history),
        tuple(history),
        _manifest(
            last_report,
            last_cv,
            last_letter,
            selected_job_id=selected_job_id,
            profile_version=profile_version,
            generation_id=generation_id,
            backtest_score=backtest_score,
        ),
    )


def _manifest(
    report: ArtifactReport,
    cv: RenderResult,
    letter: RenderResult,
    *,
    selected_job_id: str,
    profile_version: str,
    generation_id: str,
    backtest_score: float | None,
) -> ArtifactManifest:
    """Convert the final audit into the single download-status contract."""
    return ArtifactManifest(
        status="ready" if report.passed else "withheld",
        cv_pdf=str(cv.pdf_path) if report.passed and cv.pdf_path else None,
        cv_tex=str(cv.tex_path) if cv.tex_path else None,
        cover_letter_pdf=str(letter.pdf_path) if report.passed and letter.pdf_path else None,
        cover_letter_tex=str(letter.tex_path) if letter.tex_path else None,
        cv_pages=report.cv_pages,
        cv_words=report.cv_words,
        cover_letter_words=report.cover_letter_words,
        source_links=report.source_links,
        generated_links=report.generated_links,
        annotation_count=report.annotation_count,
        issue_codes=tuple(issue.code for issue in report.issues),
        selected_job_id=selected_job_id,
        profile_version=profile_version,
        generation_id=generation_id,
        backtest_score=backtest_score,
    )
