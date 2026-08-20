"""Deterministic quality checks for generated cover letters.

This is intentionally separate from the fabrication validator. Fabrication asks
whether claims are grounded; this gate asks whether the letter is substantial,
specific, and addressed to the actual job. It never calls an LLM.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from job_scout.graph.schemas import CoverLetterQualityReport

_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "from",
    "have",
    "into",
    "that",
    "the",
    "their",
    "this",
    "with",
    "your",
    "will",
    "you",
    "for",
    "our",
    "they",
    "who",
    "what",
    "work",
    "role",
    "team",
    "job",
    "must",
}
_GENERIC = (
    "i am excited to apply",
    "passionate professional",
    "dynamic team",
    "fast-paced environment",
    "i believe i would be a great fit",
    "thank you for considering my application",
)
_REQUIREMENT_MARKERS = (
    "required",
    "must",
    "experience",
    "proficiency",
    "knowledge",
    "ability",
    "skills",
    "familiarity",
    "looking for",
)


def _words(text: str) -> list[str]:
    return [word for word in re.findall(r"[A-Za-z][A-Za-z0-9+#/-]{2,}", text.lower()) if word not in _STOPWORDS]


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if part.strip()]


def _requirements(description: str) -> list[str]:
    sentences = _sentences(description)
    marked = [sentence for sentence in sentences if any(marker in sentence.lower() for marker in _REQUIREMENT_MARKERS)]
    return marked or sentences[:4]


def requirement_targets(description: str) -> list[str]:
    """Return the concrete job-description clauses used by the quality gate."""
    return _requirements(description)[:6]


def evaluate_cover_letter(letter: str, job_description: str, corpus_text: str) -> CoverLetterQualityReport:
    """Return a reproducible quality report for one draft."""
    from job_scout.graph.schemas import CoverLetterQualityReport

    normalized = re.sub(r"<[^>]+>", " ", str(letter)).strip()
    word_count = len(re.findall(r"\b[\w][\w'-]*\b", normalized))
    reasons: list[str] = []
    generic_phrases = [phrase for phrase in _GENERIC if phrase in normalized.lower()]
    if word_count < 250:
        reasons.append(f"cover letter is too short ({word_count} words; minimum is 250)")
    if word_count > 350:
        reasons.append(f"cover letter is too long ({word_count} words; maximum is 350)")
    if not normalized:
        reasons.append("cover letter is empty")
    if generic_phrases:
        reasons.append("cover letter contains generic or placeholder language")

    corpus_items = [set(_words(sentence)) for sentence in _sentences(corpus_text)]
    evidence_matches = 0
    for sentence in _sentences(normalized)[1:]:
        tokens = set(_words(sentence))
        if len(tokens) >= 3 and any(len(tokens & item) >= 2 for item in corpus_items):
            evidence_matches += 1
    evidence_matches = min(evidence_matches, 3)
    if evidence_matches < 2:
        reasons.append("cover letter does not contain two concrete, resume-grounded evidence points")

    requirements = requirement_targets(job_description)
    requirement_matches = 0
    letter_tokens = set(_words(normalized))
    for requirement in requirements:
        tokens = set(_words(requirement))
        if tokens and len(tokens & letter_tokens) >= min(3, max(2, len(tokens) // 4)):
            requirement_matches += 1
    requirement_matches = min(requirement_matches, 3)
    if requirement_matches < 2:
        reasons.append("cover letter does not address two specific requirements from the job description")

    passed = not reasons
    return CoverLetterQualityReport(
        word_count=word_count,
        evidence_matches=evidence_matches,
        requirement_matches=requirement_matches,
        requirement_targets=requirements,
        generic_phrases=generic_phrases,
        passed=passed,
        reasons=reasons,
    )
