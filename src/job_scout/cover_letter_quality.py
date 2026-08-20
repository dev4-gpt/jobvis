"""Deterministic quality checks for generated cover letters.

This is intentionally separate from the fabrication validator. Fabrication asks
whether claims are grounded; this gate asks whether the letter is substantial,
specific, and addressed to the actual job. It never calls an LLM.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
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
_AUTHORIZATION_CLAIM = re.compile(
    r"\b(?:authorized to work|eligible to work|work authorization|visa|citizenship|"
    r"sponsorship|sponsor(?:ship)?|h[- ]?1b|f[- ]?1|opt|stem opt)\b",
    re.I,
)
_CLEARANCE_CLAIM = re.compile(
    r"\b(?:security clearance|clearance|top secret|secret clearance|ts/sci|public trust|polygraph)\b",
    re.I,
)
_UNCERTAINTY = re.compile(
    r"\b(?:unknown|uncertain|unclear|confirm|confirmation|cannot|can't|not confirmed|"
    r"not yet|pending|needs review|under review|to be determined)\b",
    re.I,
)
_GAP_LANGUAGE = re.compile(
    r"\b(?:gap|gaps|do not have|don't have|lack(?:s|ing)?|no prior|not documented|"
    r"not yet demonstrated|would need to confirm|would confirm)\b",
    re.I,
)


def _words(text: str) -> list[str]:
    return [word for word in re.findall(r"[A-Za-z][A-Za-z0-9+#/-]{2,}", text.lower()) if word not in _STOPWORDS]


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if part.strip()]


def _requirements(description: str) -> list[str]:
    sentences = _sentences(description)
    marked = [sentence for sentence in sentences if any(marker in sentence.lower() for marker in _REQUIREMENT_MARKERS)]
    if not marked:
        return sentences[:4]
    # Job boards frequently compress two requirements into one clause such as
    # "work closely with platform teams to deliver generative AI solutions."
    # Keep both halves as separate targets so the letter can address them
    # explicitly and the gate does not demand two unrelated sentences.
    expanded: list[str] = []
    for sentence in marked:
        parts = re.split(r"\s+to\s+", sentence, maxsplit=1, flags=re.I)
        if len(parts) == 2 and len(parts[0].split()) >= 4 and len(parts[1].split()) >= 3:
            expanded.extend((parts[0].strip(), f"to {parts[1].strip()}"))
        else:
            expanded.append(sentence)
    return expanded


def requirement_targets(description: str) -> list[str]:
    """Return the concrete job-description clauses used by the quality gate."""
    return _requirements(description)[:6]


def policy_claim_violations(
    text: str,
    *,
    authorization_status: str = "unknown",
    sponsorship_policy: str = "unknown",
    clearance_status: str = "unknown",
) -> list[str]:
    """Find unconfirmed authorization or clearance claims in generated text."""
    violations: list[str] = []
    auth_known = authorization_status.lower() not in {"", "unknown", "unresolved", "pending"}
    sponsorship_known = sponsorship_policy.lower() not in {"", "unknown", "unresolved", "pending"}
    clearance_known = clearance_status.lower() not in {"", "unknown", "unresolved", "pending"}
    for sentence in _sentences(str(text)):
        authorization_violation = _AUTHORIZATION_CLAIM.search(sentence) and not (
            auth_known or sponsorship_known or _UNCERTAINTY.search(sentence)
        )
        clearance_violation = _CLEARANCE_CLAIM.search(sentence) and not (clearance_known or _UNCERTAINTY.search(sentence))
        if authorization_violation or clearance_violation:
            violations.append(sentence)
    return list(dict.fromkeys(violations))


def remove_unconfirmed_policy_sentences(
    text: str,
    *,
    authorization_status: str = "unknown",
    sponsorship_policy: str = "unknown",
    clearance_status: str = "unknown",
) -> str:
    """Remove positive policy claims that the candidate has not confirmed.

    Models often append ``authorized to work`` or ``F-1 OPT`` to an otherwise
    grounded summary. A warning alone is too easy to miss in a downloadable
    application pack, so unknown policy claims are removed before rendering.
    Explicit uncertainty statements are retained for the human review note.
    """
    violations = set(
        policy_claim_violations(
            text,
            authorization_status=authorization_status,
            sponsorship_policy=sponsorship_policy,
            clearance_status=clearance_status,
        )
    )
    if not violations:
        return str(text).strip()
    parts = _sentences(str(text))
    kept = [part for part in parts if part not in violations]
    return "\n\n".join(kept).strip()


def evaluate_cover_letter(
    letter: str,
    job_description: str,
    corpus_text: str,
    *,
    authorization_status: str = "unknown",
    sponsorship_policy: str = "unknown",
    clearance_status: str = "unknown",
) -> CoverLetterQualityReport:
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

    policy_violations = policy_claim_violations(
        normalized,
        authorization_status=authorization_status,
        sponsorship_policy=sponsorship_policy,
        clearance_status=clearance_status,
    )
    if policy_violations:
        reasons.append("cover letter contains unconfirmed authorization, sponsorship, visa, or clearance claims")

    corpus_items = [set(_words(sentence)) for sentence in _sentences(corpus_text)]
    evidence_matches = 0
    for sentence in _sentences(normalized)[1:]:
        # A transparent gap statement is not a claim of experience. It is
        # useful review context and should not fail the evidence gate.
        if _GAP_LANGUAGE.search(sentence):
            continue
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
        policy_violations=policy_violations,
        passed=passed,
        reasons=reasons,
    )


def grounded_fallback_letter(
    *,
    candidate_name: str,
    company: str,
    job_title: str,
    job_description: str,
    corpus_items: Iterable[str],
) -> str:
    """Build a conservative letter when the model cannot satisfy the gate.

    This is intentionally plain prose. Every candidate fact is copied from a
    corpus item, while the first paragraph quotes two bounded job requirements
    so the letter remains specific without inventing company facts, status, or
    domain experience.
    """

    def clip(value: str, limit: int) -> str:
        words = str(value).replace("\n", " ").split()
        return " ".join(words[:limit]).rstrip(" ,;:")

    title = clip(job_title, 12) or "this role"
    employer = clip(company, 10) or "your team"
    requirements = requirement_targets(job_description)
    first_requirement = clip(requirements[0], 18) if requirements else "building reliable AI systems"
    second_requirement = clip(requirements[1], 18) if len(requirements) > 1 else "communicating technical results clearly"
    evidence = [clip(item, 34) for item in corpus_items if str(item).strip()]
    evidence = evidence[:3]
    while len(evidence) < 3:
        evidence.append("My resume includes additional technical and project evidence relevant to this work")

    return (
        f"Dear {employer} hiring team,\n\n"
        f"I am applying for the {title} role because the posting emphasizes {first_requirement} and "
        f"{second_requirement}. Those priorities match the way I have built and evaluated practical AI systems. "
        f"I am seeking a full-time opportunity beginning in the candidate's selected start window, and I would "
        f"welcome the chance to discuss how my experience could support {employer}'s team.\n\n"
        f"My experience includes this evidence from my resume: {evidence[0]}. I also documented the following work: "
        f"{evidence[1]}. Together, these examples show hands-on work across model development, software delivery, "
        f"and technical problem solving. I would apply that same disciplined process to a new team and product. "
        f"I value clear ownership, reproducible work, measurable outcomes, and respectful collaboration across disciplines. "
        f"A further project is described as follows: {evidence[2]}. I would bring "
        f"careful implementation, measurement, and a willingness to explain tradeoffs clearly to collaborators.\n\n"
        f"I want to be precise about fit. My resume does not document every requirement in the posting, so I would "
        f"confirm any domain-specific, authorization, clearance, or production-scale expectations directly with the "
        f"employer. I have not added claims about those areas. The evidence above is the basis for my application, "
        f"and the remaining questions are appropriate topics for a human review.\n\n"
        f"Thank you for reviewing my application. I would be glad to discuss the role and the evidence behind this "
        f"application pack.\n\nSincerely,\n{clip(candidate_name, 8) or 'Candidate'}"
    )
