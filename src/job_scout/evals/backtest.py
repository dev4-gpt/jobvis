"""Offline backtesting and bounded improvement for application packs.

This module is deliberately deterministic. It does not call a model, browse a
job board, or decide that a weak draft is acceptable because it sounds good.
An optional repair callback may propose a new pack, but every proposal is
measured against the same backtest and a finite attempt budget.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from job_scout.corpus import build_corpus
from job_scout.cover_letter_quality import evaluate_cover_letter
from job_scout.graph.schemas import (
    CandidatePreferences,
    CoverLetterQualityReport,
    CVLink,
    FabricationReport,
    TailoringPack,
)
from job_scout.validation import validate_pack


@dataclass(frozen=True)
class BacktestThresholds:
    """The minimum quality contract for a pack that may be presented as ready."""

    min_letter_words: int = 250
    max_letter_words: int = 350
    min_evidence_matches: int = 2
    min_requirement_matches: int = 2
    max_fabrication_rate: float = 0.0
    min_cv_words: int = 600
    min_cv_bullets: int = 10
    min_cv_experience_entries: int = 2
    min_cv_project_entries: int = 2
    min_cv_skills: int = 12


@dataclass(frozen=True)
class BacktestMetric:
    """One human-readable, machine-checkable backtest dimension."""

    name: str
    value: float
    passed: bool
    detail: str


@dataclass(frozen=True)
class BacktestReport:
    """A complete deterministic decision for one application pack."""

    passed: bool
    score: float
    metrics: tuple[BacktestMetric, ...]
    failures: tuple[str, ...]
    cover_letter_quality: CoverLetterQualityReport
    fabrication_report: FabricationReport
    missing_links: tuple[str, ...] = ()
    failure_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImprovementResult:
    """The best pack found by a finite, non-regressing repair loop."""

    pack: TailoringPack
    report: BacktestReport
    attempts: int
    history: tuple[BacktestReport, ...]


RepairPack = Callable[[TailoringPack, BacktestReport, int], TailoringPack | None]


def _preference_value(preferences: CandidatePreferences | dict | None, name: str, default: str = "unknown") -> str:
    """Read a policy value from either the typed model or a fixture mapping."""
    if preferences is None:
        return default
    if isinstance(preferences, dict):
        return str(preferences.get(name, default))
    return str(getattr(preferences, name, default))


def _cv_word_count(pack: TailoringPack) -> int:
    parts = [pack.cv.headline, pack.cv.summary, *pack.cv.skills, *pack.cv.education]
    for entry in (*pack.cv.experience, *pack.cv.projects):
        parts.extend((entry.role, entry.company, entry.dates))
        parts.extend(bullet.text for bullet in entry.bullets)
    return len(" ".join(parts).split())


def _cv_bullet_count(pack: TailoringPack) -> int:
    return sum(len(entry.bullets) for entry in (*pack.cv.experience, *pack.cv.projects))


def _score(metrics: Sequence[BacktestMetric]) -> float:
    """Weighted score; hard safety dimensions receive the largest weights."""
    weights = {
        "grounding": 0.30,
        "policy_safety": 0.25,
        "cover_letter": 0.20,
        "source_links": 0.15,
        "cv_density": 0.10,
    }
    return round(sum(weights.get(metric.name, 0.0) * min(1.0, max(0.0, metric.value)) for metric in metrics), 4)


def backtest_pack(
    pack: TailoringPack,
    cv_text: str,
    job_description: str,
    *,
    source_links: Sequence[CVLink] = (),
    candidate_preferences: CandidatePreferences | dict | None = None,
    research_notes: str | None = None,
    job_context: Sequence[str] = (),
    thresholds: BacktestThresholds | None = None,
) -> BacktestReport:
    """Evaluate a pack without an LLM.

    ``source_links`` is optional for generic fixture cases. When supplied, all
    source URLs must survive in the generated CV; a missing link is a hard
    failure. Job-description language is allowed as context for the letter,
    but cannot make candidate evidence pass the grounding validator.
    """
    limits = thresholds or BacktestThresholds()
    corpus = build_corpus(cv_text)
    quality = evaluate_cover_letter(
        pack.cover_letter,
        job_description,
        "\n".join(item.text for item in corpus.items),
        authorization_status=_preference_value(candidate_preferences, "authorization_status"),
        sponsorship_policy=_preference_value(candidate_preferences, "sponsorship_policy"),
        clearance_status=_preference_value(candidate_preferences, "clearance_status"),
    )
    fabrication = validate_pack(
        pack,
        corpus,
        research_notes=research_notes,
        job_context=list(job_context),
        candidate_preferences=candidate_preferences,
    )

    source_urls = {link.url for link in source_links if link.url}
    generated_urls = {link.url for link in pack.cv.links if link.url}
    missing_links = tuple(sorted(source_urls - generated_urls))
    fabrication_rate = fabrication.flags / fabrication.claims_checked if fabrication.claims_checked else 0.0
    cv_words = _cv_word_count(pack)
    cv_bullets = _cv_bullet_count(pack)
    density_values = (
        cv_words >= limits.min_cv_words,
        cv_bullets >= limits.min_cv_bullets,
        len(pack.cv.experience) >= limits.min_cv_experience_entries,
        len(pack.cv.projects) >= limits.min_cv_project_entries,
        len(pack.cv.skills) >= limits.min_cv_skills,
    )
    density = sum(density_values) / len(density_values)
    policy_ok = not fabrication.policy_violations and not quality.policy_violations
    letter_ok = (
        quality.passed
        and limits.min_letter_words <= quality.word_count <= limits.max_letter_words
        and quality.evidence_matches >= limits.min_evidence_matches
        and quality.requirement_matches >= limits.min_requirement_matches
    )
    grounding_ok = fabrication_rate <= limits.max_fabrication_rate
    links_ok = not missing_links

    metrics = (
        BacktestMetric(
            "grounding",
            1.0 - fabrication_rate,
            grounding_ok,
            f"{fabrication.flags}/{fabrication.claims_checked} claims flagged",
        ),
        BacktestMetric(
            "policy_safety",
            1.0 if policy_ok else 0.0,
            policy_ok,
            f"{len(fabrication.policy_violations) + len(quality.policy_violations)} policy violations",
        ),
        BacktestMetric(
            "cover_letter",
            float(
                sum(
                    (
                        quality.word_count >= limits.min_letter_words,
                        quality.word_count <= limits.max_letter_words,
                        quality.evidence_matches >= limits.min_evidence_matches,
                        quality.requirement_matches >= limits.min_requirement_matches,
                        quality.passed,
                    )
                )
                / 5
            ),
            letter_ok,
            f"{quality.word_count} words, {quality.evidence_matches} evidence matches, "
            f"{quality.requirement_matches} requirement matches",
        ),
        BacktestMetric(
            "source_links",
            1.0 if links_ok else max(0.0, 1.0 - len(missing_links) / max(1, len(source_urls))),
            links_ok,
            f"{len(generated_urls & source_urls)}/{len(source_urls)} source links preserved",
        ),
        BacktestMetric(
            "cv_density",
            density,
            all(density_values),
            f"{cv_words} words, {cv_bullets} bullets, {len(pack.cv.experience)} experience, "
            f"{len(pack.cv.projects)} projects, {len(pack.cv.skills)} skills",
        ),
    )
    failures = [failure for failure in quality.reasons]
    failure_codes: list[str] = []
    if quality.word_count < limits.min_letter_words or quality.word_count > limits.max_letter_words:
        failure_codes.append("cover_letter_length")
    if quality.evidence_matches < limits.min_evidence_matches:
        failure_codes.append("cover_letter_evidence")
    if quality.requirement_matches < limits.min_requirement_matches:
        failure_codes.append("cover_letter_requirements")
    if not quality.passed:
        failure_codes.append("cover_letter_quality")
    if not grounding_ok:
        failure_codes.append("fabrication")
        failures.append(f"fabrication rate {fabrication_rate:.3f} exceeds {limits.max_fabrication_rate:.3f}")
        failures.extend(flag.reason for flag in fabrication.flagged[:3])
    if not policy_ok:
        failure_codes.append("policy_safety")
        failures.append("pack contains unconfirmed authorization, sponsorship, visa, or clearance claims")
    if missing_links:
        failure_codes.append("source_links")
        failures.append(f"missing source links: {', '.join(missing_links)}")
    if not all(density_values):
        failure_codes.append("cv_density")
        failures.append("CV density contract is not met")
    return BacktestReport(
        passed=all(metric.passed for metric in metrics),
        score=_score(metrics),
        metrics=metrics,
        failures=tuple(dict.fromkeys(failures)),
        cover_letter_quality=quality,
        fabrication_report=fabrication,
        missing_links=missing_links,
        failure_codes=tuple(dict.fromkeys(failure_codes)),
    )


def improve_pack(
    initial: TailoringPack,
    evaluate: Callable[[TailoringPack], BacktestReport],
    repair: RepairPack | None = None,
    *,
    max_attempts: int = 2,
) -> ImprovementResult:
    """Try bounded repairs and retain only strictly better candidates.

    This is intentionally not an autonomous infinite loop. It stops as soon
    as the pack passes, when no repair is available, when a repair is not an
    improvement, or after ``max_attempts``. A provider cannot trade away
    grounding or policy safety for a nicer-looking letter because the score
    is computed from those hard dimensions first.
    """
    current = initial
    report = evaluate(current)
    history = [report]
    attempts = 0
    for attempt in range(1, max(0, max_attempts) + 1):
        if report.passed or repair is None:
            break
        candidate = repair(current, report, attempt)
        attempts += 1
        if candidate is None:
            break
        candidate_report = evaluate(candidate)
        history.append(candidate_report)
        if candidate_report.score <= report.score:
            break
        current, report = candidate, candidate_report
    return ImprovementResult(current, report, attempts, tuple(history))
