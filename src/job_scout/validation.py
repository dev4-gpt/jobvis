"""Deterministic fabrication validator (Phase 2's "verifiable checks" half).

``validate_pack`` checks every claim in a ``TailoringPack`` against the
candidate corpus with plain string similarity — no LLM, no cost, same result
every run. It flags what it cannot ground; it never retries or repairs. Flags
flow to state, trace metadata, and a visible UI warning.

Heuristics and their failure modes (documented, deliberately simple — tuning
them is the reader's exercise):

- Bullet ratio (``SCOUT_FAB_BULLET_RATIO``, default 0.65): a reworded bullet
  must reach this normalized ``difflib.SequenceMatcher`` ratio against its
  referenced corpus item. Numbers and units are canonicalized first
  (10M ~ 10 million, "/day" ~ "per day" ~ "daily"), so unit spelling alone
  cannot sink an honest rewrite; a fabricated detail inside an
  otherwise-copied bullet can still pass (the remaining false negative).
- Letter ratio (``SCOUT_FAB_LETTER_RATIO``, default 0.55): cover letters
  legitimately paraphrase, so the sentence-level threshold is softer, and only
  "factual-looking" sentences (a digit, a year, or a multi-word capitalized
  name) are checked at all. A sentence that misses on every single reference
  gets one more chance against two-reference combinations, because true
  sentences are often assembled from two corpus items. Motivation sentences
  are unverifiable and skipped.
- Skill ratio (``SCOUT_FAB_SKILL_RATIO``, default 0.85) applies when the CV
  yielded a skills section; a skill whose words are a subset of a corpus
  skill's words also passes ("AWS" is grounded by "basic AWS" — claiming less
  is honest). When segmentation found NO skills section, each skill is
  grounded in the full corpus text instead — flagging everything at 0.00
  against an empty vocabulary is noise, not signal.

All three are env knobs, and every ``FabricationReport`` records the values it
ran with — the report reaches state, trace metadata, and the UI, so a tuning
change is measurable run-to-run in Opik. (Phase 2's committed baselines were
measured at 0.75/0.9/0.55.)
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from job_scout.config import get_settings
from job_scout.corpus import CandidateCorpus
from job_scout.cover_letter_quality import policy_claim_violations
from job_scout.graph.schemas import FabricationReport, FlaggedClaim, TailoringPack

_MIN_FACTUAL_SENTENCE_WORDS = 6
_MIN_SKILL_TOKEN_CHARS = 3

_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")

# Deterministic canonicalization (Phase 3, weakness #8): "10M requests/day"
# and "10 million requests per day" should compare as the same claim. A tiny
# lookup table, not NLP — both sides of every comparison go through it, so it
# can only merge spellings, never invent meaning.
_NUM_SUFFIXES = {"k": "thousand", "m": "million", "bn": "billion", "b": "billion"}
_NUM_SUFFIX_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(bn|k|m|b)\b")
_PER_RE = re.compile(r"\bper\s+(\w+)\b")
_FREQ_WORDS = {"daily": "day", "weekly": "week", "monthly": "month", "yearly": "year", "annually": "year", "hourly": "hour"}
_NEAR_MISS_MARGIN = 0.10


def _claim_classification(ratio: float, threshold: float) -> str:
    """Classify a failed match for human review without changing the gate."""
    return "near_miss" if ratio >= max(0.0, threshold - _NEAR_MISS_MARGIN) else "unsupported"


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, canonicalize units.

    Canonicalization: numeric suffixes expand (10m -> 10 million), "per X"
    collapses to "X" (punctuation already turned "/day" into "day"), and
    frequency adverbs map to their nouns (daily -> day).
    """
    text = _WS.sub(" ", _PUNCT.sub(" ", text.lower()))
    text = _NUM_SUFFIX_RE.sub(lambda m: f"{m.group(1)} {_NUM_SUFFIXES[m.group(2)]}", text)
    text = _PER_RE.sub(r"\1", text)
    return " ".join(_FREQ_WORDS.get(word, word) for word in text.split()).strip()


def _ratio(a: str, b: str) -> float:
    """Similarity of two strings after normalization (0..1)."""
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _best_ratio(text: str, references: list[str]) -> float:
    """Best similarity of ``text`` against any reference string."""
    return max((_ratio(text, ref) for ref in references), default=0.0)


def _best_pair_ratio(text: str, references: list[str]) -> float:
    """Best similarity against a two-reference combination (Phase 3, #9).

    A true sentence assembled from two corpus items matches neither alone.
    Concatenating the top three single-match references pairwise (both orders)
    keeps this deterministic and cheap while catching the compositional case.
    """
    top = sorted(references, key=lambda ref: _ratio(text, ref), reverse=True)[:3]
    return max(
        (_ratio(text, f"{a} {b}") for a in top for b in top if a is not b),
        default=0.0,
    )


def _best_token_recall(text: str, references: list[str]) -> float:
    """Measure grounded vocabulary across several source items.

    A cover-letter sentence commonly combines two or three adjacent project
    bullets. SequenceMatcher penalizes that legitimate composition because no
    single bullet contains the full sentence. Select the references with the
    strongest token overlap, then require most meaningful sentence terms to be
    present in that small evidence set. This stays local to the best matches;
    it does not compare against the whole CV as one unrestricted bag of words.
    """
    sentence_tokens = set(_normalize(text).split())
    if len(sentence_tokens) < _MIN_FACTUAL_SENTENCE_WORDS:
        return 0.0
    scored = []
    for reference in references:
        reference_tokens = set(_normalize(reference).split())
        scored.append((len(sentence_tokens & reference_tokens), reference_tokens))
    evidence_tokens: set[str] = set()
    for _overlap, tokens in sorted(scored, key=lambda item: item[0], reverse=True)[:4]:
        evidence_tokens.update(tokens)
    return len(sentence_tokens & evidence_tokens) / len(sentence_tokens)


def _looks_factual(sentence: str) -> bool:
    """Whether a cover-letter sentence makes a checkable claim.

    A digit (metrics, years) or a capitalized multi-word token mid-sentence
    (product/company/tool names) marks a sentence as factual. Everything else
    (motivation, enthusiasm) is unverifiable by design and skipped.
    """
    if len(sentence.split()) < _MIN_FACTUAL_SENTENCE_WORDS:
        return False
    # Explicitly stated gaps are review information, not claims that require
    # evidence. Positive follow-up claims in the same paragraph remain checked.
    if re.search(
        r"\b(?:gap|gaps|do not have|don't have|lack(?:s|ing)?|no prior|not documented|"
        r"not yet demonstrated|would need to confirm|would confirm)\b",
        sentence,
        re.I,
    ):
        return False
    if re.search(r"\d", sentence):
        return True
    return bool(re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+", sentence[1:]))


def _split_sentences(text: str) -> list[str]:
    """Naive sentence split on newlines and ., ! and ? boundaries.

    Newlines split too because greetings ("Dear team,") end without sentence
    punctuation and would otherwise swallow the following claim.
    """
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def validate_pack(
    pack: TailoringPack,
    corpus: CandidateCorpus,
    research_notes: str | None = None,
    job_context: list[str] | None = None,
    candidate_preferences: object | None = None,
) -> FabricationReport:
    """Check every claim in the pack against the corpus. Deterministic, no LLM.

    ``job_context`` (e.g. job title and company) is added to the cover-letter
    reference texts so "I am applying for X at Y" is not flagged.
    """
    settings = get_settings()
    bullet_ratio = settings.fab_bullet_ratio
    skill_ratio = settings.fab_skill_ratio
    letter_ratio = settings.fab_letter_ratio
    flagged: list[FlaggedClaim] = []
    claims_checked = 0
    confirmed_claims = 0
    near_miss_claims = 0
    unsupported_claims = 0

    preferences = candidate_preferences
    if isinstance(preferences, dict):
        authorization_status = str(preferences.get("authorization_status", "unknown"))
        sponsorship_policy = str(preferences.get("sponsorship_policy", "unknown"))
        clearance_status = str(preferences.get("clearance_status", "unknown"))
    else:
        authorization_status = str(getattr(preferences, "authorization_status", "unknown"))
        sponsorship_policy = str(getattr(preferences, "sponsorship_policy", "unknown"))
        clearance_status = str(getattr(preferences, "clearance_status", "unknown"))

    # 1. CV bullets: the corpus_ref must resolve and the rewrite must stay close.
    for entry in pack.cv.experience:
        for bullet in entry.bullets:
            claims_checked += 1
            item = corpus.get(bullet.corpus_ref)
            if item is None:
                unsupported_claims += 1
                flagged.append(
                    FlaggedClaim(
                        where=f"cv_bullet:{bullet.corpus_ref}",
                        text=bullet.text,
                        reason="corpus_ref does not resolve to any corpus item",
                        classification="unsupported",
                    )
                )
                continue
            ratio = _ratio(bullet.text, item.text)
            if ratio < bullet_ratio:
                classification = _claim_classification(ratio, bullet_ratio)
                if classification == "near_miss":
                    near_miss_claims += 1
                else:
                    unsupported_claims += 1
                flagged.append(
                    FlaggedClaim(
                        where=f"cv_bullet:{bullet.corpus_ref}",
                        text=bullet.text,
                        reason=f"rewrite drifted too far from its corpus item (ratio {ratio:.2f} < {bullet_ratio})",
                        best_match_ratio=round(ratio, 3),
                        classification=classification,
                    )
                )
            else:
                confirmed_claims += 1

    # 2. Skills: must come from the corpus skill vocabulary. When segmentation
    #    parsed no skills section at all, fall back to grounding each skill's
    #    words in the full corpus text — still deterministic, and the reason
    #    names the real gap instead of flagging everything at 0.00.
    corpus_skills = corpus.skills()
    full_corpus_text = _normalize(" ".join(item.text for item in corpus.items))
    for skill in pack.cv.skills:
        claims_checked += 1
        if corpus_skills:
            best = _best_ratio(skill, corpus_skills)
            # Containment (Phase 3, #8): claiming LESS than the corpus states
            # is honest — "AWS" is grounded by "basic AWS". The reverse (adding
            # qualifiers the corpus never made) still has to pass the ratio.
            claimed = set(_normalize(skill).split())
            contained = bool(claimed) and any(claimed <= set(_normalize(cs).split()) for cs in corpus_skills)
            if best < skill_ratio and not contained:
                classification = _claim_classification(best, skill_ratio)
                if classification == "near_miss":
                    near_miss_claims += 1
                else:
                    unsupported_claims += 1
                flagged.append(
                    FlaggedClaim(
                        where=f"skill:{skill}",
                        text=skill,
                        reason=f"skill is not in the candidate's corpus (best match {best:.2f})",
                        best_match_ratio=round(best, 3),
                        classification=classification,
                    )
                )
            else:
                confirmed_claims += 1
        else:
            tokens = [t for t in _normalize(skill).split() if len(t) >= _MIN_SKILL_TOKEN_CHARS]
            missing = [t for t in tokens if t not in full_corpus_text]
            if missing:
                found = 1 - len(missing) / len(tokens) if tokens else 0.0
                classification = _claim_classification(found, skill_ratio)
                if classification == "near_miss":
                    near_miss_claims += 1
                else:
                    unsupported_claims += 1
                flagged.append(
                    FlaggedClaim(
                        where=f"skill:{skill}",
                        text=skill,
                        reason=(
                            "no skills section was parsed from the CV, and these words appear "
                            f"nowhere in its text: {', '.join(missing)}"
                        ),
                        best_match_ratio=round(found, 3),
                        classification=classification,
                    )
                )
            else:
                confirmed_claims += 1

    # 3. Cover letter: factual sentences must trace to the corpus, the research
    #    notes, or the job context.
    references = [item.text for item in corpus.items] + list(job_context or [])
    if research_notes:
        references.extend(_split_sentences(research_notes))
    sentences = _split_sentences(pack.cover_letter)
    for n, sentence in enumerate(sentences, start=1):
        if n == 1 or n == len(sentences):  # greeting / sign-off lines
            continue
        if not _looks_factual(sentence):
            continue
        claims_checked += 1
        best = _best_ratio(sentence, references)
        if best < letter_ratio:
            best = max(best, _best_pair_ratio(sentence, references), _best_token_recall(sentence, references))
        if best < letter_ratio:
            classification = _claim_classification(best, letter_ratio)
            if classification == "near_miss":
                near_miss_claims += 1
            else:
                unsupported_claims += 1
            flagged.append(
                FlaggedClaim(
                    where=f"cover_letter:sentence:{n}",
                    text=sentence,
                    reason=f"factual claim not traceable to the corpus or research (best match {best:.2f})",
                    best_match_ratio=round(best, 3),
                    classification=classification,
                )
            )
        else:
            confirmed_claims += 1

    # 4. Human-policy claims: authorization, visa, sponsorship, and clearance
    #    are never facts that may be inferred from a CV. Check the whole pack,
    #    including the summary, because a model can put the claim outside the
    #    cover letter where the sentence-level evidence gate would not see it.
    policy_text = "\n".join(
        [
            pack.cv.headline,
            pack.cv.summary,
            *(bullet.text for entry in (*pack.cv.experience, *pack.cv.projects) for bullet in entry.bullets),
            *pack.cv.skills,
            *pack.cv.education,
            pack.cover_letter,
        ]
    )
    policy_violations = policy_claim_violations(
        policy_text,
        authorization_status=authorization_status,
        sponsorship_policy=sponsorship_policy,
        clearance_status=clearance_status,
    )
    for sentence in policy_violations:
        claims_checked += 1
        unsupported_claims += 1
        flagged.append(
            FlaggedClaim(
                where="policy:unconfirmed_status",
                text=sentence,
                reason="authorization, visa, sponsorship, or clearance status is not confirmed by candidate policy",
                classification="unsupported",
            )
        )

    return FabricationReport(
        flags=len(flagged),
        claims_checked=claims_checked,
        flagged=flagged,
        confirmed_claims=confirmed_claims,
        near_miss_claims=near_miss_claims,
        unsupported_claims=unsupported_claims,
        policy_violations=policy_violations,
        thresholds={"bullet": bullet_ratio, "skill": skill_ratio, "letter": letter_ratio},
    )
