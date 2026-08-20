# Candidate-fit architecture

Jobvis treats matching as two separate decisions:

1. The ranking model estimates role fit and evidence fit from the resume.
2. Deterministic policy code decides eligibility from the candidate's confirmed
   employment, graduation, location, work-mode, authorization, and clearance
   choices.

This prevents a high semantic score from promoting an internship, a clearance
job, or a posting with an incompatible start date. Unknown metadata is shown as
`borderline` and requires review; it is never silently treated as a match.

## Aryaman's default intent

The local candidate store starts with an editable policy: M.S. Artificial
Intelligence graduation in December 2026, full-time roles beginning December
2026 through March 2027, any US location, all three work modes, and AI/ML,
Data Science, GenAI, forward-deployed AI, Computer Vision, and MLOps/ML
Platform as primary families. Forward-deployed and platform work are deliberate
for Aryaman's Veloce AgenticOS and product-facing systems work; Computer Vision
is supported by the Bioqube and NUS evidence. BI, Java-heavy, generic analyst,
and broad software roles remain visible as adjacent roles. Authorization and
clearance stay unknown until the candidate answers them.

The profile extractor may identify education and evidence, but the UI-owned
`CandidatePreferences` object is authoritative. Job descriptions and company
research are untrusted reference data.

## Search flow

Deterministic role-family queries fan out through the existing JSearch, Adzuna,
Remotive, and cache adapters. Role queries run with a bounded concurrency limit,
while the total candidate set is capped by `SCOUT_MAX_JOBS`; a broad policy
cannot silently turn into 25 ranking candidates. Results are deduplicated by
stable posting identity, ranked in bounded batches, normalized for
employment/work mode/clearance signals, then divided into primary, adjacent,
and blocked/review-required sections. Source diagnostics remain independent so
a result set that came only from Adzuna is visible as such.

If every configured ranking provider is rate-limited, unavailable, or times
out, Jobvis does not discard the fetched postings. It emits conservative local
review scores from exact title/description skill matches, preserves the same
eligibility blockers, and labels the result as a deterministic review score.
The reformulation loop stops in that state so the product does not spend more
model calls repeating a provider outage.

## Tailoring and application safety

Tailoring chooses a resume persona (AI/ML and Data Science, GenAI/RAG,
Forward-Deployed AI, Computer Vision, or MLOps/ML Platform) and keeps the
existing corpus references, link preservation, PDF artifacts, fabrication
validator, and cover-letter quality gate. Browser automation remains visible
and review-gated. Jobvis can open an ATS form and fill explicitly approved safe
fields, but it has no submit action.

## Updating the policy

Use the JSON target-policy editor in the Profile step for a local run. Keep
authorization and sponsorship unknown when uncertain. A missing answer should
produce a review state, not an inferred citizenship or visa answer.
