# ADR 0001: Separate candidate policy from probabilistic fit

Status: accepted

## Context

The original matcher combined semantic skill similarity, location hints, and
job metadata into one LLM score. That caused internships, clearance roles,
BI/Java adjacent roles, and jobs with incompatible timing to compete with the
candidate's intended full-time AI/ML search.

## Decision

Keep `Profile` as extracted evidence and store human choices in typed
`CandidatePreferences`. Normalize posting metadata and run deterministic
eligibility before displaying the final priority score. Preserve the LLM's
role/evidence explanation, but never let it override hard policy outcomes.

## Trade-offs

- More fields and UI review are required, but the result is explainable and
  testable without an API call.
- Missing job metadata creates borderline results instead of false certainty.
- Role-family fan-out costs more source requests, but prevents one broad query
  from hiding relevant AI/ML and GenAI openings.

## Consequences

The LangGraph checkpoint carries the typed intent, API/voice payloads expose
eligibility reasons, and tests can verify graduate timing and clearance policy
without network access.
