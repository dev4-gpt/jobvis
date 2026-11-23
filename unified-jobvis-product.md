# Unified Jobvis

One dashboard for the full-time AI/ML search. See today's ranked openings and follow-ups, tailor an application, review and authorize it, then track the outcome and prepare for interviews. Jobvis owns the profile, saved searches, job IDs, application history, and permissions.

| Repository | Contribution |
| --- | --- |
| [Jobvis](https://github.com/dev4-gpt/jobvis) | Keep the Python backend, Next.js console, résumé matching, tailored packs, application review, and voice controls. |
| [JobReach](https://github.com/dev4-gpt/jobreach) | Port the daily shortlist, digest, follow-up reminders, and activity feed. Its [scheduled handler](https://github.com/dev4-gpt/jobreach/blob/main/functions/dailyJobSearch.ts) seeds static internship examples. Use Jobvis discovery and omit those examples from migration. |
| [Career-Ops](https://github.com/dev4-gpt/career-ops-private) | Add interview prep and an editable STAR story bank. Reuse Jobvis scoring, documents, and tracker. |
| [n8n](https://github.com/dev4-gpt/n8n) | Offer a scheduler and digest-delivery adapter. Jobvis runs discovery and owns the run log. |
| [AIHawk bridge](https://github.com/dev4-gpt/career-ops-aihawk-bridge) | Keep Jobvis's existing reviewed export as the optional handoff. Require explicit confirmation before recording a returned submission status. |
| [Internships](https://github.com/dev4-gpt/Internships/blob/main/index.html) | Optional profile link to supporting materials, selected by the user. |

## Build order

- [x] **Sync the baseline.** The local and published review-gated submission commits share tree `7a1b2b1`. Implementation branch: `codex/unified-jobvis`.
- [x] **Unify records.** Preview and import Base44, Career-Ops, and Sheets exports. Fixtures exercise IDs, URL aliases, backup, dates, notes, and no status downgrade on reimport.
- [x] **Build Today.** Saved searches feed a dated shortlist, due follow-ups, and source diagnostics. Fixtures verify duplicate URLs collapse, distinct requisitions stay separate, and a card hands the selected job to tailoring.
- [x] **Implement the digest.** The console previews the digest. The daily command reuses the stored run and can send to one configured recipient. A mocked SMTP test verifies a retry sends once.
- [x] **Add interview prep.** Two applications share the same editable STAR story while keeping separate notes. An optional supporting-materials link is saved locally.
- [ ] **Verify and cut over.** Run `make test` and the console typecheck, lint, and build. Replay discovery → shortlist → tailored pack → application review → follow-up → interview with fixtures, including scheduler retries and a rejected stale confirmation. Confirm import counts and rollback from a backup before switching off any legacy schedule.

Ship the tracker import and Today first. Keep recruiter outreach as drafts and employer submission behind the existing per-application confirmation. Approve any live schedule or email configuration at cutover.
