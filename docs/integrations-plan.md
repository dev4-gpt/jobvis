# Complete Jobvis integrations

Accepted scope: curated discovery, listing checks, manual Apify expansion, encrypted local
memory, and reviewed AIHawk export. Workers are launched separately. Final submission and
outreach sending remain user actions.

## Implementation

- Curated Greenhouse, Lever and Ashby boards feed existing adapters when direct sources
  are enabled. Explicit lists replace defaults. Board responses are shared across role queries
  for fifteen minutes. Imported seniority exclusions do not override candidate policy.
- Liveness runs before ranking: four shared workers, eight-second request timeout,
  fifteen-second stage budget and fifteen-minute cache. Only confident expiry removes jobs.
  Blocked requests, server errors, shells and unfinished checks remain uncertain. Application
  opening and export recheck URLs. Diagnostics retain counts and reasons.
- Manual Apify expansion runs a configured saved task with a 120-second deadline and at most
  25 normalized results. The existing run coordinator prevents duplicate in-flight actions.
  Results enter the common ranking/eligibility path and total cap. Timeout attempts cancellation;
  cancellation can fail and does not reverse charges. Ordinary searches never invoke Apify.
- Memory uses Keychain-backed encryption, atomic writes, candidate identity and explicit
  confirmation. Edits clear confirmation. Legacy plaintext imports remain unconfirmed and preserve
  the source. Confirmed, nonsensitive resume highlights can assist tailoring only when exactly
  matched to a reference in the current corpus. Answer suggestions require review before use.
- Durable reviews store PDFs, hashes, job identity, audit evidence and approval time.
  Regeneration invalidates approval. Export requires approval, score >=70/100, valid URL,
  no eligibility blockers, passing artifact checks and acknowledgment for uncertain liveness.
  Scores are divided by twenty. Persistent three-digit IDs stop at 999. Repeated exports verify
  hashes before reusing their bundles. Export events never mark jobs submitted.
- Submission/interview recording computes follow-up dates (days 7/14 and +24h) for the tracker.
  These dates do not create scheduled notifications or send messages.

## Interfaces

| Route | Purpose |
| --- | --- |
| `GET /api/integrations` | Board configuration and Apify readiness |
| `POST/GET /api/apify/expansion` | Explicit start and provider progress |
| `GET/POST /api/memory` | Availability/list and unconfirmed creation |
| `PUT/DELETE /api/memory/{id}` | Edit (clears confirmation) or delete |
| `POST /api/memory/{id}/confirm` | Confirm a reviewed entry |
| `POST /api/memory/import-legacy` | Import the original plaintext vault for review |
| `POST /api/memory/import-resume` | Import current resume facts for review |
| `GET /api/memory/suggestions?question=...` | Confirmed keyword suggestions |
| `GET/POST /api/pack/review` | Audit preview and snapshot approval |
| `GET /api/packs/reviewed` | Candidate-scoped snapshot list |
| `POST /api/packs/export` | Idempotent export bundle |
| `GET /api/packs/exports/{id}/queue` | Candidate-scoped queue download |

Jobs add optional `liveness`; diagnostics add `expired` and `incomplete_checks`.
Application records add `interview` and due dates. New stores use version 1 under
`JOBVIS_DATA_DIR`, defaulting to `~/Library/Application Support/Jobvis`.

## Configuration and compatibility

Install `uv sync --extra application` for encrypted memory and local browser support.
Encryption failures are visible and never fall back to plaintext. Memory is local keyword
retrieval; no external Synaptic synchronization or vector database is implemented.

Apify requires `APIFY_API_TOKEN`, `APIFY_TASK_ID`, `APIFY_INPUT_TEMPLATE`, and
`APIFY_OUTPUT_MAPPING`. The template is a JSON object. Values exactly equal to `$query`,
`$location`, `$country`, `$remote`, or `$limit` become current search values; other values are
task-specific constants. No CV or memory is inserted. The output mapping requires `title`,
`company`, and `url`, with values naming dataset fields (including dotted paths). Optional
normalized fields include `description`, `location`, `id`, `remote`, `applyUrl`, `tags`, and
`salary`. Configure these against the actual saved task before enabling manual expansion.

Provider contracts: [run task](https://docs.apify.com/api/v2/actor-task-runs-post) and
[API reference](https://docs.apify.com/api/v2). Credentials travel in authorization headers.

The legacy queue schema is vendored unchanged. Extra Jobvis identity mappings, cover-letter
paths and audit evidence reside in a separate bundle manifest. Export uses durable local paths
for a worker on the same host. The old bridge's dry run can prepare files, so automated tests
validate fixtures/schema without invoking the installed worker.

## Verification

Acceptance covers source wiring/caching, candidate-policy conflicts, liveness deadlines,
explicit Apify execution/failure handling, encrypted restart and isolation, corpus grounding,
stale reviews, durable export hashes, schema compatibility and repeat operations.
See `integration-acceptance.md` for commands, results and live-service limitations.
