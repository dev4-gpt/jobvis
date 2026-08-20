# Acceptance record

## Current local acceptance (2026-08-20)

The canonical checkout is `/Users/aryamandev/Documents/Codex/Projects/Jobvis/observable-job-agent`,
on branch `main`. The runtime preflight now rejects a non-primary
Git worktree before importing the application and prints the source path,
branch, and commit.

- `329 passed` in the offline Python suite; Ruff and `git diff --check` passed.
- The built voice console passed TypeScript typecheck, ESLint, and Next static build.
- A live same-process smoke on wizard `:7864` and console `:8005` returned HTTP 200
  for the wizard, `/api/config`, and `/api/state`.
- The uploaded local resume fact check found 12 PDF links, phone contact, and the
  Penn State M.S. Artificial Intelligence end date of December 2026.
- Candidate-aware searches use deterministic role-family fan-out and do not enter
  the legacy LLM reformulation loop; provider retries are disabled at the client
  boundary and explicit fallbacks remain bounded.
- The console now separates primary, adjacent, and blocked/review-required roles,
  including source diagnostics.

The last actual-resume pack backtest should be rerun once the local PDF/parser
process is responsive; the offline backtest and link-preservation acceptance
fixtures are green. No remote push or deployment was performed.

Verified on 2026-08-13 from branch `codex/harden-job-agent`.

## Passing checks

- `make ci`: passed.
- `142 passed, 1 skipped` in the offline suite.
- Ruff format and lint: passed.
- Pyright: passed for the typed application contracts configured in
  `pyrightconfig.json`.
- Keyless health check: passed; the graph imports and compiles without an LLM
  key or network call.
- Keyless Gradio construction: passed (`Blocks`).
- Search diagnostics: tested for source counts, latency, contribution, and
  cache fallback while preserving the legacy `(jobs, sources)` API.
- Checkpoint continuity and stale-selection tests: passed.
- LaTeX escaping and `.tex` degradation tests: passed.
- Prompt-injection boundary instructions are present for search, ranking,
  reformulation, and tailoring inputs.

## Expected skips or unavailable checks

- The renderer compile test is skipped because the `tectonic` binary is not
  installed. The fallback `.tex` path is tested.
- The Opik deterministic gate is skipped when `OPIK_API_KEY` is absent. This is
  intentional; it does not claim a remote evaluation passed without access.
- Docker packaging could not be built in this environment because the Docker
  daemon is not running. Run `docker build -t job-scout:local .` after starting
  Docker Desktop.

## Release state

The local tag `v0.1.0` points at the final acceptance commit. No remote push or
deployment was performed.
