# Acceptance record

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
