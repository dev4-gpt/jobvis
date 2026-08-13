# Job Scout baseline import

This workspace starts from the upstream Part 3 release referenced by the three
teaching PDFs.

## Source control

- Upstream repository: `https://github.com/jamwithai/observable-job-agent`
- Imported reference: `part3.0`
- Checked-out upstream commit: `674802e971a3cc5543cc5d5fd3f73d6bcff7849`
- Working branch: `codex/harden-job-agent`
- Local working directory: `observable-job-agent/`

The `part3.0` tag is an annotated release reference. The clone initially
checked it out detached; the working branch above was created explicitly before
any project changes.

## Baseline verification

Run from the repository root:

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run pytest gates/ -v
```

Results on 2026-08-13:

- `140 passed, 1 skipped` in the offline test suite.
- The skipped test requires the optional `tectonic` binary.
- Ruff passed with no findings.
- The deterministic evaluation gate was skipped because `OPIK_API_KEY` is not configured; it does not silently pass without Opik access.

The baseline is therefore usable before hardening. Later commits should keep
the no-key test path deterministic and should call out any intentionally changed
behavior or lockfile update.
