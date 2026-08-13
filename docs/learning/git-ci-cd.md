# Git, CI/CD, and release lessons

## The branch model

The working branch is `codex/harden-job-agent`. Each milestone is an atomic
commit with one reason to exist. A pull request should be reviewable by reading
the commit sequence, not by reconstructing one giant diff.

Recommended workflow:

```bash
git switch -c feature/name
git status
git diff
git add path/to/changed/files
git commit -m "fix: describe one behavior"
git log --oneline --decorate -6
```

Never commit `.env`, API keys, uploaded CVs, LinkedIn exports, generated PDFs,
or local virtual environments. A release tag points at a verified commit and
is the rollback unit:

```bash
git tag -a v0.1.0 -m "first hardened local release"
git show v0.1.0
```

## CI layers

Pull-request CI is intentionally keyless and deterministic:

1. install from `uv.lock`;
2. check formatting and lint;
3. run type checks;
4. run unit/integration tests with mocked network and LLM calls;
5. run deterministic gates;
6. build and smoke-test the container.

Live APIs, paid model calls, Opik experiments, and latency benchmarks belong in
a manually triggered workflow because they spend money, depend on external
state, and should not block ordinary code review.

## Deployment boundary

The application is packaged as one container with configuration supplied by
environment variables. The image is reproducible from the lockfile; secrets
are injected by the runtime; user data is written to temporary/supported local
storage only. A deployment platform can be selected later without changing the
graph or source adapters.

## Rollback thinking

If a prompt, source adapter, or evaluation threshold regresses quality, revert
the smallest offending commit or redeploy the previous tag. Do not “fix” a
quality regression by hiding the metric. Record the before/after metric and the
tradeoff in the evaluation report.
