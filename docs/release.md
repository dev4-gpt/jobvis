# Release and deployment runbook

## Local container

Build and run the deployable image without putting secrets into the image:

```bash
docker build -t job-scout:local .
docker run --rm -p 7860:7860 --env-file .env job-scout:local
```

Open <http://127.0.0.1:7860>. The image contains the offline cache; API keys,
Opik settings, and model configuration are runtime environment variables.

## Pull requests

Every pull request must pass keyless quality checks and the container build.
Live job APIs, paid LLM calls, Opik experiments, and search benchmarks are
available only through the manually triggered `Optional live integration`
workflow.

## Versioned releases

After review and acceptance:

```bash
git switch main
git pull --ff-only
git tag -a v0.1.0 -m "first hardened local release"
git push origin v0.1.0
```

The final push is intentionally a human action. To roll back locally, check out
the previous verified tag and rebuild the image:

```bash
git switch --detach v0.1.0
docker build -t job-scout:rollback .
```
