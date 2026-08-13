# Data and tracing policy

Job Scout processes resumes and optional LinkedIn exports, which contain
personal data.

- The application never submits a job application.
- LinkedIn ZIP files are passed by filesystem path and are not attached to
  traces or committed to Git.
- With `OPIK_ENABLED=true`, the CV PDF and model/tool inputs may leave the
  machine and be stored in the configured Opik project. This is intentional for
  the learning workflow and must be disclosed to anyone using the app.
- API keys belong only in `.env` or a deployment secret store. They must never
  appear in logs, fixtures, screenshots, notebooks, or commits.
- The offline test suite uses fixture CVs and mocked providers, so it does not
  require real candidate data or network access.

For a privacy-sensitive deployment, set `OPIK_ENABLED=false` and use the local
application logs/metrics path instead of uploading trace attachments.
