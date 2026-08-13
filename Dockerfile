FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    GRADIO_SERVER_NAME=0.0.0.0

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY data/cached_jobs.json ./data/cached_jobs.json

RUN uv sync --frozen --no-dev

EXPOSE 7860
CMD ["python", "-m", "job_scout.app"]
