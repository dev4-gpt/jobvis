"""AIHawk Queue Exporter.

Exports approved Jobvis application packs into the standardized AIHawk queue format
(conforming to aihawk-queue.schema.json) for automated submission via AIHawk or external workers.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator, FormatChecker


def format_aihawk_job(
    job_id: str | int,
    company: str,
    role: str,
    score: float,
    status: str = "Evaluated",
    job_url: str | None = None,
    pdf_path: str | Path | None = None,
    report_path: str | Path | None = None,
    notes: str = "",
    date_evaluated: str | None = None,
) -> dict[str, Any]:
    """Formats a single job pack into an AIHawk queue entry."""
    # Ensure zero-padded 3-digit id if integer or digit string
    str_id = str(job_id).strip()
    if str_id.isdigit():
        str_id = f"{int(str_id):03d}"
    elif not str_id:
        str_id = "001"

    # This boundary accepts Jobvis scores only, always 0–100.
    if not math.isfinite(float(score)) or not 0 <= score <= 100:
        raise ValueError("Jobvis score must be finite and between 0 and 100")
    normalized_score = round(float(score) / 20.0, 2)

    resolved_pdf = str(Path(pdf_path).resolve()) if pdf_path else None
    resolved_report = str(Path(report_path).resolve()) if report_path else None
    eval_date = date_evaluated or datetime.now(UTC).strftime("%Y-%m-%d")

    return {
        "id": str_id,
        "date": eval_date,
        "company": company.strip(),
        "role": role.strip(),
        "score": normalized_score,
        "status": status,
        "hasPdf": resolved_pdf is not None,
        "pdfPath": resolved_pdf,
        "reportPath": resolved_report,
        "jobUrl": job_url,
        "notes": notes.strip(),
    }


def export_aihawk_queue(
    packs: list[dict[str, Any]],
    output_path: str | Path | None = None,
    score_threshold: float = 3.5,
    repo_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Exports multiple application packs into a standardized AIHawk queue document."""
    now_iso = datetime.now(UTC).isoformat()
    jobs = []

    for idx, pack in enumerate(packs, start=1):
        job_id = pack.get("id", idx)
        company = pack["company"]
        role = pack["role"]
        score = float(pack["score"])
        job_url = pack.get("job_url", pack.get("url", pack.get("application_url")))
        pdf_path = pack.get("pdf_path", pack.get("tailored_cv_pdf_path"))
        report_path = pack.get("report_path")
        notes = pack.get("notes", "")
        date_eval = pack.get("date")

        formatted = format_aihawk_job(
            job_id=job_id,
            company=company,
            role=role,
            score=score,
            status="Evaluated",
            job_url=job_url,
            pdf_path=pdf_path,
            report_path=report_path,
            notes=notes,
            date_evaluated=date_eval,
        )

        if formatted["score"] >= score_threshold:
            jobs.append(formatted)

    queue_doc = {
        "schema_version": "1.0",
        "exported_at": now_iso,
        "score_threshold": score_threshold,
        "status_filter": ["Evaluated"],
        "career_ops_dir": str(Path(repo_dir).resolve()) if repo_dir else str(Path.cwd()),
        "jobs": jobs,
    }

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        from job_scout.application.handoff import atomic_json

        validate_queue(queue_doc)
        atomic_json(out, queue_doc)

    validate_queue(queue_doc)

    return queue_doc


def validate_queue(document: dict) -> None:
    schema = json.loads(Path(__file__).with_name("aihawk-queue.schema.json").read_text(encoding="utf-8"))
    Draft7Validator(schema, format_checker=FormatChecker()).validate(document)
