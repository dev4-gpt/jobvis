"""Preview and apply user-provided legacy tracker exports."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from job_scout.application.identity import posting_url
from job_scout.application.models import ApplicationEvent, ApplicationRecord, ApplicationStatus, now_iso
from job_scout.application.store import ApplicationStore, record_urls, write_private_json

_STATUSES: dict[str, ApplicationStatus] = {
    "found": "discovered",
    "new": "discovered",
    "evaluated": "saved",
    "interested": "saved",
    "applied": "submitted_by_user",
    "submitted": "submitted_by_user",
    "interviewing": "interview",
    "interview scheduled": "interview",
    "offer received": "offer",
    "not selected": "rejected",
    **{x: x for x in ("discovered", "saved", "tailored", "reviewed", "interview", "offer", "rejected", "withdrawn")},
}
_JOBREACH_EXAMPLES = {
    "AI/ML Research Intern (Summer 2026)",
    "Machine Learning Intern - Computer Vision",
    "Data Science Intern - Recommendation Systems",
    "AI Systems Engineer Intern (Summer 2026)",
    "NLP/LLM Research Intern (Summer 2026)",
}


def _rows(content: str) -> list[dict]:
    if len(content) > 1_000_000:
        raise ValueError("Import is limited to 1 MB of text.")
    content = content.strip().lstrip("\ufeff")
    if content.startswith(("[", "{")):
        data = json.loads(content)
        if isinstance(data, dict):
            data = data.get("data", data.get("applications", data.get("items")))
        if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
            raise ValueError("Expected a JSON array of records.")
        return data
    tables = [line.strip().strip("|").split("|") for line in content.splitlines() if line.strip().startswith("|")]
    for index, row in enumerate(tables[:-1]):
        if "#" not in {cell.strip() for cell in row} and "company" not in {cell.strip().lower() for cell in row}:
            continue
        result = []
        for cells in tables[index + 2 :]:
            if len(cells) != len(row) or any(re.fullmatch(r"[: -]+", cell) for cell in cells):
                break
            result.append(dict(zip(row, cells, strict=True)))
        return result
    return list(csv.DictReader(io.StringIO(content)))


def _date(value: str) -> str | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).astimezone(UTC).isoformat()


def _plan(records: list[ApplicationRecord], source: str, content: str) -> tuple[dict, list[ApplicationRecord]]:
    records = [record.model_copy(deep=True) for record in records]
    rows = _rows(content)
    if not rows or len(rows) > 2000:
        raise ValueError("Import must contain 1 to 2,000 rows.")
    summary: dict = {"created": 0, "matched": 0, "skipped": 0, "warnings": [], "rows": []}
    for i, raw in enumerate(rows, start=1):
        row = {re.sub(r"[^a-z0-9#]+", "_", str(k).strip().lower()): str(v or "").strip() for k, v in raw.items()}
        title, company = row.get("title") or row.get("role") or row.get("job_title"), row.get("company")
        source_id = row.get("id") or row.get("job_id") or row.get("application_id") or row.get("#") or ""
        url = row.get("application_url") or row.get("job_url") or row.get("url") or row.get("link") or ""
        match = re.fullmatch(r"\[[^\]]*\]\((https?://[^)]+)\)", url)
        url = match[1] if match else url
        listing = row.get("listing_url") or url
        keys = {key for value in (url, listing) if (key := posting_url(value))}
        if not title or not company or (not keys and not source_id) or (source == "base44" and title in _JOBREACH_EXAMPLES):
            summary["skipped"] += 1
            summary["warnings"].append(f"Row {i}: missing identity or a bundled JobReach example.")
            continue
        matches = [
            record
            for record in records
            if (source_id and source_id in record.source_ids.get(source, [])) or keys & record_urls(record)
        ]
        if len(matches) > 1:
            summary["skipped"] += 1
            summary["warnings"].append(f"Row {i}: multiple tracker records match.")
            continue
        raw_status = row.get("status", "")
        status = _STATUSES.get(raw_status.lower().strip(), "discovered")
        created = _date(row.get("created_at") or row.get("created_date") or row.get("date") or "") or now_iso()
        updated = _date(row.get("updated_at") or row.get("updated_date") or "") or created
        submitted = _date(row.get("submitted_at") or row.get("applied_at") or row.get("applied_date") or "")
        notes = row.get("notes", "")
        if matches:
            record = matches[0]
            summary["matched"] += 1
            if notes and notes not in record.notes.split("\n\n"):
                record.notes = "\n\n".join(filter(None, (record.notes, notes)))
            if record.status == "discovered" and status != "discovered":
                record.status = status
                record.imported_status = raw_status
        else:
            record = ApplicationRecord(
                job_id=f"import-{uuid4().hex}",
                title=title,
                company=company,
                source=source,
                status=status,
                imported_status=raw_status,
                notes=notes,
                created_at=created,
                updated_at=updated,
            )
            records.append(record)
            summary["created"] += 1
        if source_id:
            record.source_ids[source] = sorted({*record.source_ids.get(source, []), source_id})
        record.posting_aliases = sorted(set(record.posting_aliases) | keys)
        record.application_url = record.application_url or (url if posting_url(url) else "")
        record.listing_url = record.listing_url or (listing if posting_url(listing) else "")
        if record.status in {"submitted_by_user", "interview", "offer"} and not record.submitted_at:
            record.submitted_at = submitted or (updated if status == "submitted_by_user" else None)
            if record.submitted_at:
                date = datetime.fromisoformat(record.submitted_at)
                record.followup_due = [(date + timedelta(days=days)).isoformat() for days in (7, 14)]
        if record.status == "interview" and not record.interviewed_at:
            record.interviewed_at = _date(row.get("interviewed_at", ""))
        detail = f"{source} · {source_id or url} · {raw_status or 'unspecified'}"
        if not any(event.name == "imported" and event.detail == detail for event in record.events):
            record.events.append(ApplicationEvent(name="imported", detail=detail, at=updated))
        summary["rows"].append({"company": company, "title": title, "status": record.status, "matched": bool(matches)})
    return summary, records


def import_tracker(store: ApplicationStore, source: str, content: str, approval: str | None = None) -> dict:
    if source not in {"base44", "career-ops", "sheets"}:
        raise ValueError("Choose Base44, Career-Ops, or Sheets.")
    with store.transaction(strict=True):
        records = store.persisted_records()
        before = [record.model_dump(mode="json") for record in records]
        fingerprint = hashlib.sha256(json.dumps([source, content, before], sort_keys=True).encode()).hexdigest()
        summary, after = _plan(records, source, content)
        if approval is not None:
            if approval != fingerprint:
                raise ValueError("The export or tracker changed. Preview again.")
            if summary["rows"] and after != records:
                backup = store.data_dir / "tracker-backups" / f"{uuid4().hex}.json"
                write_private_json(backup, {"version": 1, "applications": before})
                store.save_records(after)
                summary["backup"] = backup.name
        return {**summary, "preview_id": fingerprint}
