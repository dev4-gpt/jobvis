"""Atomic local JSON persistence for application records and asset manifests."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path

from job_scout.application.models import ApplicationRecord
from job_scout.config import get_settings


def default_data_dir() -> Path:
    configured = get_settings().jobvis_data_dir.strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Library" / "Application Support" / "Jobvis"


class ApplicationStore:
    """Small versioned store; browser state and secrets are intentionally absent."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or default_data_dir()
        self.path = self.data_dir / "applications.json"
        self._memory: list[ApplicationRecord] = []

    def _load(self) -> dict:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, payload: dict) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.chmod(0o700)
        fd, name = tempfile.mkstemp(prefix="applications.", suffix=".tmp", dir=self.data_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
            Path(name).chmod(0o600)
            os.replace(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)

    def list(self) -> list[ApplicationRecord]:
        loaded = [ApplicationRecord.model_validate(item) for item in self._load().get("applications", [])]
        return loaded or list(self._memory)

    def get(self, application_id: str) -> ApplicationRecord | None:
        return next((item for item in self.list() if item.application_id == application_id), None)

    def upsert(self, record: ApplicationRecord) -> ApplicationRecord:
        records = self.list()
        for index, existing in enumerate(records):
            if existing.application_id == record.application_id:
                records[index] = record
                break
        else:
            records.append(record)
        self._memory = records
        with suppress(OSError):
            self._save({"version": 1, "applications": [item.model_dump(mode="json") for item in records]})
        return record

    def create_for_job(
        self, *, job_id: str, title: str, company: str, listing_url: str, application_url: str, source: str
    ) -> ApplicationRecord:
        existing = next((item for item in self.list() if item.job_id == job_id), None)
        if existing:
            return existing
        record = ApplicationRecord(
            job_id=job_id,
            title=title,
            company=company,
            listing_url=listing_url,
            application_url=application_url,
            source=source,
        )
        return self.upsert(record)

    def mark(self, application_id: str, status: str, detail: str = "") -> ApplicationRecord | None:
        record = self.get(application_id)
        if record is None:
            return None
        allowed = {
            "discovered",
            "saved",
            "tailored",
            "reviewed",
            "opened",
            "safe_fields_filled",
            "final_review",
            "submitted_by_user",
        }
        if status not in allowed:
            raise ValueError("invalid application status")
        record.transition(status, detail)  # type: ignore[arg-type]
        return self.upsert(record)
