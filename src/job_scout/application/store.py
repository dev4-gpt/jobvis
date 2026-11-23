"""Atomic local JSON persistence for application records and asset manifests."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import ExitStack, contextmanager, suppress
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from threading import RLock
from typing import get_args

from job_scout.application.identity import posting_url
from job_scout.application.models import ApplicationRecord, ApplicationStatus
from job_scout.config import get_settings


def default_data_dir() -> Path:
    configured = get_settings().jobvis_data_dir.strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Library" / "Application Support" / "Jobvis"


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with os.fdopen(os.open(path, os.O_CREAT | os.O_RDWR, 0o600), "r+") as handle:
        flock(handle.fileno(), LOCK_EX)
        try:
            yield
        finally:
            flock(handle.fileno(), LOCK_UN)


def write_private_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class ApplicationStore:
    """Small versioned store; browser state and secrets are intentionally absent."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or default_data_dir()
        self.path = self.data_dir / "applications.json"
        self._memory: list[ApplicationRecord] = []
        self._lock = RLock()

    @contextmanager
    def transaction(self, *, strict: bool = False):
        with self._lock, ExitStack() as stack:
            try:
                stack.enter_context(file_lock(self.path.with_suffix(".lock")))
            except OSError:
                if strict:
                    raise
            yield

    def _load(self) -> dict:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, payload: dict) -> None:
        write_private_json(self.path, payload)

    def list(self) -> list[ApplicationRecord]:
        payload = self._load()
        if "applications" not in payload:
            return list(self._memory)
        return [ApplicationRecord.model_validate(item) for item in payload["applications"]]

    def persisted_records(self) -> list[ApplicationRecord]:
        """Fail closed if a file exists but cannot be read or validated."""
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return list(self._memory)
        return [ApplicationRecord.model_validate(item) for item in payload["applications"]]

    def save_records(self, records: list[ApplicationRecord]) -> None:
        self._save({"version": 1, "applications": [item.model_dump(mode="json") for item in records]})
        self._memory = records

    def get(self, application_id: str) -> ApplicationRecord | None:
        return next((item for item in self.list() if item.application_id == application_id), None)

    def _upsert(self, record: ApplicationRecord) -> ApplicationRecord:
        records = self.list()
        for index, existing in enumerate(records):
            if existing.application_id == record.application_id:
                records[index] = record
                break
        else:
            records.append(record)
        self._memory = records
        with suppress(OSError):
            self.save_records(records)
        return record

    def upsert(self, record: ApplicationRecord) -> ApplicationRecord:
        with self.transaction():
            return self._upsert(record)

    def create_for_job(
        self, *, job_id: str, title: str, company: str, listing_url: str, application_url: str, source: str
    ) -> ApplicationRecord:
        with self.transaction():
            records = self.list()
            known = {key for url in (listing_url, application_url) if (key := posting_url(url))}
            existing = next((item for item in records if item.job_id == job_id), None)
            if existing is None and known:
                matches = [item for item in records if known & record_urls(item)]
                existing = matches[0] if len(matches) == 1 else None
            if existing:
                aliases = sorted(set(existing.posting_aliases) | known)
                if aliases != existing.posting_aliases:
                    existing.posting_aliases = aliases
                    self._upsert(existing)
                return existing
            record = ApplicationRecord(
                job_id=job_id,
                title=title,
                company=company,
                listing_url=listing_url,
                application_url=application_url,
                source=source,
                posting_aliases=sorted(known),
            )
            return self._upsert(record)

    def mark(self, application_id: str, status: str, detail: str = "") -> ApplicationRecord | None:
        with self.transaction():
            record = self.get(application_id)
            if record is None:
                return None
            if status not in get_args(ApplicationStatus):
                raise ValueError("invalid application status")
            record.transition(status, detail)  # type: ignore[arg-type]
            return self._upsert(record)


def record_urls(record: ApplicationRecord) -> set[str]:
    return {key for url in [record.application_url, record.listing_url, *record.posting_aliases] if (key := posting_url(url))}
