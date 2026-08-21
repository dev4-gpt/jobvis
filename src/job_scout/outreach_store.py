"""Private local persistence for contact candidates and outreach drafts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from job_scout.application.store import default_data_dir
from job_scout.outreach import OutreachDraft


class OutreachStore:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or default_data_dir()
        self.path = self.data_dir / "outreach_drafts.json"

    def list(self) -> list[OutreachDraft]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [OutreachDraft.model_validate(item) for item in payload.get("drafts", [])]

    def save(self, draft: OutreachDraft) -> OutreachDraft:
        drafts = [item for item in self.list() if item.draft_id != draft.draft_id]
        drafts.append(draft)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.chmod(0o700)
        fd, name = tempfile.mkstemp(prefix="outreach.", suffix=".tmp", dir=self.data_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"version": 1, "drafts": [item.model_dump(mode="json") for item in drafts]}, handle, indent=2)
                handle.write("\n")
            Path(name).chmod(0o600)
            os.replace(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)
        return draft
