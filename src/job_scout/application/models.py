"""Local, review-gated application tracking contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

ApplicationStatus = Literal[
    "discovered",
    "saved",
    "tailored",
    "reviewed",
    "opened",
    "safe_fields_filled",
    "final_review",
    "submitted_by_user",
]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class AssetManifest(BaseModel):
    """Paths and deterministic checks for one selected application pack."""

    manifest_id: str = Field(default_factory=lambda: str(uuid4()))
    job_id: str
    cv_pdf: str = ""
    cv_tex: str = ""
    cover_letter_pdf: str = ""
    cover_letter_tex: str = ""
    links_verified: bool = False
    annotations_verified: bool = False
    quality_verified: bool = False
    ready: bool = False
    warnings: list[str] = Field(default_factory=list)


class ApplicationEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    detail: str = ""
    at: str = Field(default_factory=now_iso)


class ApplicationRecord(BaseModel):
    """Durable local tracker state; it has no submit or credential fields."""

    application_id: str = Field(default_factory=lambda: str(uuid4()))
    job_id: str
    title: str = ""
    company: str = ""
    listing_url: str = ""
    application_url: str = ""
    source: str = ""
    status: ApplicationStatus = "discovered"
    asset_manifest_id: str = ""
    events: list[ApplicationEvent] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)

    def transition(self, status: ApplicationStatus, detail: str = "") -> None:
        self.status = status
        self.updated_at = now_iso()
        self.events.append(ApplicationEvent(name=status, detail=detail))
