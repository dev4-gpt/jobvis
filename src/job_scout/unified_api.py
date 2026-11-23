"""Console routes for the daily shortlist, tracker imports, and interview prep."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, HttpUrl

from job_scout.application.models import ApplicationEvent
from job_scout.application.store import default_data_dir, file_lock, write_private_json
from job_scout.application.tracker_import import import_tracker
from job_scout.daily import DailyShortlist, _read
from job_scout.graph.schemas import RankedJob

router = APIRouter(prefix="/api")


class TrackerImport(BaseModel):
    source: str
    content: str = Field(max_length=1_000_000)
    preview_id: str | None = None


class Story(BaseModel):
    title: str = Field(min_length=1, max_length=150)
    situation: str = Field(default="", max_length=5000)
    task: str = Field(default="", max_length=5000)
    action: str = Field(default="", max_length=5000)
    result: str = Field(default="", max_length=5000)
    evidence: str = Field(default="", max_length=5000)


class Prep(BaseModel):
    notes: str = Field(default="", max_length=20000)
    story_ids: list[str] = Field(default_factory=list, max_length=100)


class Resource(BaseModel):
    url: HttpUrl | None = None


class Followup(BaseModel):
    due: str


def _store():
    from job_scout.api import _APPLICATION_STORE

    return _APPLICATION_STORE


def _daily() -> DailyShortlist:
    try:
        return DailyShortlist(_store().data_dir)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise HTTPException(400, "Set JOBVIS_DIGEST_TIMEZONE to a valid IANA time zone.") from exc


def _prep_path() -> Path:
    return default_data_dir() / "interview-prep.json"


def _prep() -> dict:
    return _read(_prep_path()) or {"stories": {}, "supporting_url": ""}


@router.post("/applications/import/preview")
def preview_import(body: TrackerImport) -> dict:
    return _import(body, None)


@router.post("/applications/import")
def apply_import(body: TrackerImport) -> dict:
    if not body.preview_id:
        raise HTTPException(400, "Preview the import before applying it.")
    return _import(body, body.preview_id)


def _import(body: TrackerImport, approval: str | None) -> dict:
    try:
        return import_tracker(_store(), body.source, body.content, approval)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(409 if approval else 400, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503, "Could not save or read the private tracker.") from exc


@router.get("/today")
def today() -> dict:
    daily = _daily()
    run = daily.get()
    return {
        "date": daily.date,
        "timezone": daily.zone,
        "run": run,
        "followups": daily.followups(),
        "digest": daily.render(run) if run else "",
    }


@router.post("/today/run")
def run_today() -> dict:
    try:
        _daily().run()
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return today()


@router.post("/today/jobs/{job_id}/tailor")
def tailor_today_job(job_id: str) -> dict:
    from job_scout import api
    from job_scout.candidate_fit import assess_eligibility, preferences_from_dict
    from job_scout.voice import bridge as voice_bridge

    run = _daily().get() or {}
    row = next((item for item in run.get("ranked", []) if item["job"]["job_id"] == job_id), None)
    if row is None:
        raise HTTPException(404, "Job not found in today's shortlist.")
    ranked = RankedJob.model_validate(row)
    api.ensure_session()
    bridge = voice_bridge.get_bridge()
    snap = bridge.snapshot()
    if snap.profile is None or bridge.run_status()["running"]:
        raise HTTPException(409, "Load your résumé and wait for the current run to finish.")
    values = voice_bridge.checkpoint_values(snap.thread_id)
    assessment = assess_eligibility(ranked.job, snap.profile, preferences_from_dict(values.get("candidate_preferences")))
    if ranked.eligibility_status == "blocked" or assessment.status == "blocked":
        raise HTTPException(409, "This job is blocked by your search preferences.")
    job = ranked.job
    record = _store().create_for_job(
        job_id=job.job_id,
        title=job.title,
        company=job.company,
        source=job.source,
        listing_url=job.listing_url or job.source_url or job.url,
        application_url=job.application_url or job.url,
    )
    canonical_id = record.job_id
    job = job.model_copy(update={"job_id": canonical_id})
    ranked = ranked.model_copy(update={"job": job})
    retained = [item for item in values.get("ranked_jobs") or [] if item.job.job_id not in {job_id, canonical_id}]
    jobs = [item for item in values.get("jobs") or [] if item.job_id not in {job_id, canonical_id}]
    voice_bridge.get_compiled_graph().update_state(
        {"configurable": {"thread_id": snap.thread_id}},
        {"jobs": [*jobs, job], "ranked_jobs": [*retained, ranked]},
    )
    if refusal := bridge.start_tailoring(canonical_id):
        raise HTTPException(409, refusal)
    return {"started": True, "job_id": canonical_id}


@router.get("/interview-prep")
def get_prep() -> dict:
    return _prep()


@router.put("/interview-prep/resource")
def save_resource(body: Resource) -> dict:
    path = _prep_path()
    with file_lock(path.with_suffix(".lock")):
        data = _prep()
        data["supporting_url"] = str(body.url) if body.url else ""
        write_private_json(path, data)
    return data


@router.post("/interview-prep/stories")
def create_story(body: Story) -> dict:
    return _save_story(uuid4().hex, body, create=True)


@router.put("/interview-prep/stories/{story_id}")
def update_story(story_id: str, body: Story) -> dict:
    return _save_story(story_id, body, create=False)


def _save_story(story_id: str, body: Story, *, create: bool) -> dict:
    path = _prep_path()
    with file_lock(path.with_suffix(".lock")):
        data = _prep()
        if not create and story_id not in data["stories"]:
            raise HTTPException(404, "Story not found.")
        data["stories"][story_id] = body.model_dump()
        write_private_json(path, data)
    return data


@router.put("/applications/{application_id}/interview-prep")
def save_interview(application_id: str, body: Prep) -> dict:
    if not set(body.story_ids) <= _prep()["stories"].keys():
        raise HTTPException(400, "Choose stories from your story bank.")
    store = _store()
    with store.transaction(strict=True):
        records = store.persisted_records()
        record = next((record for record in records if record.application_id == application_id), None)
        if record is None:
            raise HTTPException(404, "Application not found.")
        record.interview_notes = body.notes
        record.interview_story_ids = list(dict.fromkeys(body.story_ids))
        store.save_records(records)
        return record.model_dump(mode="json")


@router.post("/applications/{application_id}/followup")
def complete_followup(application_id: str, body: Followup) -> dict:
    store = _store()
    with store.transaction(strict=True):
        records = store.persisted_records()
        record = next((record for record in records if record.application_id == application_id), None)
        if record is None or not body.due or body.due not in [*record.followup_due, record.thank_you_due]:
            raise HTTPException(404, "Follow-up not found.")
        if not any(event.name == "followup_done" and event.detail == body.due for event in record.events):
            record.events.append(ApplicationEvent(name="followup_done", detail=body.due))
            store.save_records(records)
    return {"completed": True}
