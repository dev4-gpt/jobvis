"""Console routes for explicit expansion, private memory, and reviewed handoff."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from job_scout.application.handoff import HandoffStore, digest
from job_scout.application.models import ApplicationEvent
from job_scout.application.security import BrowserSecurityError
from job_scout.config import get_settings
from job_scout.memory.service import MemoryEntry, MemoryService, candidate_key
from job_scout.tools.apify_client import ConfiguredApifyTask

router = APIRouter(prefix="/api")
_TASK: ConfiguredApifyTask | None = None


def context():
    from job_scout import api

    api.ensure_session()
    bridge = api.voice_bridge.get_bridge()
    snap = bridge.snapshot()
    if snap.profile is None:
        raise HTTPException(409, "Load a candidate first")
    return api, bridge, snap, candidate_key(snap.cv_text, snap.profile.name or "")


def invoke(fn):
    try:
        return fn()
    except (ValueError, OSError, BrowserSecurityError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/integrations")
def integrations():
    settings = get_settings()
    task = ConfiguredApifyTask(settings)
    error = ""
    try:
        task.configuration()
    except ValueError as exc:
        error = str(exc)
    return {
        "direct_sources_enabled": settings.direct_sources_enabled,
        "boards": {"greenhouse": settings.greenhouse_boards, "lever": settings.lever_accounts, "ashby": settings.ashby_boards},
        "liveness_enabled": settings.liveness_enabled,
        "apify_ready": not error,
        "apify_hint": error,
    }


@router.get("/apify/expansion")
def expansion_status():
    return dict(_TASK.state) if _TASK else {"status": "idle", "run_id": None}


@router.post("/apify/expansion")
def expand_search():
    global _TASK
    if _TASK and _TASK.state.get("status") in {"starting", "running", "ranking"}:
        raise HTTPException(409, "An Apify expansion is already in flight")
    api, bridge, snap, identity = context()
    values = api.voice_bridge.checkpoint_values(snap.thread_id)
    if not values.get("profile"):
        raise HTTPException(409, "Run an ordinary search before expanding")
    task = ConfiguredApifyTask()
    invoke(task.configuration)
    fingerprint = digest(values)

    def work(cancelled):
        from job_scout.graph import get_compiled_graph
        from job_scout.graph.nodes.fetch_jobs import _candidate_queries
        from job_scout.graph.nodes.rank_jobs import rank_jobs
        from job_scout.tools.jobs_api import _dedupe

        preferences = api.preferences_from_dict(values.get("candidate_preferences") or snap.preferences)
        queries = _candidate_queries(snap.profile, preferences, 0)
        jobs = task.run(
            {
                "query": " | ".join(queries),
                "location": ", ".join(preferences.locations),
                "country": preferences.country_scope,
                "remote": preferences.accepted_work_modes == ["remote"],
                "limit": 25,
            },
            cancelled=cancelled,
        )
        current = api.voice_bridge.checkpoint_values(snap.thread_id)
        latest = bridge.snapshot()
        if (
            cancelled()
            or latest.thread_id != snap.thread_id
            or latest.profile is None
            or candidate_key(latest.cv_text, latest.profile.name or "") != identity
            or digest(current) != fingerprint
        ):
            raise ValueError("Search changed during expansion; results were not applied")
        # Prioritize expansion candidates within the same total ranking budget.
        merged = _dedupe(jobs + list(values.get("jobs", [])))[: get_settings().scout_max_jobs]
        from job_scout.graph.schemas import SourceDiagnostic

        diagnostics = list(values.get("source_diagnostics", []))
        diagnostics = [item for item in diagnostics if item.source != "apify"]
        diagnostics.append(SourceDiagnostic(source="apify", requested=True, completed=True, returned=len(jobs)))
        task.state.update(status="ranking")
        update = rank_jobs({**values, "jobs": merged, "source_diagnostics": diagnostics})
        if cancelled() or digest(api.voice_bridge.checkpoint_values(snap.thread_id)) != fingerprint:
            raise ValueError("Search changed while ranking expanded results")
        for diagnostic in update.get("source_diagnostics", []):
            diagnostic.contributed = any(job.source == diagnostic.source for job in update.get("jobs", []))
        update.update(
            selected_job_id=None,
            tailoring=None,
            jobs_sources=list(dict.fromkeys([*values.get("jobs_sources", []), *(["apify"] if jobs else [])])),
        )
        get_compiled_graph().update_state({"configurable": {"thread_id": snap.thread_id}}, update)
        task.state.update(status="complete", merged=len(update.get("jobs", [])))

    def drive(cancelled):
        try:
            work(cancelled)
        except Exception:
            if task.state["status"] not in {"failed", "timed_out"}:
                task.state.update(status="failed", error="Expansion could not be applied to the current search")
            raise

    refusal = bridge.start_expansion(drive)
    if refusal:
        raise HTTPException(409, refusal)
    _TASK = task
    return {"started": True}


def memory_service():
    return MemoryService(context()[3])


@router.get("/memory")
def memory_list():
    service = memory_service()
    try:
        from job_scout.application.security import _fernet

        _fernet()
        return {"available": True, "entries": [item.model_dump() for item in service.list()]}
    except Exception:
        return {
            "available": False,
            "entries": [],
            "error": "Encrypted memory unavailable. Check the application extra, Keychain access, and stored file.",
        }


class MemoryInput(BaseModel):
    content: str = Field(min_length=1, max_length=10000)
    category: str = "qa_answer"
    question_key: str = ""
    sensitive: bool = False


@router.post("/memory")
def memory_add(body: MemoryInput):
    return invoke(lambda: memory_service().save(MemoryEntry(**body.model_dump())).model_dump())


@router.put("/memory/{entry_id}")
def memory_edit(entry_id: str, body: MemoryInput):
    service = memory_service()

    def edit():
        old = next((entry for entry in service.list() if entry.id == entry_id), None)
        if old is None:
            raise ValueError("Memory not found")
        return service.save(old.model_copy(update=body.model_dump())).model_dump()

    return invoke(edit)


@router.post("/memory/{entry_id}/confirm")
def memory_confirm(entry_id: str):
    service = memory_service()

    def confirm():
        entry = next((entry for entry in service.list() if entry.id == entry_id), None)
        if entry is None:
            raise ValueError("Memory not found")
        return service.save(entry, confirm=True).model_dump()

    return invoke(confirm)


@router.delete("/memory/{entry_id}")
def memory_delete(entry_id: str):
    invoke(lambda: memory_service().delete(entry_id))
    return {"deleted": True}


@router.post("/memory/import-legacy")
def memory_import():
    legacy = Path(__file__).resolve().parents[2] / "data" / "synaptic" / "memory_vault.json"
    if get_settings().jobvis_data_dir:
        configured = Path(get_settings().jobvis_data_dir) / "memory_vault.json"
        if configured.exists():
            legacy = configured
    count = invoke(lambda: memory_service().import_legacy(legacy))
    return {"imported": count, "note": "Imported entries need confirmation. The original plaintext file still exists."}


@router.get("/memory/suggestions")
def memory_suggestions(question: str):
    return {"suggestions": invoke(lambda: [item.model_dump() for item in memory_service().search(question)])}


@router.post("/memory/import-resume")
def memory_import_resume():
    from job_scout.corpus import build_corpus

    snap = context()[2]
    count = invoke(lambda: memory_service().import_corpus(build_corpus(snap.cv_text)))
    return {"imported": count, "note": "Imported resume facts need confirmation before reuse."}


def current_pack():
    api, bridge, snap, identity = context()
    if bridge.run_status().get("running"):
        raise HTTPException(409, "Wait for the current run to finish")
    values = api.voice_bridge.checkpoint_values(snap.thread_id)
    pack = values.get("tailoring")
    ranked = next((item for item in values.get("ranked_jobs", []) if item.job.job_id == values.get("selected_job_id")), None)
    if pack is None or ranked is None:
        raise HTTPException(409, "Select and tailor a job first")
    pack_key = digest([ranked.model_dump(mode="json"), pack.model_dump(mode="json"), snap.cv_text])
    audit = api._pack_audit()
    pack_key = digest([pack_key, audit.get("hashes", {})])
    cv, _, letter, _ = api._render_paths()
    return identity, ranked, pack_key, {"cv_pdf": cv, "cover_letter_pdf": letter}, audit


@router.get("/pack/review")
def review_preview():
    _, ranked, key, _, audit = current_pack()
    return {"job_id": ranked.job.job_id, "pack_key": key, "audit": audit}


class ReviewInput(BaseModel):
    pack_key: str
    approved: bool = False
    acknowledge_uncertain: bool = False


@router.post("/pack/review")
def review_pack(body: ReviewInput):
    identity, ranked, key, artifacts, audit = current_pack()
    if key != body.pack_key:
        raise HTTPException(409, "Pack changed since preview; review again")
    row = invoke(
        lambda: HandoffStore().review(
            identity, ranked, key, artifacts, audit, approved=body.approved, acknowledge_uncertain=body.acknowledge_uncertain
        )
    )
    api = context()[0]
    record = api._APPLICATION_STORE.create_for_job(
        job_id=ranked.job.job_id,
        title=ranked.job.title,
        company=ranked.job.company,
        listing_url=ranked.job.listing_url or ranked.job.url,
        application_url=ranked.job.application_url or ranked.job.url,
        source=ranked.job.source,
    )
    record.asset_manifest_id = row["id"]
    record.transition("reviewed", "Approved durable application snapshot")
    api._APPLICATION_STORE.upsert(record)
    return row


@router.get("/packs/reviewed")
def reviewed_packs():
    return {"snapshots": invoke(lambda: HandoffStore().list(context()[3]))}


class ExportInput(BaseModel):
    snapshot_ids: list[str]
    acknowledge_uncertain: bool = False


@router.post("/packs/export")
def export_packs(body: ExportInput):
    api, bridge, _, identity = context()
    if bridge is not None and bridge.run_status().get("running"):
        raise HTTPException(409, "Wait for the current run before exporting")
    result = invoke(lambda: HandoffStore().export(identity, body.snapshot_ids, acknowledge_uncertain=body.acknowledge_uncertain))
    for record in api._APPLICATION_STORE.list():
        if record.asset_manifest_id in result["snapshot_ids"] and not any(
            event.name == "exported" and event.detail == result["id"] for event in record.events
        ):
            record.events.append(ApplicationEvent(name="exported", detail=result["id"]))
            api._APPLICATION_STORE.upsert(record)
    return result


@router.get("/packs/exports/{export_id}/queue")
def download_export(export_id: str):
    from fastapi.responses import FileResponse

    identity = context()[3]
    exported = invoke(lambda: HandoffStore().read()["exports"].get(export_id))
    if not exported or exported["candidate_id"] != identity:
        raise HTTPException(404, "Export not found")
    return FileResponse(exported["queue_path"], media_type="application/json", filename="aihawk-queue.json")
