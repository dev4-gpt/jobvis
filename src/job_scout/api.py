"""The browser console's server side: tokens, tools, state, events, downloads.

Jobvis moved into the browser. The conversation itself now runs there over
WebRTC — which is what buys real barge-in and the browser's own echo
cancellation — but nothing the agent may *know* moved with it. The client tools
it calls are still the Python functions in ``voice.tools``, reading the same
LangGraph checkpoint the wizard reads; the browser only forwards the call. That
split is the point: a new modality, the same grounding contract.

The ElevenLabs key never leaves this process. The browser asks for a
short-lived conversation token, and that is the only credential it ever holds.

Routes, all under ``/api``: ``config`` (what the console may assume),
``voice/token``, ``tools/{name}``, ``state``, ``events`` (SSE), and the four
pack downloads (CV and cover letter as PDF or LaTeX). Everything else is the
built console, served from ``web/out``.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

import httpx
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from job_scout import candidate_store
from job_scout.application.controller import get_application_controller
from job_scout.candidate_fit import assess_eligibility, preferences_from_dict
from job_scout.config import get_settings
from job_scout.evals.artifacts import audit_generated_pack
from job_scout.evals.pack_loop import render_verified_pack
from job_scout.graph.schemas import JobPosting, Profile, RankedJob, TailoringPack
from job_scout.outreach import ContactCandidate, PublicContactSource, build_outreach_draft, discover_public_contacts
from job_scout.outreach_store import OutreachStore
from job_scout.renderer import render_cover_letter_pdf, render_pdf
from job_scout.tools.jobs_api import run_search_detailed
from job_scout.voice import bridge as voice_bridge
from job_scout.voice import is_voice_available
from job_scout.voice.announce import run_announcement
from job_scout.voice.persona import greeting_variables
from job_scout.voice.tools import CLIENT_TOOL_HANDLERS
from job_scout.watchlists import Watchlist, WatchlistStore

TOKEN_URL = "https://api.elevenlabs.io/v1/convai/conversation/token"  # noqa: S105 - an endpoint, not a credential

# The console is a static export, built into web/out. JOBVIS_WEB_DIR overrides
# it (a different build output, or a checkout that is not the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_APPLICATION = get_application_controller()
_APPLICATION_STORE = _APPLICATION.store
_WATCHLIST_STORE = WatchlistStore()
_OUTREACH_STORE = OutreachStore()


def web_dir() -> Path:
    return Path(os.environ.get("JOBVIS_WEB_DIR") or _REPO_ROOT / "web" / "out")


def ensure_session() -> None:
    """Claim a thread for this console and seed the saved candidate into it.

    The bridge is process-wide, so when the wizard already registered a thread
    this is a no-op and the console simply views the same session.
    """
    bridge = voice_bridge.get_bridge()
    if bridge.snapshot().thread_id:
        return
    thread_id = str(uuid4())
    bridge.register_thread(thread_id)
    stored = candidate_store.load_candidate()
    if stored is not None:
        profile = candidate_store.effective_profile(stored.profile, stored.preferences)
        bridge.record_profile(profile, stored.cv_text, thread_id, stored.cv_links)


def _job_row(ranked: RankedJob, rank: int) -> dict:
    row = {
        "rank": rank,
        "job_id": ranked.job.job_id,
        "title": ranked.job.title,
        "company": ranked.job.company,
        "location": ranked.job.location,
        "url": ranked.job.url,
        "fit_score": ranked.fit_score,
        "why": ranked.fit_explanation,
    }
    has_policy = (
        ranked.final_priority_score
        or ranked.role_fit_score
        or ranked.evidence_fit_score
        or ranked.primary_or_adjacent != "review"
    )
    if has_policy:
        row.update(
            listing_url=ranked.job.listing_url or ranked.job.source_url or ranked.job.url,
            application_url=ranked.job.application_url or ranked.job.url,
            final_priority_score=ranked.final_priority_score or ranked.fit_score,
            role_fit_score=ranked.role_fit_score,
            evidence_fit_score=ranked.evidence_fit_score,
            eligibility_status=ranked.eligibility_status,
            eligibility_reasons=ranked.eligibility_reasons,
            hard_blockers=ranked.hard_blockers,
            primary_or_adjacent=ranked.primary_or_adjacent,
            start_timing_fit=ranked.start_timing_fit,
            employment_type=ranked.job.employment_type,
            work_mode=ranked.job.work_mode,
            experience_level=ranked.job.experience_level,
            posted_at=ranked.job.posted_at,
            start_date_text=ranked.job.start_date_text,
            clearance_required=ranked.job.clearance_required,
            authorization_requirement=ranked.job.authorization_requirement,
            sponsorship_signal=ranked.job.sponsorship_signal,
            salary_text=ranked.job.salary_text,
            source_url=ranked.job.source_url or ranked.job.url,
            source=ranked.job.source,
        )
    return row


def _verdict(flags: int) -> str:
    if flags == 0:
        return "Every claim checked against the CV — no flags."
    plural = "s" if flags != 1 else ""
    return f"{flags} statement{plural} could not be verified — review before sending."


def _preference_payload(value: object) -> dict | None:
    if value is None:
        return None
    return preferences_from_dict(value if isinstance(value, dict) else value).model_dump(mode="json")


def _pack_payload(values: dict) -> dict | None:
    pack: TailoringPack | None = values.get("tailoring")
    if pack is None:
        return None
    flags = int(values.get("fabrication_flags") or 0)
    return {
        "job_id": str(values.get("selected_job_id") or ""),
        "headline": pack.cv.headline,
        "summary": pack.cv.summary,
        "cover_letter": pack.cover_letter,
        "honesty_note": pack.honesty_note,
        "links": [link.model_dump() for link in pack.cv.links],
        "flags": flags,
        "quality_issue_codes": list(values.get("tailor_issue_codes") or []),
        "artifact_manifest": _RENDER_CACHE.get("manifest"),
        "verdict": _verdict(flags),
        "cover_letter_quality": values.get("cover_letter_quality").model_dump()
        if values.get("cover_letter_quality") is not None
        else None,
    }


def _candidate_payload(profile: Profile | None) -> dict | None:
    if profile is None:
        return None
    return {
        "name": profile.name or "",
        "role": (profile.primary_roles or [""])[0],
        "seniority": profile.seniority,
        "locations": list(profile.locations),
        "remote_ok": profile.remote_ok,
        "phone": profile.phone,
        "professional_experience_months": profile.professional_experience_months,
        "education_history": [entry.model_dump(mode="json") for entry in profile.education_history],
        "expected_graduation_date": profile.expected_graduation_date.isoformat() if profile.expected_graduation_date else None,
        "current_program": profile.current_program,
        "degree_fields": profile.degree_fields,
    }


def current_state() -> dict:
    """Everything the console paints, read straight from the bridge + checkpoint."""
    ensure_session()
    bridge = voice_bridge.get_bridge()
    snap = bridge.snapshot()
    values = voice_bridge.checkpoint_values(snap.thread_id)
    ranked = list(values.get("ranked_jobs") or [])
    primary = [r for r in ranked if r.primary_or_adjacent == "primary" and r.eligibility_status != "blocked"]
    adjacent = [r for r in ranked if r.primary_or_adjacent == "adjacent" and r.eligibility_status != "blocked"]
    blocked = [r for r in ranked if r.eligibility_status == "blocked" or r.primary_or_adjacent == "review"]
    ranked_rows = [_job_row(r, i) for i, r in enumerate(ranked, 1)]
    return {
        "step": snap.step,
        "thread_id": snap.thread_id,
        "candidate": _candidate_payload(snap.profile),
        # Keep `jobs` for older console clients, but expose explicit lanes so
        # an adjacent or review-only role cannot look like a primary match.
        "jobs": ranked_rows[:5],
        "primary_jobs": [
            row
            for row in ranked_rows
            if row.get("primary_or_adjacent") == "primary" and row.get("eligibility_status") != "blocked"
        ],
        "adjacent_jobs": [
            row
            for row in ranked_rows
            if row.get("primary_or_adjacent") == "adjacent" and row.get("eligibility_status") != "blocked"
        ],
        "blocked_or_review_jobs": [
            row for row in ranked_rows if row.get("eligibility_status") == "blocked" or row.get("primary_or_adjacent") == "review"
        ],
        "candidate_preferences": _preference_payload(values.get("candidate_preferences")),
        "source_coverage": {
            "fetched": len(values.get("jobs") or []),
            "ranked": len(ranked),
            "primary": len(primary),
            "adjacent": len(adjacent),
            "blocked_or_review": len(blocked),
            # ``jobs`` is the raw fetch output (``JobPosting`` objects); the
            # ranked output below is the separate ``RankedJob`` collection.
            "sources": sorted({job.source for job in values.get("jobs") or []}),
            "diagnostics": [diagnostic.model_dump(mode="json") for diagnostic in values.get("source_diagnostics") or []],
        },
        "pack": _pack_payload(values),
        "run": bridge.run_status(include_lifecycle=True),
        "application": _APPLICATION.state.public(),
    }


# One render per pack, keyed by its contents, so repeated download clicks do
# not recompile the same LaTeX twice.
_RENDER_CACHE: dict = {
    "key": None,
    "cv_pdf": None,
    "cv_tex": None,
    "letter_pdf": None,
    "letter_tex": None,
    "manifest": None,
}


def _render_paths() -> tuple[Path | None, Path | None, Path | None, Path | None]:
    ensure_session()
    bridge = voice_bridge.get_bridge()
    snap = bridge.snapshot()
    values = voice_bridge.checkpoint_values(snap.thread_id)
    pack: TailoringPack | None = values.get("tailoring")
    if pack is None:
        return None, None, None, None
    key = hash(
        (
            str(values.get("selected_job_id") or ""),
            pack.cover_letter,
            pack.cv.model_dump_json(),
            tuple(str(item) for item in values.get("cv_links") or pack.cv.links),
            values.get("tailor_backtest_score"),
        )
    )
    if _RENDER_CACHE["key"] != key:
        name = (snap.profile.name if snap.profile else None) or "Candidate"
        out_dir = Path(tempfile.mkdtemp(prefix="job_scout_render_"))
        selected = next(
            (item for item in values.get("ranked_jobs") or [] if item.job.job_id == values.get("selected_job_id")),
            None,
        )
        source_text = str(values.get("cv_text") or "")
        source_links = list(values.get("cv_links") or pack.cv.links)
        if source_text or source_links or selected:
            verified = render_verified_pack(
                pack,
                candidate_name=name,
                source_text=source_text,
                source_links=source_links,
                job_description=selected.job.description if selected else "",
                company=selected.job.company if selected else "your team",
                job_title=selected.job.title if selected else "this role",
                out_dir=out_dir,
                selected_job_id=str(values.get("selected_job_id") or ""),
                generation_id=snap.thread_id,
                backtest_score=values.get("tailor_backtest_score"),
            )
            cv_result = verified.cv
            letter_result = verified.cover_letter
            manifest = verified.manifest.as_dict()
        else:
            cv_result = render_pdf(pack.cv, name, out_dir)
            letter_result = render_cover_letter_pdf(pack.cover_letter, name, out_dir)
            manifest = {
                "status": "withheld",
                "cv_pdf": None,
                "cv_tex": str(cv_result.tex_path) if cv_result.tex_path else None,
                "cover_letter_pdf": None,
                "cover_letter_tex": str(letter_result.tex_path) if letter_result.tex_path else None,
                "pdfs_ready": False,
                "issue_codes": ["missing_audit_context"],
            }
        _RENDER_CACHE.update(
            key=key,
            cv_pdf=cv_result.pdf_path,
            cv_tex=cv_result.tex_path,
            letter_pdf=letter_result.pdf_path,
            letter_tex=letter_result.tex_path,
            manifest=manifest,
        )
    return (
        _RENDER_CACHE["cv_pdf"],
        _RENDER_CACHE["cv_tex"],
        _RENDER_CACHE["letter_pdf"],
        _RENDER_CACHE["letter_tex"],
    )


def _pack_audit() -> dict:
    """Run the no-LLM artifact gate against the current checkpoint pack."""
    ensure_session()
    bridge = voice_bridge.get_bridge()
    snap = bridge.snapshot()
    values = voice_bridge.checkpoint_values(snap.thread_id)
    pack: TailoringPack | None = values.get("tailoring")
    if pack is None:
        raise HTTPException(status_code=404, detail="no tailored pack to audit")
    cv_pdf, cv_tex, letter_pdf, letter_tex = _render_paths()
    selected = next(
        (item for item in values.get("ranked_jobs") or [] if item.job.job_id == values.get("selected_job_id")),
        None,
    )
    report = audit_generated_pack(
        str(values.get("cv_text") or ""),
        snap.cv_links or pack.cv.links,
        cv_pdf or "",
        cover_letter_pdf=letter_pdf,
        cover_letter_text=pack.cover_letter,
        job_description=selected.job.description if selected else "",
        cv_tex=cv_tex,
        cover_letter_tex=letter_tex,
    )
    return report.as_dict()


def _require_pack_audit() -> None:
    """Prevent a failed rendered pack from being presented as ready."""
    report = _pack_audit()
    if not report["passed"]:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "pack failed deterministic artifact checks; download the .tex fallback and review the report",
                "audit": report,
            },
        )


def _jsonable(value: Any, path: str = "") -> Any:
    """Coerce a payload to something json.dumps accepts, loudly.

    One unserializable leaf used to raise straight out of the SSE generator,
    which kills the connection — the console then sits there looking fine while
    every live update silently stops. A degraded frame beats a dead stream, so
    the offender becomes its repr and says where it was.
    """
    if isinstance(value, dict):
        return {k: _jsonable(v, f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    print(f"jobvis: {path or 'payload'} is not JSON-serializable ({type(value).__name__}); sending its repr")
    return repr(value)


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(_jsonable(payload, event))}\n\n"


async def event_stream(disconnected: Callable[[], Awaitable[bool]], poll: float = 0.25) -> AsyncIterator[str]:
    """Finished runs and run-status changes, as server-sent events.

    A voice-triggered search returns to the agent immediately and lands a
    minute later; this stream is how the agent finds out. The console replays
    `run_finished` as a user message, which is the only thing that makes Jobvis
    actually speak, and `context` as a contextual update, which is silent by
    design — the wizard's screen events should inform the agent, not interrupt
    it.

    Takes its disconnect check as an argument rather than reading the Request,
    so it can be driven directly in tests: TestClient buffers a whole response
    body before returning, and an SSE stream never has one.
    """
    bridge = voice_bridge.get_bridge()
    feed = bridge.subscribe()
    last_run: str | None = None
    idle = 0
    try:
        while not await disconnected():
            forced = False
            try:
                event = feed.get_nowait()
            except queue.Empty:
                pass
            else:
                if event.kind == "run_finished" and event.run is not None:
                    yield _sse("run_finished", {"kind": event.run.kind, "text": run_announcement(event.run)})
                elif event.kind == "screen":
                    yield _sse("context", {"text": event.text})
                # Any event means the checkpoint may have moved. Run status alone
                # is not enough to notice: a search started by CLICKING in the
                # wizard never touches the run manager, so without this the
                # console would sit on "say find me jobs" while the wizard is
                # already showing ranked results.
                forced = True
            snapshot = json.dumps(bridge.run_status(), sort_keys=True)
            if forced or snapshot != last_run:
                last_run = snapshot
                yield _sse("state", current_state())
                idle = 0
            idle += 1
            if idle >= 60:  # ~15s of quiet: keep proxies from reaping the connection
                idle = 0
                yield ": heartbeat\n\n"
            await asyncio.sleep(poll)
    finally:
        bridge.unsubscribe(feed)


def create_app() -> FastAPI:
    app = FastAPI(title="Jobvis", docs_url=None, redoc_url=None)

    # `npm run dev` serves the console from :3000 against this API. The built
    # console is same-origin, so this only ever matters in development.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/config")
    def config() -> dict:
        """What the console may assume before it renders anything."""
        voice_ok, hint = is_voice_available()
        return {
            "voice_ok": voice_ok,
            "voice_hint": hint,
            "wizard_url": f"http://localhost:{WIZARD_PORT}",
            "has_candidate": candidate_store.load_candidate() is not None,
        }

    @app.post("/api/voice/token")
    def voice_token() -> dict:
        """Mint a short-lived conversation token. The API key stays here."""
        voice_ok, hint = is_voice_available()
        if not voice_ok:
            raise HTTPException(status_code=503, detail=hint)
        settings = get_settings()
        try:
            response = httpx.get(
                TOKEN_URL,
                params={"agent_id": settings.elevenlabs_agent_id},
                headers={"xi-api-key": settings.elevenlabs_api_key.get_secret_value()},
                timeout=15,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"ElevenLabs unreachable: {exc}") from exc
        if response.status_code != 200:
            # 401 here almost always means the key lacks the Agents scopes.
            raise HTTPException(status_code=502, detail=f"token request failed ({response.status_code}): {response.text[:200]}")
        token = response.json().get("token")
        if not token:
            raise HTTPException(status_code=502, detail="ElevenLabs returned no token")
        # The greeting is a template (persona.FIRST_MESSAGE). Unfilled, the agent
        # opens the conversation by reading "{{part_of_day}}" out loud, so these
        # ride along with the token rather than being the console's to invent.
        # ensure_session() first: on a cold visit the stored candidate has not
        # been seeded yet, and "Good morning." is a sadder greeting than
        # "Good morning, Shirin."
        ensure_session()
        snap = voice_bridge.get_bridge().snapshot()
        name = snap.profile.name if snap.profile else None
        return {"token": token, "dynamic_variables": greeting_variables(name, datetime.now().hour)}

    @app.get("/api/voice/last-error")
    def voice_last_error() -> dict:
        """Why the last conversation died, in ElevenLabs' own words.

        The browser SDK reports a bare "Server error: Unknown error" for every
        failure, including the one readers will hit first — running out of free
        conversation minutes, which arrives as code 1002 with a perfectly clear
        reason attached. Asking after the fact is the only way to say something
        useful, so the console calls this when a session drops.
        """
        settings = get_settings()
        if not settings.elevenlabs_agent_id:
            return {"reason": "", "quota": False}
        headers = {"xi-api-key": settings.elevenlabs_api_key.get_secret_value()}
        try:
            listing = httpx.get(
                "https://api.elevenlabs.io/v1/convai/conversations",
                headers=headers,
                params={"agent_id": settings.elevenlabs_agent_id, "page_size": 1},
                timeout=10,
            )
            conversations = listing.json().get("conversations") or []
            if not conversations or conversations[0].get("status") != "failed":
                return {"reason": "", "quota": False}
            detail = httpx.get(
                f"https://api.elevenlabs.io/v1/convai/conversations/{conversations[0]['conversation_id']}",
                headers=headers,
                timeout=10,
            )
            reason = str(detail.json().get("metadata", {}).get("termination_reason") or "")
        except (httpx.HTTPError, ValueError, KeyError):
            return {"reason": "", "quota": False}
        return {"reason": reason, "quota": "quota" in reason.lower()}

    @app.post("/api/tools/{name}")
    def call_tool(name: str, parameters: Annotated[dict | None, Body()] = None) -> dict:
        """Run one client tool. Defined sync on purpose: FastAPI gives it a
        worker thread, and these handlers read the checkpoint synchronously."""
        handler = CLIENT_TOOL_HANDLERS.get(name)
        if handler is None:
            raise HTTPException(status_code=404, detail=f"unknown tool: {name}")
        ensure_session()
        return handler(parameters or {})

    @app.post("/api/run/cancel")
    def cancel_run(parameters: Annotated[dict | None, Body()] = None) -> dict:
        """Request cancellation of the current voice/background run."""
        return voice_bridge.get_bridge().cancel_run(str((parameters or {}).get("run_id") or "") or None)

    @app.get("/api/state")
    def state() -> dict:
        # Same guard as the SSE frames: the console losing its state endpoint is
        # a worse failure than one field arriving as a repr.
        return _jsonable(current_state(), "state")

    @app.get("/api/application/state")
    def application_state() -> dict:
        return _APPLICATION.state.public()

    @app.get("/api/applications")
    def applications() -> dict:
        """Return the local tracker only; no employer site is contacted."""
        return {"applications": [record.model_dump(mode="json") for record in _APPLICATION_STORE.list()]}

    @app.get("/api/outreach")
    def outreach_drafts() -> dict:
        """Return locally saved outreach drafts; no message is sent."""
        return {"drafts": [draft.model_dump(mode="json") for draft in _OUTREACH_STORE.list()]}

    @app.post("/api/contacts/discover")
    def discover_contacts(parameters: Annotated[dict | None, Body()] = None) -> dict:
        """Extract explicit contacts from user-approved public page text.

        Jobvis does not scrape LinkedIn, guess email patterns, or contact a
        person. The caller must provide the page URL and the text they chose to
        import or approve.
        """
        body = parameters or {}
        raw_sources = body.get("sources") or []
        if not isinstance(raw_sources, list) or not raw_sources:
            raise HTTPException(status_code=400, detail="provide one or more user-approved public sources")
        try:
            sources = [PublicContactSource.model_validate(item) for item in raw_sources]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid contact source: {exc}") from exc
        contacts = discover_public_contacts(sources)
        return {
            "contacts": [contact.model_dump(mode="json") for contact in contacts],
            "warning": "Verify each contact and employer policy manually. Jobvis never sends outreach.",
        }

    @app.post("/api/outreach/generate")
    def generate_outreach(parameters: Annotated[dict | None, Body()] = None) -> dict:
        """Generate a grounded email/video draft for a checkpointed job pack."""
        ensure_session()
        bridge = voice_bridge.get_bridge()
        snap = bridge.snapshot()
        values = voice_bridge.checkpoint_values(snap.thread_id)
        body = parameters or {}
        job_id = str(body.get("job_id") or "")
        ranked = next((item for item in values.get("ranked_jobs") or [] if item.job.job_id == job_id), None)
        pack = values.get("tailoring")
        if ranked is None:
            raise HTTPException(status_code=404, detail="ranked job not found in the current checkpoint")
        if pack is None:
            raise HTTPException(status_code=409, detail="tailor the selected job before generating outreach")
        try:
            contact = ContactCandidate.model_validate(body["contact"]) if body.get("contact") else None
            draft = build_outreach_draft(ranked, snap.profile, pack, contact) if snap.profile else None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid contact: {exc}") from exc
        if draft is None:
            raise HTTPException(status_code=409, detail="candidate profile is not available")
        _OUTREACH_STORE.save(draft)
        return draft.model_dump(mode="json")

    @app.post("/api/applications/{application_id}/status")
    def application_status(application_id: str, parameters: Annotated[dict | None, Body()] = None) -> dict:
        body = parameters or {}
        status = str(body.get("status") or "")
        if status == "submitted":
            status = "submitted_by_user"
        try:
            record = _APPLICATION_STORE.mark(application_id, status, str(body.get("detail") or ""))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if record is None:
            raise HTTPException(status_code=404, detail="application record not found")
        return record.model_dump(mode="json")

    @app.post("/api/jobs/import")
    def import_job(parameters: Annotated[dict | None, Body()] = None) -> dict:
        """Import a user-selected listing; this never crawls controlled boards."""
        body = parameters or {}
        url = str(body.get("application_url") or body.get("listing_url") or body.get("url") or "").strip()
        if not url.startswith(("https://", "http://")):
            raise HTTPException(status_code=400, detail="a user-provided HTTP(S) job URL is required")
        job_id = str(body.get("job_id") or f"manual-{uuid4()}")
        posting = JobPosting(
            job_id=job_id,
            title=str(body.get("title") or "Imported job"),
            company=str(body.get("company") or "Unknown"),
            location=str(body.get("location") or "Unspecified"),
            description=str(body.get("description") or ""),
            url=url,
            listing_url=str(body.get("listing_url") or url),
            application_url=str(body.get("application_url") or url),
            source="manual",
            employment_type=str(body.get("employment_type") or "unknown"),
            work_mode=str(body.get("work_mode") or "unknown"),
            source_url=str(body.get("listing_url") or url),
            metadata_confidence=0.0,
        )
        bridge = voice_bridge.get_bridge()
        snap = bridge.snapshot()
        if snap.profile is not None and snap.thread_id:
            values = voice_bridge.checkpoint_values(snap.thread_id)
            assessment = assess_eligibility(posting, snap.profile, preferences_from_dict(values.get("candidate_preferences")))
            ranked = RankedJob(
                job=posting,
                fit_score=assessment.final_priority_score,
                final_priority_score=assessment.final_priority_score,
                role_fit_score=assessment.role_fit_score,
                evidence_fit_score=assessment.evidence_fit_score,
                fit_explanation="Imported by the user; review the posting and evidence before tailoring.",
                eligibility_status=assessment.status,
                eligibility_reasons=assessment.reasons,
                hard_blockers=assessment.hard_blockers,
                primary_or_adjacent=assessment.role_bucket,
                start_timing_fit=assessment.start_timing_fit,
            )
            voice_bridge.get_compiled_graph().update_state(
                {"configurable": {"thread_id": snap.thread_id}},
                {"jobs": [*(values.get("jobs") or []), posting], "ranked_jobs": [*(values.get("ranked_jobs") or []), ranked]},
            )
        record = _APPLICATION_STORE.create_for_job(
            job_id=job_id,
            title=str(body.get("title") or "Imported job"),
            company=str(body.get("company") or "Unknown"),
            listing_url=str(body.get("listing_url") or url),
            application_url=str(body.get("application_url") or url),
            source=str(body.get("source") or "manual"),
        )
        return record.model_dump(mode="json")

    @app.get("/api/watchlists")
    def watchlists() -> dict:
        return {"watchlists": [item.model_dump(mode="json") for item in _WATCHLIST_STORE.list()]}

    @app.post("/api/watchlists")
    def create_watchlist(parameters: Annotated[dict | None, Body()] = None) -> dict:
        body = parameters or {}
        query = str(body.get("query") or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="watchlist query is required")
        item = Watchlist(
            name=str(body.get("name") or query),
            query=query,
            location=str(body.get("location") or ""),
            country=str(body.get("country") or "us"),
            remote=bool(body.get("remote", False)),
            enabled=bool(body.get("enabled", True)),
        )
        return _WATCHLIST_STORE.create(item).model_dump(mode="json")

    @app.delete("/api/watchlists/{watchlist_id}")
    def delete_watchlist(watchlist_id: str) -> dict:
        if not _WATCHLIST_STORE.delete(watchlist_id):
            raise HTTPException(status_code=404, detail="watchlist not found")
        return {"deleted": True}

    @app.post("/api/watchlists/{watchlist_id}/refresh")
    def refresh_watchlist(watchlist_id: str) -> dict:
        item = next((candidate for candidate in _WATCHLIST_STORE.list() if candidate.watchlist_id == watchlist_id), None)
        if item is None:
            raise HTTPException(status_code=404, detail="watchlist not found")
        result = run_search_detailed(item.query, item.location or None, item.country, item.remote)
        _WATCHLIST_STORE.mark_refreshed(item)
        return {
            "watchlist": item.model_dump(mode="json"),
            "count": len(result.jobs),
            "sources": result.sources_used,
            "diagnostics": [d.model_dump(mode="json") for d in result.diagnostics],
        }

    @app.post("/api/application/open")
    def application_open(parameters: Annotated[dict | None, Body()] = None) -> dict:
        """Open an ATS application in the visible local browser; never submit."""
        ensure_session()
        bridge = voice_bridge.get_bridge()
        values = voice_bridge.checkpoint_values(bridge.snapshot().thread_id)
        job_id = str((parameters or {}).get("job_id") or "")
        ranked = next((item for item in values.get("ranked_jobs") or [] if item.job.job_id == job_id), None)
        if ranked is None or bridge.snapshot().profile is None:
            raise HTTPException(status_code=404, detail="ranked job not found in the current checkpoint")
        return _APPLICATION.open(ranked, bridge.snapshot().profile, bridge.snapshot().cv_links or [])

    @app.post("/api/application/inspect")
    def application_inspect(parameters: Annotated[dict | None, Body()] = None) -> dict:
        """Inspect supplied local fixture HTML; production flow reads the visible page."""
        body = parameters or {}
        return _APPLICATION.inspect_html(str(body.get("url") or "http://fixture.local"), str(body.get("html") or ""))

    @app.post("/api/application/fill-safe")
    def application_fill_safe(parameters: Annotated[dict | None, Body()] = None) -> dict:
        _require_pack_audit()
        approved = {str(item) for item in (parameters or {}).get("approved_field_ids", [])}
        cv_pdf, _, letter_pdf, _ = _render_paths()
        artifacts = {key: path for key, path in (("resume", cv_pdf), ("cover", letter_pdf)) if path is not None}
        return _APPLICATION.fill_safe(approved, artifacts)

    @app.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        return StreamingResponse(event_stream(request.is_disconnected), media_type="text/event-stream")

    @app.get("/api/pack/pdf")
    def pack_pdf() -> FileResponse:
        _require_pack_audit()
        pdf, _, _, _ = _render_paths()
        if pdf is None:
            raise HTTPException(status_code=404, detail="no tailored CV yet (tectonic missing? try the .tex)")
        return FileResponse(pdf, filename="tailored_cv.pdf", media_type="application/pdf")

    @app.get("/api/pack/tex")
    def pack_tex() -> FileResponse:
        _, tex, _, _ = _render_paths()
        if tex is None:
            raise HTTPException(status_code=404, detail="no tailored CV yet")
        return FileResponse(tex, filename="tailored_cv.tex", media_type="application/x-tex")

    @app.get("/api/pack/cover-letter/pdf")
    def cover_letter_pdf() -> FileResponse:
        _require_pack_audit()
        _, _, pdf, _ = _render_paths()
        if pdf is None:
            raise HTTPException(status_code=404, detail="no cover-letter PDF yet (tectonic missing? try the .tex)")
        return FileResponse(pdf, filename="cover_letter.pdf", media_type="application/pdf")

    @app.get("/api/pack/cover-letter/tex")
    def cover_letter_tex() -> FileResponse:
        _, _, _, tex = _render_paths()
        if tex is None:
            raise HTTPException(status_code=404, detail="no tailored cover letter yet")
        return FileResponse(tex, filename="cover_letter.tex", media_type="application/x-tex")

    @app.get("/api/pack/audit")
    def pack_audit() -> dict:
        """Return deterministic PDF/link/letter checks for the current pack."""
        return _pack_audit()

    @app.get("/api/pack/manifest")
    def pack_manifest() -> dict:
        """Return the authoritative download manifest and withheld reason."""
        _render_paths()
        manifest = _RENDER_CACHE.get("manifest")
        if manifest is None:
            raise HTTPException(status_code=404, detail="no tailored pack yet")
        return manifest

    console = web_dir()
    if (console / "index.html").exists():
        app.mount("/", StaticFiles(directory=console, html=True), name="console")
    else:
        print(f"jobvis: no built console at {console} — API only. Build it with `make web-build`.")
    return app


# The default matches the documented local setup.  Keep this configurable so
# Jobvis can coexist with another local service that already owns :8000.
CONSOLE_PORT = int(os.getenv("JOBVIS_CONSOLE_PORT", "8000"))
WIZARD_PORT = int(os.getenv("JOBVIS_WIZARD_PORT", "7860"))


def serve_in_thread(port: int = CONSOLE_PORT) -> None:
    """Run the console's API on a daemon thread, for callers that own the main one.

    The wizard and the console MUST live in one process: the bridge and the
    LangGraph ``MemorySaver`` are both process-wide, so two processes would mean
    two sessions wearing the same name — the wizard would find jobs the console
    could not see. `job_scout.app` starts this before it launches Gradio.
    """
    import threading

    import uvicorn

    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="jobvis-api").start()


def main() -> None:
    """API only, on the main thread — for frontend work with `make web-dev`.

    Note that a console served this way sees an EMPTY session: the wizard is not
    running, so nothing has claimed a thread or extracted a profile. Use
    `make app` for the real thing.
    """
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=CONSOLE_PORT)


if __name__ == "__main__":
    main()
