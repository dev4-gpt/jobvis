"""Behavioral integration contracts with no external providers or personal stores."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from job_scout.application import security
from job_scout.application.handoff import HandoffStore, file_hash
from job_scout.application.models import ApplicationRecord
from job_scout.application.store import ApplicationStore
from job_scout.config import get_settings
from job_scout.graph.schemas import JobPosting, RankedJob, SourceDiagnostic
from job_scout.memory.service import MemoryEntry, MemoryService
from job_scout.tools import liveness
from job_scout.tools.aihawk_exporter import export_aihawk_queue, format_aihawk_job
from job_scout.tools.apify_client import ApifyTaskError, ConfiguredApifyTask
from job_scout.tools.direct_sources import GreenhouseSource


def posting(identifier="job-1", **changes):
    return JobPosting(
        job_id=identifier,
        title="Junior AI Engineer",
        company="Example",
        location="New York, US",
        source="greenhouse",
        url="https://jobs.example.com/1",
        **changes,
    )


@pytest.fixture
def encrypted(monkeypatch):
    cipher = Fernet(Fernet.generate_key())
    monkeypatch.setattr(security, "_fernet", lambda: cipher)
    return cipher


def test_registry_defaults_explicit_override_and_policy(monkeypatch):
    monkeypatch.setenv("JOBVIS_DIRECT_SOURCES_ENABLED", "true")
    monkeypatch.setenv("GREENHOUSE_BOARD_TOKENS", "")
    get_settings.cache_clear()
    assert "anthropic" in get_settings().greenhouse_boards
    from job_scout.tools.portals_registry import PortalsRegistry

    assert PortalsRegistry().filter_title("Junior AI Software Engineer")
    monkeypatch.setenv("GREENHOUSE_BOARD_TOKENS", "chosen,chosen")
    get_settings.cache_clear()
    assert get_settings().greenhouse_boards == ["chosen"]


def test_board_fetch_shared_across_queries(monkeypatch):
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        time.sleep(0.01)
        return httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": 1,
                        "title": "Junior AI Engineer",
                        "content": "Python ML",
                        "location": {"name": "New York, US"},
                        "absolute_url": "https://jobs.example.com/1",
                    }
                ]
            },
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx, "get", get)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda query: GreenhouseSource(["chosen"]).fetch(query, None, "us", False, 10), ["AI", "ML"]))
    assert all(results)
    assert len(calls) == 1
    assert not results[0][0].remote


@pytest.mark.parametrize("code", [401, 403, 429, 500, 503])
def test_failed_http_is_not_expired_or_active(code):
    assert liveness.classify_liveness(code, body_text="Apply now")["status"] == "uncertain"


def test_search_page_precedes_apply_and_shell_uncertain():
    assert liveness.classify_liveness(body_text="12 jobs found. Apply now")["status"] == "expired"
    assert liveness.classify_liveness(body_text="Loading app…")["status"] == "uncertain"


def test_liveness_cache_and_forced_recheck(monkeypatch):
    check = Mock(return_value={"liveness": "active", "reason": "button"})
    monkeypatch.setattr(liveness, "check_job_liveness", check)
    liveness.verified_liveness("https://jobs.example.com/1")
    liveness.verified_liveness("https://jobs.example.com/1")
    assert check.call_count == 1
    liveness.verified_liveness("https://jobs.example.com/1", force=True)
    assert check.call_count == 2


def test_liveness_budget_keeps_unfinished_candidates(monkeypatch):
    gate = threading.Event()

    def check(url, **kwargs):
        gate.wait(0.3)
        return {"liveness": "active", "reason": "button"}

    monkeypatch.setattr(liveness, "verified_liveness", check)
    diagnostic = SourceDiagnostic(source="greenhouse")
    start = time.monotonic()
    try:
        result = liveness.filter_live_jobs([posting()], [diagnostic], budget=0.01)
        assert time.monotonic() - start < 0.2
        assert result[0].liveness["liveness"] == "uncertain"
        assert diagnostic.incomplete_checks == 1
    finally:
        gate.set()


def test_encrypted_memory_review_restart_isolation_edit_delete(tmp_path, encrypted):
    service = MemoryService("candidate-a", tmp_path)
    entry = service.save(MemoryEntry(content="Python projects", question_key="experience", sensitive=True))
    assert b"Python projects" not in service.path.read_bytes()
    assert service.search("Python") == []
    service.save(entry, confirm=True)
    assert MemoryService("candidate-a", tmp_path).search("Python")[0].sensitive
    assert MemoryService("candidate-b", tmp_path).list() == []
    service.save(entry.model_copy(update={"content": "Rust projects"}))
    assert service.search("Rust") == []
    service.delete(entry.id)
    assert service.list() == []


def test_memory_encryption_failure_does_not_write_plaintext(tmp_path, monkeypatch):
    monkeypatch.setattr(security, "_fernet", Mock(side_effect=security.BrowserSecurityError("locked")))
    service = MemoryService("a", tmp_path)
    with pytest.raises(security.BrowserSecurityError):
        service.save(MemoryEntry(content="secret"))
    assert not service.path.exists()


def test_explicit_legacy_import_is_unconfirmed_and_preserves_source(tmp_path, encrypted):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps([{"content": "Python engineer", "verified": True}]))
    service = MemoryService("a", tmp_path)
    assert service.import_legacy(legacy) == 1
    assert service.import_legacy(legacy) == 0
    assert not service.list()[0].confirmed
    assert service.search("Python") == []
    assert legacy.exists()


def task_settings():
    return SimpleNamespace(
        apify_api_token=SecretStr("secret-token"),
        apify_task_id="saved-task",
        apify_input_template=json.dumps({"search": "$query", "limit": "$limit"}),
        apify_output_mapping=json.dumps({"title": "job.title", "company": "company", "url": "url", "remote": "remote"}),
    )


def test_apify_task_uses_explicit_mapping_and_only_search_input():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/runs"):
            return httpx.Response(201, json={"data": {"id": "run1", "status": "SUCCEEDED", "defaultDatasetId": "data1"}})
        return httpx.Response(
            200,
            json=[{"job": {"title": "AI Engineer"}, "company": "Example", "url": "https://example.com/job", "remote": "false"}],
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    task = ConfiguredApifyTask(task_settings())
    jobs = task.run(
        {"query": "AI Engineer", "location": "US", "country": "us", "remote": False, "limit": 25, "cv_text": "PRIVATE"},
        client=client,
    )
    assert len(jobs) == 1 and jobs[0].remote is False
    assert task.state["run_id"] == "run1"
    assert "PRIVATE" not in requests[0].content.decode()
    assert "secret-token" not in str(requests[0].url)


def test_apify_timeout_attempts_abort(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"data": {"id": "run1", "status": "RUNNING"}})

    task = ConfiguredApifyTask(task_settings())
    with httpx.Client(transport=httpx.MockTransport(handler)) as client, pytest.raises(ApifyTaskError):
        task.run({"query": "AI", "location": "", "country": "us", "remote": False, "limit": 25}, client=client, deadline=0.01)
    assert any(request.url.path.endswith("/run1/abort") for request in requests)
    assert task.state["cancellation_requested"]


def test_apify_configuration_invalid_without_mapping():
    settings = task_settings()
    settings.apify_output_mapping = "{}"
    with pytest.raises(ApifyTaskError):
        ConfiguredApifyTask(settings).configuration()


@pytest.fixture
def reviewed(tmp_path, monkeypatch):
    monkeypatch.setattr("job_scout.application.handoff.verified_liveness", lambda *a, **k: {"liveness": "active"})
    artifacts = {"cv_pdf": tmp_path / "cv.pdf", "cover_letter_pdf": tmp_path / "letter.pdf"}
    for path in artifacts.values():
        path.write_bytes(b"%PDF-verified-fixture")
    ranked = RankedJob(job=posting(), fit_score=90, fit_explanation="match", eligibility_status="eligible")
    store = HandoffStore(tmp_path)
    audit = {
        "passed": True,
        "hashes": {"tailored_cv": file_hash(artifacts["cv_pdf"]), "cover_letter": file_hash(artifacts["cover_letter_pdf"])},
    }
    row = store.review("candidate", ranked, "generation-1", artifacts, audit, approved=True)
    return store, row, ranked, artifacts


def test_handoff_export_schema_idempotence_durable_artifacts(reviewed):
    store, row, _, artifacts = reviewed
    result = store.export("candidate", [row["id"]])
    for path in artifacts.values():
        path.unlink()
    assert store.export("candidate", [row["id"]]) == result
    queue = json.loads(__import__("pathlib").Path(result["queue_path"]).read_text())
    assert queue["jobs"][0]["id"] == "001"
    assert queue["jobs"][0]["score"] == 4.5
    assert HandoffStore(store.root.parent).list("candidate")[0]["approved_at"]


def test_handoff_stale_and_changed_artifacts_rejected(reviewed):
    store, row, _, _ = reviewed
    store.invalidate(row["job_id"])
    with pytest.raises(ValueError, match="stale"):
        store.export("candidate", [row["id"]])


def test_handoff_requires_acknowledgment_and_candidate_scope(reviewed, monkeypatch):
    store, row, _, _ = reviewed
    with pytest.raises(ValueError):
        store.export("different-candidate", [row["id"]])
    monkeypatch.setattr("job_scout.application.handoff.verified_liveness", lambda *a, **k: {"liveness": "uncertain"})
    with pytest.raises(ValueError, match="acknowledgment"):
        store.export("candidate", [row["id"]])
    assert store.export("candidate", [row["id"]], acknowledge_uncertain=True)


@pytest.mark.parametrize("score,expected", [(0, 0), (4, 0.2), (5, 0.25), (70, 3.5), (100, 5)])
def test_score_scale_is_explicit(score, expected):
    assert format_aihawk_job("001", "Example", "AI", score)["score"] == expected


def test_invalid_bridge_id_and_nan_are_rejected():
    with pytest.raises(Exception):
        export_aihawk_queue([{"id": "greenhouse-123", "company": "Example", "role": "AI", "score": 90}])
    with pytest.raises(ValueError):
        format_aihawk_job("001", "Example", "AI", float("nan"))


def test_export_checks_audit_and_bundle_tampering(reviewed):
    from pathlib import Path

    store, row, ranked, artifacts = reviewed
    with pytest.raises(ValueError, match="audit hashes"):
        store.review("candidate", ranked, "bad", artifacts, {"passed": True}, approved=True)
    result = store.export("candidate", [row["id"]])
    Path(result["queue_path"]).write_text("{}")
    with pytest.raises(ValueError, match="queue changed"):
        store.export("candidate", [row["id"]])


def test_export_capacity_never_recycles_ids(reviewed):
    from job_scout.application.handoff import atomic_json

    store, row, _, _ = reviewed
    data = store.read()
    data["ids"] = {f"old-{index}": f"{index:03d}" for index in range(1, 1000)}
    atomic_json(store.path, data)
    with pytest.raises(ValueError, match="capacity"):
        store.export("candidate", [row["id"]])
    assert len(store.read()["ids"]) == 999


def test_export_rejects_artifact_change_during_copy(reviewed, monkeypatch):
    from pathlib import Path

    store, row, _, _ = reviewed
    monkeypatch.setattr("job_scout.application.handoff.shutil.copyfile", lambda source, dest: Path(dest).write_bytes(b"changed"))
    with pytest.raises(ValueError, match="changed during export"):
        store.export("candidate", [row["id"]])
    assert store.read()["exports"] == {}


def test_keychain_failure_has_safe_visible_error(monkeypatch):
    import keyring

    monkeypatch.setattr(keyring, "get_password", Mock(side_effect=RuntimeError("provider internals")))
    with pytest.raises(security.BrowserSecurityError, match="check Keychain access"):
        security._fernet()


def test_resume_memory_only_repeats_current_corpus_evidence(tmp_path, encrypted):
    from job_scout.corpus import CandidateCorpus, CorpusItem

    corpus = CandidateCorpus(
        items=[CorpusItem(id="cv1", text="Built Python pipelines", kind="bullet", source="cv", section="Work")]
    )
    service = MemoryService("candidate", tmp_path)
    assert service.import_corpus(corpus) == 1
    assert service.grounded_context(corpus) == ""
    entry = service.list()[0]
    service.save(entry, confirm=True)
    assert "Built Python pipelines" in service.grounded_context(corpus)
    service.save(entry.model_copy(update={"content": "Invented leadership claim"}), confirm=True)
    assert service.grounded_context(corpus) == ""


def test_run_coordinator_refuses_duplicate_expansion():
    from job_scout.voice.bridge import VoiceBridge

    bridge = VoiceBridge()
    gate = threading.Event()
    finished = threading.Event()

    def work(cancelled):
        gate.wait(2)
        finished.set()

    try:
        assert bridge.start_expansion(work) is None
        assert "busy" in bridge.start_expansion(work)
    finally:
        gate.set()
        assert finished.wait(2)


def test_ordinary_search_never_runs_apify(monkeypatch):
    from job_scout.tools.jobs_api import run_search_detailed

    run = Mock(side_effect=AssertionError("ordinary search started Apify"))
    monkeypatch.setattr(ConfiguredApifyTask, "run", run)
    disabled = SimpleNamespace(available=False)
    source = SimpleNamespace(fetch=lambda *args: [posting()])
    run_search_detailed("AI", jsearch=disabled, adzuna=disabled, remotive=source, cache=source)
    run.assert_not_called()


def test_expansion_endpoint_merges_and_caps_current_search(monkeypatch, sample_profile):
    import importlib

    import job_scout.graph as graph_module
    from job_scout import integrations_api as routes
    from job_scout.graph.schemas import CandidatePreferences
    from job_scout.memory.service import candidate_key
    from job_scout.voice.bridge import VoiceBridge

    rank_module = importlib.import_module("job_scout.graph.nodes.rank_jobs")
    bridge = VoiceBridge()
    bridge.record_profile(sample_profile, "CV", "thread")
    values = {"profile": sample_profile, "cv_text": "CV", "jobs": [], "candidate_preferences": CandidatePreferences()}
    fake_api = SimpleNamespace(
        voice_bridge=SimpleNamespace(checkpoint_values=lambda thread: dict(values)), preferences_from_dict=lambda value: value
    )
    identity = candidate_key("CV", sample_profile.name)
    monkeypatch.setattr(routes, "context", lambda: (fake_api, bridge, bridge.snapshot(), identity))
    fake_task = ConfiguredApifyTask(task_settings())
    jobs = [
        posting(f"apify-{index}").model_copy(
            update={"title": f"AI role {index}", "source": "apify", "url": f"https://jobs.example.com/{index}"}
        )
        for index in range(25)
    ]
    fake_task.run = Mock(return_value=jobs)
    monkeypatch.setattr(routes, "ConfiguredApifyTask", lambda: fake_task)
    monkeypatch.setattr(
        rank_module, "rank_jobs", lambda state: {"jobs": state["jobs"], "source_diagnostics": state["source_diagnostics"]}
    )
    completed = threading.Event()

    def update(config, changes):
        values.update(changes)
        completed.set()

    monkeypatch.setattr(graph_module, "get_compiled_graph", lambda: SimpleNamespace(update_state=update))
    assert routes.expand_search()["started"]
    assert completed.wait(2)
    assert len(values["jobs"]) == get_settings().scout_max_jobs
    assert values["jobs_sources"] == ["apify"]
    assert values["source_diagnostics"][0].contributed
    fake_task.run.assert_called_once()


def test_followup_dates_and_repeated_status():
    record = ApplicationRecord(job_id="one")
    record.transition("submitted_by_user")
    due = record.followup_due[:]
    record.transition("submitted_by_user")
    assert record.followup_due == due and len(due) == 2
    record.transition("interview")
    assert record.thank_you_due


def test_console_review_export_and_memory_routes(reviewed, encrypted, monkeypatch):
    from job_scout import integrations_api as routes

    store, row, ranked, artifacts = reviewed
    monkeypatch.setattr(routes, "HandoffStore", lambda: store)
    fake_api = SimpleNamespace(_APPLICATION_STORE=ApplicationStore(store.root.parent))
    monkeypatch.setattr(routes, "context", lambda: (fake_api, None, None, "candidate"))
    monkeypatch.setattr(routes, "current_pack", lambda: ("candidate", ranked, "generation-1", artifacts, row["audit"]))
    monkeypatch.setattr(routes, "memory_service", lambda: MemoryService("candidate", store.root.parent))
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        assert client.post("/api/pack/review", json={"pack_key": "old", "approved": True}).status_code == 409
        approved = client.post("/api/pack/review", json={"pack_key": "generation-1", "approved": True})
        assert approved.status_code == 200
        exported = client.post("/api/packs/export", json={"snapshot_ids": [approved.json()["id"]]})
        assert exported.status_code == 200
        assert fake_api._APPLICATION_STORE.list()[0].status == "reviewed"
        assert fake_api._APPLICATION_STORE.list()[0].events[-1].name == "exported"
        saved = client.post("/api/memory", json={"content": "Python projects", "question_key": "experience"}).json()
        assert saved["confirmed"] is False
        assert client.post(f"/api/memory/{saved['id']}/confirm").status_code == 200
        assert client.get("/api/memory/suggestions?question=Python").json()["suggestions"]
        assert client.delete(f"/api/memory/{saved['id']}").status_code == 200
        assert client.get("/api/memory/suggestions?question=Python").json()["suggestions"] == []
