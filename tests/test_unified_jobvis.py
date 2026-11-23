import json
from unittest.mock import Mock

from fastapi.testclient import TestClient

from job_scout import api, candidate_store, daily
from job_scout.application.store import ApplicationStore
from job_scout.application.tracker_import import import_tracker
from job_scout.graph.schemas import RankedJob, SourceDiagnostic
from job_scout.tools.jobs_api import SearchResult
from job_scout.voice import bridge as voice_bridge
from tests.conftest import make_job


def test_tracker_import_preview_dedupe_and_backup(tmp_path):
    store = ApplicationStore(tmp_path)
    content = (
        "id,company,title,job_url,status,applied_date,notes\n"
        "one,Acme,ML Engineer,https://board.example/jobs?id=1&utm_source=a,Applied,2025-01-01,Keep this note\n"
        "two,Acme,ML Engineer,https://board.example/jobs?id=2,Found,,\n"
    )
    preview = import_tracker(store, "sheets", content)
    assert preview["created"] == 2 and not store.path.exists()
    applied = import_tracker(store, "sheets", content, preview["preview_id"])
    old = (tmp_path / "tracker-backups" / applied["backup"]).read_text()
    restored = ApplicationStore(tmp_path / "restored")
    daily.write_private_json(restored.path, json.loads(old))
    assert restored.list() == []
    weaker = """| # | Company | Role | URL | Status |
|---|---|---|---|---|
| 1 | Acme | ML Engineer | https://board.example/jobs?id=1 | Evaluated |
"""
    preview = import_tracker(store, "career-ops", weaker)
    assert preview["matched"] == 1
    import_tracker(store, "career-ops", weaker, preview["preview_id"])
    preview = import_tracker(store, "sheets", content)
    import_tracker(store, "sheets", content, preview["preview_id"])
    assert len(store.list()) == 2
    submitted = store.list()[0]
    assert submitted.status == "submitted_by_user"
    assert submitted.submitted_at == "2025-01-01T00:00:00+00:00"
    assert submitted.notes == "Keep this note"
    assert submitted.source_ids == {"sheets": ["one"], "career-ops": ["1"]}


def test_daily_pipeline_and_interview_routes(tmp_path, monkeypatch, sample_profile):
    store = ApplicationStore(tmp_path)
    monkeypatch.setattr(api, "_APPLICATION_STORE", store)
    monkeypatch.setenv("JOBVIS_DATA_DIR", str(tmp_path))
    candidate_store.save_candidate(sample_profile, "Test CV")
    client = TestClient(api.create_app())
    legacy = {
        "source": "base44",
        "content": '[{"id":"a","role":"Old role","company":"Example","status":"Applied","applied_date":"2025-01-01"},'
        '{"id":"b","role":"Other role","company":"Example","status":"Found"},'
        '{"id":"seed","role":"AI/ML Research Intern (Summer 2026)","company":"DeepMind"}]',
    }
    preview = client.post("/api/applications/import/preview", json=legacy).json()
    assert preview["created"] == 2 and preview["skipped"] == 1
    assert client.post("/api/applications/import", json={**legacy, "preview_id": "stale"}).status_code == 409
    assert client.post("/api/applications/import", json={**legacy, "preview_id": preview["preview_id"]}).status_code == 200
    first, second = store.list()
    first_job = make_job("one", "ML Engineer", "Acme", "lever").model_copy(update={"url": "https://example.com/jobs?id=1"})
    second_job = first_job.model_copy(update={"job_id": "two", "url": "https://example.com/jobs?id=2"})
    alias = first_job.model_copy(update={"job_id": "alias", "url": "https://example.com/jobs?id=1&utm_source=feed"})
    search = Mock(
        return_value=SearchResult(
            jobs=[first_job, alias, second_job],
            diagnostics=[SourceDiagnostic(source="adzuna", requested=True, error="timeout")],
        )
    )
    monkeypatch.setattr(daily, "run_search_detailed", search)
    monkeypatch.setattr(
        daily,
        "rank_jobs",
        lambda state: {
            "ranked_jobs": [RankedJob(job=job, fit_score=80, fit_explanation="Matches profile") for job in state["jobs"]],
        },
    )
    response = client.post("/api/today/run")
    assert response.status_code == 200, response.text
    today = response.json()
    assert [row["job"]["job_id"] for row in today["run"]["ranked"]] == ["one", "two"]
    assert "adzuna: timeout" in today["digest"]
    assert len(today["followups"]) == 2
    due = today["followups"][0]["due"]
    assert client.post(f"/api/applications/{first.application_id}/followup", json={"due": due}).status_code == 200
    assert len(client.post("/api/today/run").json()["followups"]) == 1
    assert search.call_count == 2
    bridge = voice_bridge.VoiceBridge()
    bridge.register_thread("today-integration")
    bridge.record_profile(sample_profile, "Test CV", "today-integration", [])
    tailor = Mock(return_value=None)
    graph = Mock()
    monkeypatch.setattr(voice_bridge, "_BRIDGE", bridge)
    monkeypatch.setattr(bridge, "start_tailoring", tailor)
    monkeypatch.setattr(voice_bridge, "checkpoint_values", lambda _: {})
    monkeypatch.setattr(voice_bridge, "get_compiled_graph", lambda: graph)
    response = client.post("/api/today/jobs/one/tailor")
    assert response.status_code == 200, response.text
    tailor.assert_called_once_with("one")
    assert graph.update_state.call_args.args[1]["ranked_jobs"][0].job.job_id == "one"

    response = client.post("/api/interview-prep/stories", json={"title": "Research pipeline", "action": "Built it"})
    assert response.status_code == 200
    story = next(iter(response.json()["stories"]))
    for record, notes in ((first, "Round one"), (second, "Technical screen")):
        response = client.put(
            f"/api/applications/{record.application_id}/interview-prep",
            json={
                "notes": notes,
                "story_ids": [story],
            },
        )
        assert response.status_code == 200
    assert [record.interview_notes for record in store.list()[:2]] == ["Round one", "Technical screen"]
    assert all(record.interview_story_ids == [story] for record in store.list()[:2])


def test_digest_delivery_is_idempotent(tmp_path, monkeypatch):
    shortlist = daily.DailyShortlist(tmp_path, zone="America/New_York")
    daily.write_private_json(shortlist.path, {"ranked": [], "diagnostics": [], "delivery": {"status": "preview"}})
    for key, value in {
        "JOBVIS_SMTP_HOST": "smtp.example.com",
        "JOBVIS_SMTP_FROM": "sender@example.com",
        "JOBVIS_DIGEST_EMAIL": "owner@example.com",
    }.items():
        monkeypatch.setenv(key, value)
    smtp = Mock()
    factory = Mock()
    factory.return_value.__enter__ = Mock(return_value=smtp)
    factory.return_value.__exit__ = Mock(return_value=None)
    monkeypatch.setattr(daily.smtplib, "SMTP_SSL", factory)
    assert shortlist.send()["status"] == "sent"
    assert shortlist.send()["status"] == "sent"
    assert smtp.send_message.call_count == 1
