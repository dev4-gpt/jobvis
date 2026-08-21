import httpx

from job_scout.tools.direct_sources import AshbySource, GreenhouseSource, LeverSource, USAJobsSource
from job_scout.tools.jobs_api import run_search_detailed


def test_greenhouse_maps_canonical_listing_and_application_urls(respx_mock):
    respx_mock.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": 42,
                        "title": "Applied ML Engineer",
                        "content": "Full-time Python role",
                        "location": {"name": "Remote - United States"},
                        "absolute_url": "https://boards.greenhouse.io/acme/jobs/42",
                    }
                ]
            },
        )
    )
    jobs = GreenhouseSource(["acme"]).fetch("ML", None, "us", True, 5)
    assert jobs[0].source_record_id == "42"
    assert jobs[0].listing_url == jobs[0].application_url == jobs[0].url
    assert jobs[0].content_hash


def test_direct_sources_are_injectable_without_enabling_network(monkeypatch, respx_mock):
    monkeypatch.setenv("JOBVIS_DIRECT_SOURCES_ENABLED", "false")
    respx_mock.get("https://api.lever.co/v0/postings/acme").mock(return_value=httpx.Response(200, json=[]))
    result = run_search_detailed(
        "data scientist",
        lever=LeverSource(["acme"]),
    )
    assert any(d.source == "lever" for d in result.diagnostics)


def test_malformed_ashby_payload_degrades_to_empty(respx_mock):
    respx_mock.post("https://api.ashbyhq.com/posting-api/job-board/acme").mock(
        return_value=httpx.Response(200, json={"jobs": {"unexpected": True}})
    )
    assert AshbySource(["acme"]).fetch("ML", None, "us", False, 5) == []


def test_usajobs_requires_explicit_credentials():
    assert USAJobsSource().available is False
