"""Node behaviour with mocked LLMs and search."""

from __future__ import annotations

import job_scout.graph.nodes.fetch_jobs as fetch_mod
import job_scout.graph.nodes.rank_jobs as rank_mod
import job_scout.graph.nodes.reformulate_query as reformulate_mod
from job_scout.graph.nodes.fetch_jobs import fetch_jobs
from job_scout.graph.nodes.rank_jobs import rank_jobs
from job_scout.graph.nodes.reformulate_query import reformulate_query
from job_scout.graph.schemas import CandidatePreferences, JobScore, JobScores, SearchRequest
from tests.conftest import make_job, plain_llm, structured_llm, tool_calling_llm


def test_fetch_jobs_uses_llm_tool_args(monkeypatch, sample_profile, sample_jobs):
    llm = tool_calling_llm([{"name": "search_jobs", "args": {"query": "ml engineer", "country": "de", "remote": True}}])
    monkeypatch.setattr(fetch_mod, "get_chat_model", lambda *a, **k: llm)
    captured = {}

    def fake_run_search(query, location, country, remote, limit):
        captured.update(query=query, country=country, remote=remote)
        return sample_jobs, ["adzuna"]

    monkeypatch.setattr(fetch_mod, "run_search", fake_run_search)
    out = fetch_jobs({"profile": sample_profile, "llm_calls": 1})
    assert captured == {"query": "ml engineer", "country": "de", "remote": True}
    assert out["jobs"] == sample_jobs
    assert out["jobs_sources"] == ["adzuna"]
    assert out["search_query"] == "ml engineer"
    assert out["llm_calls"] == 2


def test_fetch_jobs_uses_structured_args_for_groq(monkeypatch, sample_profile, sample_jobs):
    """Groq fetches avoid the fragile LangChain function-call schema."""
    llm = structured_llm(SearchRequest(query="ml engineer", country="de", remote=True))
    monkeypatch.setattr(fetch_mod, "get_chat_model", lambda *a, **k: llm)
    monkeypatch.setattr(
        fetch_mod,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "scout_fetch_model": "groq:qwen/qwen3.6-27b",
                "scout_model": "groq:qwen/qwen3.6-27b",
                "scout_fallback_models": "",
                "max_llm_calls_per_run": 25,
                "scout_max_jobs": 10,
            },
        )(),
    )
    captured = {}

    def fake_run_search(query, location, country, remote, limit):
        captured.update(query=query, country=country, remote=remote)
        return sample_jobs, ["adzuna"]

    monkeypatch.setattr(fetch_mod, "run_search", fake_run_search)
    out = fetch_jobs({"profile": sample_profile, "llm_calls": 0})
    assert captured == {"query": "ml engineer", "country": "de", "remote": True}
    assert out["jobs"] == sample_jobs
    assert out["llm_calls"] == 1


def test_fetch_jobs_uses_explicit_fallback_after_provider_failure(monkeypatch, sample_profile, sample_jobs):
    """The legacy query-selector path honors the same failover chain as ranking."""
    primary = structured_llm(None)
    primary.with_structured_output.return_value.invoke.side_effect = RuntimeError("429 rate limit")
    fallback = tool_calling_llm([{"name": "search_jobs", "args": {"query": "ml engineer", "country": "us"}}])
    models = iter([primary, fallback])
    monkeypatch.setattr(fetch_mod, "get_chat_model", lambda *args, **kwargs: next(models))
    monkeypatch.setattr(
        fetch_mod,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "scout_fetch_model": "groq:primary",
                "scout_model": "groq:primary",
                "scout_fallback_models": "nvidia:fallback",
                "max_llm_calls_per_run": 4,
                "scout_max_jobs": 10,
            },
        )(),
    )
    monkeypatch.setattr(fetch_mod, "run_search", lambda **kwargs: (sample_jobs, ["cache"]))

    out = fetch_jobs({"profile": sample_profile, "llm_calls": 0})

    assert out["search_query"] == "ml engineer"
    assert out["llm_calls"] == 2
    assert any("primary" in error and "rate limit" in error for error in out["errors"])


def test_fetch_jobs_falls_back_to_profile_query_when_all_models_fail(monkeypatch, sample_profile, sample_jobs):
    """Fetch-model failure must not discard an otherwise usable source search."""
    failing = structured_llm(None)
    failing.with_structured_output.return_value.invoke.side_effect = RuntimeError("model unavailable")
    failing.bind_tools.return_value.invoke.side_effect = RuntimeError("model unavailable")
    monkeypatch.setattr(fetch_mod, "get_chat_model", lambda *args, **kwargs: failing)
    monkeypatch.setattr(
        fetch_mod,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "scout_fetch_model": "groq:retired",
                "scout_model": "groq:retired",
                "scout_fallback_models": "nvidia:also-unavailable",
                "max_llm_calls_per_run": 4,
                "scout_max_jobs": 10,
            },
        )(),
    )
    captured = {}

    def fake_run_search(**kwargs):
        captured.update(kwargs)
        return sample_jobs, ["cache"]

    monkeypatch.setattr(fetch_mod, "run_search", fake_run_search)

    out = fetch_jobs({"profile": sample_profile, "llm_calls": 0})

    assert out["jobs"] == sample_jobs
    assert captured["query"] == "Data Scientist ML Engineer"
    assert out["llm_calls"] == 2
    assert any("profile-derived query" in error for error in out["errors"])


def test_fetch_jobs_trims_keyword_soup_to_a_title(monkeypatch, sample_profile, sample_jobs):
    """A long query is cut to title length, and the trim is recorded.

    Boards match the query against posting titles, so the soup both models used
    to produce matched nothing and the cascade fell through to the remote-only
    board. The prompt asks for a title; this guarantees one.
    """
    soup = "Senior Data Scientist AI Engineer deep learning neural networks LLMs RAG systems"
    llm = tool_calling_llm([{"name": "search_jobs", "args": {"query": soup, "country": "de", "remote": False}}])
    monkeypatch.setattr(fetch_mod, "get_chat_model", lambda *a, **k: llm)
    captured = {}

    def fake_run_search(query, location, country, remote, limit):
        captured["query"] = query
        return sample_jobs, ["adzuna"]

    monkeypatch.setattr(fetch_mod, "run_search", fake_run_search)
    out = fetch_jobs({"profile": sample_profile, "llm_calls": 0})

    assert captured["query"] == "Senior Data Scientist AI Engineer deep"
    assert out["search_query"] == captured["query"]
    # Silently trimming would hide exactly the signal worth seeing in the trace.
    assert any("query trimmed" in e and "learning neural networks" in e for e in out["errors"])


def test_fetch_jobs_leaves_a_title_query_alone(monkeypatch, sample_profile, sample_jobs):
    llm = tool_calling_llm([{"name": "search_jobs", "args": {"query": "senior data scientist", "country": "de"}}])
    monkeypatch.setattr(fetch_mod, "get_chat_model", lambda *a, **k: llm)
    monkeypatch.setattr(fetch_mod, "run_search", lambda **k: (sample_jobs, ["adzuna"]))
    out = fetch_jobs({"profile": sample_profile, "llm_calls": 0})
    assert out["search_query"] == "senior data scientist"
    assert not any("trimmed" in e for e in out["errors"])


def test_fetch_jobs_no_tool_call_fallback(monkeypatch, sample_profile, sample_jobs):
    llm = tool_calling_llm([])  # model issued no tool call
    monkeypatch.setattr(fetch_mod, "get_chat_model", lambda *a, **k: llm)
    monkeypatch.setattr(fetch_mod, "run_search", lambda **k: (sample_jobs, ["cache"]))
    out = fetch_jobs({"profile": sample_profile, "llm_calls": 0})
    assert any("no tool call" in e for e in out["errors"])
    assert out["jobs"] == sample_jobs


def test_fetch_jobs_country_scope_overrides_model_location_and_remote(monkeypatch, sample_profile, sample_jobs):
    llm = tool_calling_llm([{"name": "search_jobs", "args": {"query": "ml engineer", "country": "de", "remote": False}}])
    monkeypatch.setattr(fetch_mod, "get_chat_model", lambda *a, **k: llm)
    captured = {}

    def fake_run_search(query, location, country, remote, limit):
        captured.update(query=query, location=location, country=country, remote=remote)
        return sample_jobs, ["cache"]

    monkeypatch.setattr(fetch_mod, "run_search", fake_run_search)
    profile = sample_profile.model_copy(update={"locations": ["Anywhere in the United States"], "remote_ok": True})
    fetch_jobs({"profile": profile, "llm_calls": 0})

    assert captured == {"query": "ml engineer", "location": None, "country": "us", "remote": True}


def test_fetch_jobs_uses_role_family_fanout_for_candidate_preferences(monkeypatch, sample_profile, sample_jobs):
    queries = []

    def fake_run_search(query, location, country, remote, limit):
        queries.append(query)
        return [sample_jobs[0].model_copy(update={"job_id": query, "title": query})], ["cache"]

    monkeypatch.setattr(fetch_mod, "run_search", fake_run_search)
    prefs = CandidatePreferences(country_scope="us", primary_role_families=["ai_ml", "genai"])
    out = fetch_jobs({"profile": sample_profile, "candidate_preferences": prefs, "llm_calls": 0})
    assert set(queries) == {"Applied ML Engineer", "AI/ML Engineer", "GenAI Engineer", "LLM RAG Engineer"}
    assert out["llm_calls"] == 0
    assert len(out["jobs"]) == 4


def test_default_candidate_queries_include_forward_deployed_roles():
    from job_scout.graph.nodes.fetch_jobs import _candidate_queries

    prefs = CandidatePreferences()
    queries = _candidate_queries(None, prefs, 0, max_queries=6)
    assert "Forward Deployed Engineer" in queries


def test_candidate_search_does_not_filter_hybrid_or_onsite_when_all_modes_are_accepted(monkeypatch, sample_profile, sample_jobs):
    captured = []

    def fake_run_search(**kwargs):
        captured.append(kwargs)
        return [sample_jobs[0]], ["cache"]

    monkeypatch.setattr(fetch_mod, "run_search", fake_run_search)
    out = fetch_jobs(
        {
            "profile": sample_profile,
            "candidate_preferences": CandidatePreferences(primary_role_families=["ai_ml"]),
            "llm_calls": 0,
        }
    )
    assert out["jobs"]
    assert captured and all(item["remote"] is False for item in captured)


def test_candidate_role_fanout_honors_global_job_cap(monkeypatch, sample_profile, sample_jobs):
    """Multiple role queries must not bypass the ranking budget."""

    def fake_run_search(query, location, country, remote, limit):
        return [
            sample_jobs[0].model_copy(
                update={"job_id": f"{query}-{i}", "title": f"{query} {i}", "url": f"https://example.com/{query}/{i}"}
            )
            for i in range(3)
        ], ["cache"]

    monkeypatch.setattr(fetch_mod, "run_search", fake_run_search)
    monkeypatch.setattr(
        fetch_mod,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "scout_max_jobs": 6,
                "scout_max_role_queries": 6,
                "scout_query_concurrency": 2,
                "max_llm_calls_per_run": 25,
            },
        )(),
    )
    prefs = CandidatePreferences(country_scope="us", primary_role_families=["ai_ml", "genai"])
    out = fetch_jobs({"profile": sample_profile, "candidate_preferences": prefs, "llm_calls": 0})
    assert len(out["jobs"]) == 6


def test_candidate_role_fanout_does_not_spend_or_require_an_llm_call(monkeypatch, sample_profile, sample_jobs):
    monkeypatch.setattr(fetch_mod, "run_search", lambda **k: ([sample_jobs[0]], ["cache"]))
    monkeypatch.setattr(
        fetch_mod,
        "get_settings",
        lambda: type(
            "S",
            (),
            {"scout_max_jobs": 3, "scout_max_role_queries": 1, "scout_query_concurrency": 1, "max_llm_calls_per_run": 0},
        )(),
    )
    prefs = CandidatePreferences(country_scope="us", primary_role_families=["ai_ml"])
    out = fetch_jobs({"profile": sample_profile, "candidate_preferences": prefs, "llm_calls": 0})
    assert out["llm_calls"] == 0


def test_rank_jobs_batches_by_five(monkeypatch, sample_profile):
    jobs = [make_job(f"j{i}", f"Role {i}", f"Co{i}") for i in range(7)]
    calls = []

    def fake_model(*a, **k):
        llm = structured_llm(None)

        def invoke(prompt):
            # return a score for whichever ids appear in this batch prompt
            ids = [j.job_id for j in jobs if f"job_id: {j.job_id}\n" in prompt]
            calls.append(len(ids))
            return JobScores(scores=[JobScore(job_id=i, fit_score=80, fit_explanation="ok") for i in ids])

        llm.with_structured_output.return_value.invoke.side_effect = invoke
        return llm

    monkeypatch.setattr(rank_mod, "get_chat_model", fake_model)
    out = rank_jobs({"profile": sample_profile, "jobs": jobs, "llm_calls": 2})
    assert len(calls) == 2  # 7 jobs -> batches of 5 + 2
    assert out["llm_calls"] == 4  # 2 + 2 batches
    assert len(out["ranked_jobs"]) == 7


def test_rank_jobs_uses_explicit_fallback_after_provider_failure(monkeypatch, sample_profile):
    jobs = [make_job("j1", "Data Scientist", "Acme")]
    primary = structured_llm(None)
    primary.with_structured_output.return_value.invoke.side_effect = RuntimeError("429 rate limit")
    fallback = structured_llm(JobScores(scores=[JobScore(job_id="j1", fit_score=82, fit_explanation="fallback")]))
    models = iter([primary, fallback])
    monkeypatch.setattr(rank_mod, "get_chat_model", lambda *args, **kwargs: next(models))
    monkeypatch.setattr(
        rank_mod,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "scout_model": "groq:primary",
                "scout_fallback_models": "nvidia:fallback",
                "scout_rank_batch": 4,
                "scout_rank_timeout": 45.0,
                "max_llm_calls_per_run": 4,
            },
        )(),
    )

    out = rank_jobs({"profile": sample_profile, "jobs": jobs, "llm_calls": 0})

    assert out["ranked_jobs"][0].fit_score == 82
    assert out["llm_calls"] == 2


def test_rank_jobs_keeps_postings_when_all_providers_fail(monkeypatch, sample_profile):
    """A rate limit must degrade to transparent local scores, not zero results."""
    failing = structured_llm(None)
    failing.with_structured_output.return_value.invoke.side_effect = RuntimeError("429 rate limit")
    monkeypatch.setattr(rank_mod, "get_chat_model", lambda *args, **kwargs: failing)
    monkeypatch.setattr(
        rank_mod,
        "get_settings",
        lambda: type(
            "S",
            (),
            {
                "scout_model": "groq:primary",
                "scout_fallback_models": "nvidia:fallback",
                "scout_rank_batch": 4,
                "scout_rank_timeout": 45.0,
                "max_llm_calls_per_run": 4,
            },
        )(),
    )
    jobs = [make_job("j1", "Data Scientist", "Acme")]

    out = rank_jobs({"profile": sample_profile, "jobs": jobs, "llm_calls": 0})

    assert len(out["ranked_jobs"]) == 1
    assert "Deterministic review score" in out["ranked_jobs"][0].fit_explanation
    assert any("providers unavailable" in error for error in out["errors"])
    assert out["llm_calls"] == 2


def test_rank_jobs_empty_jobs(sample_profile):
    out = rank_jobs({"profile": sample_profile, "jobs": [], "llm_calls": 0})
    assert out["ranked_jobs"] == []


def test_rank_jobs_applies_deterministic_candidate_policy(monkeypatch, sample_profile):
    jobs = [
        make_job("primary", "Data Scientist", "Acme").model_copy(
            update={"description": "Full-time onsite role starting January 2027; Python and SQL."}
        ),
        make_job("intern", "Data Scientist Intern", "Acme").model_copy(
            update={"description": "Summer internship requiring Python."}
        ),
    ]
    model = structured_llm(
        JobScores(
            scores=[
                JobScore(job_id="primary", fit_score=90, fit_explanation="strong evidence", matched_skills=["python"]),
                JobScore(job_id="intern", fit_score=95, fit_explanation="intern fit", matched_skills=["python"]),
            ]
        )
    )
    monkeypatch.setattr(rank_mod, "get_chat_model", lambda *a, **k: model)
    prefs = CandidatePreferences()
    out = rank_jobs({"profile": sample_profile, "jobs": jobs, "candidate_preferences": prefs, "llm_calls": 0})
    by_id = {item.job.job_id: item for item in out["ranked_jobs"]}
    assert by_id["primary"].eligibility_status in {"eligible", "borderline"}
    assert by_id["intern"].eligibility_status == "blocked"
    assert by_id["intern"].fit_score < by_id["primary"].fit_score


def test_reformulate_increments_counter(monkeypatch, sample_profile):
    monkeypatch.setattr(reformulate_mod, "get_chat_model", lambda *a, **k: plain_llm("data analyst"))
    state = {"profile": sample_profile, "search_query": "data scientist", "reformulation_count": 0, "llm_calls": 3}
    out = reformulate_query(state)
    assert out["search_query"] == "data analyst"
    assert out["reformulation_count"] == 1
    assert out["llm_calls"] == 4
    assert out["llm_calls"] == 4


def test_rank_jobs_skips_already_scored(monkeypatch, sample_profile):
    """Reformulation loops must not re-spend LLM calls on jobs already scored."""
    from job_scout.graph.schemas import RankedJob

    jobs = [make_job(f"j{i}", f"Role {i}", f"Co{i}") for i in range(4)]
    prior = [
        RankedJob(job=jobs[0], fit_score=90, fit_explanation="kept"),
        RankedJob(job=jobs[1], fit_score=40, fit_explanation="kept"),
    ]
    scored_ids = []

    def fake_model(*a, **k):
        llm = structured_llm(None)

        def invoke(prompt):
            ids = [j.job_id for j in jobs if f"job_id: {j.job_id}\n" in prompt]
            scored_ids.extend(ids)
            return JobScores(scores=[JobScore(job_id=i, fit_score=70, fit_explanation="new") for i in ids])

        llm.with_structured_output.return_value.invoke.side_effect = invoke
        return llm

    monkeypatch.setattr(rank_mod, "get_chat_model", fake_model)
    out = rank_jobs({"profile": sample_profile, "jobs": jobs, "ranked_jobs": prior, "llm_calls": 0})
    assert scored_ids == ["j2", "j3"]  # only the new postings hit the model
    assert out["llm_calls"] == 1
    assert len(out["ranked_jobs"]) == 4
    assert out["ranked_jobs"][0].fit_score == 90  # prior scores kept, list re-sorted


def test_rank_jobs_all_already_scored_is_free(sample_profile):
    from job_scout.graph.schemas import RankedJob

    jobs = [make_job("j1", "Role", "Co")]
    prior = [RankedJob(job=jobs[0], fit_score=75, fit_explanation="kept")]
    out = rank_jobs({"profile": sample_profile, "jobs": jobs, "ranked_jobs": prior, "llm_calls": 3})
    assert out["ranked_jobs"] == prior  # no model construction, no llm_calls key needed


def test_fetch_jobs_limit_from_settings(monkeypatch, sample_profile, sample_jobs):
    monkeypatch.setenv("SCOUT_MAX_JOBS", "3")
    from job_scout.config import get_settings

    get_settings.cache_clear()
    seen = {}

    def fake_run_search(**kwargs):
        seen.update(kwargs)
        return sample_jobs, ["cache"]

    monkeypatch.setattr(fetch_mod, "run_search", fake_run_search)
    monkeypatch.setattr(fetch_mod, "get_chat_model", lambda *a, **k: tool_calling_llm([{"args": {"query": "ds"}}]))
    fetch_jobs({"profile": sample_profile, "llm_calls": 0})
    assert seen["limit"] == 3


def test_rank_jobs_batches_run_in_parallel(monkeypatch, sample_profile):
    """Two batches must overlap in time — ranking latency is max(batch), not sum."""
    import time as _time

    jobs = [make_job(f"j{i}", f"Role {i}", f"Co{i}") for i in range(10)]

    def fake_model(*a, **k):
        llm = structured_llm(None)

        def invoke(prompt):
            _time.sleep(0.3)
            ids = [j.job_id for j in jobs if f"job_id: {j.job_id}\n" in prompt]
            return JobScores(scores=[JobScore(job_id=i, fit_score=70, fit_explanation="ok") for i in ids])

        llm.with_structured_output.return_value.invoke.side_effect = invoke
        return llm

    monkeypatch.setattr(rank_mod, "get_chat_model", fake_model)
    start = _time.monotonic()
    out = rank_jobs({"profile": sample_profile, "jobs": jobs, "llm_calls": 0})
    elapsed = _time.monotonic() - start
    assert len(out["ranked_jobs"]) == 10
    assert elapsed < 0.55, f"batches ran serially ({elapsed:.2f}s for 2×0.3s sleeps)"


def test_fetch_jobs_model_override(monkeypatch, sample_profile, sample_jobs):
    monkeypatch.setenv("SCOUT_FETCH_MODEL", "openai:tiny-model")
    from job_scout.config import get_settings

    get_settings.cache_clear()
    seen = {}

    def fake_get_chat_model(name, **kwargs):
        seen["model"] = name
        return tool_calling_llm([{"args": {"query": "ds"}}])

    monkeypatch.setattr(fetch_mod, "get_chat_model", fake_get_chat_model)
    monkeypatch.setattr(fetch_mod, "run_search", lambda **k: (sample_jobs, ["cache"]))
    fetch_jobs({"profile": sample_profile, "llm_calls": 0})
    assert seen["model"] == "openai:tiny-model"
