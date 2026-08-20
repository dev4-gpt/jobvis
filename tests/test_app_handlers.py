"""App handler wiring: on_find/on_tailor generators (Gradio-free, mocked runner)."""

from __future__ import annotations

from dataclasses import replace

import pytest

import job_scout.app as app_mod
from job_scout.app import _cv_preview_html, on_find, on_tailor, on_zip, reset
from job_scout.graph.schemas import CVContent, ExperienceEntry, RankedJob, TailoredBullet, TailoringPack
from job_scout.renderer import RenderResult
from job_scout.runner import RunResult, TailorResult
from tests.conftest import make_job


def _search_result() -> RunResult:
    ranked = [RankedJob(job=make_job("j1", "Data Scientist", "Acme"), fit_score=88, fit_explanation="fits")]
    return RunResult(ranked_jobs=ranked, jobs_sources=["cache"], n_jobs_ranked=1)


def _tailor_result(flags: int = 0) -> TailorResult:
    pack = TailoringPack(cv=CVContent(headline="DS", summary="s"), cover_letter="Dear team,", honesty_note="Gap: SQL.")
    return TailorResult(pack=pack, fabrication_flags=flags)


def _fake_stream(result):
    def stream(*args, **kwargs):
        yield ("status", "working…")
        yield ("result", result)

    return stream


@pytest.fixture(autouse=True)
def no_latex_compilation(monkeypatch):
    """Keep handler tests deterministic even when Tectonic is installed."""
    monkeypatch.setattr(
        app_mod,
        "render_pdf",
        lambda cv, name, out_dir: RenderResult(tex_path=out_dir / "tailored_cv.tex"),
    )
    monkeypatch.setattr(
        app_mod,
        "render_cover_letter_pdf",
        lambda letter, name, out_dir: RenderResult(tex_path=out_dir / "cover_letter.tex"),
    )


def test_on_find_populates_job_dropdown(monkeypatch, sample_profile):
    monkeypatch.setattr(app_mod, "stream_search", _fake_stream(_search_result()))
    final = list(on_find("cv text", sample_profile, "t1", []))[-1]
    select = final[4]
    assert select["choices"] == [("Data Scientist — Acme (fit 88)", "j1")]
    assert select["visible"] is True


def test_location_chooser_always_includes_us_scope(sample_profile):
    choices, _ = app_mod._preference_selection(sample_profile, None)
    assert app_mod.US_SCOPE_CHOICE in choices


def test_cv_preview_hides_internal_corpus_ids():
    pack = _tailor_result().pack
    pack.cv.experience = [
        ExperienceEntry(
            role="Data Scientist",
            company="Example",
            bullets=[TailoredBullet(text="Built a grounded model", corpus_ref="cv-bullet-001")],
        )
    ]
    html = _cv_preview_html(pack)
    assert "cv-bullet-001" not in html
    assert "verified evidence" in html


def test_target_policy_controls_are_authoritative(sample_profile):
    raw = app_mod.CandidatePreferences().model_dump(mode="json")
    prefs, typed = app_mod._preferences_from_ui(
        [app_mod.US_SCOPE_CHOICE],
        raw,
        ["full_time"],
        "2026-12-01",
        "2027-03-31",
        ["hybrid", "onsite"],
        ["ai_ml", "forward_deployed"],
        "unknown",
        "required",
        "confirmed",
    )
    assert typed is not None
    assert prefs["country_scope"] == "us"
    assert prefs["employment_types"] == ["full_time"]
    assert prefs["accepted_work_modes"] == ["hybrid", "onsite"]
    assert prefs["primary_role_families"] == ["ai_ml", "forward_deployed"]
    assert prefs["sponsorship_policy"] == "required"
    assert prefs["clearance_status"] == "confirmed"


def test_on_tailor_renders_pack_and_honesty_note(monkeypatch, sample_profile):
    monkeypatch.setattr(app_mod, "stream_tailor", _fake_stream(_tailor_result()))
    final = list(on_tailor("j1", "t1", None, sample_profile))[-1]
    html = final[2]
    assert "Dear team," in html
    assert "Honesty note" in html
    assert "traced back to your CV" in html  # zero flags → quiet green line
    # .tex download becomes visible (PDF depends on tectonic availability).
    assert final[5]["visible"] is True


def test_on_tailor_shows_fabrication_warning(monkeypatch, sample_profile):
    result = _tailor_result(flags=2)
    monkeypatch.setattr(app_mod, "stream_tailor", _fake_stream(result))
    final = list(on_tailor("j1", "t1", None, sample_profile))[-1]
    assert "could not be verified against your CV" in final[2]


def test_on_tailor_without_selection_stays_on_results(monkeypatch, sample_profile):
    monkeypatch.setattr(app_mod.gr, "Warning", lambda *a, **k: None)
    final = list(on_tailor(None, "t1", None, sample_profile))[-1]
    assert final[0]["visible"] is True  # page_results stays visible
    assert final[1]["visible"] is False


def test_on_tailor_renders_graceful_error(monkeypatch, sample_profile):
    result = replace(TailorResult(), errors=["tailor: no search state on this thread — run a job search first"])
    monkeypatch.setattr(app_mod, "stream_tailor", _fake_stream(result))
    final = list(on_tailor("j1", "t1", None, sample_profile))[-1]
    assert "Could not tailor this job" in final[2]
    assert "run a job search first" in final[2]


def test_reset_issues_fresh_thread_id():
    first, second = reset(), reset()
    thread_index = 9  # position of thread_id in the reset outputs
    assert first[thread_index] != second[thread_index]


def test_on_zip_passes_path_through():
    assert on_zip("/tmp/export.zip") == "/tmp/export.zip"
    assert on_zip(None) is None
