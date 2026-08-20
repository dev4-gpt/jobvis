"""Tailor node: guards, budget, corpus grounding, research degradation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import job_scout.graph.nodes.tailor as tailor_mod
from job_scout.corpus import build_corpus
from job_scout.graph.nodes.tailor import (
    _cv_bullet_count,
    _cv_word_count,
    _enforce_cv_contract,
    _restore_source_headers,
    tailor,
)
from job_scout.graph.schemas import (
    CandidatePreferences,
    CVContent,
    CVLink,
    ExperienceEntry,
    Profile,
    RankedJob,
    TailoredBullet,
    TailoringPack,
)
from job_scout.llm import LLMBudgetExceededError
from tests.conftest import make_job, plain_llm, structured_llm
from tests.test_corpus import SAMPLE_CV


def _ranked(job_id: str = "j1") -> RankedJob:
    return RankedJob(job=make_job(job_id, "Data Engineer", "Initech"), fit_score=80, fit_explanation="good fit")


def _pack() -> TailoringPack:
    return TailoringPack(
        cv=CVContent(
            headline="Data Engineer",
            summary="Builds pipelines.",
            skills=["Python"],
        ),
        cover_letter="Dear team,",
        honesty_note="No Kubernetes experience.",
    )


def _state(**overrides) -> dict:
    state = {
        "cv_text": SAMPLE_CV,
        "profile": overrides.pop("profile"),
        "ranked_jobs": [_ranked()],
        "selected_job_id": "j1",
        "llm_calls": 3,
        "errors": [],
    }
    state.update(overrides)
    return state


def test_happy_path_returns_pack_and_increments_budget(monkeypatch, sample_profile):
    llm = structured_llm(_pack())
    monkeypatch.setattr(tailor_mod, "get_chat_model", lambda *a, **k: llm)

    update = tailor(_state(profile=sample_profile))

    assert isinstance(update["tailoring"], TailoringPack)
    # The short fixture letter triggers the bounded quality-repair call.
    assert update["llm_calls"] == 5
    assert update["research_notes"] is None  # keyless env skips Tavily
    prompt = llm.with_structured_output().invoke.call_args[0][0]
    assert "[cv-bullet-001]" in prompt  # corpus rendered with ids
    assert "Initech" in prompt


def test_unknown_job_id_errors_without_llm_call(monkeypatch, sample_profile):
    llm = structured_llm(_pack())
    monkeypatch.setattr(tailor_mod, "get_chat_model", lambda *a, **k: llm)

    update = tailor(_state(profile=sample_profile, selected_job_id="nope"))

    assert update["tailoring"] is None
    assert any("'nope'" in e for e in update["errors"])
    llm.with_structured_output().invoke.assert_not_called()


def test_virgin_thread_errors_without_llm_call(monkeypatch):
    llm = structured_llm(_pack())
    monkeypatch.setattr(tailor_mod, "get_chat_model", lambda *a, **k: llm)

    update = tailor({"selected_job_id": "j1"})

    assert update["tailoring"] is None
    assert any("run a job search first" in e for e in update["errors"])
    llm.with_structured_output().invoke.assert_not_called()


def test_empty_cv_text_errors_gracefully(monkeypatch, sample_profile):
    llm = structured_llm(_pack())
    monkeypatch.setattr(tailor_mod, "get_chat_model", lambda *a, **k: llm)

    update = tailor(_state(profile=sample_profile, cv_text=""))

    assert update["tailoring"] is None
    assert any("empty candidate corpus" in e for e in update["errors"])


def test_bad_linkedin_zip_degrades_to_cv_only(monkeypatch, sample_profile, tmp_path):
    llm = structured_llm(_pack())
    monkeypatch.setattr(tailor_mod, "get_chat_model", lambda *a, **k: llm)
    bogus = tmp_path / "export.zip"
    bogus.write_text("not a zip")

    update = tailor(_state(profile=sample_profile, linkedin_zip_path=str(bogus)))

    assert isinstance(update["tailoring"], TailoringPack)
    assert any("continuing with the CV only" in e for e in update["errors"])


def test_budget_exceeded_raises(monkeypatch, sample_profile):
    llm = structured_llm(_pack())
    monkeypatch.setattr(tailor_mod, "get_chat_model", lambda *a, **k: llm)

    with pytest.raises(LLMBudgetExceededError):
        tailor(_state(profile=sample_profile, llm_calls=25))


def test_bullet_requiring_corpus_ref_flows_through(monkeypatch, sample_profile):
    pack = _pack()
    pack.cv.experience = [
        # Schema-level guarantee exercised end to end: bullets carry refs.
        ExperienceEntry(
            role="Data Engineer",
            company="PipeCorp",
            bullets=[TailoredBullet(text="Built streaming ingestion handling 2M events/day", corpus_ref="cv-bullet-002")],
        )
    ]
    llm = structured_llm(pack)
    monkeypatch.setattr(tailor_mod, "get_chat_model", lambda *a, **k: llm)

    update = tailor(_state(profile=sample_profile))
    bullets = update["tailoring"].cv.experience[0].bullets
    assert bullets[0].corpus_ref == "cv-bullet-002"


def test_model_cannot_replace_source_employer_header():
    corpus = build_corpus(
        """Jane Doe
Experience
Data Engineer, PipeCorp (2022-2026)
- Built reliable streaming ingestion pipelines handling millions of events daily.
- Migrated the warehouse from Redshift to BigQuery with tested repeatable workflows.
"""
    )
    pack = TailoringPack(
        cv=CVContent(
            headline="Data Engineer",
            summary="Builds data systems.",
            experience=[
                ExperienceEntry(
                    role="AI Intern",
                    company="Undisclosed",
                    dates="2024",
                    bullets=[TailoredBullet(text="Rewritten evidence", corpus_ref="cv-bullet-002")],
                )
            ],
        ),
        cover_letter="Draft",
    )

    restored = _restore_source_headers(pack, corpus)

    # The source fixture's first entry is the only available experience entry;
    # the model cannot anonymize or rewrite its employer/date metadata.
    assert restored.cv.experience[0].role == "Data Engineer"
    assert restored.cv.experience[0].company == "PipeCorp"
    assert restored.cv.experience[0].dates == "2022-2026"


def test_empty_openrouter_structured_envelope_recovers_with_validated_json(monkeypatch, sample_profile):
    pack = _pack().model_copy(
        update={
            "cover_letter": (
                "Dear Initech team, "
                + "I built practical Python and SQL systems and measured their outcomes. " * 45
                + "Sincerely, Aryaman"
            )
        }
    )
    base_model = structured_llm(pack)
    typed_model = base_model.with_structured_output.return_value
    typed_model.invoke.side_effect = [
        ValueError("Structured Output response does not have a 'parsed' field nor a 'refusal' field"),
        pack,
    ]
    recovery_model = plain_llm(pack.model_dump_json())
    models = iter([base_model, recovery_model])
    monkeypatch.setattr(tailor_mod, "get_chat_model", lambda *a, **k: next(models))

    update = tailor(_state(profile=sample_profile))

    assert isinstance(update["tailoring"], TailoringPack)
    assert update["llm_calls"] == 6  # typed attempt + recovery + bounded quality repair
    assert not any("no draft was created" in error for error in update["errors"])


def test_all_tailoring_providers_failed_still_returns_grounded_pack(monkeypatch, sample_profile):
    """An empty OpenRouter response must degrade to a reviewable local draft."""
    import job_scout.graph.nodes.tailor as tailor_module

    failed_model = MagicMock()
    failed_structured = MagicMock()
    failed_structured.invoke.side_effect = ValueError(
        "Structured Output response does not have a 'parsed' field nor a 'refusal' field"
    )
    failed_model.with_structured_output.return_value = failed_structured
    empty_message = MagicMock()
    empty_message.content = ""
    failed_model.invoke.return_value = empty_message
    monkeypatch.setattr(tailor_module, "get_chat_model", lambda *a, **k: failed_model)

    update = tailor(
        _state(
            profile=sample_profile,
            cv_links=[CVLink(label="Portfolio", url="https://portfolio.example", page=1)],
        )
    )

    assert isinstance(update["tailoring"], TailoringPack)
    assert update["tailoring"].cv.links[0].url == "https://portfolio.example"
    assert 250 <= update["cover_letter_quality"].word_count <= 350
    assert any("deterministic CV/letter draft" in error for error in update["errors"])


def test_unverified_fhir_claim_is_removed_from_generated_pack(monkeypatch, sample_profile):
    pack = _pack().model_copy(update={"cv": CVContent(headline="AI engineer", summary="Completed self-study FHIR.")})
    llm = structured_llm(pack)
    monkeypatch.setattr(tailor_mod, "get_chat_model", lambda *a, **k: llm)

    update = tailor(_state(profile=sample_profile))

    assert "FHIR" not in update["tailoring"].cv.summary


def test_sparse_model_cv_is_rebuilt_from_source_with_density_contract():
    cv_text = """\
Professional Experience
SymphonyAI, NY
Data Scientist January 2024 - July 2024
- Built predictive maintenance models with Python and PyTorch, reducing downtime by 25%.
- Improved model accuracy by 20% through evaluation and feature engineering.
- Deployed FastAPI services and reduced response time by 40%.
- Analyzed multivariate data, identified bottlenecks, and documented improvements for stakeholders and engineering teams.
- Validated model behavior with repeatable experiments, error analysis, and monitoring checks before deployment.
- Translated technical findings into clear implementation notes covering data quality, model limitations, and next steps.

Bioqube, NY
AI Engineer July 2023 - August 2023
- Developed a computer vision system achieving 75% accuracy on video data.
- Built data pipelines processing large-scale video datasets.
- Improved reliability through testing and optimization.
- Designed evaluation workflows for precision, recall, false positives, and data coverage across representative samples.
- Collaborated on service integration and documented reproducible steps for testing model outputs and API behavior.
- Investigated failure cases and converted observations into targeted improvements for model quality and operational reliability.

Academic Reports
Autonomous Legal Document Analyzer, NY
- Built a GenAI legal document analysis system for contract ingestion and risk assessment.
- Designed a RAG pipeline using LangChain, FAISS, and ChromaDB.
- Developed a multi-step LLM agent with LangGraph and Docker deployment.
- Structured the workflow around ingestion, retrieval, clause extraction, and review-oriented outputs.
- Connected the system components into a prototype with clear boundaries between retrieval, reasoning, and presentation.
- Focused evaluation on traceable outputs, reproducibility, and grounded evidence rather than unsupported generated claims.

Veloce AgenticOS, NY
- Built an AI command center for agent execution, memory, and workflow orchestration.
- Designed operator interfaces for status, monitoring, and live activity streams.
- Implemented backend API routes for agent chat, previews, and memory search.
- Organized agent capabilities behind a consistent interface for planning, execution, review, and handoffs.
- Connected local context and activity records so operators could inspect decisions and system state across sessions.
- Documented the platform architecture and integration points to make future extensions easier to evaluate and maintain.
- Evaluated orchestration across planning, tool use, state transitions, and review checkpoints.
- Built reusable surfaces for inspecting agent status, active work, stored context, generated outputs, and task relationships.
- Connected product workflows to backend services while keeping development, deployment, and integration testing understandable.
- Applied structured logging and visible activity records to make failures and handoffs easier to diagnose.
- Designed the system so new providers and specialized agents could be introduced without changing the review workflow.
- Tested interactions across frontend components, API routes, model calls, local context, and generated artifacts.
- Documented decisions, constraints, and tradeoffs so the platform could evolve with grounded outputs.
- Focused implementation on AI product delivery, reproducible workflows, observable state, and human review.

TECHNICAL SKILLS & SOFTWARES PROFICIENCY
Python, SQL, PyTorch, Docker, FastAPI, LangChain, FAISS, ChromaDB, LangGraph, Git, AWS, GCP, React, TypeScript.
Education
Penn State — M.S. Artificial Intelligence (2026)
"""
    corpus = build_corpus(cv_text)
    ranked = _ranked()
    profile = Profile(name="Candidate", primary_roles=["AI Engineer"])
    sparse = TailoringPack(
        cv=CVContent(headline="Relevant resume evidence", summary="Short summary", skills=[]),
        cover_letter="Draft",
    )
    rebuilt, changed = _enforce_cv_contract(
        sparse,
        profile,
        corpus,
        ranked,
        [CVLink(label="Portfolio", url="https://portfolio.example", page=1)],
        CandidatePreferences(),
    )

    assert changed is True
    # A source CV that is itself shorter than the contract must remain
    # grounded; the gate cannot invent filler just to hit a page target.
    assert _cv_word_count(rebuilt) >= 250
    assert _cv_bullet_count(rebuilt) >= 10
    assert len(rebuilt.cv.experience) >= 2
    assert len(rebuilt.cv.projects) >= 2
    assert len(rebuilt.cv.skills) >= 12
    assert rebuilt.cv.links[0].url == "https://portfolio.example"


def test_legacy_tailoring_json_is_normalized_before_validation():
    pack = TailoringPack.model_validate(
        {
            "cv": {
                "summary": "AI/ML engineer building production systems.",
                "projects": [
                    {
                        "name": "Legal Document Analyzer",
                        "description": "RAG system",
                        "bullets": [{"text": "Built retrieval", "corpus_ref": "cv-bullet-019"}],
                        "links": ["GitHub"],
                    }
                ],
                "education": [{"institution": "Penn State", "degree": "M.S. AI", "graduation": "2026-12"}],
                "links": [{"label": "Portfolio", "url": "https://example.com"}],
            },
            "cover_letter": "Evidence-backed letter.",
        }
    )

    assert pack.cv.headline == "AI/ML engineer building production systems"
    assert pack.cv.projects[0].role == "Legal Document Analyzer"
    assert pack.cv.projects[0].company == "Project"
    assert pack.cv.education == ["Penn State — M.S. AI (2026-12)"]
    assert pack.cv.links[0].page == 1


def test_semantic_link_page_labels_are_safe_placeholders():
    pack = TailoringPack.model_validate(
        {
            "cv": {
                "headline": "AI engineer",
                "summary": "Builds ML systems.",
                "links": [{"label": "GitHub", "url": "https://github.com/example", "page": "github"}],
            },
            "cover_letter": "Draft.",
        }
    )

    assert pack.cv.links[0].page == 1
