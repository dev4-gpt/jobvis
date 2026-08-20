"""Offline acceptance tests for resume fidelity, quality gates, and ATS safety."""

from __future__ import annotations

import shutil

import pytest
from fpdf import FPDF
from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Link

from job_scout.application.answers import AnswerMemory
from job_scout.application.ats import ApplicantFacts, ATSName, discover_fields, propose_mappings
from job_scout.application.controller import ApplicationController
from job_scout.cover_letter_quality import (
    evaluate_cover_letter,
    grounded_fallback_letter,
    policy_claim_violations,
    remove_unconfirmed_policy_sentences,
)
from job_scout.graph.schemas import CVContent, CVLink
from job_scout.renderer import render_pdf
from job_scout.tools.cv_reader import extract_cv_document


def test_pdf_link_annotations_are_extracted(tmp_path):
    path = tmp_path / "linked.pdf"
    base = tmp_path / "base.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(text="Person - data engineer resume")
    pdf.output(str(base))
    writer = PdfWriter()
    source = PdfReader(base)
    writer.add_page(source.pages[0])
    writer.add_annotation(0, Link(rect=(0, 0, 100, 20), url="https://portfolio.example/"))
    writer.add_annotation(0, Link(rect=(0, 30, 100, 50), url="mailto:person@example.com"))
    with path.open("wb") as stream:
        writer.write(stream)

    text, links = extract_cv_document(path)
    assert "data engineer" in text
    assert {link.url for link in links} == {"https://portfolio.example/", "mailto:person@example.com"}


def test_generated_cv_preserves_clickable_links(tmp_path):
    cv = CVContent(
        headline="Data Engineer",
        summary="Builds grounded data systems.",
        links=[
            CVLink(label="Portfolio", url="https://portfolio.example/", page=1),
            CVLink(label="Email", url="mailto:person@example.com", page=1),
        ],
    )
    if shutil.which("tectonic") is None:
        pytest.skip("tectonic is optional")
    result = render_pdf(cv, "Person", tmp_path)
    assert result.pdf_path is not None
    annotations = [annotation.get_object() for annotation in (PdfReader(result.pdf_path).pages[0].get("/Annots") or [])]
    urls = {str(annotation.get("/A").get_object().get("/URI")) for annotation in annotations if annotation.get("/A")}
    assert {"https://portfolio.example/", "mailto:person@example.com"} <= urls


def test_cover_letter_quality_rejects_empty_and_generic():
    report = evaluate_cover_letter("I am excited to apply.", "Python experience required. SQL skills required.", "Python and SQL")
    assert report.passed is False
    assert report.reasons
    assert report.generic_phrases


def test_cover_letter_quality_rejects_unconfirmed_policy_claims():
    report = evaluate_cover_letter(
        "Dear team. "
        + "I built Python systems and measured outcomes. " * 30
        + "I am authorized to work in the United States. Best, Person",
        "Python experience required.",
        "Built Python systems and measured outcomes.",
    )
    assert report.passed is False
    assert report.policy_violations
    assert policy_claim_violations("Authorization remains unconfirmed.") == []


def test_unconfirmed_policy_claims_are_removed_before_rendering():
    cleaned = remove_unconfirmed_policy_sentences(
        "AI engineer with Python experience. I am authorized to work in the United States."
    )
    assert "authorized to work" not in cleaned.lower()
    assert "Python experience" in cleaned


def test_grounded_fallback_letter_meets_quality_gate_with_two_job_clauses():
    description = "Candidates must work closely with Software Engineers and Platform teams to deliver Generative AI solutions."
    letter = grounded_fallback_letter(
        candidate_name="Person",
        company="Acme",
        job_title="AI Engineer",
        job_description=description,
        corpus_items=[
            "Built Python pipelines and measured model outcomes.",
            "Deployed FastAPI services for reproducible workflows.",
            "Evaluated scikit-learn models and documented results.",
        ],
    )
    corpus = "\n".join(
        [
            "Built Python pipelines and measured model outcomes.",
            "Deployed FastAPI services for reproducible workflows.",
            "Evaluated scikit-learn models and documented results.",
        ]
    )
    report = evaluate_cover_letter(letter, description, corpus)
    assert report.passed is True
    assert report.requirement_matches >= 2


def test_cover_letter_quality_accepts_evidence_and_requirements():
    letter = (
        "Dear hiring team, I am applying because my work building Python data pipelines and evaluating scikit-learn models "
        "matches this role. "
        + "I built Python pipelines with SQL and pandas, measured model outcomes, and delivered a documented project. " * 14
        + "I would bring careful testing, clear communication, and practical collaboration to the team. "
        + "Thank you for reviewing my application. Best, Person"
    )
    report = evaluate_cover_letter(
        letter,
        "The role requires Python experience and SQL skills. Candidates must communicate results and build data pipelines.",
        "Built Python pipelines with SQL and pandas. Evaluated scikit-learn models and measured model outcomes.",
    )
    assert report.passed is True
    assert 250 <= report.word_count <= 350
    assert report.evidence_matches >= 2
    assert report.requirement_matches >= 2


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://boards.greenhouse.io/acme/jobs/1", ATSName.GREENHOUSE),
        ("https://jobs.lever.co/acme/1", ATSName.LEVER),
        ("https://jobs.ashbyhq.com/acme/1", ATSName.ASHBY),
    ],
)
def test_ats_detection_and_safe_mapping(url, expected):
    html = """
    <form>
      <label for="name">Full name</label><input id="name" name="name" required>
      <label for="email">Email</label><input id="email" name="email" type="email">
      <label for="resume">Resume</label><input id="resume" name="resume" type="file">
      <label for="sponsor">Will you require sponsorship?</label><input id="sponsor" name="sponsor">
    </form>
    """
    inspection = discover_fields(url, html)
    assert inspection.ats == expected
    proposals = propose_mappings(inspection.fields, ApplicantFacts(name="Person", email="person@example.com"))
    assert next(item for item in proposals if item.field_id == "name").confidence > 0.9
    assert next(item for item in proposals if item.field_id == "sponsor").sensitive is True
    assert next(item for item in proposals if item.field_id == "resume").value == ""


def test_sensitive_answers_require_consent_and_are_not_auto_reused(monkeypatch, tmp_path):
    import job_scout.application.answers as answers_module

    saved: dict[str, bytes] = {}
    monkeypatch.setattr(answers_module, "save_encrypted", lambda path, payload: saved.__setitem__("payload", payload))
    monkeypatch.setattr(answers_module, "load_encrypted", lambda path: saved.get("payload", b"{}"))
    memory = AnswerMemory(tmp_path / "answers.enc")
    memory.remember("email", "person@example.com", sensitive=False)
    assert memory.reusable("email", sensitive=False).answer == "person@example.com"
    with pytest.raises(ValueError):
        memory.remember("sponsorship", "no", sensitive=True)
    memory.remember("sponsorship", "no", sensitive=True, consent=True)
    suggested = memory.reusable("sponsorship", sensitive=True)
    assert suggested is not None and suggested.sensitive is True


def test_application_controller_has_no_submit_operation():
    controller = ApplicationController()
    assert not hasattr(controller, "submit")
    assert not hasattr(controller.browser, "submit")
