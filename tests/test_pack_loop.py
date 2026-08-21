from pathlib import Path

import pytest
from pypdf import PdfReader

from job_scout.evals.pack_loop import render_verified_pack
from job_scout.graph.schemas import CVContent, CVLink, ExperienceEntry, TailoredBullet, TailoringPack


def _pack() -> TailoringPack:
    def bullet(index: int) -> TailoredBullet:
        return TailoredBullet(
            text=(
                f"Built and evaluated Python model pipeline {index}, documented deployment tradeoffs, "
                "measured failure cases, and communicated reproducible results to technical collaborators "
                "using versioned experiments and clear technical documentation."
            ),
            corpus_ref=f"cv-bullet-{index}",
        )

    links = [CVLink(label="Portfolio", url="https://example.com", page=1)]
    return TailoringPack(
        cv=CVContent(
            headline="AI Engineer",
            summary="Grounded systems with measured outcomes and reproducible deployment.",
            links=links,
            experience=[ExperienceEntry(role="Data Scientist", company="Acme", bullets=[bullet(i) for i in range(12)])],
            projects=[
                ExperienceEntry(role="Veloce AgenticOS", company="", bullets=[bullet(i) for i in range(12, 16)]),
                ExperienceEntry(
                    role="Research Assistantship and operating system (Cursor, Python)",
                    company="",
                    bullets=[bullet(i) for i in range(16, 20)],
                ),
            ],
            skills=["Python", "SQL", "FAISS", "RAG", "LLM", "API", "Docker", "FastAPI", "Pandas", "NumPy", "Git", "LangChain"],
            education=["M.S. Artificial Intelligence"],
        ),
        cover_letter=(
            "Dear team, because it calls for we are looking for someone who is a talented software engineer at their core, "
            "but has contributed and to AI research. "
            + "I built Python model evaluation and deployment systems. " * 8
            + "Sincerely, Candidate"
        ),
    )


@pytest.mark.compile
@pytest.mark.skipif(__import__("shutil").which("tectonic") is None, reason="tectonic binary not installed")
def test_render_loop_repairs_known_pdf_failures(tmp_path: Path):
    source = "Python model evaluation and deployment evidence. " * 180
    result = render_verified_pack(
        _pack(),
        candidate_name="Candidate",
        source_text=source,
        source_links=[CVLink(label="Portfolio", url="https://example.com", page=1)],
        job_description=(
            "Python programming experience required. Model evaluation experience required. Deployment experience required."
        ),
        company="Acme",
        job_title="Applied ML Engineer",
        out_dir=tmp_path,
        backtest_score=0.91,
    )
    assert result.report.passed
    assert result.attempts == 2
    assert result.cv.pdf_path is not None
    text = "\n".join(page.extract_text() or "" for page in PdfReader(str(result.cv.pdf_path)).pages)
    assert "V eloce" not in text
    assert "Research Assistantship and operating system" not in text
    assert result.report.generated_links == 1
    assert result.manifest.status == "ready"
    assert result.manifest.pdfs_ready is True
    assert result.manifest.annotation_count == 1
    assert result.manifest.backtest_score == 0.91


@pytest.mark.compile
@pytest.mark.skipif(__import__("shutil").which("tectonic") is None, reason="tectonic binary not installed")
def test_render_loop_withholds_then_repairs_overlength_letter(tmp_path: Path):
    pack = _pack()
    pack.cover_letter = pack.cover_letter + (" I measured Python model outcomes and documented deployment results." * 60)
    result = render_verified_pack(
        pack,
        candidate_name="Candidate",
        source_text=("Python model evaluation and deployment evidence. " * 180),
        source_links=[CVLink(label="Portfolio", url="https://example.com", page=1)],
        job_description="Python programming experience required. Model evaluation experience required.",
        company="Acme",
        job_title="Applied ML Engineer",
        out_dir=tmp_path,
    )
    assert result.attempts == 2
    assert result.report.passed
    assert 250 <= result.report.cover_letter_words <= 350
    assert result.manifest.pdfs_ready


def test_render_loop_stops_at_a_finite_budget(monkeypatch, tmp_path: Path):
    import job_scout.evals.pack_loop as loop

    monkeypatch.setattr(loop, "render_pdf", lambda cv, name, out: loop.RenderResult(out / "tailored_cv.tex"))
    monkeypatch.setattr(loop, "render_cover_letter_pdf", lambda letter, name, out: loop.RenderResult(out / "cover_letter.tex"))
    result = loop.render_verified_pack(
        _pack(),
        candidate_name="Candidate",
        source_text="short source",
        source_links=[],
        job_description="Python required.",
        out_dir=tmp_path,
        max_attempts=3,
    )
    assert result.attempts <= 3
    assert len(result.history) <= 3
    assert not result.report.passed
    assert result.manifest.status == "withheld"
    assert result.manifest.pdfs_ready is False
