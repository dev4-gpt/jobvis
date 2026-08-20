"""LaTeX renderer: escaping, template rendering, tectonic degradation."""

from __future__ import annotations

import shutil

import pytest

from job_scout.graph.schemas import CVContent, CVLink, ExperienceEntry, TailoredBullet
from job_scout.renderer import (
    OVERLEAF_HINT,
    latex_escape,
    normalize_cover_letter,
    render_cover_letter_pdf,
    render_cover_letter_tex,
    render_pdf,
    render_tex,
)

_HAS_TECTONIC = shutil.which("tectonic") is not None


@pytest.mark.parametrize(
    ("raw", "escaped"),
    [
        ("A & B", r"A \& B"),
        ("100%", r"100\%"),
        ("$5", r"\$5"),
        ("#1", r"\#1"),
        ("snake_case", r"snake\_case"),
        ("{braces}", r"\{braces\}"),
        ("~home", r"\textasciitilde{}home"),
        ("x^2", r"x\textasciicircum{}2"),
        ("a\\b", r"a\textbackslash{}b"),
        ("plain text", "plain text"),
    ],
)
def test_latex_escape(raw, escaped):
    assert latex_escape(raw) == escaped


def test_latex_escape_backslash_then_special_chars():
    # A backslash followed by an escapable char must not double-escape.
    assert latex_escape("\\&") == r"\textbackslash{}\&"


def _cv() -> CVContent:
    return CVContent(
        headline="ML Engineer & Data Scientist",  # deliberate LaTeX special char
        summary="Ships models with 100% test coverage.",
        experience=[
            ExperienceEntry(
                role="ML Engineer",
                company="C&A Analytics",
                dates="2020-2026",
                bullets=[TailoredBullet(text="Cut costs by 20% via smart_caching", corpus_ref="cv-bullet-001")],
            )
        ],
        skills=["Python", "SQL", "CI/CD"],
        education=["M.Sc. Computer Science, TU Munich"],
    )


def test_render_tex_escapes_user_content():
    tex = render_tex(_cv(), candidate_name="Jane & Co")
    assert r"Jane \& Co" in tex
    assert r"ML Engineer \& Data Scientist" in tex
    assert r"C\&A Analytics" in tex
    assert r"20\%" in tex
    assert r"smart\_caching" in tex
    assert r"S\kern-0.08em{}Q" in tex
    # No unescaped specials sneak through from user values.
    assert "Jane & Co" not in tex


def test_render_tex_is_a_complete_document():
    tex = render_tex(_cv(), candidate_name="Jane")
    assert tex.strip().startswith(r"\documentclass")
    assert tex.strip().endswith(r"\end{document}")
    assert r"\begin{itemize}" in tex
    assert r"\Needspace{5\baselineskip}" in tex
    assert r"\begin{samepage}" not in tex


def test_render_tex_includes_selected_projects_and_clickable_links():
    cv = CVContent(
        headline="AI Engineer",
        summary="Full-time AI/ML candidate.",
        projects=[
            ExperienceEntry(
                role="Legal Document Analyzer",
                company="Portfolio project",
                bullets=[TailoredBullet(text="Built a grounded RAG workflow", corpus_ref="cv-bullet-010")],
            )
        ],
        links=[
            CVLink(label="Portfolio", url="https://portfolio.example/a_b", page=1),
            CVLink(label="Email", url="mailto:person@example.com", page=1),
        ],
    )
    tex = render_tex(cv, candidate_name="Jane")
    assert "Selected Projects" in tex
    assert r"Legal Document Analyzer" in tex
    assert r"\href{https://portfolio.example/a\_b}{Portfolio}" in tex
    assert r"\href{mailto:person@example.com}{Email}" in tex


def test_render_cover_letter_normalizes_html_and_escapes_latex():
    letter = "Dear team,<br><br>I built 100% reliable APIs & data tools.<br>Best, Jane"
    assert normalize_cover_letter(letter) == "Dear team,\n\nI built 100% reliable APIs & data tools.\nBest, Jane"
    tex = render_cover_letter_tex(letter, "Jane & Co")
    assert r"Jane \& Co" in tex
    assert r"100\% reliable APIs \& data tools." in tex
    assert "<br>" not in tex


def test_render_tex_handles_empty_sections():
    cv = CVContent(headline="X", summary="Y")
    tex = render_tex(cv, candidate_name="Jane")
    assert "Experience" not in tex
    assert "Skills" not in tex
    assert "Education" not in tex


def test_render_pdf_degrades_without_tectonic(tmp_path, monkeypatch):
    import job_scout.renderer as renderer_mod

    monkeypatch.setattr(renderer_mod, "tectonic_path", lambda: None)
    result = render_pdf(_cv(), "Jane", tmp_path)
    assert result.tex_path.exists()
    assert result.pdf_path is None
    assert result.message == OVERLEAF_HINT


def test_render_cover_letter_pdf_degrades_without_tectonic(tmp_path, monkeypatch):
    import job_scout.renderer as renderer_mod

    monkeypatch.setattr(renderer_mod, "tectonic_path", lambda: None)
    result = render_cover_letter_pdf("Dear team,<br>Best, Jane", "Jane", tmp_path)
    assert result.tex_path.name == "cover_letter.tex"
    assert result.tex_path.exists()
    assert result.pdf_path is None
    assert result.message == OVERLEAF_HINT


@pytest.mark.compile
@pytest.mark.skipif(not _HAS_TECTONIC, reason="tectonic binary not installed")
def test_render_pdf_compiles_with_tectonic(tmp_path):
    result = render_pdf(_cv(), "Jane & Co", tmp_path)
    assert result.pdf_path is not None
    assert result.pdf_path.exists()
    assert result.pdf_path.stat().st_size > 1000
