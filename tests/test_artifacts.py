from pathlib import Path

from fpdf import FPDF
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject

from job_scout.evals.artifacts import audit_rendered_pack


def _pdf(path: Path, text: str, links: list[str] = ()) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=8)
    pdf.multi_cell(0, 5, text)
    pdf.output(str(path))
    if links:
        reader = PdfReader(str(path))
        writer = PdfWriter()
        writer.add_page(reader.pages[0])
        for url in links:
            writer.add_uri(0, url, RectangleObject((10, 10, 100, 20)))
        with path.open("wb") as handle:
            writer.write(handle)


def test_artifact_audit_requires_original_links_and_quality(tmp_path: Path):
    original = tmp_path / "original.pdf"
    tailored = tmp_path / "tailored.pdf"
    letter = tmp_path / "letter.pdf"
    source = "Portfolio https://example.com\n" + ("Python data model deployment evidence. " * 80)
    _pdf(original, source, ["https://example.com"])
    _pdf(tailored, "Python data model deployment evidence. " * 80, [])
    _pdf(letter, "Dear team. " + ("I built Python data pipelines and evaluated models. " * 45))
    report = audit_rendered_pack(
        original, tailored, cover_letter_pdf=letter, job_description="Python and model evaluation experience required."
    )
    codes = {issue.code for issue in report.issues}
    assert not report.passed
    assert "missing_source_links" in codes


def test_artifact_audit_passes_with_preserved_links_and_substantive_letter(tmp_path: Path):
    original = tmp_path / "original.pdf"
    tailored = tmp_path / "tailored.pdf"
    letter = tmp_path / "letter.pdf"
    source = "Portfolio https://example.com\n" + ("Python data model deployment evidence. " * 80)
    links = ["https://example.com"]
    _pdf(original, source, links)
    _pdf(tailored, "Python data model deployment evidence. " * 220, links)
    _pdf(
        letter,
        "Dear team. "
        + ("I used Python programming for data pipelines and documented model evaluation. " * 27)
        + "Sincerely, Candidate",
    )
    report = audit_rendered_pack(
        original,
        tailored,
        cover_letter_pdf=letter,
        job_description="Python programming required. Experience with model evaluation required.",
    )
    assert report.passed
    assert report.missing_links == ()


def test_artifact_audit_rejects_layout_and_project_title_corruption(tmp_path: Path):
    original = tmp_path / "original.pdf"
    tailored = tmp_path / "tailored.pdf"
    letter = tmp_path / "letter.pdf"
    source = "Portfolio https://example.com\n" + ("Python data model deployment evidence. " * 80)
    _pdf(original, source, ["https://example.com"])
    _pdf(
        tailored,
        "V eloce AgenticOS - Research Assistantship and operating system (Cursor, Python). "
        + ("Python data model deployment evidence. " * 80),
        ["https://example.com"],
    )
    _pdf(
        letter,
        "Dear team. "
        + ("I used Python programming for data pipelines and documented model evaluation. " * 27)
        + "Sincerely, Candidate",
    )
    report = audit_rendered_pack(
        original,
        tailored,
        cover_letter_pdf=letter,
        job_description="Python programming required. Experience with model evaluation required.",
    )
    codes = {issue.code for issue in report.issues}
    assert "split_layout_word" in codes
    assert "malformed_project_title" in codes
    assert not report.passed
