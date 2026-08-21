from job_scout.graph.schemas import CVContent, ExperienceEntry, RankedJob, TailoredBullet, TailoringPack
from job_scout.outreach import ContactCandidate, PublicContactSource, build_outreach_draft, discover_public_contacts
from tests.conftest import make_job


def _pack() -> TailoringPack:
    return TailoringPack(
        cv=CVContent(
            headline="Applied ML Engineer",
            summary="Applied ML evidence",
            experience=[],
            projects=[],
        ),
        cover_letter="Grounded letter",
    )


def test_contact_discovery_extracts_only_explicit_public_email():
    contacts = discover_public_contacts(
        [
            PublicContactSource(
                url="https://example.com/team",
                text="Jordan Lee, Head of Engineering — contact jordan@example.com for recruiting questions.",
                source_type="company_page",
            )
        ]
    )
    assert contacts[0].email == "jordan@example.com"
    assert contacts[0].confidence > 0.8
    assert discover_public_contacts([PublicContactSource(url="https://example.com", text="Jordan is hiring")]) == []


def test_outreach_draft_is_grounded_and_manual_only(sample_profile):
    pack = _pack()
    pack.cv.projects = [
        ExperienceEntry(
            role="Legal analyzer",
            company="",
            bullets=[TailoredBullet(text="Built a LangGraph RAG service with FastAPI.", corpus_ref="p1")],
        )
    ]
    posting = make_job("j1", "Applied ML Engineer", "Acme")
    posting.description = "Experience with Python and model evaluation required."
    job = RankedJob(
        job=posting,
        fit_score=90,
        fit_explanation="fits",
    )
    draft = build_outreach_draft(
        job,
        sample_profile,
        pack,
        ContactCandidate(name="Jordan Lee", email="jordan@example.com", source_url="https://example.com"),
    )
    assert "Applied ML Engineer" in draft.video_script
    assert "LangGraph" in draft.video_script
    assert draft.contact.email == "jordan@example.com"
    assert any("does not send" in note for note in draft.review_notes)
