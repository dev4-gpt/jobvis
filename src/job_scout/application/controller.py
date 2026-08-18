"""Process-local application workflow used by the API and voice console."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from job_scout.application.ats import ApplicantFacts, FormInspection, discover_fields
from job_scout.application.browser import BrowserUnavailable, VisibleApplicationBrowser
from job_scout.graph.schemas import CVLink, Profile, RankedJob


@dataclass
class ApplicationState:
    job_id: str = ""
    url: str = ""
    status: str = "idle"
    message: str = ""
    inspection: FormInspection | None = None

    def public(self) -> dict:
        inspection = self.inspection
        return {
            "job_id": self.job_id,
            "url": self.url,
            "status": self.status,
            "message": self.message,
            "ats": inspection.ats.value if inspection else None,
            "fields": [
                {
                    "field_id": proposal.field_id,
                    "label": proposal.label,
                    "confidence": proposal.confidence,
                    "sensitive": proposal.sensitive,
                    "reason": proposal.reason,
                    "has_value": bool(proposal.value),
                }
                for proposal in (inspection.proposals if inspection else [])
            ],
        }


class ApplicationController:
    """Review-gated controller. There is intentionally no submit operation."""

    def __init__(self) -> None:
        self.browser = VisibleApplicationBrowser()
        self.state = ApplicationState()

    def open(self, ranked: RankedJob, profile: Profile, links: list[CVLink]) -> dict:
        if not ranked.job.url:
            self.state = ApplicationState(status="blocked", message="This job has no application URL.")
            return self.state.public()
        try:
            page = self.browser.open(ranked.job.url)
            inspection = discover_fields(page.url, page.content())
        except Exception as exc:  # noqa: BLE001 - browser errors become a visible pause, never a crash
            self.state = ApplicationState(job_id=ranked.job.job_id, url=ranked.job.url, status="blocked", message=str(exc))
            return self.state.public()
        inspection.proposals = _proposals_for(inspection, profile, links)
        self.state = ApplicationState(
            job_id=ranked.job.job_id,
            url=ranked.job.url,
            status="review_mapping",
            message="Review the proposed mappings. Login, MFA, and CAPTCHA remain manual.",
            inspection=inspection,
        )
        return self.state.public()

    def inspect_html(self, url: str, html: str) -> dict:
        inspection = discover_fields(url, html)
        self.state = ApplicationState(job_id=self.state.job_id, url=url, status="review_mapping", inspection=inspection)
        return self.state.public()

    def fill_safe(self, approved_ids: set[str], artifacts: dict[str, Path] | None = None) -> dict:
        if self.state.inspection is None:
            self.state.status = "blocked"
            self.state.message = "Open and inspect an application before filling fields."
            return self.state.public()
        try:
            filled = self.browser.fill_safe_fields(self.state.inspection, approved_ids, artifacts)
        except BrowserUnavailable as exc:
            self.state.status = "blocked"
            self.state.message = str(exc)
            return self.state.public()
        self.state.status = "final_review"
        self.state.message = f"Filled {len(filled)} approved safe fields. Review everything and click Submit yourself."
        return self.state.public()


def _proposals_for(inspection: FormInspection, profile: Profile, links: list[CVLink]):
    email = next((link.url.removeprefix("mailto:") for link in links if link.url.startswith("mailto:")), "")
    linkedin = next((link.url for link in links if "linkedin.com" in link.url), "")
    portfolio = next((link.url for link in links if "portfolio" in link.label.lower() or "vercel.app" in link.url), "")
    from job_scout.application.ats import propose_mappings

    facts = ApplicantFacts(
        name=profile.name or "",
        email=email,
        linkedin_url=linkedin,
        portfolio_url=portfolio,
        location=(profile.locations or [""])[0],
    )
    return propose_mappings(inspection.fields, facts)


_CONTROLLER = ApplicationController()


def get_application_controller() -> ApplicationController:
    """The process-wide controller shared by API and voice tools."""
    return _CONTROLLER
