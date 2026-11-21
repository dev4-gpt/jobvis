"""Process-local application workflow used by the API and voice console."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from uuid import uuid4

from job_scout.application.ats import ApplicantFacts, FormInspection, discover_fields
from job_scout.application.browser import BrowserUnavailable, VisibleApplicationBrowser
from job_scout.application.store import ApplicationStore
from job_scout.graph.schemas import CVLink, Profile, RankedJob


def _browser_thread(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        # Playwright's synchronous objects must stay on their creating thread.
        return self._executor.submit(method, self, *args, **kwargs).result()

    return run


@dataclass
class ApplicationState:
    application_id: str = ""
    job_id: str = ""
    url: str = ""
    listing_url: str = ""
    application_url: str = ""
    status: str = "idle"
    message: str = ""
    inspection: FormInspection | None = None
    review_id: str = ""

    def public(self) -> dict:
        inspection = self.inspection
        return {
            "application_id": self.application_id,
            "job_id": self.job_id,
            "url": self.url,
            "listing_url": self.listing_url,
            "application_url": self.application_url,
            "status": self.status,
            "message": self.message,
            "review_id": self.review_id,
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
    """Review-gated controller with an explicit, opt-in submit operation."""

    def __init__(self) -> None:
        self.browser = VisibleApplicationBrowser()
        self.store = ApplicationStore()
        self.state = ApplicationState()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jobvis-application")

    @_browser_thread
    def open(self, ranked: RankedJob, profile: Profile, links: list[CVLink]) -> dict:
        listing_url = ranked.job.listing_url or ranked.job.source_url or ranked.job.url
        application_url = ranked.job.application_url or ranked.job.url or listing_url
        from job_scout.config import get_settings
        from job_scout.tools.liveness import verified_liveness

        if get_settings().liveness_enabled and verified_liveness(application_url, force=True)["liveness"] == "expired":
            self.state = ApplicationState(status="blocked", message="This posting is confirmed expired.")
            return self.state.public()
        if not application_url:
            self.state = ApplicationState(status="blocked", message="This job has no application URL.")
            return self.state.public()
        record = self.store.create_for_job(
            job_id=ranked.job.job_id,
            title=ranked.job.title,
            company=ranked.job.company,
            listing_url=listing_url,
            application_url=application_url,
            source=ranked.job.source,
        )
        try:
            page = self.browser.open(application_url)
            if page is None:
                self.state = ApplicationState(
                    application_id=record.application_id,
                    job_id=ranked.job.job_id,
                    url=application_url,
                    listing_url=listing_url,
                    application_url=application_url,
                    status="opened_manual",
                    message=(
                        "Opened the application in a private browser window. "
                        "Install `uv sync --extra application` and run `uv run playwright install chromium` "
                        "to enable form inspection and approved safe-field filling."
                    ),
                )
                self.store.mark(record.application_id, "opened", "Application opened in visible browser.")
                return self.state.public()
            inspection = discover_fields(page.url, page.content())
        except Exception as exc:  # noqa: BLE001 - browser errors become a visible pause, never a crash
            self.state = ApplicationState(
                application_id=record.application_id,
                job_id=ranked.job.job_id,
                url=application_url,
                listing_url=listing_url,
                application_url=application_url,
                status="blocked",
                message=str(exc),
            )
            return self.state.public()
        inspection.proposals = _proposals_for(inspection, profile, links)
        self.state = ApplicationState(
            application_id=record.application_id,
            job_id=ranked.job.job_id,
            url=page.url,
            listing_url=listing_url,
            application_url=application_url,
            status="review_mapping",
            message="Review the proposed mappings. Login, MFA, and CAPTCHA remain manual.",
            inspection=inspection,
        )
        self.store.mark(record.application_id, "opened", "Application inspected for review.")
        return self.state.public()

    @_browser_thread
    def inspect_html(self, url: str, html: str) -> dict:
        inspection = discover_fields(url, html)
        self.state = ApplicationState(
            url=url,
            application_url=url,
            status="review_mapping",
            inspection=inspection,
        )
        return self.state.public()

    @_browser_thread
    def fill_safe(
        self,
        approved_ids: set[str],
        artifacts: dict[str, Path] | None = None,
        *,
        application_id: str | None = None,
        pack_job_id: str = "",
    ) -> dict:
        if application_id is not None and application_id != self.state.application_id:
            return {**self.state.public(), "message": "The application changed. Review its fields again."}
        if self.state.status != "review_mapping" or self.state.inspection is None or not self.state.application_id:
            return {**self.state.public(), "message": "Open and inspect an application before filling fields."}
        if pack_job_id != self.state.job_id:
            artifacts = None
        try:
            filled = self.browser.fill_safe_fields(self.state.inspection, approved_ids, artifacts)
        except Exception as exc:  # noqa: BLE001 - browser errors pause the workflow
            self.state.status = "blocked"
            self.state.message = str(exc)
            return self.state.public()
        from job_scout.config import get_settings

        self.state.status = "final_review"
        if get_settings().auto_submit_enabled:
            self.state.review_id = uuid4().hex
        self.state.message = f"Filled {len(filled)} approved safe fields. Review everything before submitting."
        if self.state.application_id:
            self.store.mark(self.state.application_id, "final_review", self.state.message)
        return self.state.public()

    @_browser_thread
    def submit(self, confirmation: str, review_id: str) -> dict:
        """Submit the one visible, reviewed application after exact confirmation."""
        from job_scout.config import get_settings

        if not get_settings().auto_submit_enabled:
            return {**self.state.public(), "message": "Jobvis submission is disabled. Submit manually in the browser."}
        if self.state.status != "final_review" or not review_id or review_id != self.state.review_id:
            return {**self.state.public(), "message": "This review is no longer active. Check the current application."}
        if confirmation.strip() != "SUBMIT APPLICATION":
            self.state.message = "Type SUBMIT APPLICATION to confirm this single submission."
            return self.state.public()
        self.state.review_id = ""
        try:
            self.browser.submit(self.state.url)
        except BrowserUnavailable as exc:
            self.state.status = "blocked"
            self.state.message = str(exc)
            return self.state.public()
        except Exception:  # noqa: BLE001 - an interrupted click may already have reached the employer
            self.state.status = "submission_unconfirmed"
            self.state.message = "Check the employer page for confirmation before attempting another submission."
            if self.state.application_id:
                self.store.mark(self.state.application_id, "submission_unconfirmed", self.state.message)
            return self.state.public()
        self.state.status = "submitted_by_jobvis"
        self.state.message = "Submission was triggered in the visible browser. Check the employer confirmation page."
        if self.state.application_id:
            self.store.mark(self.state.application_id, "submitted_by_jobvis", self.state.message)
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
