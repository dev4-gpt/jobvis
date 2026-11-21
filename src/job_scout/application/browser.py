"""Visible local browser control for reviewed application submission."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from job_scout.application.ats import FormInspection, _field_key, discover_fields
from job_scout.application.security import load_encrypted, save_encrypted


class BrowserUnavailable(RuntimeError):
    """The optional Playwright runtime is not installed or configured."""


class VisibleApplicationBrowser:
    """A visible Chromium-compatible context; credentials always stay local."""

    def __init__(self, state_path: Path | None = None) -> None:
        self.state_path = state_path or Path("data/private/application/browser_state.enc")
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def open(self, url: str):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            # Opening the posting is useful even when the optional automation
            # extra is not installed. Use a fresh, private browser profile so
            # credentials/cookies from the user's normal browser are neither
            # read nor persisted by Jobvis. Form inspection and safe filling
            # still require Playwright and remain unavailable until the extra
            # is installed.
            return self._open_without_playwright(url, cause=exc)
        self._playwright = sync_playwright().start()
        executable = os.environ.get("JOBVIS_BROWSER_EXECUTABLE") or _brave_path()
        kwargs = {"headless": False}
        if executable:
            kwargs["executable_path"] = executable
        storage_state = None
        if self.state_path.exists():
            storage_state = json.loads(load_encrypted(self.state_path).decode("utf-8"))
        # A regular browser context keeps credentials in memory; saved state is
        # encrypted through Keychain-backed storage only when save_state() is called.
        self._browser = self._playwright.chromium.launch(**kwargs)
        self._context = self._browser.new_context(storage_state=storage_state) if storage_state else self._browser.new_context()
        self._page = self._context.new_page()
        self._page.goto(url, wait_until="domcontentloaded")
        return self._page

    def _open_without_playwright(self, url: str, *, cause: ImportError):
        executable = os.environ.get("JOBVIS_BROWSER_EXECUTABLE") or _brave_path()
        if not executable:
            raise BrowserUnavailable(
                "Could not find Brave or another Chromium browser. Set JOBVIS_BROWSER_EXECUTABLE, "
                "or install the optional application extra with `uv sync --extra application`."
            ) from cause
        profile_dir = Path(tempfile.mkdtemp(prefix="jobvis-application-"))
        try:
            subprocess.Popen(
                [executable, f"--user-data-dir={profile_dir}", "--incognito", "--new-window", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise BrowserUnavailable(f"Could not open the local browser: {exc}") from exc
        return None

    def inspect(self, url: str | None = None) -> FormInspection:
        if self._page is None:
            raise BrowserUnavailable("Open the application in the visible browser first.")
        if url and self._page.url != url:
            self._page.goto(url, wait_until="domcontentloaded")
        return discover_fields(self._page.url, self._page.content())

    def save_state(self) -> None:
        if self._context is None:
            raise BrowserUnavailable("No browser context is open.")
        save_encrypted(self.state_path, json.dumps(self._context.storage_state()).encode("utf-8"))

    def fill_safe_fields(
        self, inspection: FormInspection, approved_ids: set[str], artifacts: dict[str, Path] | None = None
    ) -> list[str]:
        if self._page is None:
            raise BrowserUnavailable("Open the application in the visible browser first.")
        if self._page.url != inspection.url:
            raise BrowserUnavailable("The browser moved to a different page. Reopen the application to review its fields.")
        filled: list[str] = []
        for proposal in inspection.proposals:
            if proposal.field_id not in approved_ids or proposal.sensitive or not proposal.value:
                continue
            field = next((item for item in inspection.fields if item.field_id == proposal.field_id), None)
            if field is None or not field.selector:
                continue
            self._page.locator(field.selector).fill(proposal.value)
            filled.append(proposal.field_id)
        for key, artifact in (artifacts or {}).items():
            field = next(
                (
                    item
                    for item in inspection.fields
                    if item.input_type == "file"
                    and (item.field_id == key or key in _field_key(item.label, item.name, item.field_id))
                ),
                None,
            )
            # File uploads are consequential personal-document transfers and
            # need their own approval. Approving a name/email field must never
            # implicitly upload the tailored CV or cover letter as a side
            # effect of the artifacts being available.
            if field and field.field_id in approved_ids and field.selector:
                self._page.locator(field.selector).set_input_files(str(artifact))
                filled.append(key)
        return filled

    def submit(self, reviewed_url: str) -> None:
        """Click one unambiguous, visible final-submit control.

        The caller is responsible for the per-application confirmation and
        feature flag. We intentionally refuse ambiguous forms rather than
        guessing which of several buttons sends an application.
        """
        if self._page is None:
            raise BrowserUnavailable("Open the application in the visible browser first.")
        hostname = urlsplit(reviewed_url).hostname or ""
        supported = ("greenhouse.io", "lever.co", "ashbyhq.com")
        if self._page.url != reviewed_url or not any(hostname.endswith(f".{host}") for host in supported):
            raise BrowserUnavailable("The reviewed form is no longer open or needs manual submission in the browser.")
        selector = "form button:visible:enabled, form input[type='submit']:visible:enabled"
        candidates = []
        for index in range(self._page.locator(selector).count()):
            candidate = self._page.locator(selector).nth(index)
            label = (candidate.inner_text() or candidate.get_attribute("value") or "").strip().lower()
            if " ".join(label.split()) in {"submit", "submit application", "submit my application"}:
                candidates.append(candidate)
        if len(candidates) != 1:
            raise BrowserUnavailable(
                "Could not identify one unambiguous final Submit button; submit manually in the visible browser."
            )
        candidates[0].click()


def _brave_path() -> str | None:
    candidates = (
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        "/Applications/Brave Browser Beta.app/Contents/MacOS/Brave Browser Beta",
    )
    return next((path for path in candidates if Path(path).exists()), shutil.which("brave"))
