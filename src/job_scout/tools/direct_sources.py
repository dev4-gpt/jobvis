"""Opt-in direct job-board adapters.

These adapters only call public, documented listing endpoints or a configured
public board. They never submit applications. Company/board identifiers are
explicit configuration so a broad search cannot accidentally crawl an entire
site or leak credentials.
"""

from __future__ import annotations

import hashlib
import html
import re
from html.parser import HTMLParser
from urllib.parse import urljoin

import httpx

from job_scout.config import get_settings
from job_scout.graph.schemas import JobPosting

_DESCRIPTION_LIMIT = 4000


def _text(value: object) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip()


def _strip_html(value: object) -> str:
    return _text(re.sub(r"<[^>]+>", " ", str(value or "")))[:_DESCRIPTION_LIMIT]


def _hash(*parts: object) -> str:
    return hashlib.sha256("|".join(_text(part) for part in parts).encode()).hexdigest()[:16]


def _employment(value: object) -> str:
    lowered = _text(value).lower().replace("-", "_")
    if "intern" in lowered or "student" in lowered:
        return "internship"
    if "co_op" in lowered or "coop" in lowered:
        return "co_op"
    if "part" in lowered:
        return "part_time"
    if "contract" in lowered or "temporary" in lowered:
        return "contract"
    if "full" in lowered or "permanent" in lowered:
        return "full_time"
    return "unknown"


def _work_mode(value: object, remote: bool = False) -> str:
    lowered = _text(value).lower()
    if remote or "remote" in lowered:
        return "remote"
    if "hybrid" in lowered:
        return "hybrid"
    if "onsite" in lowered or "on-site" in lowered or "in office" in lowered:
        return "onsite"
    return "unknown"


def _matches(query: str, title: str, description: str) -> bool:
    terms = [term for term in re.split(r"\W+", query.lower()) if term]
    if not terms:
        return True
    haystack = f"{title} {description}".lower()
    return any(term in haystack for term in terms)


def _base(
    *,
    source: str,
    record_id: str,
    title: str,
    company: str,
    location: str,
    description: str,
    listing_url: str,
    application_url: str = "",
    posted_at: str | None = None,
    employment_type: str = "unknown",
    work_mode: str = "unknown",
    ats: str = "unknown",
    tags: list[str] | None = None,
    salary_text: str = "",
) -> JobPosting:
    application_url = application_url or listing_url
    return JobPosting(
        job_id=f"{source}-{record_id}",
        title=_text(title) or "Untitled",
        company=_text(company) or "Unknown",
        location=_text(location) or "Unspecified",
        remote=work_mode == "remote",
        description=_strip_html(description),
        url=application_url,
        tags=[_text(tag) for tag in (tags or []) if _text(tag)],
        source=source,
        employment_type=employment_type,
        work_mode=work_mode,
        posted_at=_text(posted_at) or None,
        salary_text=_text(salary_text),
        source_url=listing_url,
        listing_url=listing_url,
        application_url=application_url,
        source_record_id=_text(record_id),
        ats=ats,
        freshness="fresh",
        content_hash=_hash(title, company, location, description),
    )


class GreenhouseSource:
    name = "greenhouse"

    def __init__(self, board_tokens: list[str] | None = None, timeout: float | None = None) -> None:
        settings = get_settings()
        self.board_tokens = board_tokens if board_tokens is not None else settings.greenhouse_boards
        self.timeout = timeout or settings.scout_source_timeout

    @property
    def available(self) -> bool:
        return bool(self.board_tokens)

    def fetch(self, query: str, location: str | None, country: str | None, remote: bool, limit: int) -> list[JobPosting]:
        found: list[JobPosting] = []
        for token in self.board_tokens:
            try:
                response = httpx.get(
                    f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs",
                    params={"content": "true"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                rows = response.json().get("jobs", [])
            except (httpx.HTTPError, ValueError, TypeError):
                continue
            for row in rows:
                title = _text(row.get("title"))
                description = _strip_html(row.get("content"))
                office = row.get("location") or {}
                location_text = _text(office.get("name") if isinstance(office, dict) else office)
                listing = _text(row.get("absolute_url"))
                job = _base(
                    source=self.name,
                    record_id=row.get("id"),
                    title=title,
                    company=token,
                    location=location_text,
                    description=description,
                    listing_url=listing,
                    application_url=listing,
                    posted_at=row.get("updated_at"),
                    employment_type=_employment(description),
                    work_mode=_work_mode(description, remote),
                    ats="greenhouse",
                )
                if _matches(query, title, description) and (not location or location.lower() in job.location.lower()):
                    found.append(job)
                    if len(found) >= limit:
                        return found
        return found


class LeverSource:
    name = "lever"

    def __init__(self, companies: list[str] | None = None, timeout: float | None = None) -> None:
        settings = get_settings()
        self.companies = companies if companies is not None else settings.lever_accounts
        self.timeout = timeout or settings.scout_source_timeout

    @property
    def available(self) -> bool:
        return bool(self.companies)

    def fetch(self, query: str, location: str | None, country: str | None, remote: bool, limit: int) -> list[JobPosting]:
        found: list[JobPosting] = []
        for company in self.companies:
            try:
                response = httpx.get(
                    f"https://api.lever.co/v0/postings/{company}",
                    params={"mode": "json"},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                rows = response.json()
            except (httpx.HTTPError, ValueError, TypeError):
                continue
            for row in rows if isinstance(rows, list) else []:
                categories = row.get("categories") or {}
                title = _text(row.get("text"))
                description = _strip_html(row.get("descriptionPlain") or row.get("description"))
                location_text = _text(categories.get("location") or row.get("workplaceType"))
                listing = _text(row.get("hostedUrl"))
                application = _text(row.get("applyUrl")) or listing
                mode = _work_mode(row.get("workplaceType"), remote)
                job = _base(
                    source=self.name,
                    record_id=row.get("id"),
                    title=title,
                    company=company,
                    location=location_text,
                    description=description,
                    listing_url=listing,
                    application_url=application,
                    posted_at=row.get("createdAt"),
                    employment_type=_employment(categories.get("commitment")),
                    work_mode=mode,
                    ats="lever",
                    tags=categories.values() if isinstance(categories, dict) else [],
                )
                if _matches(query, title, description) and (not location or location.lower() in job.location.lower()):
                    found.append(job)
                    if len(found) >= limit:
                        return found
        return found


class AshbySource:
    name = "ashby"

    def __init__(self, boards: list[str] | None = None, timeout: float | None = None) -> None:
        settings = get_settings()
        self.boards = boards if boards is not None else settings.ashby_boards
        self.timeout = timeout or settings.scout_source_timeout

    @property
    def available(self) -> bool:
        return bool(self.boards)

    def fetch(self, query: str, location: str | None, country: str | None, remote: bool, limit: int) -> list[JobPosting]:
        found: list[JobPosting] = []
        for board in self.boards:
            try:
                response = httpx.post(
                    f"https://api.ashbyhq.com/posting-api/job-board/{board}",
                    json={"includeCompensation": True},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                rows = payload.get("jobs", payload) if isinstance(payload, dict) else payload
            except (httpx.HTTPError, ValueError, TypeError):
                continue
            for row in rows if isinstance(rows, list) else []:
                title = _text(row.get("title"))
                description = _strip_html(row.get("descriptionPlain") or row.get("description"))
                location_text = _text(row.get("location") or ", ".join(row.get("locations", []) or []))
                listing = _text(row.get("jobUrl") or row.get("jobPostingUrl"))
                application = _text(row.get("applyUrl")) or listing
                mode = _work_mode(row.get("workplaceType"), remote)
                job = _base(
                    source=self.name,
                    record_id=row.get("jobPostingId") or row.get("id"),
                    title=title,
                    company=board,
                    location=location_text,
                    description=description,
                    listing_url=listing,
                    application_url=application,
                    posted_at=row.get("publishedAt"),
                    employment_type=_employment(row.get("employmentType")),
                    work_mode=mode,
                    ats="ashby",
                    salary_text=_text(row.get("compensation")),
                )
                if _matches(query, title, description) and (not location or location.lower() in job.location.lower()):
                    found.append(job)
                    if len(found) >= limit:
                        return found
        return found


class USAJobsSource:
    name = "usajobs"
    BASE = "https://data.usajobs.gov/api/search"

    def __init__(self, api_key: str = "", user_agent: str = "", timeout: float | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.usajobs_api_key.get_secret_value()
        self.user_agent = user_agent or settings.usajobs_user_agent
        self.timeout = timeout or settings.scout_source_timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.user_agent)

    def fetch(self, query: str, location: str | None, country: str | None, remote: bool, limit: int) -> list[JobPosting]:
        if not self.available:
            return []
        params: dict[str, object] = {"Keyword": query, "ResultsPerPage": min(limit, 25), "Fields": "Full"}
        if location:
            params["LocationName"] = location
        if remote:
            params["RemoteIndicator"] = "True"
        try:
            response = httpx.get(
                self.BASE,
                params=params,
                headers={"Host": "data.usajobs.gov", "User-Agent": self.user_agent, "Authorization-Key": self.api_key},
                timeout=self.timeout,
            )
            response.raise_for_status()
            rows = response.json().get("SearchResult", {}).get("SearchResultItems", [])
        except (httpx.HTTPError, ValueError, TypeError):
            return []
        found: list[JobPosting] = []
        for item in rows:
            descriptor = item.get("MatchedObjectDescriptor", {})
            locations = descriptor.get("PositionLocation", []) or []
            location_text = _text(descriptor.get("PositionLocationDisplay")) or _text(
                ", ".join(str(row.get("LocationName", "")) for row in locations if isinstance(row, dict))
            )
            listing = _text(descriptor.get("PositionURI"))
            job = _base(
                source=self.name,
                record_id=descriptor.get("PositionID") or item.get("MatchedObjectId"),
                title=descriptor.get("PositionTitle"),
                company=descriptor.get("OrganizationName"),
                location=location_text,
                description=descriptor.get("UserArea", {}).get("Details", {}).get("JobSummary", ""),
                listing_url=listing,
                application_url=listing,
                posted_at=descriptor.get("PublicationStartDate"),
                employment_type="full_time",
                work_mode=_work_mode(location_text, remote),
                ats="usajobs",
            )
            if _matches(query, job.title, job.description):
                found.append(job)
        return found[:limit]


class ProtocolLabsSource:
    """Best-effort parser for the public Protocol Labs directory board.

    The directory is a human-facing board rather than a stable public API, so
    failures are intentionally empty and the source remains clearly labeled.
    """

    name = "protocol_labs"

    def __init__(self, url: str | None = None, timeout: float | None = None) -> None:
        settings = get_settings()
        self.url = url or settings.protocol_labs_jobs_url
        self.timeout = timeout or settings.scout_source_timeout

    @property
    def available(self) -> bool:
        return bool(self.url)

    def fetch(self, query: str, location: str | None, country: str | None, remote: bool, limit: int) -> list[JobPosting]:
        try:
            response = httpx.get(self.url, timeout=self.timeout, follow_redirects=True)
            response.raise_for_status()
        except httpx.HTTPError:
            return []
        parser = _AnchorParser()
        parser.feed(response.text)
        found: list[JobPosting] = []
        for index, (label, href) in enumerate(parser.items):
            if not label or href.rstrip("/") == self.url.rstrip("/"):
                continue
            if not _matches(query, label, label):
                continue
            listing = urljoin(str(response.url), href)
            found.append(
                _base(
                    source=self.name,
                    record_id=_hash(listing) or str(index),
                    title=label,
                    company="Protocol Labs Network",
                    location="Remote" if remote else (location or "Unspecified"),
                    description=label,
                    listing_url=listing,
                    application_url=listing,
                    employment_type="full_time",
                    work_mode="remote" if remote else "unknown",
                )
            )
            if len(found) >= limit:
                break
        return found


class _AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.items: list[tuple[str, str]] = []
        self._href = ""
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._href = dict(attrs).get("href") or ""
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href:
            label = _text(" ".join(self._parts))
            if label:
                self.items.append((label, self._href))
            self._href = ""
            self._parts = []
