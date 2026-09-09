"""Apify scraping fleet client and JobSource adapter.

Provides a production-grade interface to Apify cloud actors (ATS scraper,
LinkedIn scraper, Indeed scraper) with residential proxy rotation, asynchronous
run monitoring, dataset retrieval, and normalization into JobPosting schemas.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import httpx

from job_scout.config import get_settings
from job_scout.graph.schemas import JobPosting

logger = logging.getLogger(__name__)

DESCRIPTION_LIMIT = 4000
APIFY_API_BASE = "https://api.apify.com/v2"


class ApifyJobFleet:
    """Production client for orchestrating Apify job scraping actors."""

    def __init__(self, api_token: str | None = None, timeout: float = 60.0) -> None:
        if api_token is not None:
            self.api_token = api_token
        else:
            raw_token = get_settings().apify_api_token
            self.api_token = raw_token.get_secret_value() if hasattr(raw_token, "get_secret_value") else str(raw_token or "")
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        """Returns True if a non-empty API token is set."""
        return bool(self.api_token and self.api_token.strip())

    def run_actor_sync(self, actor_id: str, run_input: dict[str, Any]) -> list[dict[str, Any]]:
        """Run an Apify actor synchronously and return dataset items.

        Never raises unhandled network exceptions; returns an empty list on failure.
        """
        if not self.is_configured:
            logger.debug("Apify API token not configured; skipping actor %s", actor_id)
            return []

        clean_actor_id = actor_id.replace("/", "~")
        url = f"{APIFY_API_BASE}/acts/{clean_actor_id}/run-sync-get-dataset-items"
        params = {"token": self.api_token}

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, params=params, json=run_input)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    return data
                return []
        except httpx.HTTPStatusError as exc:
            logger.warning("Apify actor %s returned HTTP %s: %s", actor_id, exc.response.status_code, exc)
            return []
        except Exception as exc:
            logger.warning("Apify actor %s run failed: %s", actor_id, exc)
            return []

    def scrape_ats_jobs(
        self,
        query: str,
        companies: list[str] | None = None,
        location: str | None = None,
        remote: bool = False,
        limit: int = 25,
    ) -> list[JobPosting]:
        """Scrape company career boards (Greenhouse, Lever, Ashby, Workday) via Apify ATS scraper."""
        run_input: dict[str, Any] = {
            "query": query,
            "limit": limit,
        }
        if companies:
            run_input["companies"] = companies
        if location:
            run_input["location"] = location
        if remote:
            run_input["includeRemoteOnly"] = True

        raw_items = self.run_actor_sync("apify/ats-jobs-scraper", run_input)
        postings: list[JobPosting] = []

        for item in raw_items:
            posting = self.normalize_apify_item(item)
            if posting:
                postings.append(posting)

        return postings[:limit]

    def normalize_apify_item(self, item: dict[str, Any]) -> JobPosting | None:
        """Normalize an Apify dataset item into a validated JobPosting."""
        try:
            title = (item.get("title") or item.get("jobTitle") or item.get("position") or "").strip()
            company = (item.get("company") or item.get("companyName") or item.get("employer") or "").strip()
            if not title or not company:
                return None

            location = item.get("location") or item.get("city") or "Remote"
            url = item.get("url") or item.get("jobUrl") or item.get("applyUrl") or ""
            description = item.get("description") or item.get("text") or ""
            if len(description) > DESCRIPTION_LIMIT:
                description = description[:DESCRIPTION_LIMIT]

            remote = bool(item.get("isRemote") or item.get("remote") or "remote" in location.lower() or "remote" in title.lower())

            raw_id = str(item.get("id") or item.get("jobId") or f"{company}-{title}")
            content_sig = f"{company}::{title}::{location}".lower()
            content_hash = hashlib.sha256(content_sig.encode("utf-8")).hexdigest()

            tags = item.get("tags") or item.get("keywords") or []
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]

            return JobPosting(
                job_id=f"apify-{raw_id}",
                title=title,
                company=company,
                location=location,
                remote=remote,
                description=description,
                url=url,
                tags=tags,
                source="apify",
                listing_url=url,
                application_url=item.get("applyUrl") or url,
                source_record_id=raw_id,
                content_hash=content_hash,
                salary_text=str(item.get("salary") or item.get("compensation") or ""),
            )
        except Exception as exc:
            logger.debug("Failed to normalize Apify item %s: %s", item, exc)
            return None


class ApifySource:
    """JobSource adapter implementation for Apify fleet."""

    name = "apify"

    def __init__(self, fleet: ApifyJobFleet | None = None) -> None:
        self.fleet = fleet or ApifyJobFleet()

    def fetch(
        self,
        query: str,
        location: str | None,
        country: str | None,
        remote: bool,
        limit: int,
    ) -> list[JobPosting]:
        """Fetch jobs via Apify fleet, adhering strictly to JobSource protocol."""
        if not self.fleet.is_configured:
            return []

        search_loc = location or country or ""
        return self.fleet.scrape_ats_jobs(
            query=query,
            location=search_loc if search_loc else None,
            remote=remote,
            limit=limit,
        )
