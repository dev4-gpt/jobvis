"""Apify normalization helpers and explicit configured-task execution."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any

import httpx

from job_scout.config import get_settings
from job_scout.graph.schemas import JobPosting

logger = logging.getLogger(__name__)

DESCRIPTION_LIMIT = 4000
APIFY_API_BASE = "https://api.apify.com/v2"


class ApifyTaskError(ValueError):
    pass


class ConfiguredApifyTask:
    """One explicitly requested saved task; never part of ordinary source fan-out."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.state: dict[str, Any] = {"status": "idle", "run_id": None}

    def configuration(self) -> tuple[dict, dict]:
        settings = self.settings
        if not settings.apify_api_token.get_secret_value() or not re.fullmatch(r"[\w~-]+", settings.apify_task_id):
            raise ApifyTaskError("Configure APIFY_API_TOKEN and APIFY_TASK_ID")
        try:
            template = json.loads(settings.apify_input_template)
            mapping = json.loads(settings.apify_output_mapping)
            if not isinstance(template, dict) or not isinstance(mapping, dict):
                raise ValueError
            if not all(isinstance(mapping.get(key), str) and mapping[key] for key in ("title", "company", "url")):
                raise ValueError
            if not all(isinstance(path, str) and path for path in mapping.values()):
                raise ValueError
            self._input(template, {"query": "test", "location": "", "country": "us", "remote": False, "limit": 25})
        except (ValueError, TypeError, KeyError) as exc:
            raise ApifyTaskError("Configure a JSON input template and title/company/url output field mapping") from exc
        return template, mapping

    def _input(self, value, criteria):
        if isinstance(value, dict):
            return {key: self._input(item, criteria) for key, item in value.items()}
        if isinstance(value, list):
            return [self._input(item, criteria) for item in value]
        if isinstance(value, str) and value.startswith("$"):
            return criteria[value[1:]]
        return value

    def run(self, criteria: dict, *, client=None, deadline: float = 120, cancelled=lambda: False) -> list[JobPosting]:
        template, mapping = self.configuration()
        criteria = {key: criteria[key] for key in ("query", "location", "country", "remote", "limit")}
        criteria["limit"] = min(25, int(criteria["limit"]))
        own = client is None
        client = client or httpx.Client(headers={"Authorization": f"Bearer {self.settings.apify_api_token.get_secret_value()}"})
        end = time.monotonic() + deadline
        run_id = None
        self.state = {"status": "starting", "run_id": None}

        def request(method, path, **kwargs):
            remaining = end - time.monotonic()
            if remaining <= 0 or cancelled():
                raise TimeoutError
            response = client.request(method, APIFY_API_BASE + path, timeout=min(10, remaining), **kwargs)
            response.raise_for_status()
            return response.json()

        try:
            run = request(
                "POST",
                f"/actor-tasks/{self.settings.apify_task_id}/runs",
                params={"timeout": 120, "maxItems": 25},
                json=self._input(template, criteria),
            )["data"]
            run_id = run["id"]
            self.state = {"status": "running", "run_id": run_id}
            while run.get("status") not in {"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"}:
                if cancelled() or time.monotonic() >= end:
                    raise TimeoutError
                time.sleep(min(1, max(0, end - time.monotonic())))
                run = request("GET", f"/actor-runs/{run_id}")["data"]
            if run["status"] != "SUCCEEDED":
                raise ApifyTaskError(f"Apify run ended with {run['status']}")
            items = request("GET", f"/datasets/{run['defaultDatasetId']}/items", params={"limit": 25, "clean": "true"})
            if not isinstance(items, list):
                raise ApifyTaskError("Apify dataset response is not a list")
            fleet = ApifyJobFleet(api_token="")
            jobs = []
            for raw in items[:25]:
                if not isinstance(raw, dict):
                    continue
                mapped = {}
                for key, source in mapping.items():
                    value = raw
                    for part in source.split("."):
                        value = value.get(part) if isinstance(value, dict) else None
                    mapped[key] = value
                job = fleet.normalize_apify_item(mapped)
                from job_scout.tools.liveness import valid_job_url

                if job and valid_job_url(job.application_url or job.url):
                    jobs.append(job)
            if items and not jobs:
                raise ApifyTaskError("Dataset returned records but none matched the configured output mapping")
            self.state.update(status="complete", returned=len(jobs))
            return jobs
        except (TimeoutError, httpx.TimeoutException) as exc:
            aborted = False
            if run_id:
                try:
                    response = client.request("POST", f"{APIFY_API_BASE}/actor-runs/{run_id}/abort", timeout=5)
                    aborted = response.is_success
                except httpx.HTTPError:
                    pass
            self.state.update(
                status="timed_out",
                cancellation_requested=aborted,
                error="Run stopped waiting; cancellation was attempted. Provider charges may still apply.",
            )
            raise ApifyTaskError(self.state["error"]) from exc
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            message = str(exc) if isinstance(exc, ApifyTaskError) else f"Apify request or response failed ({type(exc).__name__})"
            self.state.update(status="failed", error=message)
            raise ApifyTaskError(message) from exc
        finally:
            if own:
                client.close()


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

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, headers={"Authorization": f"Bearer {self.api_token}"}, json=run_input)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    return data
                return []
        except httpx.HTTPStatusError as exc:
            logger.warning("Apify actor returned HTTP %s", exc.response.status_code)
            return []
        except Exception as exc:
            logger.warning("Apify actor run failed: %s", type(exc).__name__)
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

        task = ConfiguredApifyTask()
        return task.run({"query": query, "location": location or "", "country": "", "remote": remote, "limit": min(limit, 25)})

    def normalize_apify_item(self, item: dict[str, Any]) -> JobPosting | None:
        """Normalize an Apify dataset item into a validated JobPosting."""
        try:
            title = (item.get("title") or item.get("jobTitle") or item.get("position") or "").strip()
            company = (item.get("company") or item.get("companyName") or item.get("employer") or "").strip()
            if not title or not company:
                return None

            location = item.get("location") or item.get("city") or "Unspecified"
            url = item.get("url") or item.get("jobUrl") or item.get("applyUrl") or ""
            description = item.get("description") or item.get("text") or ""
            if len(description) > DESCRIPTION_LIMIT:
                description = description[:DESCRIPTION_LIMIT]

            flag = item.get("isRemote", item.get("remote", False))
            remote = flag is True or str(flag).lower() == "true" or "remote" in location.lower()

            raw_id = str(item.get("id") or item.get("jobId") or hashlib.sha256(url.encode()).hexdigest()[:20])
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
            logger.debug("Failed to normalize Apify item: %s", type(exc).__name__)
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
