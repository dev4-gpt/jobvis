"""One persisted Jobvis shortlist per local date, shared by the console and local scheduler."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from job_scout import candidate_store
from job_scout.application.identity import posting_url
from job_scout.application.store import ApplicationStore, default_data_dir, file_lock, record_urls, write_private_json
from job_scout.config import get_settings
from job_scout.graph.nodes.rank_jobs import rank_jobs
from job_scout.graph.schemas import JobPosting
from job_scout.tools.jobs_api import run_search_detailed
from job_scout.watchlists import Watchlist, WatchlistStore


def date_key(zone: str) -> tuple[str, ZoneInfo]:
    tz = ZoneInfo(zone)
    return datetime.now(tz).date().isoformat(), tz


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def shortlist_jobs(jobs: list[JobPosting], excluded: set[str] | None = None) -> list[JobPosting]:
    groups: list[tuple[JobPosting, set[str]]] = []
    for job in jobs:
        keys = {f"id:{job.source}:{job.source_record_id or job.job_id}"}
        keys |= {url for value in (job.application_url, job.listing_url, job.url, job.source_url) if (url := posting_url(value))}
        overlaps = [i for i, (_, seen) in enumerate(groups) if seen & keys]
        if not overlaps:
            groups.append((job, keys))
            continue
        first = overlaps[0]
        for index in reversed(overlaps[1:]):
            keys |= groups.pop(index)[1]
        groups[first][1].update(keys)
    return [job for job, keys in groups if not excluded or not keys & excluded]


class DailyShortlist:
    def __init__(self, data_dir: Path | None = None, *, zone: str | None = None) -> None:
        self.data_dir = data_dir or default_data_dir()
        self.zone = zone or os.environ.get("JOBVIS_DIGEST_TIMEZONE", "America/New_York")
        self.date, self.tz = date_key(self.zone)
        zone_key = hashlib.sha256(self.zone.encode()).hexdigest()[:12]
        self.path = self.data_dir / "daily" / f"{self.date}-{zone_key}.json"

    def get(self) -> dict | None:
        return _read(self.path)

    def followups(self) -> list[dict]:
        pending = []
        for record in ApplicationStore(self.data_dir).list():
            if record.status not in {"submitted_by_user", "submitted_by_jobvis", "interview"}:
                continue
            completed = {event.detail for event in record.events if event.name == "followup_done"}
            scheduled = record.followup_due if record.status != "interview" else [record.thank_you_due]
            for due in filter(None, scheduled):
                try:
                    day = datetime.fromisoformat(due).astimezone(self.tz).date().isoformat()
                except ValueError:
                    continue
                if day <= self.date and due not in completed:
                    pending.append(
                        {
                            "application_id": record.application_id,
                            "company": record.company,
                            "title": record.title,
                            "due": due,
                            "kind": "thank_you" if record.status == "interview" else "followup",
                        }
                    )
        return sorted(pending, key=lambda row: row["due"])

    def run(self) -> dict:
        with file_lock(self.path.with_suffix(".lock")):
            existing = self.get()
            if existing:
                return existing
            candidate = candidate_store.load_candidate()
            if candidate is None:
                raise ValueError("Upload a résumé in Jobvis before running the shortlist.")
            settings = get_settings()
            profile = candidate_store.effective_profile(candidate.profile, candidate.preferences)
            saved = WatchlistStore(self.data_dir).list()
            location = (profile.locations or [""])[0]
            searches = (
                [item for item in saved if item.enabled]
                if saved
                else [
                    Watchlist(name=role, query=role, location=location, remote=profile.remote_ok)
                    for role in ("Machine Learning Engineer", "Applied AI Engineer")
                ]
            )
            if not searches:
                raise ValueError("Enable a saved search before running the shortlist.")
            jobs, diagnostics, searches_run = [], [], []
            for watch in searches[: settings.scout_max_role_queries]:
                result = run_search_detailed(watch.query, watch.location or None, watch.country, watch.remote)
                searches_run.append(watch.name)
                jobs.extend(job for job in result.jobs if job.source != "cache")
                diagnostics.extend(result.diagnostics)
            tracked = ApplicationStore(self.data_dir).list()
            seen = set().union(*(record_urls(record) for record in tracked))
            seen.update(f"id:{record.source}:{record.job_id}" for record in tracked)
            seen_ids = {record.job_id for record in tracked}
            postings = [job for job in shortlist_jobs(jobs, seen) if job.job_id not in seen_ids][: settings.scout_max_jobs]
            ranked = rank_jobs(
                {
                    "profile": profile,
                    "jobs": postings,
                    "source_diagnostics": diagnostics,
                    "candidate_preferences": candidate.candidate_preferences,
                }
            )
            payload = {
                "date": self.date,
                "timezone": self.zone,
                "searches": searches_run,
                "ranked": [job.model_dump(mode="json") for job in ranked.get("ranked_jobs", [])],
                "diagnostics": [row.model_dump(mode="json") for row in ranked.get("source_diagnostics", diagnostics)],
                "warnings": ranked.get("errors", []),
                "delivery": {"status": "preview"},
            }
            write_private_json(self.path, payload)
            return payload

    def render(self, run: dict) -> str:
        lines = [f"Jobvis daily shortlist · {self.date} ({self.zone})", "", "New matches"]
        eligible = [row for row in run["ranked"] if row["eligibility_status"] != "blocked"]
        for row in eligible:
            job = row["job"]
            link = job["application_url"] or job["url"]
            lines.append(f"{row['fit_score']:3}/100 · {job['title']} at {job['company']} · {link}")
        if not eligible:
            lines.append("No new eligible matches.")
        lines.extend(["", "Follow-ups due"])
        followups = self.followups()
        lines.extend(f"{row['company']} · {row['title']} · {row['kind']}" for row in followups)
        if not followups:
            lines.append("None due.")
        failures = {
            f"{row['source']}: {row.get('error') or 'timed out'}"
            for row in run["diagnostics"]
            if row.get("requested") and (row.get("error") or row.get("timed_out"))
        }
        if failures:
            lines.extend(["", "Source issues", *sorted(failures)])
        return "\n".join(lines)

    def send(self) -> dict:
        self.run()
        with file_lock(self.path.with_suffix(".lock")):
            run = self.get()
            assert run is not None
            if run["delivery"]["status"] != "preview":
                return run["delivery"]
            host = os.environ.get("JOBVIS_SMTP_HOST", "")
            sender = os.environ.get("JOBVIS_SMTP_FROM", "")
            recipient = os.environ.get("JOBVIS_DIGEST_EMAIL", "")
            if not all((host, sender, recipient)) or any(c in recipient for c in ",;\r\n"):
                raise ValueError("Set SMTP host, sender, and one digest recipient before sending.")
            message = EmailMessage()
            message["From"], message["To"] = sender, recipient
            message["Subject"] = f"Jobvis shortlist for {self.date}"
            message.set_content(self.render(run))
            # Claim durably before handing the email to SMTP. Never resend automatically
            # when an interrupted attempt might already have been accepted.
            run["delivery"] = {"status": "unconfirmed"}
            write_private_json(self.path, run)
            try:
                with smtplib.SMTP_SSL(
                    host, int(os.environ.get("JOBVIS_SMTP_PORT", "465")), timeout=20, context=ssl.create_default_context()
                ) as client:
                    if username := os.environ.get("JOBVIS_SMTP_USERNAME"):
                        client.login(username, os.environ.get("JOBVIS_SMTP_PASSWORD", ""))
                    client.send_message(message, from_addr=sender, to_addrs=[recipient])
                run["delivery"] = {"status": "sent"}
            except (OSError, ValueError, smtplib.SMTPException) as exc:
                run["delivery"]["error"] = f"{type(exc).__name__}: check the mail provider before retrying."
            write_private_json(self.path, run)
            return run["delivery"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--send", action="store_true", help="Email the daily preview to the configured recipient.")
    args = parser.parse_args()
    try:
        daily = DailyShortlist()
        if args.send:
            delivery = daily.send()
            print(json.dumps(delivery))
            if delivery["status"] != "sent":
                raise SystemExit(1)
        else:
            print(daily.render(daily.run()))
    except (OSError, ValueError, KeyError, ZoneInfoNotFoundError) as exc:
        parser.exit(1, f"Jobvis: {exc}\n")


if __name__ == "__main__":
    main()
