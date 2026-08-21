# Extending job sources

Job Scout fetches postings through a small, pluggable interface. The agent only
ever sees **one** LangChain tool, `search_jobs`; behind it, a `JobSource`
adapter (or several, merged) does the actual fetching. This doc shows how to add
a new source and explains why scraping-based sources are deliberately excluded.

## The `JobSource` protocol

```python
class JobSource(Protocol):
    name: str
    def fetch(
        self, query: str, location: str | None, country: str | None,
        remote: bool, limit: int,
    ) -> list[JobPosting]: ...
```

Contract:

- **Never raise on a network/parse error.** Return an empty list instead, so the
  orchestrator (`run_search`) can fall through to the next source and ultimately
  to the committed cache. A raising adapter would break offline reproducibility.
- **Normalize into `JobPosting`.** Map the provider's fields onto our schema
  (`job_id`, `title`, `company`, `location`, `remote`, `description`, `url`,
  `tags`, `source`). Prefix `job_id` with the source name to avoid collisions.
- **Truncate descriptions** to `DESCRIPTION_LIMIT` (4000 chars).

The shipped adapters include the current aggregators in `src/job_scout/tools/jobs_api.py`
and opt-in direct adapters in `src/job_scout/tools/direct_sources.py`:

- Greenhouse, Lever, Ashby: explicit public board/account identifiers only.
- USAJOBS: explicit API key and user-agent only.
- Protocol Labs Network: best-effort public directory parser, clearly labeled.
- Adzuna, JSearch, Remotive, and the committed cache remain available as before.

Enable direct adapters deliberately with `JOBVIS_DIRECT_SOURCES_ENABLED=true`
and configure only the boards you intend to query. A direct adapter failure is
recorded in source diagnostics and does not turn into a false zero-result claim.
Every normalized posting carries separate `listing_url`, `application_url`,
`source_record_id`, `content_hash`, and freshness metadata.

Remote.co, FlexJobs, Instahyre, and Protocol Jobs AI are controlled/manual
connectors. Use the local **Import this job** flow with a user-selected URL;
Jobvis does not crawl or aggregate those sites. The imported URL remains the
provenance link and enters the same eligibility, tailoring, and review-gated
application flow. LinkedIn and Indeed scraping are intentionally unsupported.

## Worked example: adding a hypothetical official API

Say "JobsCoAPI" offers an official REST endpoint with an API key.

```python
class JobsCoSource:
    name = "jobsco"
    BASE = "https://api.jobsco.example/v1/search"

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key or get_settings().jobsco_api_key.get_secret_value()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def fetch(self, query, location, country, remote, limit) -> list[JobPosting]:
        if not self.available:
            return []
        try:
            resp = httpx.get(
                self.BASE,
                params={"q": query, "loc": location, "limit": limit},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10.0,
            )
            resp.raise_for_status()
            rows = resp.json().get("results", [])
        except (httpx.HTTPError, ValueError):
            return []
        return [
            JobPosting(
                job_id=f"jobsco-{r['id']}",
                title=r["title"],
                company=r["employer"],
                location=r.get("location", ""),
                remote=r.get("remote", False),
                description=(r.get("description") or "")[:DESCRIPTION_LIMIT],
                url=r.get("apply_url", ""),
                tags=r.get("skills", []),
                source="jobsco",  # add to the JobSourceName Literal in schemas.py
            )
            for r in rows
        ]
```

Then wire it into `run_search(...)` in the order you want it tried, add its key
to `config.py` + `.env.example`, and extend the `JobSourceName` literal in
`schemas.py`. That's it — the agent's tool signature does not change.

## Why no scrapers

This repo will not include LinkedIn/Indeed scrapers or third-party scraping
actors (e.g. Apify LinkedIn actors), **even as optional adapters**:

- It violates those platforms' Terms of Service.
- It risks readers' own accounts being flagged or banned.
- It costs credits and is brittle, breaking reproducibility for the course.

If you wire up a scraper privately, you accept those risks yourself. The
supported, reproducible path is official APIs behind the `JobSource` interface.

## Watchlists

Watchlists are local query definitions, not a second job database. Create and
refresh them through `/api/watchlists`; each refresh is on-demand, bounded by
the same source timeouts, and stores only the query and last-refresh timestamp.
An external launchd/cron schedule may call the refresh endpoint if the user
explicitly wants a daily refresh. Job postings are fetched fresh and are not
silently written into the candidate store.
