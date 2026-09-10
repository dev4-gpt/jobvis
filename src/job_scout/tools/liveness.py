"""Job Listing Liveness Classifier & Verification Engine.

Ported and extended from career-ops liveness engine.
Determines whether a job posting is active, expired, or uncertain
before wasting LLM tokens or submitting forms.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
from typing import Any
from urllib.parse import urlsplit

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

HARD_EXPIRED_PATTERNS = [
    re.compile(r"job (is )?no longer available", re.IGNORECASE),
    re.compile(r"job.*no longer open", re.IGNORECASE),
    re.compile(r"position has been filled", re.IGNORECASE),
    re.compile(r"this job has expired", re.IGNORECASE),
    re.compile(r"job posting has expired", re.IGNORECASE),
    re.compile(r"no longer accepting applications", re.IGNORECASE),
    re.compile(r"this (position|role|job) (is )?no longer", re.IGNORECASE),
    re.compile(r"this job (listing )?is closed", re.IGNORECASE),
    re.compile(r"job (listing )?not found", re.IGNORECASE),
    re.compile(r"the page you are looking for doesn.t exist", re.IGNORECASE),
    re.compile(r"diese stelle (ist )?(nicht mehr|bereits) besetzt", re.IGNORECASE),
    re.compile(r"offre (expirée|n'est plus disponible)", re.IGNORECASE),
]

LISTING_PAGE_PATTERNS = [
    re.compile(r"\d+\s+jobs?\s+found", re.IGNORECASE),
    re.compile(r"search for jobs page is loaded", re.IGNORECASE),
]

EXPIRED_URL_PATTERNS = [
    re.compile(r"[?&]error=true", re.IGNORECASE),
]

APPLY_PATTERNS = [
    re.compile(r"\bapply\b", re.IGNORECASE),
    re.compile(r"\bsolicitar\b", re.IGNORECASE),
    re.compile(r"\bbewerben\b", re.IGNORECASE),
    re.compile(r"\bpostuler\b", re.IGNORECASE),
    re.compile(r"submit application", re.IGNORECASE),
    re.compile(r"easy apply", re.IGNORECASE),
    re.compile(r"start application", re.IGNORECASE),
    re.compile(r"ich bewerbe mich", re.IGNORECASE),
]

MIN_CONTENT_CHARS = 300
_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_LOCK = Lock()
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="job-liveness")


def valid_job_url(url: str) -> bool:
    """Only absolute public HTTP URLs belong in application handoffs."""
    import ipaddress

    try:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        if parsed.scheme not in {"https", "http"} or not host or parsed.username or parsed.password:
            return False
        if host == "localhost" or host.endswith((".local", ".localhost", ".internal")):
            return False
        try:
            return ipaddress.ip_address(host).is_global
        except ValueError:
            return "." in host
    except ValueError:
        return False


def verified_liveness(url: str, *, force: bool = False) -> dict:
    now = time.time()
    with _CACHE_LOCK:
        cached = _CACHE.get(url)
        if not force and cached and now - cached[0] < 900:
            return dict(cached[1])
    if not valid_job_url(url):
        result = {"liveness": "uncertain", "reason": "Missing or unsupported public job URL", "url": url}
    else:
        result = check_job_liveness(url, timeout=8)
    result["checked_at"] = now
    with _CACHE_LOCK:
        if len(_CACHE) >= 2000:
            _CACHE.clear()
        _CACHE[url] = (now, result)
    return dict(result)


def filter_live_jobs(jobs: list, diagnostics: list | None = None, *, budget: float = 15.0) -> list:
    """Bounded shared verification stage; unfinished candidates remain visible."""
    pending = {}
    for job in jobs:
        url = job.application_url or job.url
        if url not in pending:
            pending[url] = _POOL.submit(verified_liveness, url)
    done, unfinished = wait(pending.values(), timeout=budget)
    for future in unfinished:
        future.cancel()
    kept = []
    for job in jobs:
        future = pending[job.application_url or job.url]
        result = {"liveness": "uncertain", "reason": "Verification stage deadline reached"}
        if future in done:
            try:
                result = future.result()
            except Exception:
                result = {"liveness": "uncertain", "reason": "Verification unavailable"}
        job = job.model_copy(update={"liveness": result})
        for diagnostic in diagnostics or []:
            if diagnostic.source == job.source:
                diagnostic.expired += int(result["liveness"] == "expired")
                diagnostic.incomplete_checks += int(result["liveness"] == "uncertain")
        if result["liveness"] != "expired":
            kept.append(job)
    return kept


def classify_liveness(
    status_code: int = 200,
    final_url: str = "",
    body_text: str = "",
    apply_controls: list[str] | None = None,
) -> dict[str, str]:
    """Classifies whether a job listing is active, expired, or uncertain."""
    if status_code in (404, 410):
        return {"status": "expired", "reason": f"HTTP {status_code}"}
    if status_code < 200 or status_code >= 400:
        return {"status": "uncertain", "reason": f"HTTP {status_code}; could not verify listing"}

    for pattern in EXPIRED_URL_PATTERNS:
        if pattern.search(final_url):
            return {"status": "expired", "reason": f"redirect to expired url: {final_url}"}

    for pattern in HARD_EXPIRED_PATTERNS:
        match = pattern.search(body_text)
        if match:
            return {"status": "expired", "reason": f"pattern matched: {pattern.pattern}"}

    for pattern in LISTING_PAGE_PATTERNS:
        if pattern.search(body_text):
            return {"status": "expired", "reason": "redirected to general search page"}

    controls = apply_controls or []
    has_apply = False
    for control in controls:
        for pattern in APPLY_PATTERNS:
            if pattern.search(control):
                has_apply = True
                break
        if has_apply:
            break

    if has_apply:
        return {"status": "active", "reason": "visible apply control detected"}

    # Also check if body_text contains apply patterns directly
    for pattern in APPLY_PATTERNS:
        if pattern.search(body_text):
            return {"status": "active", "reason": f"apply pattern found in text: {pattern.pattern}"}

    for pattern in LISTING_PAGE_PATTERNS:
        if pattern.search(body_text):
            return {"status": "expired", "reason": f"redirected to general search page: {pattern.pattern}"}

    if len(body_text.strip()) < MIN_CONTENT_CHARS:
        return {"status": "uncertain", "reason": "insufficient content (empty page or JavaScript shell)"}

    return {"status": "uncertain", "reason": "content present but no explicit apply control found"}


def check_job_liveness(
    url: str,
    timeout: float = 8.0,
    client: Any | None = None,
) -> dict[str, Any]:
    """Performs an HTTP GET verification on the job URL and classifies its liveness.

    Returns:
        dict with status ('active', 'expired', 'uncertain'), reason, status_code, and final_url.
    """
    if client is None and httpx is None:
        return {
            "url": url,
            "final_url": url,
            "status_code": 0,
            "liveness": "uncertain",
            "reason": "httpx library not available for live network check",
        }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    own_client = client is None
    http_client = client or httpx.Client(follow_redirects=True, timeout=timeout, headers=headers)

    try:
        response = http_client.get(url)
        raw_text = response.text
        visible_text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", raw_text, flags=re.IGNORECASE | re.DOTALL)
        clean_text = re.sub(r"<[^>]+>", " ", visible_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        button_matches = re.findall(
            r"<(?:button|a)[^>]*>(.*?)</(?:button|a)>",
            raw_text,
            re.IGNORECASE | re.DOTALL,
        )
        apply_controls = [re.sub(r"<[^>]+>", "", b).strip() for b in button_matches]

        classification = classify_liveness(
            status_code=response.status_code,
            final_url=str(response.url),
            body_text=clean_text,
            apply_controls=apply_controls,
        )

        return {
            "url": url,
            "final_url": str(response.url),
            "status_code": response.status_code,
            "liveness": classification["status"],
            "reason": classification["reason"],
        }
    except Exception as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", 0)
        if status_code in (404, 410):
            return {
                "url": url,
                "final_url": url,
                "status_code": status_code,
                "liveness": "expired",
                "reason": f"HTTP {status_code}",
            }
        logger.warning(f"Liveness check for {url} failed with error: {exc}")
        return {
            "url": url,
            "final_url": url,
            "status_code": status_code,
            "liveness": "uncertain",
            "reason": f"Check failed: {type(exc).__name__}",
        }
    finally:
        if own_client and hasattr(http_client, "close"):
            http_client.close()
