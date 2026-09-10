#!/usr/bin/env python3
"""Run one real local search→tailor→render→audit smoke test.

This is opt-in because it uses the configured job sources and model provider.
It never uploads a resume to GitHub or Opik as an evaluation dataset and never
submits an application.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from job_scout.evals.pack_loop import render_verified_pack  # noqa: E402
from job_scout.graph.schemas import CandidatePreferences  # noqa: E402
from job_scout.runner import RunResult, extract_profile, stream_search, stream_tailor  # noqa: E402
from job_scout.tools.cv_reader import extract_cv_document  # noqa: E402
from job_scout.voice import bridge as voice_bridge  # noqa: E402


class PackTimeoutError(TimeoutError):
    """Raised when pack verification exceeds the configured timeout."""

    def __init__(self, phase: str, timeout_seconds: int):
        super().__init__(f"pack_e2e_timeout: exceeded {timeout_seconds}s limit during phase '{phase}'")
        self.phase = phase
        self.timeout_seconds = timeout_seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", required=True, type=Path, help="Local resume PDF")
    parser.add_argument("--yes", action="store_true", help="Run the live provider-backed smoke test")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("JOBVIS_PACK_TIMEOUT", "300")),
        help="Outer timeout in seconds (default: 300; 0 disables)",
    )
    args = parser.parse_args()
    if not args.yes:
        print("This command uses configured providers and job sources. Re-run with --yes to execute.", flush=True)
        return 2
    if not args.cv.is_file():
        print(f"CV not found: {args.cv}", file=sys.stderr, flush=True)
        return 2

    overall_start = time.monotonic()
    current_phase = "init"

    def log_phase(phase: str, description: str) -> None:
        nonlocal current_phase
        current_phase = phase
        now = datetime.now().strftime("%H:%M:%S")
        elapsed = round(time.monotonic() - overall_start, 1)
        print(f"[{now}] (+{elapsed:>5.1f}s) [{phase}] {description}", flush=True)

    thread_id = str(uuid4())

    def alarm_handler(signum: int, frame: object) -> None:
        raise PackTimeoutError(current_phase, args.timeout)

    use_alarm = args.timeout > 0 and hasattr(signal, "SIGALRM")
    old_handler = None
    if use_alarm:
        old_handler = signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(args.timeout)

    try:
        # Phase 1: Extract candidate resume document
        log_phase("cv_extraction", f"Extracting candidate resume from {args.cv}...")
        cv_text, cv_links = extract_cv_document(args.cv)
        print(f"  → Extracted {len(cv_text):,} characters and {len(cv_links)} link(s)", flush=True)

        # Phase 2: Extract candidate profile and search/rank jobs
        log_phase("search_and_rank", "Extracting profile and searching/ranking jobs across configured sources...")
        profile = extract_profile(cv_text, thread_id=thread_id, tags=["local-pack-e2e"])
        if profile and profile.name:
            print(f"  → Candidate profile: {profile.name} ({', '.join(profile.primary_roles) or 'general'})", flush=True)

        search: RunResult | None = None
        for kind, payload in stream_search(
            profile,
            cv_text=cv_text,
            cv_path=str(args.cv),
            cv_links=cv_links,
            thread_id=thread_id,
            tags=["local-pack-e2e"],
            preferences=CandidatePreferences(),
        ):
            if kind == "status":
                print(f"  → {payload}", flush=True)
            elif kind == "result":
                search = payload  # type: ignore[assignment]

        if search is None:
            search = RunResult(failed=True, error_message="stream_search did not yield a result")

        report: dict[str, object] = {
            "status": "search_failed" if (search.failed or not search.ranked_jobs) else "search_ok",
            "thread_id": thread_id,
            "cv": str(args.cv),
            "search_failed": search.failed,
            "search_error": search.error_message,
            "ranked_jobs": len(search.ranked_jobs),
            "source_links": len(cv_links),
            "elapsed_seconds": round(time.monotonic() - overall_start, 2),
        }
        if search.failed or not search.ranked_jobs:
            log_phase("search_and_rank", f"Search completed with failure or 0 ranked jobs ({search.error_message or 'empty'})")
            rendered = json.dumps(report, indent=2, sort_keys=True)
            print(rendered, flush=True)
            if args.output:
                args.output.write_text(rendered + "\n", encoding="utf-8")
            return 1

        selected = search.ranked_jobs[0]
        log_phase(
            "tailor_pack",
            f"Tailoring application pack for '{selected.job.title}' at '{selected.job.company}' ({selected.job.job_id})...",
        )
        result = None
        for kind, payload in stream_tailor(
            thread_id=thread_id,
            selected_job_id=selected.job.job_id,
            tags=["local-pack-e2e"],
        ):
            if kind == "status":
                print(f"  → {payload}", flush=True)
            elif kind == "result":
                result = payload

        if result is None or result.pack is None:
            report.update(
                {
                    "status": "tailor_failed",
                    "tailor_failed": True,
                    "tailor_error": result.error_message if result and result.error_message else "no tailoring pack returned",
                    "elapsed_seconds": round(time.monotonic() - overall_start, 2),
                }
            )
            log_phase("tailor_pack", f"Tailoring failed: {report['tailor_error']}")
            rendered = json.dumps(report, indent=2, sort_keys=True)
            print(rendered, flush=True)
            if args.output:
                args.output.write_text(rendered + "\n", encoding="utf-8")
            return 1

        # Phase 4: Render verified pack & audit compliance
        log_phase("render_and_audit", "Rendering LaTeX PDFs and auditing verified pack...")
        values = voice_bridge.checkpoint_values(thread_id)
        directory = tempfile.mkdtemp(prefix="jobvis_local_pack_")
        verified = render_verified_pack(
            result.pack,
            candidate_name=search.profile.name if search.profile else "Candidate",
            source_text=cv_text,
            source_links=list(values.get("cv_links") or cv_links),
            job_description=selected.job.description,
            company=selected.job.company,
            job_title=selected.job.title,
            out_dir=Path(directory),
            selected_job_id=selected.job.job_id,
            generation_id=thread_id,
            backtest_score=result.backtest_score,
        )
        report.update(
            {
                "status": "verified" if verified.report.passed else "audit_failed",
                "tailor_failed": False,
                "job_id": selected.job.job_id,
                "job_title": selected.job.title,
                "company": selected.job.company,
                "manifest": verified.manifest.as_dict(),
                "issues": [issue.__dict__ for issue in verified.report.issues],
                "elapsed_seconds": round(time.monotonic() - overall_start, 2),
            }
        )
        if verified.report.passed:
            log_phase("render_and_audit", f"Verified pack passed and written to: {directory}")
        else:
            log_phase("render_and_audit", f"Verified pack audit failed with {len(verified.report.issues)} issue(s)")
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered, flush=True)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 0 if verified.report.passed else 1

    except PackTimeoutError as exc:
        elapsed = round(time.monotonic() - overall_start, 2)
        report = {
            "status": "pack_e2e_timeout",
            "timed_out": True,
            "timeout_seconds": exc.timeout_seconds,
            "phase": exc.phase,
            "error": str(exc),
            "thread_id": thread_id,
            "cv": str(args.cv),
            "elapsed_seconds": elapsed,
        }
        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] [TIMEOUT] Pack verification timed out after "
            f"{exc.timeout_seconds}s during phase '{exc.phase}'",
            file=sys.stderr,
            flush=True,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered, flush=True)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 1

    except KeyboardInterrupt:
        elapsed = round(time.monotonic() - overall_start, 2)
        report = {
            "status": "pack_e2e_interrupted",
            "interrupted": True,
            "phase": current_phase,
            "error": f"pack_e2e_interrupted: user interrupted execution during phase '{current_phase}'",
            "thread_id": thread_id,
            "cv": str(args.cv),
            "elapsed_seconds": elapsed,
        }
        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] [INTERRUPTED] Pack verification interrupted "
            f"by user during phase '{current_phase}'",
            file=sys.stderr,
            flush=True,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered, flush=True)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 130

    except Exception as exc:
        elapsed = round(time.monotonic() - overall_start, 2)
        report = {
            "status": "pack_e2e_error",
            "phase": current_phase,
            "error": f"{type(exc).__name__}: {exc}",
            "thread_id": thread_id,
            "cv": str(args.cv),
            "elapsed_seconds": elapsed,
        }
        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] [ERROR] Pack verification failed with unhandled "
            f"error during phase '{current_phase}': {exc}",
            file=sys.stderr,
            flush=True,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered, flush=True)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 1

    finally:
        if use_alarm:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


if __name__ == "__main__":
    raise SystemExit(main())
