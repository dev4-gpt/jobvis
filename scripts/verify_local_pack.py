#!/usr/bin/env python3
"""Run one real local search→tailor→render→audit smoke test.

This is opt-in because it uses the configured job sources and model provider.
It never uploads a resume to GitHub or Opik as an evaluation dataset and never
submits an application.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from job_scout.evals.pack_loop import render_verified_pack  # noqa: E402
from job_scout.runner import run_once, stream_tailor  # noqa: E402
from job_scout.tools.cv_reader import extract_cv_document  # noqa: E402
from job_scout.voice import bridge as voice_bridge  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", required=True, type=Path, help="Local resume PDF")
    parser.add_argument("--yes", action="store_true", help="Run the live provider-backed smoke test")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if not args.yes:
        print("This command uses configured providers and job sources. Re-run with --yes to execute.")
        return 2
    if not args.cv.is_file():
        print(f"CV not found: {args.cv}", file=sys.stderr)
        return 2

    thread_id = str(uuid4())
    cv_text, cv_links = extract_cv_document(args.cv)
    search = run_once(
        cv_text,
        cv_path=str(args.cv),
        cv_links=cv_links,
        thread_id=thread_id,
        tags=["local-pack-e2e"],
    )
    report: dict[str, object] = {
        "thread_id": thread_id,
        "cv": str(args.cv),
        "search_failed": search.failed,
        "search_error": search.error_message,
        "ranked_jobs": len(search.ranked_jobs),
        "source_links": len(cv_links),
    }
    if search.failed or not search.ranked_jobs:
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 1

    selected = search.ranked_jobs[0]
    result = None
    for kind, payload in stream_tailor(
        thread_id=thread_id,
        selected_job_id=selected.job.job_id,
        tags=["local-pack-e2e"],
    ):
        if kind == "result":
            result = payload
    if result is None or result.pack is None:
        report.update({"tailor_failed": True, "tailor_error": "no tailoring pack returned"})
        rendered = json.dumps(report, indent=2, sort_keys=True)
        print(rendered)
        return 1

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
            "tailor_failed": False,
            "job_id": selected.job.job_id,
            "job_title": selected.job.title,
            "company": selected.job.company,
            "manifest": verified.manifest.as_dict(),
            "issues": [issue.__dict__ for issue in verified.report.issues],
        }
    )
    if verified.report.passed:
        print(f"Verified pack written in temporary directory: {directory}")
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if verified.report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
