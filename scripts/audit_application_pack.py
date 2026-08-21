#!/usr/bin/env python3
"""Audit downloaded Jobvis CV and cover-letter artifacts without an LLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from job_scout.evals.artifacts import audit_rendered_pack  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-cv", required=True, type=Path)
    parser.add_argument("--tailored-cv", required=True, type=Path)
    parser.add_argument("--cover-letter-pdf", type=Path)
    parser.add_argument("--cover-letter-text", type=Path)
    parser.add_argument("--job-description", required=True, type=Path)
    parser.add_argument("--cv-tex", type=Path)
    parser.add_argument("--cover-letter-tex", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.cover_letter_pdf and not args.cover_letter_text:
        parser.error("provide --cover-letter-pdf or --cover-letter-text")
    report = audit_rendered_pack(
        args.original_cv,
        args.tailored_cv,
        cover_letter_pdf=args.cover_letter_pdf,
        cover_letter_text=args.cover_letter_text.read_text(encoding="utf-8") if args.cover_letter_text else None,
        job_description=args.job_description.read_text(encoding="utf-8"),
        cv_tex=args.cv_tex,
        cover_letter_tex=args.cover_letter_tex,
    )
    rendered = json.dumps(report.as_dict(), indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
