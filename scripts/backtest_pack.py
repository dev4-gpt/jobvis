#!/usr/bin/env python3
"""Run the deterministic application-pack backtest on a saved pack.

This command is intentionally offline. It reads a CV, a job description, and
the JSON representation of a ``TailoringPack``; it never calls a model or a
job source. The live tailoring graph runs the same backtest automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from job_scout.evals.backtest import backtest_pack  # noqa: E402
from job_scout.graph.schemas import TailoringPack  # noqa: E402
from job_scout.tools.cv_reader import extract_cv_document  # noqa: E402


def _jsonable(value: Any) -> Any:
    """Convert dataclasses and Pydantic models into JSON-safe primitives."""
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", required=True, type=Path, help="Original CV PDF or plain-text CV")
    parser.add_argument("--pack", required=True, type=Path, help="TailoringPack JSON produced by the application")
    parser.add_argument("--job-description", required=True, type=Path, help="Text file containing the target job description")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON report")
    args = parser.parse_args()

    if args.cv.suffix.lower() == ".pdf":
        cv_text, source_links = extract_cv_document(args.cv)
    else:
        cv_text = args.cv.read_text(encoding="utf-8")
        source_links = []
    pack = TailoringPack.model_validate(json.loads(args.pack.read_text(encoding="utf-8")))
    report = backtest_pack(
        pack,
        cv_text,
        args.job_description.read_text(encoding="utf-8"),
        source_links=source_links,
    )
    payload = _jsonable(report)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
