"""Regression tests for schema-only imports and package boundaries."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_candidate_fit_can_import_without_eagerly_building_the_graph() -> None:
    """Importing policy schemas must not recurse through graph node imports."""
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    result = subprocess.run(
        [sys.executable, "-c", "import job_scout.candidate_fit; import job_scout.graph.schemas"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
