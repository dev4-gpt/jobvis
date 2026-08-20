"""Regression tests for imports used by the local launch command."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_policy_import_does_not_compile_the_graph_eagerly() -> None:
    """Schema/policy imports must work before the graph package is compiled."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import job_scout.candidate_fit as fit; "
                "from job_scout.graph.schemas import CandidatePreferences; "
                "assert fit.preferences_from_dict({}) == CandidatePreferences()"
            ),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_app_import_succeeds_from_src_layout() -> None:
    """The same import path used by ``make app`` must remain loadable."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", "import job_scout.app"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_runtime_source_check_reports_identity() -> None:
    """The launch preflight must identify the checkout that is actually running."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "runtime_source_check.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "branch=" in result.stdout
    assert "commit=" in result.stdout
    assert str(ROOT) in result.stdout
