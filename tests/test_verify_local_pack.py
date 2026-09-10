from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import scripts.verify_local_pack as verifier  # noqa: E402
from job_scout.graph.schemas import CVContent, CVLink, TailoringPack  # noqa: E402


class TestVerifyLocalPack(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.cv_file = self.tmp_path / "resume.pdf"
        self.cv_file.write_bytes(b"%PDF-1.5 fake resume content")

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_requires_yes(self) -> None:
        with (
            patch.object(sys, "argv", ["verify_local_pack.py", "--cv", str(self.cv_file)]),
            patch("sys.stdout", new=io.StringIO()) as out,
        ):
            code = verifier.main()
            self.assertEqual(code, 2)
            self.assertIn("Re-run with --yes", out.getvalue())

    def test_requires_existing_cv(self) -> None:
        missing = self.tmp_path / "missing.pdf"
        with (
            patch.object(sys, "argv", ["verify_local_pack.py", "--cv", str(missing), "--yes"]),
            patch("sys.stderr", new=io.StringIO()) as err,
        ):
            code = verifier.main()
            self.assertEqual(code, 2)
            self.assertIn("CV not found", err.getvalue())

    def test_successful_flow_with_phases_and_report(self) -> None:
        out_json = self.tmp_path / "output.json"
        mock_profile = SimpleNamespace(name="Candidate Dev", primary_roles=["AI Engineer"])
        mock_job = SimpleNamespace(job_id="job-123", title="AI Engineer", company="Acme AI", description="Build AI.")
        mock_ranked_job = SimpleNamespace(job=mock_job)
        mock_search = SimpleNamespace(
            failed=False,
            error_message="",
            ranked_jobs=[mock_ranked_job],
            profile=mock_profile,
        )

        mock_tailor_result = SimpleNamespace(
            pack=TailoringPack(cv=CVContent(headline="AI Engineer", summary="Grounded summary"), cover_letter="Test letter"),
            backtest_score=0.95,
            error_message="",
        )

        mock_manifest = SimpleNamespace(as_dict=lambda: {"job_id": "job-123", "status": "ready", "pdfs_ready": True})
        mock_report = SimpleNamespace(passed=True, issues=[])
        mock_verified_pack = SimpleNamespace(manifest=mock_manifest, report=mock_report)

        with (
            patch.object(
                sys,
                "argv",
                ["verify_local_pack.py", "--cv", str(self.cv_file), "--yes", "--output", str(out_json), "--timeout", "10"],
            ),
            patch.object(
                verifier,
                "extract_cv_document",
                return_value=("Candidate CV text", [CVLink(label="Portfolio", url="https://example.com", page=1)]),
            ),
            patch.object(verifier, "extract_profile", return_value=mock_profile),
            patch.object(
                verifier,
                "stream_search",
                return_value=[("status", "searching jobs… 5 found"), ("result", mock_search)],
            ),
            patch.object(
                verifier,
                "stream_tailor",
                return_value=[("status", "tailoring pack…"), ("result", mock_tailor_result)],
            ),
            patch.object(verifier, "render_verified_pack", return_value=mock_verified_pack),
            patch.object(verifier.voice_bridge, "checkpoint_values", return_value={}),
            patch("sys.stdout", new=io.StringIO()) as out,
        ):
            code = verifier.main()

        self.assertEqual(code, 0)
        output_text = out.getvalue()
        self.assertIn("[cv_extraction]", output_text)
        self.assertIn("[search_and_rank]", output_text)
        self.assertIn("[tailor_pack]", output_text)
        self.assertIn("[render_and_audit]", output_text)

        self.assertTrue(out_json.is_file())
        saved_report = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(saved_report["status"], "verified")
        self.assertEqual(saved_report["job_id"], "job-123")
        self.assertFalse(saved_report["tailor_failed"])

    def test_handles_timeout_with_structured_report(self) -> None:
        out_json = self.tmp_path / "timeout_output.json"

        def hanging_search(*args, **kwargs):
            raise verifier.PackTimeoutError("search_and_rank", 5)

        with (
            patch.object(
                sys,
                "argv",
                ["verify_local_pack.py", "--cv", str(self.cv_file), "--yes", "--output", str(out_json), "--timeout", "5"],
            ),
            patch.object(verifier, "extract_cv_document", return_value=("Candidate CV text", [])),
            patch.object(verifier, "extract_profile", return_value=None),
            patch.object(verifier, "stream_search", side_effect=hanging_search),
            patch("sys.stderr", new=io.StringIO()) as err,
            patch("sys.stdout", new=io.StringIO()),
        ):
            code = verifier.main()

        self.assertEqual(code, 1)
        self.assertIn("[TIMEOUT] Pack verification timed out after 5s during phase 'search_and_rank'", err.getvalue())

        self.assertTrue(out_json.is_file())
        saved_report = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertEqual(saved_report["status"], "pack_e2e_timeout")
        self.assertTrue(saved_report["timed_out"])
        self.assertEqual(saved_report["phase"], "search_and_rank")
        self.assertEqual(saved_report["timeout_seconds"], 5)
        self.assertIn("pack_e2e_timeout", saved_report["error"])

    def test_handles_search_failure(self) -> None:
        mock_search = SimpleNamespace(
            failed=True,
            error_message="Provider 429 rate limit",
            ranked_jobs=[],
        )

        with (
            patch.object(sys, "argv", ["verify_local_pack.py", "--cv", str(self.cv_file), "--yes", "--timeout", "10"]),
            patch.object(verifier, "extract_cv_document", return_value=("Candidate CV text", [])),
            patch.object(verifier, "extract_profile", return_value=None),
            patch.object(verifier, "stream_search", return_value=[("result", mock_search)]),
            patch("sys.stdout", new=io.StringIO()) as out,
        ):
            code = verifier.main()

        self.assertEqual(code, 1)
        last_line = [line for line in out.getvalue().splitlines() if line.strip().startswith("{")][-1]
        json_start = out.getvalue().find('{\n  "cv":')
        report = json.loads(out.getvalue()[json_start:]) if json_start != -1 else json.loads(last_line)
        self.assertEqual(report["status"], "search_failed")
        self.assertTrue(report["search_failed"])
        self.assertEqual(report["search_error"], "Provider 429 rate limit")

    def test_handles_keyboard_interrupt(self) -> None:
        with (
            patch.object(sys, "argv", ["verify_local_pack.py", "--cv", str(self.cv_file), "--yes", "--timeout", "10"]),
            patch.object(verifier, "extract_cv_document", return_value=("Candidate CV text", [])),
            patch.object(verifier, "extract_profile", side_effect=KeyboardInterrupt()),
            patch("sys.stderr", new=io.StringIO()) as err,
            patch("sys.stdout", new=io.StringIO()) as out,
        ):
            code = verifier.main()

        self.assertEqual(code, 130)
        self.assertIn("[INTERRUPTED]", err.getvalue())
        json_start = out.getvalue().find('{\n  "cv":')
        self.assertNotEqual(json_start, -1)
        report = json.loads(out.getvalue()[json_start:])
        self.assertEqual(report["status"], "pack_e2e_interrupted")
        self.assertTrue(report["interrupted"])


if __name__ == "__main__":
    unittest.main()
