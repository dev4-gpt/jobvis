"""Unit tests for AIHawk Queue Exporter."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from job_scout.tools.aihawk_exporter import export_aihawk_queue, format_aihawk_job  # noqa: E402


class TestAIHawkExporter(unittest.TestCase):
    def test_format_aihawk_job_zero_pads_id(self):
        job = format_aihawk_job(job_id=7, company="Anthropic", role="AI Safety Engineer", score=4.5)
        self.assertEqual(job["id"], "007")
        self.assertEqual(job["company"], "Anthropic")
        self.assertEqual(job["score"], 4.5)
        self.assertEqual(job["status"], "Evaluated")

    def test_format_aihawk_job_normalizes_100_scale_score(self):
        job = format_aihawk_job(job_id="012", company="Hume AI", role="Speech ML Engineer", score=90.0)
        self.assertEqual(job["score"], 4.5)

    def test_export_aihawk_queue_structure(self):
        sample_packs = [
            {
                "id": 1,
                "company": "Anthropic",
                "role": "Staff AI Engineer",
                "score": 4.8,
                "job_url": "https://boards.greenhouse.io/anthropic/jobs/123",
                "pdf_path": "/tmp/anthropic_cv.pdf",
                "notes": "Verified high match",
            },
            {
                "id": 2,
                "company": "Legacy Corp",
                "role": "Legacy Analyst",
                "score": 2.0,  # Below threshold
                "job_url": "https://example.com/jobs/456",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "aihawk-queue.json"
            doc = export_aihawk_queue(sample_packs, output_path=out_file, score_threshold=3.5)

            self.assertEqual(doc["schema_version"], "1.0")
            self.assertIn("exported_at", doc)
            self.assertEqual(len(doc["jobs"]), 1)
            self.assertEqual(doc["jobs"][0]["company"], "Anthropic")
            self.assertEqual(doc["jobs"][0]["id"], "001")

            # Check file on disk
            self.assertTrue(out_file.exists())
            with open(out_file, encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["schema_version"], "1.0")
            self.assertEqual(len(loaded["jobs"]), 1)


if __name__ == "__main__":
    unittest.main()
