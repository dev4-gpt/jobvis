"""Unit tests for Job Listing Liveness Classifier."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from job_scout.tools.liveness import check_job_liveness, classify_liveness  # noqa: E402


class TestLivenessClassifier(unittest.TestCase):
    def test_http_404_and_410_are_expired(self):
        self.assertEqual(classify_liveness(status_code=404)["status"], "expired")
        self.assertEqual(classify_liveness(status_code=410)["status"], "expired")

    def test_expired_url_redirect(self):
        res = classify_liveness(status_code=200, final_url="https://boards.greenhouse.io/job?error=true")
        self.assertEqual(res["status"], "expired")
        self.assertIn("error=true", res["reason"])

    def test_hard_expired_patterns(self):
        patterns = [
            "We apologize, but this job is no longer available.",
            "This position has been filled. Thank you for your interest.",
            "This job has expired.",
            "No longer accepting applications for this opening.",
            "Job listing not found.",
        ]
        for text in patterns:
            res = classify_liveness(status_code=200, body_text=text)
            self.assertEqual(res["status"], "expired", f"Failed for text: {text}")

    def test_active_with_apply_controls(self):
        res = classify_liveness(
            status_code=200,
            body_text="Come work with us as a Senior AI Engineer building multi-agent systems.",
            apply_controls=["Submit Application", "Share on LinkedIn"],
        )
        self.assertEqual(res["status"], "active")
        self.assertIn("apply control detected", res["reason"])

    def test_active_with_apply_in_body(self):
        body = (
            "We are hiring a Machine Learning Engineer to join our core intelligence team. "
            "To apply for this role, click below: Apply Now. "
            + "Detailed requirements: Python, PyTorch, LangGraph, distributed training, "
            * 10
        )
        res = classify_liveness(status_code=200, body_text=body)
        self.assertEqual(res["status"], "active")

    def test_short_content_is_uncertain(self):
        # Page with only navigation footer (< 300 chars)
        res = classify_liveness(status_code=200, body_text="Home | About Us | Contact")
        self.assertEqual(res["status"], "uncertain")
        self.assertIn("insufficient content", res["reason"])

    def test_uncertain_when_content_present_without_apply(self):
        body = (
            "Company overview and mission. We build great products for our customers worldwide. "
            "Our values include innovation, integrity, collaboration, and continuous improvement. "
            "Office locations include San Francisco, New York, London, and Tokyo. "
        ) * 5
        res = classify_liveness(status_code=200, body_text=body)
        self.assertEqual(res["status"], "uncertain")

    def test_check_job_liveness_with_mock_client(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.url = "https://boards.greenhouse.io/anthropic/jobs/123"
        mock_response.text = (
            "<html><body><h1>AI Research Engineer</h1><p>"
            + "Long job description about LLMs, agents, reinforcement learning and evaluation. " * 8
            + "</p><button>Apply for this job</button></body></html>"
        )
        mock_client.get.return_value = mock_response

        res = check_job_liveness("https://boards.greenhouse.io/anthropic/jobs/123", client=mock_client)
        self.assertEqual(res["liveness"], "active")
        self.assertEqual(res["status_code"], 200)

    def test_check_job_liveness_network_error_graceful(self):
        mock_client = MagicMock()
        mock_client.get.side_effect = ConnectionError("Connection refused")

        res = check_job_liveness("https://invalid-nonexistent-domain.xyz", client=mock_client)
        self.assertEqual(res["liveness"], "uncertain")
        self.assertIn("ConnectionError", res["reason"])


if __name__ == "__main__":
    unittest.main()
