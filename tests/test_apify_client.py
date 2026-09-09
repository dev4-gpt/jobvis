"""Unit tests for ApifyJobFleet and ApifySource adapter."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Provide lightweight fallbacks when run outside a populated venv
if "httpx" not in sys.modules:
    try:
        import httpx  # noqa: F401
    except ImportError:
        httpx_mod = types.ModuleType("httpx")

        class HTTPStatusError(Exception):
            def __init__(self, message="", request=None, response=None):
                super().__init__(message)
                self.request = request
                self.response = response

        httpx_mod.HTTPStatusError = HTTPStatusError
        httpx_mod.Client = MagicMock()
        sys.modules["httpx"] = httpx_mod

if "pydantic" not in sys.modules:
    try:
        import pydantic  # noqa: F401
    except ImportError:
        pyd = types.ModuleType("pydantic")

        class BaseModel:
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)

            def model_dump(self):
                return self.__dict__

        pyd.BaseModel = BaseModel
        pyd.Field = lambda *a, **kw: (
            kw.get("default_factory", lambda: None)() if "default_factory" in kw else kw.get("default", None)
        )
        pyd.model_validator = lambda *a, **kw: lambda f: f
        pyd.SecretStr = lambda s: s
        pyd.field_validator = lambda *a, **kw: lambda f: f
        sys.modules["pydantic"] = pyd

if "pydantic_settings" not in sys.modules:
    try:
        import pydantic_settings  # noqa: F401
    except ImportError:
        ps = types.ModuleType("pydantic_settings")
        ps.BaseSettings = sys.modules["pydantic"].BaseModel
        ps.SettingsConfigDict = dict
        sys.modules["pydantic_settings"] = ps

import httpx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from job_scout.graph.schemas import JobPosting  # noqa: E402
from job_scout.tools.apify_client import ApifyJobFleet, ApifySource  # noqa: E402


class TestApifyJobFleet(unittest.TestCase):
    def setUp(self) -> None:
        self.fleet = ApifyJobFleet(api_token="test-token-123", timeout=10.0)

    def test_configuration_check(self) -> None:
        configured = ApifyJobFleet(api_token="valid_token")
        unconfigured = ApifyJobFleet(api_token="")
        self.assertTrue(configured.is_configured)
        self.assertFalse(unconfigured.is_configured)

    def test_normalize_apify_item_success(self) -> None:
        raw = {
            "id": "job-greenhouse-99",
            "title": "Senior AI Systems Engineer",
            "company": "DeepScale AI",
            "location": "San Francisco, CA",
            "isRemote": True,
            "url": "https://boards.greenhouse.io/deepscale/jobs/99",
            "description": "Lead the development of next-gen AI agents." * 150,
            "salary": "$180,000 - $240,000",
            "tags": ["Python", "PyTorch", "LangGraph"],
        }
        posting = self.fleet.normalize_apify_item(raw)
        self.assertIsNotNone(posting)
        assert posting is not None
        self.assertEqual(posting.job_id, "apify-job-greenhouse-99")
        self.assertEqual(posting.title, "Senior AI Systems Engineer")
        self.assertEqual(posting.company, "DeepScale AI")
        self.assertTrue(posting.remote)
        self.assertEqual(posting.source, "apify")
        self.assertLessEqual(len(posting.description), 4000)
        self.assertTrue(len(posting.content_hash) == 64)
        self.assertIn("Python", posting.tags)

    def test_normalize_apify_item_missing_required_fields(self) -> None:
        missing_title = {"company": "Acme Inc"}
        missing_company = {"title": "AI Researcher"}
        self.assertIsNone(self.fleet.normalize_apify_item(missing_title))
        self.assertIsNone(self.fleet.normalize_apify_item(missing_company))

    def test_run_actor_sync_unconfigured_returns_empty(self) -> None:
        unconfigured = ApifyJobFleet(api_token="")
        items = unconfigured.run_actor_sync("apify/ats-jobs-scraper", {})
        self.assertEqual(items, [])

    def test_run_actor_sync_success(self) -> None:
        fake_items = [{"title": "Agent Engineer", "company": "Synthetix"}]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_items
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls = MagicMock(return_value=mock_client)
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False

        with patch("httpx.Client", new=mock_client_cls):
            items = self.fleet.run_actor_sync("apify/ats-jobs-scraper", {"query": "ai"})
            self.assertEqual(items, fake_items)

    def test_run_actor_sync_http_error_returns_empty(self) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        err = httpx.HTTPStatusError("Unauthorized", request=MagicMock(), response=mock_resp)

        mock_client = MagicMock()
        mock_client.post.side_effect = err
        mock_client_cls = MagicMock(return_value=mock_client)
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False

        with patch("httpx.Client", new=mock_client_cls):
            items = self.fleet.run_actor_sync("apify/ats-jobs-scraper", {})
            self.assertEqual(items, [])

    def test_apify_source_protocol_fetch(self) -> None:
        source = ApifySource(fleet=self.fleet)
        self.assertEqual(source.name, "apify")

        fake_postings = [
            JobPosting(
                job_id="apify-1",
                title="ML Engineer",
                company="OpenScale",
                location="Remote",
                remote=True,
                source="apify",
            )
        ]
        with patch.object(self.fleet, "scrape_ats_jobs", return_value=fake_postings):
            results = source.fetch("machine learning", "San Francisco", "us", remote=True, limit=5)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].title, "ML Engineer")


if __name__ == "__main__":
    unittest.main()
