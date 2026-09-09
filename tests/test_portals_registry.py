"""Unit tests for PortalsRegistry."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from job_scout.tools.portals_registry import PortalsRegistry  # noqa: E402


class TestPortalsRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = PortalsRegistry()

    def test_loads_curated_portals(self):
        companies = self.registry.get_tracked_companies(enabled_only=False)
        self.assertGreater(len(companies), 10, "Should load curated tracked companies from portals.yml")

        # Check Anthropic exists
        anthropic = next((c for c in companies if c.get("name") == "Anthropic"), None)
        self.assertIsNotNone(anthropic)
        self.assertEqual(anthropic.get("api"), "https://boards-api.greenhouse.io/v1/boards/anthropic/jobs")

    def test_greenhouse_api_targets(self):
        gh_targets = self.registry.get_greenhouse_api_targets(enabled_only=False)
        self.assertGreater(len(gh_targets), 0)
        for target in gh_targets:
            self.assertIn("greenhouse", target["api"])

    def test_title_filtering_rules(self):
        # AI Engineer should pass
        self.assertTrue(self.registry.filter_title("Staff AI Engineer"))
        self.assertTrue(self.registry.filter_title("Machine Learning Intern"))
        self.assertTrue(self.registry.filter_title("Generative AI Solutions Architect"))

        # QA or Manual QA should be rejected
        self.assertFalse(self.registry.filter_title("AI QA Tester"))
        self.assertFalse(self.registry.filter_title("Manual QA Automation"))

        # Empty title should be rejected
        self.assertFalse(self.registry.filter_title(""))

    def test_missing_file_fallback(self):
        empty_reg = PortalsRegistry(config_path=Path("/tmp/nonexistent_portals_9999.yml"))
        self.assertEqual(empty_reg.get_tracked_companies(), [])
        self.assertEqual(empty_reg.get_greenhouse_api_targets(), [])
        self.assertEqual(empty_reg.get_search_queries(), [])


if __name__ == "__main__":
    unittest.main()
