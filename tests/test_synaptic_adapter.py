"""Unit tests for Synaptic Decentralized MCP memory adapter."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

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

from job_scout.memory.synaptic_adapter import PrivacyGuard, SynapticMemoryVault  # noqa: E402


class TestPrivacyGuard(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = PrivacyGuard()

    def test_detects_and_redacts_email_and_phone(self) -> None:
        text = "Contact me at candidate@example.com or call +1 415-555-2671."
        res = self.guard.scan_and_redact(text)
        self.assertTrue(res.has_pii)
        self.assertIn("email", res.detected_types)
        self.assertIn("phone", res.detected_types)
        self.assertNotIn("candidate@example.com", res.sanitized_text)
        self.assertIn("[EMAIL_REDACTED]", res.sanitized_text)
        self.assertIn("[PHONE_REDACTED]", res.sanitized_text)

    def test_clean_text_passes_unchanged(self) -> None:
        text = "Experienced in building PyTorch LLM agent pipelines with LangGraph."
        res = self.guard.scan_and_redact(text)
        self.assertFalse(res.has_pii)
        self.assertEqual(res.sanitized_text, text)


class TestSynapticMemoryVault(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.storage_path = Path(self.tmp_dir.name)
        self.vault = SynapticMemoryVault(storage_dir=self.storage_path)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_create_and_reload_memory(self) -> None:
        mem = self.vault.create_memory(
            content="Built distributed multi-agent systems using LangGraph and FastAPI.",
            category="experience",
            tags=["AI", "LangGraph", "FastAPI"],
        )
        self.assertIsNotNone(mem.id)
        self.assertEqual(mem.category, "experience")
        self.assertTrue(mem.verified)

        # Reload vault from disk in new instance
        reloaded_vault = SynapticMemoryVault(storage_dir=self.storage_path)
        searched = reloaded_vault.search_memories("LangGraph", category="experience")
        self.assertEqual(len(searched), 1)
        self.assertEqual(searched[0].id, mem.id)

    def test_create_memory_with_pii_redaction(self) -> None:
        raw = "Reach out to aryaman@example.com for confidential references."
        mem = self.vault.create_memory(
            content=raw,
            category="qa_answer",
            tags=["references"],
            redact_pii=True,
        )
        self.assertTrue(mem.pii_redacted)
        self.assertNotIn("aryaman@example.com", mem.content)
        self.assertIn("[EMAIL_REDACTED]", mem.content)

    def test_search_memories_filtering(self) -> None:
        self.vault.create_memory("Expert in Python, TypeScript, and Rust.", "skill", tags=["languages"])
        self.vault.create_memory("Completed MS in Computer Science at Columbia.", "education", tags=["degree"])

        skill_hits = self.vault.search_memories("Python", category="skill")
        self.assertEqual(len(skill_hits), 1)
        self.assertEqual(skill_hits[0].category, "skill")

        edu_hits = self.vault.search_memories("Columbia", category="education")
        self.assertEqual(len(edu_hits), 1)
        self.assertEqual(edu_hits[0].category, "education")

    def test_inject_context_formatting(self) -> None:
        self.vault.create_memory("Deployed high-throughput PyTorch recommendation service.", "experience", tags=["PyTorch"])
        context = self.vault.inject_context("PyTorch Engineer", "Meta")
        self.assertIn("Relevant Candidate Memory Context:", context)
        self.assertIn("[EXPERIENCE]", context)
        self.assertIn("PyTorch", context)


if __name__ == "__main__":
    unittest.main()
