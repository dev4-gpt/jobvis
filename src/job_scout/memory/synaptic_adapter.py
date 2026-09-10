"""Compatibility facade for encrypted local keyword memory (not a Synaptic client)."""

import re
from dataclasses import dataclass, field

from job_scout.memory.service import MemoryEntry, MemoryService


@dataclass
class PIIDetectionResult:
    has_pii: bool
    detected_types: list[str] = field(default_factory=list)
    sanitized_text: str = ""


class PrivacyGuard:
    """Best-effort text redaction; encrypted storage is a separate boundary."""

    def scan_and_redact(self, text: str) -> PIIDetectionResult:
        found = []
        for name, pattern in (
            ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
            ("email", r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}"),
            ("phone", r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}"),
        ):
            if re.search(pattern, text):
                found.append(name)
                text = re.sub(pattern, f"[{name.upper()}_REDACTED]", text)
        return PIIDetectionResult(bool(found), found, text)


class SynapticMemoryVault:
    """Deprecated name retained for callers; all persistence uses MemoryService."""

    def __init__(self, storage_dir=None, *, candidate_id: str):
        self.service = MemoryService(candidate_id, storage_dir)

    def create_memory(self, content, category, tags=None, redact_pii=False, metadata=None):
        if redact_pii:
            content = PrivacyGuard().scan_and_redact(content).sanitized_text
        return self.service.save(MemoryEntry(content=content, category=category, provenance="imported"))

    def search_memories(self, query, category=None, limit=5):
        return [item for item in self.service.search(query, limit) if category is None or item.category == category]

    def inject_context(self, job_title, company):
        # Free-form memory is not evidence in the resume corpus.
        return ""
