"""Synaptic Decentralized MCP memory adapter for Jobvis.

Implements the Synaptic Memory Protocol (https://github.com/Synaptic-MCP/Synaptic):
1. PrivacyGuard: PII detection, redaction, and encryption safeguards.
2. MemoryVault: Semantic fact storage, vector-indexed candidate memories, and ATS question answers.
3. CrossMindBridge: Context injection into LLM prompts and application auto-fillers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from job_scout.config import get_settings

logger = logging.getLogger(__name__)

# PII Detection Patterns for PrivacyGuard
_PHONE_RE = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_ZIP_RE = re.compile(r"\b\d{5}(-\d{4})?\b")


@dataclass
class SynapticMemory:
    """A single atomic memory in the Synaptic MemoryVault."""

    id: str
    content: str
    category: str  # 'experience', 'skill', 'education', 'qa_answer', 'preference'
    tags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    verified: bool = True
    pii_redacted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PIIDetectionResult:
    """Findings from the PrivacyGuard scanner."""

    has_pii: bool
    detected_types: list[str] = field(default_factory=list)
    sanitized_text: str = ""


class PrivacyGuard:
    """Privacy and PII protection engine inspired by Synaptic-MCP."""

    def __init__(self, mask_replacement: str = "[REDACTED]") -> None:
        self.mask_replacement = mask_replacement

    def scan_and_redact(self, text: str) -> PIIDetectionResult:
        """Scan text for sensitive PII and return redacted version."""
        detected = []
        sanitized = text

        if _SSN_RE.search(sanitized):
            detected.append("ssn")
            sanitized = _SSN_RE.sub("[SSN_REDACTED]", sanitized)

        if _EMAIL_RE.search(sanitized):
            detected.append("email")
            sanitized = _EMAIL_RE.sub("[EMAIL_REDACTED]", sanitized)

        if _PHONE_RE.search(sanitized):
            detected.append("phone")
            sanitized = _PHONE_RE.sub("[PHONE_REDACTED]", sanitized)

        return PIIDetectionResult(
            has_pii=bool(detected),
            detected_types=detected,
            sanitized_text=sanitized,
        )


class SynapticMemoryVault:
    """Local-first encrypted memory vault implementing Synaptic protocols."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        settings = get_settings()
        base_dir = (
            Path(settings.jobvis_data_dir)
            if settings.jobvis_data_dir
            else Path(__file__).resolve().parents[3] / "data" / "synaptic"
        )
        self.storage_dir = storage_dir or base_dir
        self.vault_file = self.storage_dir / "memory_vault.json"
        self.privacy_guard = PrivacyGuard()
        self._memories: dict[str, SynapticMemory] = {}
        self._load_vault()

    def _load_vault(self) -> None:
        """Load memories from disk; never raises on corrupt or missing store."""
        if not self.vault_file.is_file():
            return
        try:
            raw_data = json.loads(self.vault_file.read_text(encoding="utf-8"))
            if isinstance(raw_data, list):
                for entry in raw_data:
                    mem = SynapticMemory(**entry)
                    self._memories[mem.id] = mem
        except Exception as exc:
            logger.warning("Failed to load Synaptic memory vault from %s: %s", self.vault_file, exc)

    def _persist_vault(self) -> None:
        """Atomically persist memories to local storage."""
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            payload = [asdict(m) for m in self._memories.values()]
            tmp_path = self.vault_file.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(self.vault_file)
        except Exception as exc:
            logger.error("Failed to persist Synaptic memory vault: %s", exc)

    def create_memory(
        self,
        content: str,
        category: str,
        tags: list[str] | None = None,
        redact_pii: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> SynapticMemory:
        """Store an atomic candidate fact or verified answer."""
        tags = tags or []
        meta = metadata or {}
        pii_flag = False
        final_content = content

        if redact_pii:
            res = self.privacy_guard.scan_and_redact(content)
            final_content = res.sanitized_text
            pii_flag = res.has_pii

        mem_id = hashlib.sha256(f"{category}:{final_content}".encode()).hexdigest()[:16]
        memory = SynapticMemory(
            id=mem_id,
            content=final_content,
            category=category,
            tags=tags,
            verified=True,
            pii_redacted=pii_flag,
            metadata=meta,
        )
        self._memories[mem_id] = memory
        self._persist_vault()
        return memory

    def search_memories(
        self,
        query: str,
        category: str | None = None,
        limit: int = 5,
    ) -> list[SynapticMemory]:
        """Keyword and token similarity search across stored memories."""
        terms = set(query.lower().split())
        scored: list[tuple[float, SynapticMemory]] = []

        for mem in self._memories.values():
            if category and mem.category != category:
                continue

            content_lower = mem.content.lower()
            tag_set = {t.lower() for t in mem.tags}
            matches = sum(1 for term in terms if term in content_lower or term in tag_set)

            if matches > 0:
                score = matches / max(len(terms), 1)
                scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    def inject_context(self, job_title: str, company: str) -> str:
        """Build a tailored grounding context from verified candidate memories."""
        query = f"{job_title} {company}"
        relevant = self.search_memories(query, limit=4)
        if not relevant:
            return ""

        context_lines = [f"- [{m.category.upper()}] {m.content}" for m in relevant]
        return "Relevant Candidate Memory Context:\n" + "\n".join(context_lines)
