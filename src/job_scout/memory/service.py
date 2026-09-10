"""Candidate-scoped encrypted local memory with explicit confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import RLock
from uuid import uuid4

from pydantic import BaseModel, Field

from job_scout.application.models import now_iso
from job_scout.application.security import load_encrypted, save_encrypted
from job_scout.application.store import default_data_dir

_LOCK = RLock()


def candidate_key(cv_text: str, name: str = "") -> str:
    return hashlib.sha256((name.strip().casefold() + "\n" + cv_text.strip()).encode()).hexdigest()


class MemoryEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(min_length=1, max_length=10000)
    category: str = "qa_answer"
    question_key: str = ""
    provenance: str = "user"
    corpus_ref: str = ""
    confirmed: bool = False
    sensitive: bool = False
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    confirmed_at: str | None = None


class MemoryService:
    def __init__(self, candidate_id: str, data_dir: Path | None = None):
        self.candidate_id = candidate_id
        self.path = (data_dir or default_data_dir()) / "memory" / f"{candidate_id}.enc"

    def _read(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "candidate_id": self.candidate_id, "entries": []}
        data = json.loads(load_encrypted(self.path))
        if data.get("version") != 1 or data.get("candidate_id") != self.candidate_id:
            raise ValueError("Memory file version or candidate identity does not match")
        return data

    def list(self) -> list[MemoryEntry]:
        with _LOCK:
            return [MemoryEntry.model_validate(entry) for entry in self._read()["entries"]]

    def save(self, entry: MemoryEntry, *, confirm: bool = False) -> MemoryEntry:
        with _LOCK:
            entries = self.list()
            entry = entry.model_copy(
                update={"confirmed": confirm, "confirmed_at": now_iso() if confirm else None, "updated_at": now_iso()}
            )
            entries = [item for item in entries if item.id != entry.id] + [entry]
            self._write(entries)
            return entry

    def _write(self, entries: list[MemoryEntry]) -> None:
        save_encrypted(
            self.path,
            json.dumps(
                {"version": 1, "candidate_id": self.candidate_id, "entries": [item.model_dump() for item in entries]}
            ).encode(),
        )

    def delete(self, entry_id: str) -> None:
        with _LOCK:
            entries = self.list()
            if not any(item.id == entry_id for item in entries):
                raise ValueError("Memory not found")
            self._write([item for item in entries if item.id != entry_id])

    def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        terms = set(query.casefold().split())
        entries = [item for item in self.list() if item.confirmed]

        def score(item):
            return sum(term in (item.content + " " + item.question_key).casefold() for term in terms)

        return sorted((item for item in entries if score(item)), key=score, reverse=True)[:limit]

    def answer(self, question_key: str) -> MemoryEntry | None:
        return next(
            (
                item
                for item in self.list()
                if item.confirmed and item.category == "qa_answer" and item.question_key == question_key
            ),
            None,
        )

    def grounded_context(self, corpus) -> str:
        """Only repeat exact current-corpus evidence, never free-form memory claims."""
        selected = []
        for entry in self.list():
            item = corpus.get(entry.corpus_ref) if entry.corpus_ref else None
            if entry.confirmed and not entry.sensitive and item and item.text == entry.content:
                selected.append(f"[{item.id}] {item.text}")
        return "\n".join(selected[:4])

    def import_corpus(self, corpus) -> int:
        from job_scout.memory.synaptic_adapter import PrivacyGuard

        with _LOCK:
            entries = self.list()
            known = {(entry.corpus_ref, entry.content) for entry in entries}
            added = 0
            for item in corpus.items[:100]:
                if (item.id, item.text) in known:
                    continue
                entries.append(
                    MemoryEntry(
                        content=item.text,
                        category=item.kind,
                        corpus_ref=item.id,
                        provenance="resume",
                        sensitive=PrivacyGuard().scan_and_redact(item.text).has_pii,
                    )
                )
                added += 1
            self._write(entries)
            return added

    def import_legacy(self, path: Path) -> int:
        """Explicit import leaves the plaintext original and imports unconfirmed entries."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Legacy memory must be a list")
        with _LOCK:
            entries = self.list()
            known = {item.content for item in entries}
            added = 0
            for raw in payload:
                content = str(raw.get("content", "")).strip()
                if content and content not in known:
                    entries.append(
                        MemoryEntry(
                            content=content, category=raw.get("category", "qa_answer"), provenance="legacy_import", sensitive=True
                        )
                    )
                    known.add(content)
                    added += 1
            self._write(entries)
            return added
