"""Confirmed answer suggestions backed by the shared encrypted memory service."""

from dataclasses import dataclass
from pathlib import Path

from job_scout.memory.service import MemoryEntry, MemoryService, candidate_key


@dataclass
class AnswerRecord:
    question_key: str
    answer: str
    sensitive: bool
    confirmed_at: str


class AnswerMemory:
    def __init__(self, path: Path, *, candidate_id: str | None = None) -> None:
        # Older library callers retain a path-scoped namespace. Console callers
        # always supply the current candidate identity.
        self.service = MemoryService(candidate_id or candidate_key(str(path.resolve())), path.parent)

    def remember(self, question_key: str, answer: str, *, sensitive: bool, consent: bool = False) -> None:
        if sensitive and not consent:
            raise ValueError("Sensitive answers require explicit consent before they are remembered.")
        old = self.service.answer(question_key)
        entry = MemoryEntry(content=answer, question_key=question_key, sensitive=sensitive)
        if old:
            entry.id = old.id
        self.service.save(entry, confirm=True)

    def reusable(self, question_key: str, *, sensitive: bool) -> AnswerRecord | None:
        entry = self.service.answer(question_key)
        return (
            AnswerRecord(question_key, entry.content, sensitive or entry.sensitive, entry.confirmed_at or "") if entry else None
        )
