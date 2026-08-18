"""Explicitly consented answer memory for the local autofill workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from job_scout.application.security import load_encrypted, save_encrypted


@dataclass
class AnswerRecord:
    question_key: str
    answer: str
    sensitive: bool
    confirmed_at: str


class AnswerMemory:
    def __init__(self, path: Path) -> None:
        self.path = path

    def remember(self, question_key: str, answer: str, *, sensitive: bool, consent: bool = False) -> None:
        if sensitive and not consent:
            raise ValueError("Sensitive answers require explicit consent before they are remembered.")
        records = self._read()
        records[question_key] = AnswerRecord(question_key, answer, sensitive, datetime.now(UTC).isoformat())
        self._write(records)

    def reusable(self, question_key: str, *, sensitive: bool) -> AnswerRecord | None:
        record = self._read().get(question_key)
        if record is None:
            return None
        # Sensitive answers are suggestions only and must be confirmed again.
        if sensitive or record.sensitive:
            return AnswerRecord(record.question_key, record.answer, True, record.confirmed_at)
        return record

    def _read(self) -> dict[str, AnswerRecord]:
        try:
            data = json.loads(load_encrypted(self.path).decode("utf-8"))
            return {key: AnswerRecord(**value) for key, value in data.items()}
        except Exception:  # noqa: BLE001 - absent/corrupt memory is a safe empty store
            return {}

    def _write(self, records: dict[str, AnswerRecord]) -> None:
        payload = json.dumps({key: asdict(value) for key, value in records.items()}).encode("utf-8")
        save_encrypted(self.path, payload)
