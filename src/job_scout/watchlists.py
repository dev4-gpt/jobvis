"""Optional local watchlists; jobs themselves are never persisted."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from job_scout.application.store import default_data_dir


class Watchlist(BaseModel):
    watchlist_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    query: str
    location: str = ""
    country: str = "us"
    remote: bool = False
    enabled: bool = True
    last_refreshed_at: str | None = None


class WatchlistStore:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.path = (data_dir or default_data_dir()) / "watchlists.json"

    def _load(self) -> list[Watchlist]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return [Watchlist.model_validate(item) for item in payload.get("watchlists", [])]
        except (OSError, ValueError, TypeError):
            return []

    def _save(self, items: list[Watchlist]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)
        self.path.write_text(
            json.dumps({"version": 1, "watchlists": [item.model_dump(mode="json") for item in items]}, indent=2) + "\n",
            encoding="utf-8",
        )
        self.path.chmod(0o600)

    def list(self) -> list[Watchlist]:
        return self._load()

    def create(self, item: Watchlist) -> Watchlist:
        items = self._load()
        items.append(item)
        self._save(items)
        return item

    def delete(self, watchlist_id: str) -> bool:
        items = self._load()
        remaining = [item for item in items if item.watchlist_id != watchlist_id]
        if len(remaining) == len(items):
            return False
        self._save(remaining)
        return True

    def mark_refreshed(self, item: Watchlist) -> Watchlist:
        item.last_refreshed_at = datetime.now(UTC).isoformat()
        items = self._load()
        for index, current in enumerate(items):
            if current.watchlist_id == item.watchlist_id:
                items[index] = item
        self._save(items)
        return item
