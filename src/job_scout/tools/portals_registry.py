"""Target Company Portals Registry.

Loads and queries curated company portals, direct ATS endpoints (e.g. Greenhouse API),
and title filtering rules from data/portals.yml.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


def _simple_yaml_fallback(text: str) -> dict[str, Any]:
    """Lightweight fallback YAML parser for portals.yml when PyYAML is not installed."""
    data: dict[str, Any] = {
        "title_filter": {"positive": [], "negative": [], "seniority_boost": []},
        "tracked_companies": [],
        "search_queries": [],
    }

    current_section = None
    sub_section = None
    current_item: dict[str, Any] | None = None

    for line in text.splitlines():
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#"):
            continue

        if trimmed.startswith("title_filter:"):
            current_section = "title_filter"
            sub_section = None
            continue
        elif trimmed.startswith("tracked_companies:"):
            current_section = "tracked_companies"
            sub_section = None
            if current_item and "name" in current_item:
                data["tracked_companies"].append(current_item)
            current_item = None
            continue
        elif trimmed.startswith("search_queries:"):
            current_section = "search_queries"
            sub_section = None
            if current_item and "name" in current_item:
                data["search_queries"].append(current_item)
            current_item = None
            continue

        if current_section == "title_filter":
            if trimmed.startswith("positive:"):
                sub_section = "positive"
            elif trimmed.startswith("negative:"):
                sub_section = "negative"
            elif trimmed.startswith("seniority_boost:"):
                sub_section = "seniority_boost"
            elif trimmed.endswith(":") and not trimmed.startswith("-"):
                sub_section = None
            elif trimmed.startswith("-") and sub_section in ("positive", "negative", "seniority_boost"):
                val = trimmed[1:].strip().strip("\"'")
                if " #" in val:
                    val = val.split(" #", 1)[0].strip().strip("\"'")
                if val:
                    data["title_filter"][sub_section].append(val)

        elif current_section in ("tracked_companies", "search_queries"):
            if trimmed.startswith("- name:"):
                if current_item and "name" in current_item:
                    data[current_section].append(current_item)
                name_val = trimmed[7:].strip().strip("\"'")
                current_item = {"name": name_val, "enabled": True}
            elif current_item is not None and ":" in trimmed:
                key, val = trimmed.split(":", 1)
                key = key.strip()
                val = val.strip().strip("\"'")
                if " #" in val:
                    val = val.split(" #", 1)[0].strip().strip("\"'")
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
                current_item[key] = val

    if current_item and current_section in ("tracked_companies", "search_queries") and "name" in current_item:
        data[current_section].append(current_item)

    return data


class PortalsRegistry:
    """Manager for target company career portals and ATS endpoints."""

    def __init__(self, config_path: Path | str | None = None) -> None:
        if config_path is None:
            # Default to data/portals.yml relative to project root
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            config_path = base_dir / "data" / "portals.yml"

        self.config_path = Path(config_path)
        self.data: dict[str, Any] = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Loads portals.yml safely or returns a safe empty dict structure."""
        if not self.config_path.exists():
            return {"title_filter": {"positive": [], "negative": []}, "tracked_companies": [], "search_queries": []}

        try:
            with open(self.config_path, encoding="utf-8") as f:
                if yaml is not None:
                    content = yaml.safe_load(f)
                    return content if isinstance(content, dict) else {}
                return _simple_yaml_fallback(f.read())
        except Exception:
            return {"title_filter": {"positive": [], "negative": []}, "tracked_companies": [], "search_queries": []}

    def get_tracked_companies(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        """Returns list of tracked companies from portals.yml."""
        companies = self.data.get("tracked_companies", [])
        if not isinstance(companies, list):
            return []
        if enabled_only:
            return [c for c in companies if isinstance(c, dict) and c.get("enabled", True) is not False]
        return [c for c in companies if isinstance(c, dict)]

    def get_greenhouse_api_targets(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        """Returns companies that provide direct Greenhouse board API endpoints."""
        companies = self.get_tracked_companies(enabled_only=enabled_only)
        return [c for c in companies if c.get("api") and "greenhouse" in str(c.get("api", ""))]

    def filter_title(self, title: str) -> bool:
        """Determines if a job title passes the positive and negative keyword filters."""
        if not title:
            return False

        t_lower = title.lower()
        title_filter = self.data.get("title_filter", {})
        positives = [p.lower() for p in title_filter.get("positive", []) if isinstance(p, str)]
        negatives = [n.lower() for n in title_filter.get("negative", []) if isinstance(n, str)]

        # If positive list is empty, default to accepting
        pos_match = False
        if positives:
            for p in positives:
                # Word boundary or substring check for acronyms vs words
                if len(p) <= 2:
                    if re.search(r"\b" + re.escape(p) + r"\b", t_lower):
                        pos_match = True
                        break
                elif p in t_lower:
                    pos_match = True
                    break
        else:
            pos_match = True

        if not pos_match:
            return False

        for n in negatives:
            if len(n) <= 2:
                if re.search(r"\b" + re.escape(n) + r"\b", t_lower):
                    return False
            elif n in t_lower:
                return False

        return True

    def get_search_queries(self, enabled_only: bool = True) -> list[dict[str, Any]]:
        """Returns list of pre-configured search queries."""
        queries = self.data.get("search_queries", [])
        if not isinstance(queries, list):
            return []
        if enabled_only:
            return [q for q in queries if isinstance(q, dict) and q.get("enabled", True) is not False]
        return [q for q in queries if isinstance(q, dict)]
