"""Fail fast when a local checkout predates the import-cycle fix."""

from __future__ import annotations

import sys
from pathlib import Path


def _identity(root: Path) -> tuple[str, str]:
    """Read branch and commit without invoking Git or contacting a remote."""
    try:
        git_dir = root / ".git"
        if git_dir.is_file():
            git_dir = root / git_dir.read_text(encoding="utf-8").strip().removeprefix("gitdir: ")
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            ref = head.removeprefix("ref: ")
            branch = ref.removeprefix("refs/heads/")
            commit = (git_dir / ref).read_text(encoding="utf-8").strip()
            return branch, commit[:12]
        return "detached", head[:12]
    except (OSError, ValueError):
        return "unknown", "unknown"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    graph_init = root / "src" / "job_scout" / "graph" / "__init__.py"
    try:
        text = graph_init.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Jobvis source check failed: cannot read {graph_init}: {exc}", file=sys.stderr)
        return 1

    lazy_marker = "def build_graph(*args: Any, **kwargs: Any) -> Any:"
    eager_marker = "from job_scout.graph.graph import build_graph, get_compiled_graph"
    if lazy_marker not in text or eager_marker in text:
        print(
            "Jobvis source is stale: this checkout still has the graph import-cycle bug. "
            "Run `git fetch origin && git switch main && git pull --ff-only origin main`, "
            f"then retry. Source: {root}",
            file=sys.stderr,
        )
        return 1

    branch, commit = _identity(root)
    print(f"Jobvis source check: OK ({root}; branch={branch}; commit={commit})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
