"""Fail fast when a local checkout predates the import-cycle fix."""

from __future__ import annotations

import subprocess
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


def _primary_worktree(root: Path) -> Path | None:
    """Return the primary checkout recorded by Git, when this is a worktree."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            try:
                return Path(line.removeprefix("worktree ")).resolve()
            except OSError:
                return None
    return None


def check(root: Path) -> int:
    """Validate one checkout before any application imports are attempted."""
    root = root.resolve()
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

    primary = _primary_worktree(root)
    if primary is not None and primary != root.resolve():
        print(
            "Jobvis source is not the canonical checkout: "
            f"running from {root}, but Git records {primary} as the primary worktree. "
            "Run `make app` from the primary checkout so the wizard and voice console share the intended code.",
            file=sys.stderr,
        )
        return 1

    branch, commit = _identity(root)
    print(f"Jobvis source check: OK ({root}; branch={branch}; commit={commit})")
    return 0


def main() -> int:
    return check(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    raise SystemExit(main())
