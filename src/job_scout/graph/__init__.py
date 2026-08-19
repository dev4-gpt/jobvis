"""The Job Scout agent graph."""

from typing import Any


def build_graph(*args: Any, **kwargs: Any) -> Any:
    """Build the graph lazily so schema-only imports remain acyclic."""
    from job_scout.graph.graph import build_graph as _build_graph

    return _build_graph(*args, **kwargs)


def get_compiled_graph(*args: Any, **kwargs: Any) -> Any:
    """Compile the graph lazily for the same import-cycle boundary."""
    from job_scout.graph.graph import get_compiled_graph as _get_compiled_graph

    return _get_compiled_graph(*args, **kwargs)


__all__ = ["build_graph", "get_compiled_graph"]
