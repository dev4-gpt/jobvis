#!/usr/bin/env python3
"""One bounded pre-release audit for the local Jobvis runtime and contracts."""

from __future__ import annotations

import json
import sys

from job_scout.api import create_app
from job_scout.health import health


def main() -> int:
    result: dict[str, object] = {
        "health": health(),
        "api_routes": len(create_app().routes),
    }
    result["passed"] = bool(result["health"].get("status") == "ok")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
