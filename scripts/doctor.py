#!/usr/bin/env python3
"""Bounded local diagnostics; no network calls and no secret values printed."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
from pathlib import Path

from job_scout.health import health


def port_status(port: int) -> dict[str, object]:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
        return {"port": port, "available": True}
    except OSError as exc:
        return {"port": port, "available": False, "error": type(exc).__name__}
    finally:
        sock.close()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report: dict[str, object] = {"status": "ok", "root": str(root), "health": health()}
    report["ports"] = [
        port_status(int(os.getenv("JOBVIS_WIZARD_PORT", "7860"))),
        port_status(int(os.getenv("JOBVIS_CONSOLE_PORT", "8000"))),
    ]
    report["web_console"] = (root / "web" / "out" / "index.html").exists()
    report["browser_dependency"] = importlib.util.find_spec("playwright") is not None
    report["browser_dependency_required"] = os.getenv("JOBVIS_APPLICATION_AUTOFILL_REQUIRED", "0") == "1"
    report["pdf_dependency"] = importlib.util.find_spec("pypdf") is not None
    report["tectonic"] = shutil.which("tectonic") is not None
    report["qpdf"] = shutil.which("qpdf") is not None
    ports = report["ports"]
    if (
        not report["web_console"]
        or not report["pdf_dependency"]
        or not report["tectonic"]
        or any(not item["available"] for item in ports)
    ):  # type: ignore[index]
        report["status"] = "attention"
    if report["browser_dependency_required"] and not report["browser_dependency"]:
        report["status"] = "attention"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
