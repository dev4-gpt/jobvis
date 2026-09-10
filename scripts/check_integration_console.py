"""Exercise the built console with browser-only fixtures; no providers or personal data."""

import argparse
import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-dir", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(SimpleHTTPRequestHandler, directory=str(args.web_dir)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state = {
        "step": "results",
        "thread_id": "fixture",
        "candidate": {
            "name": "Demo Candidate",
            "role": "AI Engineer",
            "seniority": "junior",
            "locations": ["New York"],
            "remote_ok": True,
        },
        "jobs": [],
        "run": {"running": False},
        "pack": {
            "job_id": "one",
            "headline": "AI Engineer",
            "summary": "Fixture",
            "cover_letter": "Fixture letter",
            "flags": 0,
            "verdict": "Verified",
            "honesty_note": "Fixture only",
        },
        "application": {"job_id": "", "url": "", "status": "idle", "message": "", "ats": None, "fields": []},
    }
    memories = []
    requests = []

    def api(route):
        path = route.request.url.split("/api", 1)[1].split("?", 1)[0]
        method = route.request.method
        requests.append((method, path))
        data = route.request.post_data_json if route.request.post_data else {}
        result = {}
        if path == "/config":
            result = {"voice_ok": False, "voice_hint": "Offline fixture", "wizard_url": "#", "has_candidate": True}
        elif path == "/state":
            result = state
        elif path == "/events":
            route.fulfill(status=200, content_type="text/event-stream", body="")
            return
        elif path == "/integrations":
            result = {
                "direct_sources_enabled": True,
                "liveness_enabled": True,
                "boards": {"greenhouse": ["demo"]},
                "apify_ready": True,
                "apify_hint": "",
            }
        elif path == "/apify/expansion":
            result = {"status": "complete", "run_id": "fixture-run"}
        elif path == "/packs/reviewed":
            result = {
                "snapshots": [
                    {
                        "id": "snapshot",
                        "job_id": "one",
                        "company": "Demo",
                        "role": "AI Engineer",
                        "score": 90,
                        "approved_at": "2026-09-10",
                    }
                ]
            }
        elif path == "/applications":
            result = {"applications": []}
        elif path == "/memory" and method == "POST":
            memories.append({**data, "id": "memory", "confirmed": False, "provenance": "user"})
            result = memories[-1]
        elif path == "/memory":
            result = {"available": True, "entries": memories}
        elif path == "/memory/memory/confirm":
            memories[0]["confirmed"] = True
            result = memories[0]
        elif path == "/memory/memory" and method == "DELETE":
            memories.clear()
        elif path == "/pack/audit":
            result = {"passed": True, "issues": [], "artifacts": [], "hashes": {}}
        elif path == "/pack/review":
            if method == "POST":
                assert data["approved"] is True and data["pack_key"] == "fixture-key"
            result = {"pack_key": "fixture-key", "audit": {"passed": True, "issues": []}}
        elif path == "/packs/export":
            assert data["snapshot_ids"] == ["snapshot"]
            result = {"id": "export", "queue_path": "/fixture/export/aihawk-queue.json"}
        route.fulfill(status=200, content_type="application/json", body=json.dumps(result))

    try:
        with sync_playwright() as playwright:
            expect.set_options(timeout=15000)
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 1000})
            page.on("pageerror", lambda error: print(f"Browser error: {error}", flush=True))
            page.route(
                "**/*", lambda route: route.continue_() if route.request.url.startswith("http://127.0.0.1:") else route.abort()
            )
            page.route("**/api/**", api)
            page.goto(f"http://127.0.0.1:{server.server_port}")
            panel = page.locator(".integrations-panel")
            expect(panel.get_by_role("button", name="Expand search with Apify")).to_be_enabled()
            panel.get_by_role("button", name="Expand search with Apify").click()
            expect(panel.get_by_text("Expansion: complete", exact=False)).to_be_visible()
            panel.get_by_text("Encrypted local memory", exact=True).click()
            panel.get_by_label("Question or label").fill("Experience")
            panel.get_by_label("Answer or fact").fill("Built Python pipelines")
            panel.get_by_role("button", name="Save for review", exact=True).click()
            expect(panel.get_by_text("Needs confirmation", exact=False)).to_be_visible()
            panel.get_by_role("button", name="Confirm this memory").click()
            expect(panel.get_by_text("Confirmed · user", exact=True)).to_be_visible()
            panel.get_by_role("button", name="Delete", exact=True).click()
            expect(panel.get_by_text("Experience: Built Python pipelines")).to_have_count(0)
            panel.get_by_text("Review and export application packs", exact=True).click()
            panel.get_by_role("button", name="Check current pack for review").click()
            panel.get_by_role("button", name="I reviewed this pack — approve snapshot").click()
            expect(panel.get_by_text("Reviewed snapshot saved.")).to_be_visible()
            panel.get_by_label("Demo · AI Engineer · 90/100 · reviewed", exact=False).check()
            panel.get_by_role("button", name="Export selected reviewed packs").click()
            expect(panel.get_by_role("link", name="Download AIHawk queue")).to_be_visible()
            assert requests.count(("POST", "/apify/expansion")) == 1
            if args.screenshot:
                page.screenshot(path=str(args.screenshot), full_page=True)
            browser.close()
        print("Console fixture smoke passed: expansion, memory save/confirm/delete, pack review and export.")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
