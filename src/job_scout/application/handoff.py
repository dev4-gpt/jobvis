"""Durable reviewed snapshots and idempotent local AIHawk export bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from threading import RLock

from job_scout.application.models import now_iso
from job_scout.application.store import default_data_dir
from job_scout.tools.aihawk_exporter import export_aihawk_queue
from job_scout.tools.liveness import valid_job_url, verified_liveness

_LOCK = RLock()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    fd, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def digest(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HandoffStore:
    def __init__(self, data_dir: Path | None = None):
        self.root = (data_dir or default_data_dir()) / "handoff"
        self.path = self.root / "index.json"

    def read(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "snapshots": {}, "ids": {}, "exports": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("version") != 1:
            raise ValueError("Unsupported handoff store version")
        return data

    def list(self, candidate_id: str) -> list[dict]:
        with _LOCK:
            return [row for row in self.read()["snapshots"].values() if row["candidate_id"] == candidate_id]

    def invalidate(self, job_id: str) -> None:
        with _LOCK:
            data = self.read()
            changed = False
            for row in data["snapshots"].values():
                if row["job_id"] == job_id and row["approved_at"]:
                    row["approved_at"] = None
                    changed = True
            if changed:
                atomic_json(self.path, data)

    def review(
        self,
        candidate_id: str,
        ranked,
        pack_key: str,
        artifacts: dict[str, Path],
        audit: dict,
        *,
        approved: bool,
        acknowledge_uncertain: bool = False,
    ) -> dict:
        if not approved or not audit.get("passed"):
            raise ValueError("Explicit approval and a passing artifact audit are required")
        if ranked.hard_blockers or ranked.eligibility_status == "blocked":
            raise ValueError("This job has eligibility blockers")
        url = ranked.job.application_url or ranked.job.url
        if not valid_job_url(url):
            raise ValueError("A valid public application URL is required")
        if not artifacts.get("cv_pdf") or not artifacts.get("cover_letter_pdf"):
            raise ValueError("Audited resume and cover-letter PDFs are required")
        live = verified_liveness(url, force=True)
        self._require_live(live, acknowledge_uncertain)
        with _LOCK:
            data = self.read()
            hashes = {name: file_hash(path) for name, path in artifacts.items()}
            expected_hashes = audit.get("hashes", {})
            if hashes["cv_pdf"] != expected_hashes.get("tailored_cv") or hashes["cover_letter_pdf"] != expected_hashes.get(
                "cover_letter"
            ):
                raise ValueError("Artifact audit hashes do not match the files being approved")
            snapshot_id = digest([candidate_id, ranked.job.job_id, pack_key, hashes])
            target = self.root / "snapshots" / snapshot_id
            target.mkdir(parents=True, exist_ok=True)
            target.chmod(0o700)
            saved = {}
            for name, source in artifacts.items():
                dest = target / (name + source.suffix)
                shutil.copyfile(source, dest)
                dest.chmod(0o600)
                if file_hash(dest) != hashes[name]:
                    raise ValueError("Artifacts changed during review; review again")
                saved[name] = str(dest)
            for old in data["snapshots"].values():
                if old["candidate_id"] == candidate_id and old["job_id"] == ranked.job.job_id:
                    old["approved_at"] = None
            row = {
                "id": snapshot_id,
                "candidate_id": candidate_id,
                "job_id": ranked.job.job_id,
                "company": ranked.job.company,
                "role": ranked.job.title,
                "score": ranked.fit_score,
                "url": url,
                "pack_key": pack_key,
                "artifacts": saved,
                "hashes": hashes,
                "audit": audit,
                "approved_at": now_iso(),
                "liveness": live,
            }
            data["snapshots"][snapshot_id] = row
            atomic_json(self.path, data)
            return row

    @staticmethod
    def _require_live(live: dict, acknowledge: bool) -> None:
        if live["liveness"] == "expired":
            raise ValueError("Posting is confirmed expired")
        if live["liveness"] == "uncertain" and not acknowledge:
            raise ValueError("Liveness is uncertain; acknowledgment is required")

    def export(self, candidate_id: str, snapshot_ids: list[str], *, acknowledge_uncertain: bool = False) -> dict:
        if not snapshot_ids:
            raise ValueError("Select at least one reviewed snapshot")
        with _LOCK:
            data = self.read()
            rows = []
            for sid in sorted(set(snapshot_ids)):
                row = data["snapshots"].get(sid)
                if not row or row["candidate_id"] != candidate_id or not row["approved_at"]:
                    raise ValueError("Snapshot is missing or its approval is stale")
                if row["score"] < 70:
                    raise ValueError("Every selected snapshot must meet the 70/100 export threshold")
                for name, path in row["artifacts"].items():
                    if file_hash(Path(path)) != row["hashes"][name]:
                        raise ValueError("Reviewed artifacts have changed; review again")
                self._require_live(verified_liveness(row["url"], force=True), acknowledge_uncertain)
                rows.append(row)
            export_id = digest([candidate_id, [row["id"] for row in rows]])
            existing = data["exports"].get(export_id)
            if existing and Path(existing["queue_path"]).is_file():
                if file_hash(Path(existing["queue_path"])) != existing.get("queue_hash"):
                    raise ValueError("Export queue changed on disk; restore the reviewed bundle before reuse")
                for row in rows:
                    for name, source in row["artifacts"].items():
                        copied = Path(existing["queue_path"]).parent / row["id"] / Path(source).name
                        if file_hash(copied) != row["hashes"][name]:
                            raise ValueError("Exported artifacts have changed; restore the reviewed bundle before reuse")
                return existing
            missing = [row["job_id"] for row in rows if row["job_id"] not in data["ids"]]
            if len(data["ids"]) + len(set(missing)) > 999:
                raise ValueError("Legacy bridge capacity exhausted (999 IDs); no IDs were reused")
            for job_id in missing:
                data["ids"].setdefault(job_id, f"{len(data['ids']) + 1:03d}")
            # Reserve IDs before writing artifacts; interrupted exports never recycle them.
            atomic_json(self.path, data)
            bundle = self.root / "exports" / export_id
            packs = []
            for row in rows:
                dest = bundle / row["id"]
                dest.mkdir(parents=True, exist_ok=True)
                dest.chmod(0o700)
                for name, path in row["artifacts"].items():
                    copied = dest / Path(path).name
                    shutil.copyfile(path, copied)
                    copied.chmod(0o600)
                    if file_hash(copied) != row["hashes"][name]:
                        raise ValueError("Reviewed artifacts changed during export; review again")
                packs.append(
                    {
                        "id": data["ids"][row["job_id"]],
                        "company": row["company"],
                        "role": row["role"],
                        "score": row["score"],
                        "job_url": row["url"],
                        "pdf_path": dest / Path(row["artifacts"]["cv_pdf"]).name,
                    }
                )
            queue_path = bundle / "aihawk-queue.json"
            export_aihawk_queue(packs, output_path=queue_path, repo_dir=bundle)
            result = {
                "id": export_id,
                "candidate_id": candidate_id,
                "queue_path": str(queue_path),
                "queue_hash": file_hash(queue_path),
                "snapshot_ids": [row["id"] for row in rows],
                "exported_at": now_iso(),
            }
            atomic_json(bundle / "jobvis-manifest.json", {**result, "jobs": rows, "bridge_ids": data["ids"]})
            data["exports"][export_id] = result
            atomic_json(self.path, data)
            return result
