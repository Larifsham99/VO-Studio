"""Tiny JSON-file store (projects, references, jobs). One process, one lock."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path


class Store:
    COLLECTIONS = ("projects", "references", "jobs")

    def __init__(self, data_dir: Path):
        self.path = Path(data_dir) / "db.json"
        self.lock = threading.RLock()
        self.data = {c: {} for c in self.COLLECTIONS}
        if self.path.exists():
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            for c in self.COLLECTIONS:
                self.data[c] = loaded.get(c, {})
        # jobs interrupted by a restart can't resume mid-generation
        for j in self.data["jobs"].values():
            if j.get("status") in ("queued", "running"):
                j.update(status="error", error="Server dimulakan semula semasa kerja ini berjalan. Sila jana semula.")
        self._save()

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, coll: str, obj: dict) -> dict:
        with self.lock:
            obj = {"id": uuid.uuid4().hex[:12], "created": time.time(), **obj}
            self.data[coll][obj["id"]] = obj
            self._save()
            return dict(obj)

    def get(self, coll: str, oid: str) -> dict | None:
        with self.lock:
            o = self.data[coll].get(oid)
            return dict(o) if o else None

    def update(self, coll: str, oid: str, **fields) -> dict:
        with self.lock:
            self.data[coll][oid].update(fields)
            self._save()
            return dict(self.data[coll][oid])

    def delete(self, coll: str, oid: str) -> dict | None:
        with self.lock:
            o = self.data[coll].pop(oid, None)
            self._save()
            return o

    def list(self, coll: str, **match) -> list[dict]:
        with self.lock:
            rows = [dict(o) for o in self.data[coll].values()
                    if all(o.get(k) == v for k, v in match.items())]
        return sorted(rows, key=lambda o: o["created"], reverse=True)
