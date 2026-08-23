import json
import os
import threading
import time
from typing import Iterable

from backend.models import AlarmRecord


class AlertStore:
    """Append-only JSONL persistence for alarm records. In-memory cache of the
    most recent `max_memory` records for fast queries + broadcast."""

    def __init__(self, path: str, max_memory: int = 2000, retention_seconds: float = 0.0):
        self.path = path
        self.max_memory = max_memory
        # Older records are dropped entirely once this age is exceeded. A demo
        # instance accumulates every alert from every run; without an expiry the
        # panel eventually shows a wall of events from sessions nobody
        # remembers. 0 keeps everything, which is what a real site wants.
        self.retention_seconds = max(0.0, float(retention_seconds))
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._records: list[AlarmRecord] = self._load_tail()
        self._prune_locked()

    def _load_tail(self) -> list[AlarmRecord]:
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return []
        out: list[AlarmRecord] = []
        for line in lines[-self.max_memory:]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(AlarmRecord.model_validate_json(line))
            except Exception:
                continue
        return out

    def _prune_locked(self) -> None:
        """Drop expired records from memory and from the log they came from."""
        if not self.retention_seconds:
            return
        cutoff = time.time() - self.retention_seconds
        kept = [r for r in self._records if r.timestamp >= cutoff]
        if len(kept) == len(self._records):
            return
        self._records = kept
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for record in kept:
                    f.write(record.model_dump_json() + "\n")
            os.replace(tmp, self.path)
        except OSError:
            pass

    def append(self, record: AlarmRecord) -> None:
        with self._lock:
            self._prune_locked()
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(record.model_dump_json() + "\n")
            except OSError:
                pass
            self._records.append(record)
            if len(self._records) > self.max_memory:
                self._records = self._records[-self.max_memory:]

    def _rewrite_record(self, updated: AlarmRecord) -> None:
        """Atomically replace one record without truncating older JSONL rows."""
        tmp = self.path + ".tmp"
        try:
            lines: list[str] = []
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as source:
                    lines = source.readlines()
            replaced = False
            with open(tmp, "w", encoding="utf-8") as target:
                for line in lines:
                    try:
                        record = AlarmRecord.model_validate_json(line)
                    except Exception:
                        target.write(line)
                        continue
                    if record.id == updated.id:
                        target.write(updated.model_dump_json() + "\n")
                        replaced = True
                    else:
                        target.write(line if line.endswith("\n") else line + "\n")
                if not replaced:
                    target.write(updated.model_dump_json() + "\n")
            os.replace(tmp, self.path)
        except OSError:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

    def update_review(
        self,
        record_id: str,
        status: str,
        reviewed_by: str | None = None,
        note: str | None = None,
        reviewed_at: float | None = None,
    ) -> AlarmRecord | None:
        valid = {"new", "acknowledged", "confirmed", "false_positive", "escalated"}
        if status not in valid:
            raise ValueError(f"invalid review status: {status}")
        with self._lock:
            for idx in range(len(self._records) - 1, -1, -1):
                record = self._records[idx]
                if record.id != record_id:
                    continue
                updated = record.model_copy(update={
                    "review_status": status,
                    "reviewed_at": reviewed_at if reviewed_at is not None else time.time(),
                    "reviewed_by": reviewed_by.strip() if reviewed_by else None,
                    "review_note": note.strip() if note else None,
                })
                self._records[idx] = updated
                self._rewrite_record(updated)
                return updated
        return None

    def query(
        self,
        mode: str | None = None,
        severity: str | None = None,
        kind: str | None = None,
        worker_id: str | None = None,
        review_status: str | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AlarmRecord]:
        with self._lock:
            self._prune_locked()
            records = list(self._records)
        if mode:
            records = [r for r in records if r.mode == mode]
        if severity:
            records = [r for r in records if r.severity == severity]
        if kind:
            records = [r for r in records if r.kind == kind]
        if worker_id:
            records = [
                r for r in records
                if str(r.details.get("worker_id", "")) == worker_id
            ]
        if review_status:
            records = [r for r in records if r.review_status == review_status]
        if since is not None:
            records = [r for r in records if r.timestamp >= since]
        if until is not None:
            records = [r for r in records if r.timestamp <= until]
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[offset:offset + limit]

    def summary(
        self,
        mode: str | None = None,
        severity: str | None = None,
        kind: str | None = None,
        worker_id: str | None = None,
        review_status: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> dict:
        """Aggregate counts for the filtered audit-trail dashboard."""
        with self._lock:
            self._prune_locked()
            records = list(self._records)
        if mode:
            records = [r for r in records if r.mode == mode]
        if severity:
            records = [r for r in records if r.severity == severity]
        if kind:
            records = [r for r in records if r.kind == kind]
        if worker_id:
            records = [
                r for r in records
                if str(r.details.get("worker_id", "")) == worker_id
            ]
        if review_status:
            records = [r for r in records if r.review_status == review_status]
        if since is not None:
            records = [r for r in records if r.timestamp >= since]
        if until is not None:
            records = [r for r in records if r.timestamp <= until]
        by_sev: dict[str, int] = {}
        by_kind: dict[str, int] = {}
        by_worker: dict[str, int] = {}
        by_review_status: dict[str, int] = {}
        for r in records:
            by_sev[r.severity] = by_sev.get(r.severity, 0) + 1
            by_kind[r.kind] = by_kind.get(r.kind, 0) + 1
            worker_id = r.details.get("worker_id")
            if worker_id:
                worker_id = str(worker_id)
                by_worker[worker_id] = by_worker.get(worker_id, 0) + 1
            by_review_status[r.review_status] = by_review_status.get(r.review_status, 0) + 1
        return {
            "total": len(records),
            "by_severity": by_sev,
            "by_kind": by_kind,
            "by_worker": by_worker,
            "by_review_status": by_review_status,
        }

    def get(self, record_id: str) -> AlarmRecord | None:
        with self._lock:
            for r in reversed(self._records):
                if r.id == record_id:
                    return r
        return None

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def extend(self, records: Iterable[AlarmRecord]) -> None:
        for r in records:
            self.append(r)
