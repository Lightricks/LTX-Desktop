"""Persistent job queue for sequential generation processing."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


@dataclass
class QueueJob:
    id: str
    type: Literal["video", "image"]
    model: str
    params: dict[str, Any]
    status: Literal["queued", "running", "complete", "error", "cancelled"]
    slot: Literal["gpu", "api"]
    progress: int = 0
    phase: str = "queued"
    result_paths: list[str] = field(default_factory=lambda: list[str]())
    error: str | None = None
    created_at: str = ""


class JobQueue:
    def __init__(self, persistence_path: Path) -> None:
        self._path = persistence_path
        self._jobs: list[QueueJob] = []
        self._load()

    def submit(
        self,
        *,
        job_type: str,
        model: str,
        params: dict[str, Any],
        slot: str,
    ) -> QueueJob:
        job = QueueJob(
            id=uuid.uuid4().hex[:8],
            type=job_type,  # type: ignore[arg-type]
            model=model,
            params=params,
            status="queued",
            slot=slot,  # type: ignore[arg-type]
            progress=0,
            phase="queued",
            result_paths=[],
            error=None,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._jobs.append(job)
        self._save()
        return job

    def get_all_jobs(self) -> list[QueueJob]:
        return list(self._jobs)

    def get_job(self, job_id: str) -> QueueJob | None:
        for job in self._jobs:
            if job.id == job_id:
                return job
        return None

    def next_queued_for_slot(self, slot: str) -> QueueJob | None:
        for job in self._jobs:
            if job.status == "queued" and job.slot == slot:
                return job
        return None

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: int | None = None,
        phase: str | None = None,
        result_paths: list[str] | None = None,
        error: str | None = None,
    ) -> None:
        job = self.get_job(job_id)
        if job is None:
            return
        if status is not None:
            job.status = status  # type: ignore[assignment]
        if progress is not None:
            job.progress = progress
        if phase is not None:
            job.phase = phase
        if result_paths is not None:
            job.result_paths = result_paths
        if error is not None:
            job.error = error
        self._save()

    def cancel_job(self, job_id: str) -> None:
        self.update_job(job_id, status="cancelled")

    def clear_finished(self) -> None:
        self._jobs = [j for j in self._jobs if j.status not in ("complete", "error", "cancelled")]
        self._save()

    def _save(self) -> None:
        data = {"jobs": [asdict(j) for j in self._jobs]}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for item in raw.get("jobs", []):
                job = QueueJob(**item)
                if job.status == "running":
                    job.status = "queued"
                    job.progress = 0
                    job.phase = "queued"
                self._jobs.append(job)
        except (json.JSONDecodeError, TypeError, KeyError):
            self._jobs = []
