"""Provider job ids that a retry replaced, kept in a JSON file until a cancel of them succeeds.

A row of the store holds the provider job id of its newest attempt only. A retry moves the id of
the failed attempt here, so the id is not lost while the provider can still run and bill that job.
Boot recovery and the runner read the file, ask the provider to cancel each job, and then remove
the entry. The file is deleted when no entry is left.

A file is enough for this data: it is small, short lived, and no query joins it. The store writes
an entry before it commits the retry, so a crash between the two leaves the id in both places.
That is harmless: a cancel is safe to repeat, and the readers skip a duplicate.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from pydantic import BaseModel, JsonValue

from services.records import ProviderOrphan

logger = logging.getLogger(__name__)

DISPLACED_PROVIDER_JOBS_FILENAME = "displaced_provider_jobs.json"
_FORMAT_VERSION = 1

# One lock for every instance: the readers and the writers of a file can be different objects.
_LOCK = threading.Lock()


class _StoredFile(BaseModel):
    """The file as a whole. An entry is checked one by one, so one bad entry does not hide the rest."""

    jobs: list[JsonValue]


class DisplacedProviderJobs:
    def __init__(self, path: Path) -> None:
        self._path = path

    def add(self, orphan: ProviderOrphan) -> None:
        """Keep an id. Adding the same id twice keeps one entry."""
        with _LOCK:
            jobs = self._read()
            if any(_same_job(job, orphan) for job in jobs):
                return
            jobs.append(orphan)
            self._write(jobs)

    def remove(self, generation_id: str, provider_job_id: str) -> bool:
        """Forget an id once the provider dropped it. False when the file does not hold it."""
        with _LOCK:
            jobs = self._read()
            kept = [
                job
                for job in jobs
                if (job.generation_id, job.provider_job_id) != (generation_id, provider_job_id)
            ]
            if len(kept) == len(jobs):
                return False
            self._write(kept)
            return True

    def list(self, generation_id: str | None = None) -> list[ProviderOrphan]:
        """The kept ids in the order they were added, of one job or of all jobs."""
        with _LOCK:
            jobs = self._read()
        if generation_id is None:
            return jobs
        return [job for job in jobs if job.generation_id == generation_id]

    def _read(self) -> list[ProviderOrphan]:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        try:
            entries = _StoredFile.model_validate_json(raw).jobs
        except ValueError:
            self._set_aside()
            return []
        jobs: list[ProviderOrphan] = []
        for entry in entries:
            try:
                job = ProviderOrphan.model_validate(entry)
            except ValueError:
                logger.warning("Skipped an invalid entry in %s", self._path.name)
                continue
            if not any(_same_job(kept, job) for kept in jobs):
                jobs.append(job)
        return jobs

    def _write(self, jobs: list[ProviderOrphan]) -> None:
        if not jobs:
            self._path.unlink(missing_ok=True)
            return
        payload = {
            "version": _FORMAT_VERSION,
            "jobs": [job.model_dump(mode="json") for job in jobs],
        }
        temporary = self._path.with_name(f"{self._path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self._path)

    def _set_aside(self) -> None:
        """Keep an unreadable file for a person to inspect, and start with an empty list."""
        aside = self._path.with_name(f"{self._path.name}.corrupt")
        logger.warning("%s is not readable. It is moved to %s.", self._path.name, aside.name)
        try:
            os.replace(self._path, aside)
        except OSError:
            logger.warning("Could not move %s aside", self._path.name, exc_info=True)


def _same_job(a: ProviderOrphan, b: ProviderOrphan) -> bool:
    return (a.generation_id, a.provider_job_id) == (b.generation_id, b.provider_job_id)
