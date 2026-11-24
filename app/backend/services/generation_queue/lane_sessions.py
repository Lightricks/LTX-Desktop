"""The session of a job on a lane that runs beside the local GPU lane.

All of its state is its own: progress, provider status and the cancel flag. It never
touches the process-wide generation state, so two lanes cannot read each other's job.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from api_types import ApiActivityStatus
from services.generation_interrupt import GenerationCancelledError
from services.generation_queue.sessions import JobProgress

logger = logging.getLogger(__name__)

# Stores the provider job id of an attempt: (generation_id, attempt_count, provider_job_id).
# Returns False when that attempt is no longer running, so the id was not stored.
ProviderJobIdSink = Callable[[str, int, str], bool]


class LaneJobSession:
    def __init__(
        self,
        generation_id: str,
        attempt_count: int,
        record_provider_job_id: ProviderJobIdSink,
    ) -> None:
        self._generation_id = generation_id
        self._attempt_count = attempt_count
        self._record_provider_job_id = record_provider_job_id
        self._lock = threading.Lock()
        self._progress: JobProgress | None = None
        self._api_status: ApiActivityStatus | None = None
        self._cancelled = threading.Event()

    @property
    def generation_id(self) -> str:
        return self._generation_id

    @property
    def holds_gpu_slot(self) -> bool:
        return False

    def start_api(self) -> None:
        return None

    def set_progress(
        self,
        phase: str,
        progress: int,
        current_step: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        with self._lock:
            self._progress = JobProgress(phase, progress, current_step, total_steps)

    def set_api_status(self, status: ApiActivityStatus) -> None:
        with self._lock:
            self._api_status = status

    def set_provider_job_id(self, provider_job_id: str) -> bool:
        logger.info(
            "Generation %s accepted by its provider as job %s", self._generation_id, provider_job_id
        )
        # A store failure must not fail the job: the provider already runs it. The caller
        # learns from the result and can cancel at the provider.
        try:
            stored = self._record_provider_job_id(
                self._generation_id, self._attempt_count, provider_job_id
            )
        except Exception:
            logger.error(
                "Could not store provider job %s of generation %s",
                provider_job_id,
                self._generation_id,
                exc_info=True,
            )
            return False
        if not stored:
            logger.warning(
                "Provider job %s of generation %s was not stored: the job is no longer running",
                provider_job_id,
                self._generation_id,
            )
        return stored

    def progress(self) -> JobProgress | None:
        with self._lock:
            return self._progress

    def api_status(self) -> ApiActivityStatus | None:
        with self._lock:
            return self._api_status

    def raise_if_cancelled(self) -> None:
        if self._cancelled.is_set():
            raise GenerationCancelledError()

    def wait_cancelled(self, timeout: float) -> bool:
        return self._cancelled.wait(timeout)

    def cancel(self) -> None:
        self._cancelled.set()

    def complete(self, result: str | list[str] | None) -> None:
        del result

    def fail(self, error: str) -> None:
        del error
