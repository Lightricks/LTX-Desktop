"""Background queue worker that processes jobs from the job queue."""

from __future__ import annotations

import logging
import threading
from typing import Protocol

from state.job_queue import JobQueue, QueueJob

logger = logging.getLogger(__name__)


class JobExecutor(Protocol):
    def execute(self, job: QueueJob) -> list[str]:
        ...


class QueueWorker:
    def __init__(
        self,
        *,
        queue: JobQueue,
        gpu_executor: JobExecutor,
        api_executor: JobExecutor,
    ) -> None:
        self._queue = queue
        self._gpu_executor = gpu_executor
        self._api_executor = api_executor
        self._gpu_busy = False
        self._api_busy = False
        self._lock = threading.Lock()

    def tick(self) -> None:
        """Process one round: pick up available jobs for each free slot."""
        gpu_job: QueueJob | None = None
        api_job: QueueJob | None = None

        with self._lock:
            if not self._gpu_busy:
                gpu_job = self._queue.next_queued_for_slot("gpu")
                if gpu_job is not None:
                    self._gpu_busy = True
                    self._queue.update_job(gpu_job.id, status="running", phase="starting")

            if not self._api_busy:
                api_job = self._queue.next_queued_for_slot("api")
                if api_job is not None:
                    self._api_busy = True
                    self._queue.update_job(api_job.id, status="running", phase="starting")

        threads: list[threading.Thread] = []

        if gpu_job is not None:
            t = threading.Thread(target=self._run_job, args=(gpu_job, self._gpu_executor, "gpu"), daemon=True)
            threads.append(t)
            t.start()

        if api_job is not None:
            t = threading.Thread(target=self._run_job, args=(api_job, self._api_executor, "api"), daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    def _run_job(self, job: QueueJob, executor: JobExecutor, slot: str) -> None:
        try:
            result_paths = executor.execute(job)
            self._queue.update_job(job.id, status="complete", progress=100, phase="complete", result_paths=result_paths)
        except Exception as exc:
            logger.error("Job %s failed: %s", job.id, exc)
            self._queue.update_job(job.id, status="error", error=str(exc))
        finally:
            with self._lock:
                if slot == "gpu":
                    self._gpu_busy = False
                else:
                    self._api_busy = False
