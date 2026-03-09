"""Background queue worker that processes jobs from the job queue."""

from __future__ import annotations

import logging
import threading
from typing import Callable, Protocol

from services.interfaces import GpuCleaner
from state.job_queue import JobQueue, QueueJob

logger = logging.getLogger(__name__)


class JobExecutor(Protocol):
    def execute(self, job: QueueJob) -> list[str]:
        ...


class EnhancePromptProvider(Protocol):
    def enhance_i2v_motion(self, image_path: str) -> str:
        ...


class CreditDeductor(Protocol):
    def deduct_credits(
        self, generation_type: str, count: int,
        metadata: dict[str, object] | None,
    ) -> dict[str, object]:
        ...


def _credit_type_for_job(job: QueueJob) -> str | None:
    """Map a completed API job to its Palette credit type. Returns None for local GPU jobs."""
    model = job.model.lower()
    if "seedance" in model:
        return "video_seedance"
    if job.type == "image":
        return "image"
    has_image = bool(job.params.get("imagePath"))
    if has_image:
        return "video_i2v"
    return "video_t2v"


class QueueWorker:
    def __init__(
        self,
        *,
        queue: JobQueue,
        gpu_executor: JobExecutor,
        api_executor: JobExecutor,
        gpu_cleaner: GpuCleaner | None = None,
        on_batch_complete: Callable[[str, list[QueueJob]], None] | None = None,
        enhance_handler: EnhancePromptProvider | None = None,
        credit_deductor: CreditDeductor | None = None,
    ) -> None:
        self._queue = queue
        self._gpu_executor = gpu_executor
        self._api_executor = api_executor
        self._gpu_cleaner = gpu_cleaner
        self._gpu_busy = False
        self._api_busy = False
        self._lock = threading.Lock()
        self._on_batch_complete = on_batch_complete
        self._enhance_handler = enhance_handler
        self._credit_deductor = credit_deductor
        self._notified_batches: set[str] = set()

    def tick(self) -> None:
        """Process one round: pick up available jobs for each free slot.

        Non-blocking — spawns daemon threads for each job so the tick loop
        can keep checking for new jobs on other slots.
        """
        # First, fail any jobs whose dependencies have errored/cancelled
        self._fail_orphaned_dependents()

        # Recover from stuck busy flags — if slot is busy but no job is actually
        # running for that slot, reset the flag.  This handles cases where a job
        # was cancelled externally while the executor thread was still working.
        self._recover_stuck_slots()

        gpu_job: QueueJob | None = None
        api_job: QueueJob | None = None

        with self._lock:
            if not self._gpu_busy:
                gpu_job = self._next_ready_job("gpu")
                if gpu_job is not None:
                    self._gpu_busy = True
                    self._queue.update_job(gpu_job.id, status="running", phase="starting")

            if not self._api_busy:
                api_job = self._next_ready_job("api")
                if api_job is not None:
                    self._api_busy = True
                    self._queue.update_job(api_job.id, status="running", phase="starting")

        if gpu_job is not None:
            t = threading.Thread(target=self._run_job, args=(gpu_job, self._gpu_executor, "gpu"), daemon=True)
            t.start()

        if api_job is not None:
            t = threading.Thread(target=self._run_job, args=(api_job, self._api_executor, "api"), daemon=True)
            t.start()

        self._check_batch_completions()

    def _recover_stuck_slots(self) -> None:
        """Reset busy flags when no job is actually running for that slot.

        This handles the case where a job is cancelled via the API while the
        executor thread is mid-work.  The thread eventually finishes (or the
        process was restarted), but _gpu_busy / _api_busy stayed True.
        """
        has_running_gpu = any(
            j.status == "running" and j.slot == "gpu" for j in self._queue.all_jobs()
        )
        has_running_api = any(
            j.status == "running" and j.slot == "api" for j in self._queue.all_jobs()
        )
        with self._lock:
            if self._gpu_busy and not has_running_gpu:
                logger.info("Recovering stuck GPU slot — no running GPU jobs found")
                self._gpu_busy = False
            if self._api_busy and not has_running_api:
                logger.info("Recovering stuck API slot — no running API jobs found")
                self._api_busy = False

    def _next_ready_job(self, slot: str) -> QueueJob | None:
        for job in self._queue.queued_jobs_for_slot(slot):
            if job.depends_on is None:
                return job
            dep = self._queue.get_job(job.depends_on)
            if dep is None:
                return job  # Dependency missing, run anyway
            if dep.status == "complete":
                self._resolve_auto_params(job, dep)
                return job
            # dep still queued/running or already handled by _fail_orphaned_dependents
            continue
        return None

    def _fail_orphaned_dependents(self) -> None:
        for job in self._queue.all_jobs():
            if job.status != "queued" or job.depends_on is None:
                continue
            dep = self._queue.get_job(job.depends_on)
            if dep is not None and dep.status in ("error", "cancelled"):
                self._queue.update_job(
                    job.id,
                    status="error",
                    error=f"Upstream job {dep.id} failed: {dep.error or dep.status}",
                )

    def _resolve_auto_params(self, job: QueueJob, dep: QueueJob) -> None:
        for key, template in list(job.auto_params.items()):
            if template == "$dep.result_paths[0]" and dep.result_paths:
                job.params[key] = dep.result_paths[0]

        if job.auto_params.get("auto_prompt") == "true" and self._enhance_handler:
            image_path = job.params.get("imagePath", dep.result_paths[0] if dep.result_paths else "")
            if image_path:
                motion_prompt = self._enhance_handler.enhance_i2v_motion(str(image_path))
                job.params["prompt"] = motion_prompt

    def _check_batch_completions(self) -> None:
        seen: set[str] = set()
        for job in self._queue.all_jobs():
            if job.batch_id and job.batch_id not in self._notified_batches:
                seen.add(job.batch_id)
        for batch_id in seen:
            jobs = self._queue.jobs_for_batch(batch_id)
            if all(j.status in ("complete", "error", "cancelled") for j in jobs):
                self._notified_batches.add(batch_id)
                if self._on_batch_complete:
                    self._on_batch_complete(batch_id, jobs)

    def _run_job(self, job: QueueJob, executor: JobExecutor, slot: str) -> None:
        try:
            result_paths = executor.execute(job)
            self._queue.update_job(job.id, status="complete", progress=100, phase="complete", result_paths=result_paths)
            # Deduct credits for API-slot jobs (local GPU jobs are free)
            if slot == "api" and self._credit_deductor is not None:
                credit_type = _credit_type_for_job(job)
                if credit_type:
                    try:
                        self._credit_deductor.deduct_credits(
                            credit_type, 1,
                            {"model": job.model, "job_id": job.id},
                        )
                    except Exception as exc:
                        logger.warning("Credit deduction failed for job %s: %s", job.id, exc)
        except Exception as exc:
            logger.error("Job %s failed: %s", job.id, exc)
            self._queue.update_job(job.id, status="error", error=str(exc))
        finally:
            if slot == "gpu" and self._gpu_cleaner is not None:
                try:
                    self._gpu_cleaner.deep_cleanup()
                except Exception:
                    pass
            with self._lock:
                if slot == "gpu":
                    self._gpu_busy = False
                else:
                    self._api_busy = False
