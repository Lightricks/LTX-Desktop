from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from typing import Literal, TypeVar

from services.store import Store
from services.records import (
    AttemptError,
    CapabilityFailedError,
    GenerationErrorCode,
    GenerationRecord,
    MarkSucceededError,
    OutputSpec,
    ProviderOrphan,
    StatusError,
    UnavailableError,
)
from _routes._errors import HTTPError
from logging_policy import log_background_exception
from runtime_config.reject_reasons import BUDGET_MISS_CODES
from services.analytics import (
    AnalyticsService,
    QueuedEnhancement,
    QueuedExecutionModes,
    QueuedHardwareSnapshot,
    queued_generation_details,
)
from services.generation_queue.control import (
    GenerationSlotWaitAborted,
    SharedGenerationSlot,
)
from services.generation_queue.lanes import LOCAL_LANE_SPEC, LaneSpec
from services.generation_queue.orphans import cancel_provider_orphans_in_background, forget_in
from services.generation_queue.output_names import output_asset_name
from services.generation_queue.registry import ExecutorRegistry
from services.generation_queue.sessions import JobSession, SlotCancelSession
from services.generation_queue.types import ExecutionOutcome, OutputAllocation
from services.generation_interrupt import is_cancel_exception
from services.gpu_info.gpu_info import GpuInfo
from services.patches import diffusion_stage_cache
from services.patches.pinned_pool_fix import is_cuda_oom_error
from services.storage_info.storage_info import DriveType

logger = logging.getLogger(__name__)
_BACKGROUND_TASK_NAME = "generation-queue-runner"
_EXECUTOR_FAILURE_CODE: GenerationErrorCode = "EXECUTOR_FAILED"
_OOM_CODE: GenerationErrorCode = "CUDA_OOM"
_CAPABILITY_FAILURE_CODE: GenerationErrorCode = "CAPABILITY_FAILED"
_JOB_TOO_LARGE_CODE: GenerationErrorCode = "JOB_TOO_LARGE"
# BUDGET_MISS_CODES (imported above) are budget verdicts, not true failures:
# an executor raising one means the job stopped fitting between create and
# execute. Mapped to JOB_TOO_LARGE with the enriched detail preserved in
# the log; anything else keeps the branch default.
_LTX_INVALID_API_KEY_CODE: GenerationErrorCode = "LTX_INVALID_API_KEY"
_LTX_API_PROMPT_EMBEDDING_FAILED_CODE: GenerationErrorCode = "LTX_API_PROMPT_EMBEDDING_FAILED"
_IMAGE_BLOCKED_CODE: GenerationErrorCode = "IMAGE_BLOCKED_BY_SAFETY_CHECKER"


def _error_code_for(exc: BaseException, default: GenerationErrorCode) -> GenerationErrorCode:
    """OOM takes precedence over the branch default.

    Edit executors re-raise torch OOMs (a RuntimeError subclass) as
    capability failures, so the capability branch must check too — otherwise
    the policy-miss signal hides as a generic failure.
    """
    if is_cuda_oom_error(exc):
        return _OOM_CODE
    return default


_T = TypeVar("_T")


def _probe_or_none(probe: Callable[[], _T | None]) -> _T | None:
    """One tolerant probe: exceptions and missing values both become None."""
    try:
        return probe()
    except Exception:
        return None


def _capability_error_code(exc: CapabilityFailedError) -> GenerationErrorCode:
    if exc.code == _LTX_INVALID_API_KEY_CODE:
        return _LTX_INVALID_API_KEY_CODE
    if exc.code == _LTX_API_PROMPT_EMBEDDING_FAILED_CODE:
        return _LTX_API_PROMPT_EMBEDDING_FAILED_CODE
    if exc.code == _IMAGE_BLOCKED_CODE:
        return _IMAGE_BLOCKED_CODE
    if exc.code in BUDGET_MISS_CODES:
        return _JOB_TOO_LARGE_CODE
    return _CAPABILITY_FAILURE_CODE


def _http_budget_error_code(exc: HTTPError) -> GenerationErrorCode:
    """Budget-verdict HTTP codes map to JOB_TOO_LARGE; the rest are true failures."""
    if exc.code in BUDGET_MISS_CODES:
        return _JOB_TOO_LARGE_CODE
    return _EXECUTOR_FAILURE_CODE


@dataclass
class _ActiveGeneration:
    generation_id: str
    attempt_count: int
    session: JobSession


class QueueRunner:
    """Runs the queued jobs of one lane, up to the lane's concurrency."""

    def __init__(
        self,
        *,
        db: Store,
        generation: SharedGenerationSlot,
        executors: ExecutorRegistry,
        lane: LaneSpec = LOCAL_LANE_SPEC,
        session_factory: Callable[[GenerationRecord], JobSession] | None = None,
        analytics: AnalyticsService | None = None,
        enhancement_policy: Callable[[GenerationRecord], QueuedEnhancement] | None = None,
        gpu_info: GpuInfo | None = None,
        checkpoint_drive_type: DriveType | None = None,
        idle_wait_seconds: float = 0.25,
        stop_join_timeout_seconds: float = 600.0,
    ) -> None:
        self._db = db
        self._generation = generation
        self._lane = lane
        self._session_factory = session_factory
        self._executors = executors
        self._analytics = analytics
        self._enhancement_policy = enhancement_policy
        self._gpu_info = gpu_info
        self._checkpoint_drive_type: DriveType | None = checkpoint_drive_type
        self._idle_wait_seconds = idle_wait_seconds
        self._stop_join_timeout_seconds = stop_join_timeout_seconds

        self._shutdown = Event()
        self._wake_event = Event()
        self._lifecycle_lock = Lock()
        self._threads: list[Thread] = []
        self._active: dict[tuple[str, int], _ActiveGeneration] = {}
        self._finished_listener: Callable[[str], None] | None = None

    def set_finished_listener(self, listener: Callable[[str], None] | None) -> None:
        self._finished_listener = listener

    def set_checkpoint_drive_type(self, drive_type: DriveType | None) -> None:
        """Refresh the telemetry snapshot's drive class (models dir moved)."""
        self._checkpoint_drive_type = drive_type

    def start(self) -> None:
        try:
            self._db.has_queued(self._lane.id)
        except UnavailableError:
            return
        with self._lifecycle_lock:
            if any(thread.is_alive() for thread in self._threads):
                return
            self._shutdown.clear()
            self._wake_event.clear()
            self._threads = [
                Thread(
                    target=self._run,
                    name=f"generation-queue-{self._lane.id}-{index}",
                    daemon=True,
                )
                for index in range(self._lane.concurrency)
            ]
            for thread in self._threads:
                thread.start()

    def request_stop(self) -> None:
        """Tell the workers to stop. Does not wait for them.

        A lane with a stop grace runs provider jobs that a stopping app cannot wait for. They
        are cancelled now, so each one frees its provider request inside the grace time.
        """
        with self._lifecycle_lock:
            if not self._threads:
                return
            self._shutdown.set()
            self._wake_event.set()
            sessions = (
                [active.session for active in self._active.values()]
                if self._lane.stop_grace_seconds is not None
                else []
            )
        for session in sessions:
            try:
                session.cancel()
            except Exception:
                logger.warning("Could not cancel a running job at stop", exc_info=True)

    def join(self, deadline: float) -> None:
        """Wait for the workers until the monotonic deadline.

        A lane with a stop grace waits at most that long, so a slow provider call does
        not hold up shutdown.
        """
        with self._lifecycle_lock:
            threads = list(self._threads)
        if self._lane.stop_grace_seconds is not None:
            deadline = min(deadline, time.monotonic() + self._lane.stop_grace_seconds)
        for thread in threads:
            if thread is not current_thread():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        with self._lifecycle_lock:
            if self._threads == threads and not any(t.is_alive() for t in threads):
                self._threads = []

    def stop(self) -> None:
        self.request_stop()
        self.join(time.monotonic() + self._stop_join_timeout_seconds)

    def wake(self) -> None:
        self._wake_event.set()

    def owns(self, generation_id: str, attempt_count: int) -> bool:
        with self._lifecycle_lock:
            return (generation_id, attempt_count) in self._active

    def session_for(self, generation_id: str) -> JobSession | None:
        with self._lifecycle_lock:
            for (active_id, _), active in self._active.items():
                if active_id == generation_id:
                    return active.session
        return None

    def interrupt(self, generation_id: str, attempt_count: int) -> None:
        # The lock stays held while the session cancels. The worker needs it to leave the
        # active set, so the generation slot is not freed before the cancel returns.
        with self._lifecycle_lock:
            active = self._active.get((generation_id, attempt_count))
            if active is None:
                return
            try:
                active.session.cancel()
            except Exception as exc:
                logger.warning(
                    "Generation %s attempt %s interrupt failed: %s",
                    generation_id,
                    attempt_count,
                    exc,
                )

    def is_idle(self) -> bool:
        """No job runs and none waits in this lane.

        Takes no runner lock: a cancel holds the runner lock while it takes the state lock. The
        GPU lease calls it before it takes the state lock, because it reads SQLite.
        """
        return not self._active and not self._db.has_queued(self._lane.id)

    @property
    def is_running(self) -> bool:
        with self._lifecycle_lock:
            return any(thread.is_alive() for thread in self._threads)

    @property
    def lane(self) -> LaneSpec:
        return self._lane

    def _run(self) -> None:
        while True:
            if self._shutdown.is_set():
                return
            try:
                if not self._db.has_queued(self._lane.id):
                    self._wake_event.wait(timeout=self._idle_wait_seconds)
                    self._wake_event.clear()
                    continue

                with self._generation.wait_for_generation_slot(self._shutdown):
                    claimed = self._claim_and_register_active()
                    if claimed is None:
                        if self._shutdown.is_set():
                            return
                        continue
                    job, session = claimed
                    started = time.monotonic()
                    try:
                        self._process(job, session)
                        self._release_provider_job(job)
                    finally:
                        self._clear_active(job)
                        logger.info(
                            "Generation %s left lane %s after %.1fs",
                            job.id,
                            self._lane.id,
                            time.monotonic() - started,
                        )
            except GenerationSlotWaitAborted:
                return
            except Exception as exc:
                log_background_exception(_BACKGROUND_TASK_NAME, exc)
                if self._shutdown.wait(timeout=self._idle_wait_seconds):
                    return

    def _release_provider_job(self, job: GenerationRecord) -> None:
        """Ask the provider to drop the jobs of this generation that nobody runs any more.

        These are the job of an attempt that failed or was cancelled, and the ids of older
        attempts that a retry replaced (a retry can land before this runs). The provider can
        still run them. The executor only stops waiting. The call runs on its own thread, so a
        slow provider never holds this worker. An id stays stored until the call succeeds, so a
        failed call is tried again at the next boot.
        """
        if self._lane.cancel_orphan is None:
            return
        try:
            orphans = [
                orphan
                for orphan in self._db.list_displaced_provider_jobs(job.id)
                if orphan.lane == self._lane.id
            ]
            latest = self._db.get_generation(job.id)
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)
            return
        if (
            latest is not None
            and latest.attempt_count == job.attempt_count
            and latest.status in {"failed", "cancelled"}
            and latest.provider_job_id
        ):
            orphans.append(
                ProviderOrphan(
                    generation_id=latest.id,
                    lane=self._lane.id,
                    provider_job_id=latest.provider_job_id,
                )
            )
        if not orphans:
            return
        cancel_provider_orphans_in_background(
            orphans, lambda _lane_id: self._lane, forget_in(self._db)
        )

    def _new_session(self, generation: GenerationRecord) -> JobSession:
        if self._session_factory is not None:
            return self._session_factory(generation)
        return SlotCancelSession(self._generation, generation.id)

    def _claim_and_register_active(self) -> tuple[GenerationRecord, JobSession] | None:
        if self._shutdown.is_set():
            return None
        claimed = self._db.claim_next_queued(self._lane.id)
        if claimed is None:
            return None
        session = self._new_session(claimed)
        with self._lifecycle_lock:
            self._active[(claimed.id, claimed.attempt_count)] = _ActiveGeneration(
                generation_id=claimed.id,
                attempt_count=claimed.attempt_count,
                session=session,
            )
            # request_stop only sees jobs registered before it took this lock. A job that
            # registers after it must be cancelled here, or it would run past the stop.
            stopped_while_claiming = (
                self._shutdown.is_set() and self._lane.stop_grace_seconds is not None
            )
        if stopped_while_claiming:
            session.cancel()
        try:
            latest = self._db.get_generation(claimed.id)
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)
            self._clear_active(claimed)
            return None
        if latest is None or latest.attempt_count != claimed.attempt_count:
            self._clear_active(claimed)
            return None
        if latest.status == "cancelling":
            self._send_generation_started(latest)
            self._finalize_cancellation_or_unlink(claimed, [])
            self._clear_active(claimed)
            return None
        if latest.status != "running":
            self._clear_active(claimed)
            return None
        self._send_generation_started(latest)
        logger.info(
            "Generation %s (%s) started on lane %s, attempt %s",
            claimed.id,
            claimed.feature,
            self._lane.id,
            claimed.attempt_count,
        )
        return claimed, session

    def _process(self, claimed: GenerationRecord, session: JobSession) -> None:
        allocations: list[OutputAllocation] = []
        partial_paths: list[str] = []
        try:
            executor = self._executors.get(claimed.feature)
            plans = executor.plan_outputs(claimed)
            for plan in plans:
                asset_id, dest_path = self._db.allocate_output_path(
                    plan.media_kind, plan.mime_type
                )
                allocations.append(
                    OutputAllocation(
                        plan=plan,
                        asset_id=asset_id,
                        dest_path=dest_path,
                    )
                )
                partial_paths.append(dest_path)
            # Release dead pools (cached transformer + allocator reserved)
            # before the execute-time budget samples free: start_generation
            # evicts after the sample, so sampling first charges the job
            # for D it never sees. Warm live weights W still count against the
            # first sample; asks landing in (free', free'+W] get one
            # evict-and-resample inside the budget gates themselves (see
            # runtime_config.evict_retry) instead of failing outright.
            # Best-effort: a reclaim failure must never fail the job it was
            # trying to save.
            try:
                if self._lane.uses_gpu:
                    diffusion_stage_cache.evict()
            except Exception:
                logger.warning(
                    "Generation %s reclaim before execute failed; continuing",
                    claimed.id,
                    exc_info=True,
                )
            outcome = executor.execute(claimed, tuple(allocations), session)
            allocations = [
                allocation
                for allocation in allocations
                if not allocation.plan.optional or Path(allocation.dest_path).is_file()
            ]
        except Exception as exc:
            if is_cancel_exception(exc):
                self._mark_cancelled(claimed, partial_paths)
            elif isinstance(exc, CapabilityFailedError):
                logger.warning(
                    "Generation %s attempt %s capability error: %s",
                    claimed.id,
                    claimed.attempt_count,
                    exc.detail,
                )
                self._mark_failed(
                    claimed, _error_code_for(exc, _capability_error_code(exc)), partial_paths
                )
            elif isinstance(exc, HTTPError):
                logger.warning(
                    "Generation %s attempt %s HTTP error at execute time: %s",
                    claimed.id,
                    claimed.attempt_count,
                    exc.detail,
                )
                self._mark_failed(
                    claimed, _error_code_for(exc, _http_budget_error_code(exc)), partial_paths
                )
            else:
                log_background_exception(_BACKGROUND_TASK_NAME, exc)
                self._mark_failed(claimed, _error_code_for(exc, _EXECUTOR_FAILURE_CODE), partial_paths)
            return

        self._mark_succeeded(claimed, allocations, outcome)

    def _mark_succeeded(
        self,
        claimed: GenerationRecord,
        allocations: list[OutputAllocation],
        outcome: ExecutionOutcome | None,
    ) -> None:
        params = claimed.spec.get("params")
        prompt: object = None
        if isinstance(params, dict):
            prompt = params.get("prompt")
        written_prompt = outcome.written_prompt if outcome is not None else None
        if written_prompt is not None:
            prompt = written_prompt
        outputs = [
            OutputSpec(
                asset_id=allocation.asset_id,
                dest_path=allocation.dest_path,
                ordinal=index,
                mime_type=allocation.plan.mime_type,
                name=output_asset_name(
                    prompt=prompt,
                    created_at_ms=claimed.created_at,
                    asset_id=allocation.asset_id,
                    mime_type=allocation.plan.mime_type,
                ),
            )
            for index, allocation in enumerate(allocations)
        ]
        paths = [output.dest_path for output in outputs]
        if self._is_cancelling_attempt(claimed):
            self._finalize_cancellation_or_unlink(claimed, paths)
            return
        try:
            completed = self._db.mark_succeeded(
                claimed.id,
                outputs,
                attempt_count=claimed.attempt_count,
                written_prompt=written_prompt,
            )
        except MarkSucceededError as exc:
            self._mark_failed(claimed, exc.error_code, paths)
        except (StatusError, AttemptError):
            self._finalize_cancellation_or_unlink(claimed, paths)
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)
            self._unlink_unless_succeeded(claimed.id, paths)
        else:
            self._notify_finished(claimed.id)
            self._send_generation_ended(completed, outcome)

    def _is_cancelling_attempt(self, claimed: GenerationRecord) -> bool:
        try:
            latest = self._db.get_generation(claimed.id)
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)
            return False
        return (
            latest is not None
            and latest.status == "cancelling"
            and latest.attempt_count == claimed.attempt_count
        )

    def _mark_failed(
        self,
        claimed: GenerationRecord,
        error_code: GenerationErrorCode,
        partial_paths: list[str],
    ) -> None:
        wrote = self._cas_or_unlink(
            claimed,
            partial_paths,
            lambda: self._db.mark_failed(
                claimed.id,
                error_code,
                partial_paths,
                attempt_count=claimed.attempt_count,
            ),
        )
        if wrote:
            self._notify_finished(claimed.id)

    def _notify_finished(self, generation_id: str) -> None:
        listener = self._finished_listener
        if listener is None:
            return
        try:
            listener(generation_id)
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)

    def _mark_cancelled(
        self,
        claimed: GenerationRecord,
        partial_paths: list[str],
    ) -> None:
        self._cas_or_unlink(
            claimed,
            partial_paths,
            lambda: self._db.mark_cancelled(
                claimed.id,
                partial_paths,
                attempt_count=claimed.attempt_count,
            ),
        )

    def _cas_or_unlink(
        self,
        claimed: GenerationRecord,
        paths: list[str],
        write: Callable[[], object],
    ) -> bool:
        try:
            completed = write()
        except (StatusError, AttemptError):
            self._finalize_cancellation_or_unlink(claimed, paths)
            return False
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)
            self._unlink_unless_succeeded(claimed.id, paths)
            return False
        if isinstance(completed, GenerationRecord):
            self._send_generation_ended(completed)
        return True

    def _finalize_cancellation_or_unlink(
        self,
        claimed: GenerationRecord,
        paths: list[str],
    ) -> None:
        try:
            completed = self._db.mark_cancelled(
                claimed.id,
                paths,
                attempt_count=claimed.attempt_count,
            )
        except (StatusError, AttemptError):
            self._unlink_unless_succeeded(claimed.id, paths)
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)
            self._unlink_unless_succeeded(claimed.id, paths)
        else:
            self._send_generation_ended(completed)

    def _unlink_unless_succeeded(self, generation_id: str, paths: list[str]) -> None:
        try:
            generation = self._db.get_generation(generation_id)
        except Exception as exc:
            logger.warning(
                "Could not inspect generation %s after a lost finalization; "
                "preserving allocated paths: %s",
                generation_id,
                exc,
            )
            return
        if generation is None or generation.status != "succeeded":
            self._db.unlink_partial_outputs(paths)

    def _enhancement(self, generation: GenerationRecord) -> QueuedEnhancement | None:
        if self._enhancement_policy is None:
            return None
        return self._enhancement_policy(generation)

    def _snapshot_hardware(self) -> QueuedHardwareSnapshot | None:
        """Best-effort hardware snapshot for the started event.

        Per-probe tolerance: each field degrades to None independently, and
        anything unexpected degrades the whole snapshot. Telemetry never
        breaks generation.
        """
        if self._gpu_info is None or not self._lane.uses_gpu:
            return None
        try:
            return QueuedHardwareSnapshot(
                gpu_model=_probe_or_none(self._gpu_info.get_device_name),
                total_vram_bytes=_probe_or_none(self._gpu_info.get_cuda_total_vram_bytes),
                free_vram_bytes=_probe_or_none(self._gpu_info.get_free_cuda_vram_bytes),
                drive_type=self._checkpoint_drive_type,
            )
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)
            return None

    def _send_generation_started(self, generation: GenerationRecord) -> None:
        if self._analytics is None:
            return
        try:
            details = queued_generation_details(
                generation,
                enhancement=self._enhancement(generation),
                hardware=self._snapshot_hardware(),
            )
            if details is not None:
                self._analytics.send_generation_started(details)
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)

    def _send_generation_ended(
        self, generation: GenerationRecord, summary: ExecutionOutcome | None = None
    ) -> None:
        if (
            self._analytics is None
            or generation.started_at is None
            or generation.finished_at is None
        ):
            return
        try:
            modes: QueuedExecutionModes | None = (
                {
                    "load_mode": summary.load_mode,
                    "offload_mode": summary.offload_mode,
                    "prompt_rewrite": summary.prompt_rewrite,
                }
                if summary is not None
                else None
            )
            details = queued_generation_details(
                generation, enhancement=self._enhancement(generation), modes=modes
            )
            if details is None:
                return
            outcome: Literal["succeeded", "failed", "cancelled"]
            if generation.status == "succeeded":
                outcome = "succeeded"
            elif generation.status == "failed":
                outcome = "failed"
            elif generation.status == "cancelled":
                outcome = "cancelled"
            else:
                return
            self._analytics.send_generation_ended(
                details,
                outcome=outcome,
                runtime_ms=generation.finished_at - generation.started_at,
                error_code=generation.error_code,
            )
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)

    def _clear_active(self, generation: GenerationRecord) -> None:
        with self._lifecycle_lock:
            self._active.pop((generation.id, generation.attempt_count), None)
