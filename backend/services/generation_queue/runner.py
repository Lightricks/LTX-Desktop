from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from typing import Literal

from services.store import Store
from services.records import (
    AttemptError,
    CapabilityFailedError,
    GenerationErrorCode,
    GenerationRecord,
    MarkSucceededError,
    OutputSpec,
    StatusError,
    UnavailableError,
)
from logging_policy import log_background_exception
from services.analytics import AnalyticsService, QueuedEnhancement, queued_generation_details
from services.generation_queue.control import (
    GenerationSlotWaitAborted,
    SharedGenerationSlot,
)
from services.generation_queue.output_names import output_asset_name
from services.generation_queue.registry import ExecutorRegistry
from services.generation_queue.types import OutputAllocation
from services.generation_interrupt import is_cancel_exception

logger = logging.getLogger(__name__)
_BACKGROUND_TASK_NAME = "generation-queue-runner"
_EXECUTOR_FAILURE_CODE: GenerationErrorCode = "EXECUTOR_FAILED"
_CAPABILITY_FAILURE_CODE: GenerationErrorCode = "CAPABILITY_FAILED"
_LTX_INVALID_API_KEY_CODE: GenerationErrorCode = "LTX_INVALID_API_KEY"
_LTX_API_PROMPT_EMBEDDING_FAILED_CODE: GenerationErrorCode = "LTX_API_PROMPT_EMBEDDING_FAILED"


def _capability_error_code(exc: CapabilityFailedError) -> GenerationErrorCode:
    if exc.code == _LTX_INVALID_API_KEY_CODE:
        return _LTX_INVALID_API_KEY_CODE
    if exc.code == _LTX_API_PROMPT_EMBEDDING_FAILED_CODE:
        return _LTX_API_PROMPT_EMBEDDING_FAILED_CODE
    return _CAPABILITY_FAILURE_CODE


@dataclass
class _ActiveGeneration:
    generation_id: str
    attempt_count: int


class QueueRunner:
    def __init__(
        self,
        *,
        db: Store,
        generation: SharedGenerationSlot,
        executors: ExecutorRegistry,
        analytics: AnalyticsService | None = None,
        enhancement_policy: Callable[[GenerationRecord], QueuedEnhancement] | None = None,
        idle_wait_seconds: float = 0.25,
        stop_join_timeout_seconds: float = 600.0,
    ) -> None:
        self._db = db
        self._generation = generation
        self._executors = executors
        self._analytics = analytics
        self._enhancement_policy = enhancement_policy
        self._idle_wait_seconds = idle_wait_seconds
        self._stop_join_timeout_seconds = stop_join_timeout_seconds

        self._shutdown = Event()
        self._wake_event = Event()
        self._lifecycle_lock = Lock()
        self._thread: Thread | None = None
        self._active: _ActiveGeneration | None = None
        self._finished_listener: Callable[[str], None] | None = None

    def set_finished_listener(self, listener: Callable[[str], None] | None) -> None:
        self._finished_listener = listener

    def start(self) -> None:
        try:
            self._db.has_queued()
        except UnavailableError:
            return
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._shutdown.clear()
            self._wake_event.clear()
            self._thread = Thread(
                target=self._run,
                name="generation-queue-runner",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None:
                return
            self._shutdown.set()
            self._wake_event.set()

        if thread is not current_thread():
            thread.join(timeout=self._stop_join_timeout_seconds)

        with self._lifecycle_lock:
            if self._thread is thread and not thread.is_alive():
                self._thread = None

    def wake(self) -> None:
        self._wake_event.set()

    def interrupt(self, generation_id: str, attempt_count: int) -> None:
        with self._lifecycle_lock:
            active = self._active
            if (
                active is None
                or active.generation_id != generation_id
                or active.attempt_count != attempt_count
            ):
                return
            try:
                self._generation.cancel_generation()
            except Exception as exc:
                logger.warning(
                    "Generation %s attempt %s interrupt failed: %s",
                    generation_id,
                    attempt_count,
                    exc,
                )

    @property
    def is_running(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while True:
            if self._shutdown.is_set():
                return
            try:
                if not self._db.has_queued():
                    self._wake_event.wait(timeout=self._idle_wait_seconds)
                    self._wake_event.clear()
                    continue

                with self._generation.wait_for_generation_slot(self._shutdown):
                    claimed = self._claim_and_register_active()
                    if claimed is None:
                        if self._shutdown.is_set():
                            return
                        continue
                    try:
                        self._process(claimed)
                    finally:
                        self._clear_active(claimed)
            except GenerationSlotWaitAborted:
                return
            except Exception as exc:
                log_background_exception(_BACKGROUND_TASK_NAME, exc)
                if self._shutdown.wait(timeout=self._idle_wait_seconds):
                    return

    def _claim_and_register_active(self) -> GenerationRecord | None:
        if self._shutdown.is_set():
            return None
        claimed = self._db.claim_next_queued()
        if claimed is None:
            return None
        with self._lifecycle_lock:
            self._active = _ActiveGeneration(
                generation_id=claimed.id,
                attempt_count=claimed.attempt_count,
            )
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
        return claimed

    def _process(self, claimed: GenerationRecord) -> None:
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
            executor.execute(claimed, tuple(allocations))
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
                self._mark_failed(claimed, _capability_error_code(exc), partial_paths)
            else:
                log_background_exception(_BACKGROUND_TASK_NAME, exc)
                self._mark_failed(claimed, _EXECUTOR_FAILURE_CODE, partial_paths)
            return

        self._mark_succeeded(claimed, allocations)

    def _mark_succeeded(
        self,
        claimed: GenerationRecord,
        allocations: list[OutputAllocation],
    ) -> None:
        params = claimed.spec.get("params")
        prompt: object = None
        if isinstance(params, dict):
            prompt = params.get("prompt")
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
            self._send_generation_ended(completed)

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

    def _send_generation_started(self, generation: GenerationRecord) -> None:
        if self._analytics is None:
            return
        try:
            details = queued_generation_details(
                generation, enhancement=self._enhancement(generation)
            )
            if details is not None:
                self._analytics.send_generation_started(details)
        except Exception as exc:
            log_background_exception(_BACKGROUND_TASK_NAME, exc)

    def _send_generation_ended(self, generation: GenerationRecord) -> None:
        if (
            self._analytics is None
            or generation.started_at is None
            or generation.finished_at is None
        ):
            return
        try:
            details = queued_generation_details(
                generation, enhancement=self._enhancement(generation)
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
            if (
                self._active is not None
                and self._active.generation_id == generation.id
                and self._active.attempt_count == generation.attempt_count
            ):
                self._active = None
