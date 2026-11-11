"""Generation lifecycle handler."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Condition, Event, RLock
from typing import TYPE_CHECKING, Literal

from _routes._errors import HTTPError
from api_types import (
    CancelCancellingResponse,
    CancelNoActiveGenerationResponse,
    CancelResponse,
    GenerationProgressResponse,
    WallTimeEstimateResponse,
)
from handlers.base import StateHandlerBase, with_state_lock
from handlers.calibration_handler import CalibrationHandler
from runtime_config.device_calibration import JobEnvelope, RunMemoryBaseline, RunMemoryEnd
from runtime_config.eta import (
    WallTimeEstimate,
    default_machine_factor,
    estimate_envelope_wall_time,
    remaining_wall_time_s,
)
from services import generation_interrupt
from services.generation_interrupt import GenerationCancelledError
from services.generation_queue.control import GenerationSlotWaitAborted
from services.patches import diffusion_stage_cache, mps_conv3d_as_conv2d
from state.app_settings import resolved_use_conv_vae
from state.app_state_types import (
    ApiGeneration,
    AppState,
    GenerationCancelled,
    GenerationComplete,
    GenerationError,
    GenerationProgress,
    GenerationRunning,
    GenerationState,
    GpuGeneration,
)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)
GenerationSlot = Literal["gpu", "api"]
# Generous vs. any realistic pipeline load, but bounds how long a reservation can block future
# generations if some path raises before ever reaching start_generation()/fail_generation().
_RESERVATION_TIMEOUT_S = 180
# shutdown is an Event, not the Condition; waiters must re-check it after a bounded wait.
_SLOT_WAIT_POLL_S = 0.05


class GenerationHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
        *,
        calibration: CalibrationHandler | None = None,
    ) -> None:
        super().__init__(state, lock, config)
        self._calibration = calibration
        self._generation_slot = Condition(self._lock)
        self._queue_yields = 0
        from services.denoising_progress import configure_sink

        configure_sink(self._on_denoise_step)

    def _on_denoise_step(self, current: int, total: int) -> None:
        from services.denoising_progress import inference_percent

        self.update_progress("inference", inference_percent(current, total), current, total)

    def _generation_start_block_reason(self) -> str | None:
        """Return why the shared slot cannot be claimed, or None if it is free.

        Caller must hold the state lock. Same rules as try_reserve_generation_start().
        """
        since = self.state.generation_starting_since
        if self.is_generation_running():
            return "a generation is already running"
        if self.state.generation_in_flight:
            # After start_generation(), starting_since is None and in_flight stays True through
            # generate() unwind — do not expire that. Only a pre-start reservation whose
            # timestamp aged out (no finally) is reclaimable.
            if since is None or time.monotonic() - since < _RESERVATION_TIMEOUT_S:
                return "a generation is still in flight"
            return None
        if since is not None and time.monotonic() - since < _RESERVATION_TIMEOUT_S:
            return "another reservation is still active"
        return None

    @with_state_lock
    def try_reserve_generation_start(self, *, persistent: bool = False) -> bool:
        """Atomically claim the right to start a generation, before any slow pre-work.

        Shared across every generation kind (video/image/retake/extend/ic-lora all hold a
        reference to this same GenerationHandler instance) — one global gate, not one per
        endpoint. reserved_generation_start() holds generation_in_flight until the handler
        actually returns (including GPU unwind after Stop). start_generation() only clears
        generation_starting_since so progress can move off phase=starting. The timeout below
        is a backstop for a reservation that never got a context finally (legacy try_reserve
        without the context manager).
        """
        reason = self._generation_start_block_reason()
        if reason is not None:
            logger.info("Generation start reservation denied: %s", reason)
            return False
        self.state.generation_in_flight = True
        self.state.generation_starting_since = None if persistent else time.monotonic()
        return True

    @with_state_lock
    def release_generation_start_reservation(self) -> None:
        self.state.generation_in_flight = False
        self.state.generation_starting_since = None
        generation_interrupt.clear()
        self._generation_slot.notify_all()

    @contextmanager
    def wait_for_generation_slot(self, shutdown: Event) -> Iterator[None]:
        """Block until the shared generation reservation is free, then hold it.

        Unlike reserved_generation_start(), this waits instead of raising 409.
        shutdown aborts the wait without claiming the slot (the with-body is
        skipped) by raising GenerationSlotWaitAborted. Waiters are notified
        from release_generation_start_reservation().
        """
        reserved = False
        try:
            with self._generation_slot:
                while True:
                    if shutdown.is_set():
                        break
                    if (
                        self._queue_yields == 0
                        and self._generation_start_block_reason() is None
                        and self.try_reserve_generation_start()
                    ):
                        reserved = True
                        break
                    self._generation_slot.wait(timeout=_SLOT_WAIT_POLL_S)
            if not reserved:
                raise GenerationSlotWaitAborted()
            yield
        finally:
            if reserved:
                self.release_generation_start_reservation()

    @contextmanager
    def queue_yields(self) -> Iterator[None]:
        """Hold wait_for_generation_slot() waiters at the next job boundary.

        The running generation is not interrupted. A caller polling try_reserve_generation_start()
        inside this context cannot be starved by the queue re-claiming the slot between jobs.
        """
        with self._generation_slot:
            self._queue_yields += 1
        try:
            yield
        finally:
            with self._generation_slot:
                self._queue_yields -= 1
                self._generation_slot.notify_all()

    @contextmanager
    def try_lease_idle_gpu(self, *, local_lane_idle: Callable[[], bool]) -> Iterator[bool]:
        """Hold the GPU for a short job of another lane, without waiting. Yields True when held.

        The lease succeeds only when the GPU is free: no running job, no reservation, nobody
        waiting for it (assistant), and the local lane has no running or queued job. The last
        check keeps an outside job from taking the GPU between two local jobs and evicting the
        warm pipeline of the next one. The reservation does not expire while held. A granted
        lease starts with the diffusion stage cache evicted.
        """
        # The idle check reads SQLite, so it runs before the state lock is taken. The lock is
        # shared by every handler, and a slow read must not stall them (busy_timeout is 5 s).
        # Enqueue and claim do not take this lock, so holding it would not make the check exact.
        leased = False
        if local_lane_idle():
            with self._generation_slot:
                leased = self._queue_yields == 0 and self.try_reserve_generation_start(
                    persistent=True
                )
        try:
            if leased:
                # A local job cancelled before decode can leave its transformer cached. The
                # borrower loads its own model next, so free the cache first. The slot is held.
                diffusion_stage_cache.evict()
            yield leased
        finally:
            if leased:
                self.release_generation_start_reservation()

    @contextmanager
    def reserved_generation_start(self, *, persistent: bool = False) -> Iterator[None]:
        """Guarantees the reservation is released on ANY exit — a normal return, a validation
        HTTPError raised before start_generation() is ever reached, or an unexpected exception —
        not just the paths a handler remembers to route through fail_generation(). Wrap a
        handler's entire body in `with self._generation.reserved_generation_start():` right after
        the try_reserve_generation_start() gate would otherwise be checked by hand.

        Not itself @with_state_lock'd — it only sequences two independently-atomic locked calls
        (locking around the whole thing, including the caller's yielded body, would serialize
        unrelated state reads/writes for the entire generation instead of just these two).
        """
        if not self.try_reserve_generation_start(persistent=persistent):
            raise HTTPError(409, "Generation already in progress")
        try:
            yield
        finally:
            self.release_generation_start_reservation()

    @with_state_lock
    def start_generation(
        self,
        generation_id: str,
        *,
        estimated: WallTimeEstimate | None = None,
        ran: JobEnvelope | None = None,
    ) -> None:
        if self.is_generation_running():
            raise RuntimeError("Generation already in progress")
        if self.state.gpu_slot is None:
            raise RuntimeError("No active GPU pipeline")
        if generation_interrupt.is_requested():
            # Stop during reservation (enhance / pipeline load). Do not clear the Event
            # or launch Running — that would let this call proceed and a second Start
            # overlap on the GPU.
            self.state.generation_starting_since = None
            self.state.active_generation = GpuGeneration(
                state=GenerationCancelled(id=generation_id)
            )
            raise GenerationCancelledError()
        starting_since = self.state.generation_starting_since
        self.state.generation_starting_since = None

        # EXPERIMENTAL: push the live Settings toggle, then drop any transformer
        # cached from the previous generation before this one starts -- otherwise
        # it stays resident while this generation's own text encoder/VAE/etc.
        # build, double-booking VRAM. See that module's GENERATION-SCOPED
        # docstring section for the RTX 5090 repro (~42GB reported on a 32GB card).
        diffusion_stage_cache.set_enabled(self.state.app_settings.diffusion_stage_cache_enabled)
        # EXPERIMENTAL (Apple Silicon): same live-push pattern. No-op off Darwin. The Settings
        # toggle only shows under Fast decode, so the patch only runs while Fast decode is on.
        app_settings = self.state.app_settings
        conv_vae = resolved_use_conv_vae(app_settings, system=self.config.system)
        mps_conv3d_as_conv2d.set_enabled(app_settings.mps_conv3d_as_conv2d_enabled and conv_vae)
        diffusion_stage_cache.evict()
        generation_interrupt.clear()
        from services.denoising_progress import configure_sink

        configure_sink(self._on_denoise_step)

        resolved_estimated = estimated
        if resolved_estimated is None and ran is not None:
            calibrated = self.state.device_calibration.machine_factor
            factor = (
                calibrated
                if calibrated is not None
                else default_machine_factor(self.config.gpu_device_name)
            )
            system = (
                self.config.system
                if self.config.system in ("Windows", "Linux", "Darwin")
                else "Windows"
            )
            resolved_estimated = estimate_envelope_wall_time(
                ran, system=system, machine_factor=factor, conv_vae=conv_vae
            )

        self.state.active_generation = GpuGeneration(
            state=GenerationRunning(
                id=generation_id,
                progress=GenerationProgress(phase="", progress=0, current_step=0, total_steps=0),
                estimated=resolved_estimated,
                # After the evict above, so the baseline is the room this run starts with.
                memory_baseline=(
                    self._calibration.open_run() if self._calibration is not None else None
                ),
                started_monotonic=(
                    starting_since if starting_since is not None else time.monotonic()
                ),
                conv_vae=conv_vae,
            )
        )
        logger.info("Generation %s started (gpu)", generation_id)

    @with_state_lock
    def start_api_generation(self, generation_id: str) -> None:
        if self.is_generation_running():
            raise RuntimeError("Generation already in progress")
        if generation_interrupt.is_requested():
            self.state.generation_starting_since = None
            self.state.active_generation = ApiGeneration(
                state=GenerationCancelled(id=generation_id)
            )
            raise GenerationCancelledError()
        self.state.generation_starting_since = None

        # EXPERIMENTAL: see start_generation -- an API generation doesn't build a
        # local transformer itself, but evicting here still releases VRAM held by
        # a previous local generation's cached build.
        diffusion_stage_cache.evict()
        generation_interrupt.clear()

        self.state.active_generation = ApiGeneration(
            state=GenerationRunning(
                id=generation_id,
                progress=GenerationProgress(phase="", progress=0, current_step=None, total_steps=None),
            )
        )
        logger.info("Generation %s started (api)", generation_id)

    @with_state_lock
    def _gpu_generation(self) -> GenerationState | None:
        match self.state.active_generation:
            case GpuGeneration(state=generation) if self.state.gpu_slot is not None:
                return generation
            case _:
                return None

    @with_state_lock
    def _api_generation(self) -> GenerationState | None:
        match self.state.active_generation:
            case ApiGeneration(state=generation):
                return generation
            case _:
                return None

    @with_state_lock
    def _active_generation_state(self) -> tuple[GenerationSlot, GenerationState] | None:
        match self.state.active_generation:
            case GpuGeneration(state=generation) if self.state.gpu_slot is not None:
                return "gpu", generation
            case ApiGeneration(state=generation):
                return "api", generation
            case _:
                return None

    @with_state_lock
    def _running_slot(self) -> GenerationSlot | None:
        active = self._active_generation_state()
        if active is None:
            return None

        slot, generation = active
        match generation:
            case GenerationRunning():
                return slot
            case _:
                return None

    @with_state_lock
    def _running_generation(self) -> tuple[GenerationSlot, GenerationRunning] | None:
        active = self._active_generation_state()
        if active is None:
            return None

        slot, generation = active
        match generation:
            case GenerationRunning() as running:
                return slot, running
            case _:
                return None

    @with_state_lock
    def _cancelled_generation(self) -> tuple[GenerationSlot, GenerationCancelled] | None:
        active = self._active_generation_state()
        if active is None:
            return None

        slot, generation = active
        match generation:
            case GenerationCancelled() as cancelled:
                return slot, cancelled
            case _:
                return None

    @with_state_lock
    def _set_generation_state(self, slot: GenerationSlot, generation: GenerationState) -> None:
        if slot == "gpu":
            self.state.active_generation = GpuGeneration(state=generation)
            return
        self.state.active_generation = ApiGeneration(state=generation)

    @with_state_lock
    def _generation_for_polling(self) -> GenerationState | None:
        active = self._active_generation_state()
        return None if active is None else active[1]

    @with_state_lock
    def is_generation_cancelled(self) -> bool:
        match self._active_generation_state():
            case (_, GenerationCancelled()):
                return True
            case _:
                return False

    def raise_if_cancelled(self) -> None:
        """Abort if the user cancelled THIS in-flight job.

        AppState GenerationCancelled is sticky after the slot is released. Checking it
        here made the next Generate return cancelled immediately (the Windows repro after
        Stop). The interrupt Event is cleared on release; it is the signal for this
        reservation.
        """
        generation_interrupt.raise_if_requested()

    @with_state_lock
    def update_progress(
        self,
        phase: str,
        progress: int,
        current_step: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        running_generation = self._running_generation()
        if running_generation is None:
            return

        _, running = running_generation
        running.progress.phase = phase
        running.progress.progress = progress
        running.progress.current_step = current_step
        running.progress.total_steps = total_steps
        if (
            current_step is not None
            and total_steps is not None
            and total_steps > 1
            and current_step >= total_steps
        ):
            if running.decode_started_monotonic is None:
                running.decode_started_monotonic = time.monotonic()
        else:
            running.decode_started_monotonic = None

    @with_state_lock
    def _close_run_watch(
        self, running: GenerationRunning
    ) -> tuple[RunMemoryBaseline, RunMemoryEnd] | None:
        if running.memory_baseline is None or self._calibration is None:
            return None
        return running.memory_baseline, self._calibration.close_run()

    @with_state_lock
    def cancel_generation(self) -> CancelResponse:
        running_generation = self._running_generation()
        if running_generation is not None:
            generation_interrupt.request()
            slot, running = running_generation
            self._close_run_watch(running)
            self._set_generation_state(slot, GenerationCancelled(id=running.id))
            return CancelCancellingResponse(status="cancelling", id=running.id)

        if self.state.generation_in_flight:
            generation_interrupt.request()
            # After start_generation(), starting_since is None and active_generation is this
            # job. During reservation it still holds the previous job's sticky terminal id.
            if self.state.generation_starting_since is None:
                cancelled_in_flight = self._cancelled_generation()
                if cancelled_in_flight is not None:
                    return CancelCancellingResponse(status="cancelling", id=cancelled_in_flight[1].id)
            return CancelCancellingResponse(status="cancelling", id="pending")

        # Sticky GenerationCancelled after the slot is released is idle, not in-flight.
        # Returning "cancelling" here used to make a second Stop look accepted without
        # arming the Event.
        return CancelNoActiveGenerationResponse(status="no_active_generation")

    def complete_generation(
        self, result: str | list[str] | None = None, *, ran: JobEnvelope | None = None
    ) -> None:
        """``ran`` is the admitted shape at the frames that actually ran; a clean run records it.

        A cancelled run is no longer running here, so it records nothing.
        """
        finished = self._finish_running(result)
        if finished is None or ran is None or self._calibration is None:
            return
        generation_id, measured, wall_s, conv_vae = finished
        if measured is not None:
            baseline, end = measured
            self._calibration.record_if_clean(
                ran,
                baseline,
                end,
                generation_id=generation_id,
                wall_s=wall_s,
                conv_vae=conv_vae,
            )

    @with_state_lock
    def _finish_running(
        self, result: str | list[str] | None
    ) -> tuple[str, tuple[RunMemoryBaseline, RunMemoryEnd] | None, float | None, bool] | None:
        running_generation = self._running_generation()
        if running_generation is None:
            return None

        slot, running = running_generation
        # Sampled before the slot frees, since the next run's watch resets this run's peak.
        measured = self._close_run_watch(running)
        wall_s = (
            max(0.001, time.monotonic() - running.started_monotonic)
            if running.started_monotonic is not None
            else None
        )
        self._set_generation_state(slot, GenerationComplete(id=running.id, result=result))
        logger.info("Generation %s complete (%s)", running.id, slot)
        return running.id, measured, wall_s, running.conv_vae

    @with_state_lock
    def fail_generation(self, error: str) -> None:
        # Covers the reservation made by try_reserve_generation_start() even when this failure
        # happened before start_generation()/start_api_generation() ever ran (e.g. pipeline load
        # itself threw) — that path never touches generation_starting_since otherwise.
        self.state.generation_starting_since = None
        running_generation = self._running_generation()
        if running_generation is not None:
            slot, running = running_generation
            self._close_run_watch(running)
            logger.error("Generation %s failed: %s", running.id, error)
            self._set_generation_state(slot, GenerationError(id=running.id, error=error))
            return

        if self._cancelled_generation() is not None:
            return

        logger.error("Generation failed without active running job: %s", error)

    @with_state_lock
    def _local_gpu_slot_occupied(self) -> bool:
        match self.state.active_generation:
            case GpuGeneration() if self.state.gpu_slot is not None:
                return True
            case _:
                return False

    @with_state_lock
    def get_generation_progress(self) -> GenerationProgressResponse:
        # Checked before matching active_generation: try_reserve_generation_start() only succeeds
        # when is_generation_running() is false, so a live reservation never coexists with an
        # actually-running generation — but it does coexist with the PREVIOUS generation's sticky
        # Complete/Error/Cancelled state, which is otherwise still sitting in active_generation
        # until start_generation() overwrites it. Matching on gen first would report that stale
        # terminal state instead of "starting" for every generation after the first.
        if self.state.generation_starting_since is not None:
            # Reserved (try_reserve_generation_start succeeded) but pipeline load hasn't
            # finished, so start_generation() hasn't run and there's no real id yet.
            # Still report "running": a client polling for "is anything busy right now"
            # (the cross-project Generate-disable lock, or a recovery marker's own
            # baseline capture) must not see "idle" during this window and wrongly
            # conclude the single global slot is free — see try_reserve_generation_start's
            # docstring for why this window needed closing on the write side too.
            # cancellable=True: Stop during reservation sets the interrupt Event before
            # start_generation()/start_api_generation(); sticky previous slot is not this job.
            return GenerationProgressResponse(
                status="running",
                phase="starting",
                progress=0,
                currentStep=0,
                totalSteps=0,
                cancellable=True,
            )

        gen = self._generation_for_polling()
        gpu_cancellable = self._local_gpu_slot_occupied()

        match gen:
            case GenerationRunning(
                id=generation_id, progress=progress, estimated=estimated
            ) as running:
                now = time.monotonic()
                elapsed_s = (
                    max(0.0, now - running.started_monotonic)
                    if running.started_monotonic is not None
                    else 0.0
                )
                decode_elapsed_s = (
                    max(0.0, now - running.decode_started_monotonic)
                    if running.decode_started_monotonic is not None
                    else None
                )
                return GenerationProgressResponse(
                    status="running",
                    phase=progress.phase,
                    progress=progress.progress,
                    currentStep=progress.current_step,
                    totalSteps=progress.total_steps,
                    id=generation_id,
                    cancellable=gpu_cancellable,
                    estimate=WallTimeEstimateResponse(
                        seconds=estimated.seconds,
                        basis=estimated.basis,
                        t2vReuse=estimated.t2v_reuse,
                    )
                    if estimated
                    else None,
                    remainingSeconds=remaining_wall_time_s(
                        estimated,
                        elapsed_s=elapsed_s,
                        current_step=progress.current_step,
                        total_steps=progress.total_steps,
                        decode_elapsed_s=decode_elapsed_s,
                    ),
                )
            case GenerationComplete(id=generation_id, result=result):
                return GenerationProgressResponse(
                    status="complete",
                    phase="complete",
                    progress=100,
                    currentStep=0,
                    totalSteps=0,
                    result=result,
                    id=generation_id,
                    cancellable=False,
                )
            case GenerationCancelled(id=generation_id) if self.state.generation_in_flight:
                # Slot is still occupied until pipeline.generate() unwinds. Report running so
                # the cross-project Generate lock does not treat Stop as "slot is free".
                return GenerationProgressResponse(
                    status="running",
                    phase="cancelled",
                    progress=0,
                    currentStep=0,
                    totalSteps=0,
                    id=generation_id,
                    cancellable=gpu_cancellable,
                )
            case GenerationCancelled(id=generation_id):
                return GenerationProgressResponse(
                    status="cancelled",
                    phase="cancelled",
                    progress=0,
                    currentStep=0,
                    totalSteps=0,
                    id=generation_id,
                    cancellable=False,
                )
            case GenerationError(id=generation_id):
                return GenerationProgressResponse(
                    status="error",
                    phase="error",
                    progress=0,
                    currentStep=0,
                    totalSteps=0,
                    id=generation_id,
                    cancellable=False,
                )
            case _:
                return GenerationProgressResponse(
                    status="idle",
                    phase="",
                    progress=0,
                    currentStep=0,
                    totalSteps=0,
                    cancellable=False,
                )

    @with_state_lock
    def is_generation_running(self) -> bool:
        return self._running_slot() is not None
