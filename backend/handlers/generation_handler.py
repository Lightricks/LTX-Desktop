"""Generation lifecycle handler."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock
from typing import TYPE_CHECKING, Literal

from _routes._errors import HTTPError
from api_types import (
    CancelCancellingResponse,
    CancelNoActiveGenerationResponse,
    CancelResponse,
    GenerationProgressResponse,
)
from handlers.base import StateHandlerBase, with_state_lock
from services.fast_video_pipeline.mlx_fast_video_pipeline import cancel_active_mlx_sidecar
from services.patches import diffusion_stage_cache
from services.local_metal_lease import LocalMetalLeaseHandle, local_metal_lease
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


class GenerationHandler(StateHandlerBase):
    def __init__(self, state: AppState, lock: RLock, config: RuntimeConfig) -> None:
        super().__init__(state, lock, config)

    @with_state_lock
    def try_reserve_generation_start(self) -> bool:
        """Atomically claim the right to start a generation, before any slow pre-work.

        Shared across every generation kind (video/image/retake/extend/ic-lora all hold a
        reference to this same GenerationHandler instance) — one global gate, not one per
        endpoint. fail_generation() and start_generation()/start_api_generation() clear this on
        every success/failure path they're reached from; the timeout below is a backstop for any
        path that raises before reaching either (e.g. a request-validation check that runs after
        the reservation but outside the try/except that calls fail_generation).
        """
        since = self.state.generation_starting_since
        if self.is_generation_running():
            logger.info("Generation start reservation denied: a generation is already running")
            return False
        if since is not None and time.monotonic() - since < _RESERVATION_TIMEOUT_S:
            logger.info("Generation start reservation denied: another reservation is still active")
            return False
        self.state.generation_starting_since = time.monotonic()
        self.state.generation_starting_id = None
        self.state.generation_starting_phase = "starting"
        self.state.generation_start_cancelled = False
        return True

    @with_state_lock
    def release_generation_start_reservation(self) -> None:
        self.state.generation_starting_since = None
        self.state.generation_starting_id = None
        self.state.generation_starting_phase = "starting"
        self.state.generation_start_cancelled = False

    @with_state_lock
    def annotate_generation_start(self, generation_id: str, phase: str) -> None:
        if self.state.generation_starting_since is None:
            return
        self.state.generation_starting_id = generation_id
        self.state.generation_starting_phase = phase

    @with_state_lock
    def update_generation_start_phase(self, phase: str) -> None:
        if self.state.generation_starting_since is not None:
            self.state.generation_starting_phase = phase

    @contextmanager
    def reserved_generation_start(self) -> Iterator[None]:
        """Guarantees the reservation is released on ANY exit — a normal return, a validation
        HTTPError raised before start_generation() is ever reached, or an unexpected exception —
        not just the paths a handler remembers to route through fail_generation(). Wrap a
        handler's entire body in `with self._generation.reserved_generation_start():` right after
        the try_reserve_generation_start() gate would otherwise be checked by hand.

        Not itself @with_state_lock'd — it only sequences two independently-atomic locked calls
        (locking around the whole thing, including the caller's yielded body, would serialize
        unrelated state reads/writes for the entire generation instead of just these two).
        """
        if not self.try_reserve_generation_start():
            raise HTTPError(409, "Generation already in progress")
        try:
            yield
        finally:
            self.release_generation_start_reservation()

    @contextmanager
    def hold_local_metal_lease(
        self,
        *,
        generation_id: str,
        workload: str,
        reason: str,
    ) -> Iterator[None]:
        """Hold the cross-product Metal lease before any local model load."""
        self.annotate_generation_start(generation_id, "waiting_for_local_accelerator")

        def _on_wait(waited: float, owner: dict[str, object] | None) -> None:
            owner_product = owner.get("product") if owner else "another LTX app"
            owner_pid = owner.get("pid") if owner else "?"
            self.update_generation_start_phase(
                f"waiting_for_local_accelerator:{owner_product}:pid={owner_pid}:"
                f"{waited:.1f}s"
            )

        with local_metal_lease(
            job_id=generation_id,
            workload=workload,
            reason=reason,
            is_cancelled=self.is_generation_cancelled,
            on_wait=_on_wait,
        ):
            self.update_generation_start_phase("loading_model")
            yield

    def acquire_local_metal_lease(
        self,
        *,
        generation_id: str,
        workload: str,
        reason: str,
    ) -> LocalMetalLeaseHandle:
        """Acquire an explicit lease handle; caller closes it after GPU cleanup."""
        self.annotate_generation_start(generation_id, "waiting_for_local_accelerator")

        def _on_wait(waited: float, owner: dict[str, object] | None) -> None:
            owner_product = owner.get("product") if owner else "another LTX app"
            owner_pid = owner.get("pid") if owner else "?"
            self.update_generation_start_phase(
                f"waiting_for_local_accelerator:{owner_product}:pid={owner_pid}:"
                f"{waited:.1f}s"
            )

        handle = LocalMetalLeaseHandle(
            local_metal_lease(
                job_id=generation_id,
                workload=workload,
                reason=reason,
                is_cancelled=self.is_generation_cancelled,
                on_wait=_on_wait,
            )
        )
        self.update_generation_start_phase("loading_model")
        return handle

    @with_state_lock
    def start_generation(self, generation_id: str) -> None:
        if self.is_generation_running():
            raise RuntimeError("Generation already in progress")
        if self.state.generation_start_cancelled:
            raise RuntimeError("Generation was cancelled before model loading completed")
        if self.state.gpu_slot is None:
            raise RuntimeError("No active GPU pipeline")
        self.state.generation_starting_since = None
        self.state.generation_starting_id = None
        self.state.generation_starting_phase = "starting"
        self.state.generation_start_cancelled = False

        # Push the live Settings toggle, then drop any transformer left by a
        # cancelled/failed generation before this one starts. A normal two-stage
        # generation retires the cache immediately after the reuse hit, before VAE
        # decode; this is the defensive backstop. See diffusion_stage_cache.py.
        diffusion_stage_cache.set_enabled(self.state.app_settings.diffusion_stage_cache_enabled)
        diffusion_stage_cache.evict()

        self.state.active_generation = GpuGeneration(
            state=GenerationRunning(
                id=generation_id,
                progress=GenerationProgress(phase="", progress=0, current_step=0, total_steps=0),
            )
        )
        logger.info("Generation %s started (gpu)", generation_id)

    @with_state_lock
    def start_api_generation(self, generation_id: str) -> None:
        if self.is_generation_running():
            raise RuntimeError("Generation already in progress")
        self.state.generation_starting_since = None

        # EXPERIMENTAL: see start_generation -- an API generation doesn't build a
        # local transformer itself, but evicting here still releases VRAM held by
        # a previous local generation's cached build.
        diffusion_stage_cache.evict()

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
        if self.state.generation_start_cancelled:
            return True
        match self._active_generation_state():
            case (_, GenerationCancelled()):
                return True
            case _:
                return False

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

    @with_state_lock
    def cancel_generation(self) -> CancelResponse:
        if self.state.generation_starting_since is not None:
            self.state.generation_start_cancelled = True
            generation_id = self.state.generation_starting_id or "starting"
            return CancelCancellingResponse(status="cancelling", id=generation_id)
        running_generation = self._running_generation()
        if running_generation is not None:
            slot, running = running_generation
            self._set_generation_state(slot, GenerationCancelled(id=running.id))
            # Torch pipelines observe the state cooperatively. MLX is isolated
            # in a fresh child process, so cancellation tears down its whole
            # process group and releases unified memory deterministically.
            cancel_active_mlx_sidecar()
            return CancelCancellingResponse(status="cancelling", id=running.id)

        cancelled_generation = self._cancelled_generation()
        match cancelled_generation:
            case (_, GenerationCancelled(id=generation_id)):
                return CancelCancellingResponse(status="cancelling", id=generation_id)
            case _:
                return CancelNoActiveGenerationResponse(status="no_active_generation")

    @with_state_lock
    def complete_generation(self, result: str | list[str] | None = None) -> None:
        running_generation = self._running_generation()
        if running_generation is None:
            return

        slot, running = running_generation
        self._set_generation_state(slot, GenerationComplete(id=running.id, result=result))
        logger.info("Generation %s complete (%s)", running.id, slot)

    @with_state_lock
    def fail_generation(self, error: str) -> None:
        # Covers the reservation made by try_reserve_generation_start() even when this failure
        # happened before start_generation()/start_api_generation() ever ran (e.g. pipeline load
        # itself threw) — that path never touches generation_starting_since otherwise.
        self.state.generation_starting_since = None
        self.state.generation_starting_id = None
        self.state.generation_starting_phase = "starting"
        self.state.generation_start_cancelled = False
        running_generation = self._running_generation()
        if running_generation is not None:
            slot, running = running_generation
            logger.error("Generation %s failed: %s", running.id, error)
            self._set_generation_state(slot, GenerationError(id=running.id, error=error))
            return

        if self._cancelled_generation() is not None:
            return

        logger.error("Generation failed without active running job: %s", error)

    @with_state_lock
    def get_generation_progress(self) -> GenerationProgressResponse:
        # Checked before matching active_generation: try_reserve_generation_start() only succeeds
        # when is_generation_running() is false, so a live reservation never coexists with an
        # actually-running generation — but it does coexist with the PREVIOUS generation's sticky
        # Complete/Error/Cancelled state, which is otherwise still sitting in active_generation
        # until start_generation() overwrites it. Matching on gen first would report that stale
        # terminal state instead of "starting" for every generation after the first.
        if self.state.generation_starting_since is not None:
            generation_id = self.state.generation_starting_id
            if self.state.generation_start_cancelled:
                return GenerationProgressResponse(
                    status="cancelled",
                    phase="cancelled",
                    progress=0,
                    currentStep=0,
                    totalSteps=0,
                    id=generation_id,
                )
            # Reserved (try_reserve_generation_start succeeded) but pipeline load hasn't
            # finished, so start_generation() hasn't run and there's no real id yet.
            # Still report "running": a client polling for "is anything busy right now"
            # (the cross-project Generate-disable lock, or a recovery marker's own
            # baseline capture) must not see "idle" during this window and wrongly
            # conclude the single global slot is free — see try_reserve_generation_start's
            # docstring for why this window needed closing on the write side too.
            return GenerationProgressResponse(
                status="running",
                phase=self.state.generation_starting_phase,
                progress=0,
                currentStep=0,
                totalSteps=0,
                id=generation_id,
            )

        gen = self._generation_for_polling()

        match gen:
            case GenerationRunning(id=generation_id, progress=progress):
                return GenerationProgressResponse(
                    status="running",
                    phase=progress.phase,
                    progress=progress.progress,
                    currentStep=progress.current_step,
                    totalSteps=progress.total_steps,
                    id=generation_id,
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
                )
            case GenerationCancelled(id=generation_id):
                return GenerationProgressResponse(
                    status="cancelled",
                    phase="cancelled",
                    progress=0,
                    currentStep=0,
                    totalSteps=0,
                    id=generation_id,
                )
            case GenerationError(id=generation_id):
                return GenerationProgressResponse(
                    status="error",
                    phase="error",
                    progress=0,
                    currentStep=0,
                    totalSteps=0,
                    id=generation_id,
                )
            case _:
                return GenerationProgressResponse(
                    status="idle",
                    phase="",
                    progress=0,
                    currentStep=0,
                    totalSteps=0,
                )

    @with_state_lock
    def is_generation_running(self) -> bool:
        return self._running_slot() is not None
