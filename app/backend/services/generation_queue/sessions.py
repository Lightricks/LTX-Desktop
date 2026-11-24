"""Per-job view of a running queued generation.

An executor reports progress, checks for cancellation and records the end state
through its JobSession. It never reaches for process-wide state. The local lane
maps a session onto GenerationHandler (the shared GPU slot, one global job). A
lane that runs beside it keeps its own state per job (see lane_sessions.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from api_types import ApiActivityStatus, GenerationProgressResponse
from services import generation_interrupt
from services.generation_queue.control import SharedGenerationSlot


@dataclass(frozen=True)
class JobProgress:
    phase: str
    progress: int
    current_step: int | None
    total_steps: int | None
    remaining_seconds: int | None = None


class JobSession(Protocol):
    @property
    def generation_id(self) -> str: ...

    @property
    def holds_gpu_slot(self) -> bool:
        """True when this job holds the shared GPU slot, so it can use the GPU without a lease."""
        ...

    def start_api(self) -> None:
        """Mark the job as running on a provider API (no GPU)."""
        ...

    def set_progress(
        self,
        phase: str,
        progress: int,
        current_step: int | None = None,
        total_steps: int | None = None,
    ) -> None: ...

    def set_api_status(self, status: ApiActivityStatus) -> None: ...

    def set_provider_job_id(self, provider_job_id: str) -> bool:
        """Record the id of this job at its provider, right after the provider accepts it.

        The queue stores it, so a job left running at app stop can be cancelled at boot.
        Never raises for a store failure. Returns False when a store failed to keep the id. Boot
        cannot then cancel the job, so the executor must cancel it at the provider itself. A
        session with no store has nothing to keep and returns True: the job goes on.
        """
        ...

    def progress(self) -> JobProgress | None:
        """The live progress of this job, for the queue panel. None before the first report."""
        ...

    def api_status(self) -> ApiActivityStatus | None:
        """What the provider reports for this job. None for a job that has no provider."""
        ...

    def raise_if_cancelled(self) -> None: ...

    def wait_cancelled(self, timeout: float) -> bool:
        """Sleep up to ``timeout`` seconds. Return True as soon as the job is cancelled."""
        ...

    def cancel(self) -> None:
        """Ask this job to stop. Called by the queue, not by the executor.

        It must return at once and do no network I/O: the queue calls it while it holds the
        lane lock. A provider is asked to drop the job after the executor returns, with
        LaneSpec.cancel_orphan.
        """
        ...

    def complete(self, result: str | list[str] | None) -> None: ...

    def fail(self, error: str) -> None: ...


class LocalGenerationOps(Protocol):
    """The GenerationHandler methods a local-lane session calls."""

    def start_api_generation(self, generation_id: str) -> None: ...

    def get_generation_progress(self) -> GenerationProgressResponse: ...

    def update_progress(
        self,
        phase: str,
        progress: int,
        current_step: int | None = None,
        total_steps: int | None = None,
    ) -> None: ...

    def raise_if_cancelled(self) -> None: ...

    def is_generation_cancelled(self) -> bool: ...

    def cancel_generation(self) -> object: ...

    def complete_generation(self, result: str | list[str] | None = None) -> None: ...

    def fail_generation(self, error: str) -> None: ...


class LocalJobSession:
    """A session on the shared generation slot: GenerationHandler holds the one global job."""

    def __init__(self, generation: LocalGenerationOps, generation_id: str) -> None:
        self._generation = generation
        self._generation_id = generation_id

    @property
    def generation_id(self) -> str:
        return self._generation_id

    @property
    def holds_gpu_slot(self) -> bool:
        return True

    def start_api(self) -> None:
        self._generation.start_api_generation(self._generation_id)

    def set_progress(
        self,
        phase: str,
        progress: int,
        current_step: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        self._generation.update_progress(phase, progress, current_step, total_steps)

    def set_api_status(self, status: ApiActivityStatus) -> None:
        del status

    def set_provider_job_id(self, provider_job_id: str) -> bool:
        # The local lane keeps no provider ids and has no boot cancel. A fal job on this lane
        # (the kill switch) loses only that recovery: the job itself goes on.
        del provider_job_id
        return True

    def progress(self) -> JobProgress | None:
        # The global slot holds one job. Report its progress only while it is this one.
        state = self._generation.get_generation_progress()
        if state.id != self._generation_id:
            return None
        return JobProgress(
            phase=state.phase,
            progress=state.progress,
            current_step=state.currentStep,
            total_steps=state.totalSteps,
            remaining_seconds=state.remainingSeconds,
        )

    def api_status(self) -> ApiActivityStatus | None:
        return None

    def raise_if_cancelled(self) -> None:
        self._generation.raise_if_cancelled()

    def wait_cancelled(self, timeout: float) -> bool:
        # The interrupt Event is the cancel signal of the running job (see raise_if_cancelled).
        # The generation state is not: it stays "cancelled" after the slot is released.
        return generation_interrupt.wait_requested(timeout)

    def cancel(self) -> None:
        self._generation.cancel_generation()

    def complete(self, result: str | list[str] | None) -> None:
        self._generation.complete_generation(result)

    def fail(self, error: str) -> None:
        self._generation.fail_generation(error)


class SlotCancelSession:
    """The session of a lane whose only control is its slot: cancel goes to the slot."""

    def __init__(self, slot: SharedGenerationSlot, generation_id: str) -> None:
        self._slot = slot
        self._generation_id = generation_id

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
        del phase, progress, current_step, total_steps

    def set_api_status(self, status: ApiActivityStatus) -> None:
        del status

    def set_provider_job_id(self, provider_job_id: str) -> bool:
        del provider_job_id
        return False

    def progress(self) -> JobProgress | None:
        return None

    def api_status(self) -> ApiActivityStatus | None:
        return None

    def raise_if_cancelled(self) -> None:
        return None

    def wait_cancelled(self, timeout: float) -> bool:
        time.sleep(timeout)
        return False

    def cancel(self) -> None:
        self._slot.cancel_generation()

    def complete(self, result: str | list[str] | None) -> None:
        del result

    def fail(self, error: str) -> None:
        del error
