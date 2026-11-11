from __future__ import annotations

from collections.abc import Callable, Collection
from contextlib import AbstractContextManager
from threading import Event
from dataclasses import dataclass
from typing import Protocol

from api_types import ApiActivityStatus, QueueProgress
from services.store import Store
from services.records import GenerationRecord


@dataclass(frozen=True)
class QueueLiveState:
    """What the queue knows about a running job that the database does not."""

    progress: QueueProgress | None
    api_status: ApiActivityStatus | None
    # A cancel stops the job. False for a provider job that is billed once it runs.
    cancellable: bool


class QueueControl(Protocol):
    def wake(self) -> None: ...
    def interrupt(self, generation_id: str, attempt_count: int) -> None: ...
    def live_state(self, generation_id: str) -> QueueLiveState | None: ...


class SharedGenerationSlot(Protocol):
    """In-memory GPU/API slot used by the queue runner.

    Implemented by GenerationHandler. Named separately from that class's
    GenerationSlot = Literal["gpu", "api"] alias.
    """

    def wait_for_generation_slot(
        self, shutdown: Event
    ) -> AbstractContextManager[None]: ...
    def cancel_generation(self) -> object: ...


class GenerationSlotWaitAborted(RuntimeError):
    """Expected shutdown while waiting for the shared generation slot."""

    def __init__(self, message: str = "generation slot wait aborted by shutdown") -> None:
        super().__init__(message)


class NoOpQueueControl:
    def wake(self) -> None:
        return None

    def interrupt(self, generation_id: str, attempt_count: int) -> None:
        del generation_id, attempt_count

    def live_state(self, generation_id: str) -> QueueLiveState | None:
        del generation_id
        return None


def cancel_queued_generation(
    db: Store,
    generation_id: str,
    interrupt: Callable[[str, int], None],
    *,
    uncancellable_lanes: Collection[str] = (),
) -> GenerationRecord:
    generation = db.request_generation_cancellation(
        generation_id, uncancellable_lanes=uncancellable_lanes
    )
    if generation.status == "cancelling":
        interrupt(generation.id, generation.attempt_count)
    return generation
