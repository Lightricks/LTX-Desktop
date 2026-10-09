from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from threading import Event
from typing import Protocol

from services.store import Store
from services.records import GenerationRecord


class QueueControl(Protocol):
    def wake(self) -> None: ...
    def interrupt(self, generation_id: str, attempt_count: int) -> None: ...


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


def cancel_queued_generation(
    db: Store,
    generation_id: str,
    interrupt: Callable[[str, int], None],
) -> GenerationRecord:
    generation = db.request_generation_cancellation(generation_id)
    if generation.status == "cancelling":
        interrupt(generation.id, generation.attempt_count)
    return generation
