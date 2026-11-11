"""Capacity slot for a lane that does not hold the GPU."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Condition, Event

from services.generation_queue.control import GenerationSlotWaitAborted

# shutdown is an Event, not the Condition; waiters must re-check it after a bounded wait.
_WAIT_POLL_SECONDS = 0.05


class CountingLaneSlot:
    """Up to ``capacity`` jobs hold the slot at once. No global state, no GPU."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("a lane slot needs a capacity of at least 1")
        self._capacity = capacity
        self._held = 0
        self._condition = Condition()

    @contextmanager
    def wait_for_generation_slot(self, shutdown: Event) -> Iterator[None]:
        with self._condition:
            while self._held >= self._capacity:
                if shutdown.is_set():
                    raise GenerationSlotWaitAborted()
                self._condition.wait(timeout=_WAIT_POLL_SECONDS)
            if shutdown.is_set():
                raise GenerationSlotWaitAborted()
            self._held += 1
        try:
            yield
        finally:
            with self._condition:
                self._held -= 1
                self._condition.notify()

    def cancel_generation(self) -> object:
        """A lane job is stopped through its own session. The slot has nothing to stop."""
        return None
