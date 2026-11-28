"""One facade over the per-lane runners.

The rest of the app sees a single queue: start, stop, wake, interrupt. Each lane
has its own runner, so a job in one lane never waits for a job in another.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence

from api_types import QueueProgress
from services.generation_queue.control import QueueLiveState
from services.generation_queue.runner import QueueRunner
from services.storage_info.storage_info import DriveType

logger = logging.getLogger(__name__)


class QueueDispatcher:
    def __init__(
        self,
        runners: Sequence[QueueRunner],
        *,
        checkpoint_drive_type: DriveType | None = None,
        stop_join_timeout_seconds: float = 600.0,
    ) -> None:
        if not runners:
            raise ValueError("a dispatcher needs at least one lane runner")
        self._runners = tuple(runners)
        self._stop_join_timeout_seconds = stop_join_timeout_seconds
        self._checkpoint_drive_type = checkpoint_drive_type
        self._finished_listener: Callable[[str], None] | None = None

    def start(self) -> None:
        for runner in self._runners:
            runner.start()

    def stop(self) -> None:
        # Signal every lane first, so the lanes stop together against one deadline.
        for runner in self._runners:
            runner.request_stop()
        deadline = time.monotonic() + self._stop_join_timeout_seconds
        for runner in self._runners:
            runner.join(deadline)

    def wake(self) -> None:
        for runner in self._runners:
            runner.wake()

    def interrupt(self, generation_id: str, attempt_count: int) -> None:
        """Stop one running attempt. Only the lane that runs it acts."""
        for runner in self._runners:
            if runner.owns(generation_id, attempt_count):
                runner.interrupt(generation_id, attempt_count)
                return

    def lane_is_idle(self, lane_id: str) -> bool:
        """True when the lane has no running job and no queued row. False when unsure."""
        for runner in self._runners:
            if runner.lane.id == lane_id:
                try:
                    return runner.is_idle()
                except Exception:
                    # False keeps a caller from using a lane it could not prove idle.
                    logger.warning(
                        "Could not read idle state for lane %s", lane_id, exc_info=True
                    )
                    return False
        return True

    def live_state(self, generation_id: str) -> QueueLiveState | None:
        """Progress, provider status and cancel support of a running job, read from its lane."""
        for runner in self._runners:
            session = runner.session_for(generation_id)
            if session is not None:
                progress = session.progress()
                return QueueLiveState(
                    progress=None
                    if progress is None
                    else QueueProgress(
                        phase=progress.phase,
                        progress=progress.progress,
                        currentStep=progress.current_step,
                        totalSteps=progress.total_steps,
                        remainingSeconds=progress.remaining_seconds,
                    ),
                    api_status=session.api_status(),
                    cancellable=runner.lane.cancellable_while_running,
                )
        return None

    def set_finished_listener(self, listener: Callable[[str], None] | None) -> None:
        self._finished_listener = listener
        for runner in self._runners:
            runner.set_finished_listener(listener)

    def set_checkpoint_drive_type(self, drive_type: DriveType | None) -> None:
        self._checkpoint_drive_type = drive_type
        for runner in self._runners:
            runner.set_checkpoint_drive_type(drive_type)

    @property
    def is_running(self) -> bool:
        return any(runner.is_running for runner in self._runners)
