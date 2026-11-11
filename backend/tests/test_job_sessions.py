"""Job sessions: what an executor sees of cancel, progress and provider status.

Each test names the bug it catches.
"""

from __future__ import annotations

import threading
import time

from api_types import GenerationProgressResponse
from services import generation_interrupt
from services.generation_queue.lane_sessions import LaneJobSession
from services.generation_queue.sessions import LocalJobSession
from tests.fakes.generation import FakeJobSession


class _StickyCancelledOps:
    """The generation state still says cancelled from the previous job. No interrupt is set."""

    def start_api_generation(self, generation_id: str) -> None:
        del generation_id

    def get_generation_progress(self) -> GenerationProgressResponse:
        return GenerationProgressResponse(
            status="cancelled", phase="cancelled", progress=0, currentStep=0, totalSteps=0, cancellable=False
        )

    def update_progress(self, phase: str, progress: int, current_step: int | None = None, total_steps: int | None = None) -> None:
        del phase, progress, current_step, total_steps

    def raise_if_cancelled(self) -> None:
        generation_interrupt.raise_if_requested()

    def is_generation_cancelled(self) -> bool:
        return True

    def cancel_generation(self) -> object:
        return None

    def complete_generation(self, result: str | list[str] | None = None) -> None:
        del result

    def fail_generation(self, error: str) -> None:
        del error


def test_local_wait_cancelled_ignores_the_sticky_cancelled_state() -> None:
    # Bug caught: a new job reads the previous job's sticky "cancelled" state as its own cancel,
    # so a provider call is cancelled at once.
    session = LocalJobSession(_StickyCancelledOps(), "gen-1")
    started = time.monotonic()

    cancelled = session.wait_cancelled(0.1)

    assert cancelled is False
    assert time.monotonic() - started >= 0.09


def test_local_wait_cancelled_wakes_when_the_interrupt_is_requested() -> None:
    # Bug caught: a cancel waits for the end of the poll sleep.
    session = LocalJobSession(_StickyCancelledOps(), "gen-1")
    threading.Timer(0.05, generation_interrupt.request).start()
    started = time.monotonic()

    cancelled = session.wait_cancelled(5.0)

    assert cancelled is True
    assert time.monotonic() - started < 2.0


def _lane_session(
    generation_id: str, recorded: list[tuple[str, int, str]] | None = None
) -> LaneJobSession:
    sink = recorded if recorded is not None else []

    def record(gid: str, attempt: int, pid: str) -> bool:
        sink.append((gid, attempt, pid))
        return True

    return LaneJobSession(generation_id, 2, record)


def test_a_lane_session_hands_the_provider_job_id_to_the_store_sink() -> None:
    # Bug caught: the provider job id never reaches the store, so a job left running at stop cannot be cancelled at boot.
    recorded: list[tuple[str, int, str]] = []
    session = _lane_session("gen-remote", recorded)

    stored = session.set_provider_job_id("req-123")

    assert recorded == [("gen-remote", 2, "req-123")]
    assert stored is True


def test_a_lane_session_survives_a_store_failure_and_reports_it() -> None:
    # Bug caught: a store error raises into the executor. The job fails while the provider still runs it.
    def broken(generation_id: str, attempt_count: int, provider_job_id: str) -> bool:
        raise OSError("disk full")

    session = LaneJobSession("gen-remote", 1, broken)

    assert session.set_provider_job_id("req-123") is False


def test_a_lane_session_reports_an_id_the_store_refused() -> None:
    # Bug caught: the executor believes the id is stored when the job was no longer running.
    session = LaneJobSession("gen-remote", 1, lambda generation_id, attempt_count, provider_job_id: False)

    assert session.set_provider_job_id("req-123") is False


def test_the_local_session_keeps_no_provider_job_id_and_lets_the_job_go_on() -> None:
    # Bug caught: a fal job on the local lane (the kill switch) fails, as its id has nowhere to go.
    session = LocalJobSession(_StickyCancelledOps(), "gen-1")

    assert session.set_provider_job_id("req-123") is True


def test_a_lane_session_cancel_does_not_touch_the_process_wide_interrupt() -> None:
    # Bug caught: cancelling a remote job sets the global flag and stops the local job.
    session = _lane_session("gen-remote")

    session.cancel()

    assert session.wait_cancelled(0) is True
    assert generation_interrupt.is_requested() is False


def test_a_lane_session_wait_cancelled_wakes_at_once_on_cancel() -> None:
    # Bug caught: the provider poll sleeps its whole interval after a cancel and the lane stays busy.
    session = _lane_session("gen-remote")
    threading.Timer(0.05, session.cancel).start()
    started = time.monotonic()

    assert session.wait_cancelled(5.0) is True
    assert time.monotonic() - started < 2.0


def test_two_lane_sessions_keep_separate_progress_and_status() -> None:
    # Bug caught: lane jobs share one progress slot and show each other's numbers.
    first, second = _lane_session("a"), _lane_session("b")

    first.set_progress("inference", 40, None, None)
    first.set_api_status("processing")

    assert second.progress() is None
    assert second.api_status() is None
    progress = first.progress()
    assert progress is not None and progress.progress == 40
    assert first.api_status() == "processing"


def test_the_fake_session_records_progress_and_still_answers_the_progress_reader() -> None:
    # Bug caught: a recording list named `progress` hides the progress() method of the JobSession protocol.
    session = FakeJobSession()

    session.set_progress("inference", 40, None, None)

    assert session.progress_updates == [("inference", 40)]
    assert session.progress() is None
