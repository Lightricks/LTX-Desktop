"""The dispatcher runs one runner per lane. Lanes run side by side.

Each test names the bug it catches. Jobs fail on purpose after the checks: only the
timing and the claim order matter here, not the output.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

from pydantic import JsonValue

from services.displaced_provider_jobs import (
    DISPLACED_PROVIDER_JOBS_FILENAME,
    DisplacedProviderJobs,
)
from services.generation_interrupt import GenerationCancelledError
from services.generation_queue.dispatcher import QueueDispatcher
from services.generation_queue.lane_sessions import LaneJobSession
from services.generation_queue.lane_slots import CountingLaneSlot
from services.generation_queue.lanes import LOCAL_LANE_SPEC, LaneSpec
from services.generation_queue.registry import ExecutorRegistry
from services.generation_queue.runner import QueueRunner
from services.generation_queue.types import ExecutionOutcome, OutputAllocation, OutputPlan
from services.records import LOCAL_LANE, GenerationRecord, ProviderOrphan
from services.sqlite_store import SqliteStore
from tests.fakes.lanes import TEST_FEATURE, TEST_LANE, TEST_LANE_SPEC

_PLAN = OutputPlan(slot="output", media_kind="video", mime_type="video/mp4", name="output.mp4")


def _spec() -> dict[str, JsonValue]:
    return {"params": {"prompt": "fox"}, "inputs": {}}


class _OpenSlot:
    def wait_for_generation_slot(self, shutdown: threading.Event) -> AbstractContextManager[None]:
        del shutdown
        return nullcontext()

    def cancel_generation(self) -> object:
        return None


class _ProbeExecutor:
    """Records overlap and claim order, then fails the job."""

    def __init__(self, lane: str, *, hold_seconds: float = 0.0, barrier: threading.Barrier | None = None) -> None:
        self._lane = lane
        self._hold_seconds = hold_seconds
        self._barrier = barrier
        self._lock = threading.Lock()
        self.running = 0
        self.max_running = 0
        self.runs: list[str] = []
        self.release = threading.Event()
        self.release.set()

    def validate_params(self, spec: Mapping[str, object], *, contract_version: int) -> None:
        del spec, contract_version

    def lane_for(self, spec: Mapping[str, object]) -> str:
        del spec
        return self._lane

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]:
        del generation
        return (_PLAN,)

    def execute(self, generation: GenerationRecord, outputs: tuple[OutputAllocation, ...], session) -> ExecutionOutcome:
        del outputs, session
        with self._lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
            self.runs.append(generation.id)
        try:
            if self._barrier is not None:
                self._barrier.wait()
            time.sleep(self._hold_seconds)
            self.release.wait(timeout=10)
        finally:
            with self._lock:
                self.running -= 1
        raise RuntimeError("probe done")


def _runner(store: SqliteStore, executors: dict[str, _ProbeExecutor], lane: LaneSpec, **kwargs) -> QueueRunner:
    local = lane.id == LOCAL_LANE
    return QueueRunner(
        db=store,
        generation=_OpenSlot() if local else CountingLaneSlot(lane.concurrency),
        executors=ExecutorRegistry(executors),  # type: ignore[arg-type]
        lane=lane,
        session_factory=None if local else (lambda job: LaneJobSession(job.id, job.attempt_count, store.set_provider_job_id)),
        idle_wait_seconds=kwargs.pop("idle_wait_seconds", 0.05),
        **kwargs,
    )


def _wait(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _failed(store: SqliteStore, *ids: str) -> bool:
    return all(store.get_generation(generation_id).status == "failed" for generation_id in ids)


def test_a_local_job_and_a_remote_job_are_in_flight_together(tmp_path: Path) -> None:
    # Bug caught: lanes share one worker, so the second job starts only after the first ends.
    store = SqliteStore(tmp_path / "app_data")
    barrier = threading.Barrier(2, timeout=3)
    executors = {
        "text-to-video": _ProbeExecutor(LOCAL_LANE, barrier=barrier),
        TEST_FEATURE: _ProbeExecutor(TEST_LANE, barrier=barrier),
    }
    local = store.insert_generation("text-to-video", _spec())
    remote = store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)
    dispatcher = QueueDispatcher(
        [_runner(store, executors, LOCAL_LANE_SPEC), _runner(store, executors, TEST_LANE_SPEC)]
    )

    dispatcher.start()
    try:
        assert _wait(lambda: _failed(store, local.id, remote.id))
        # Both jobs end failed on purpose. A barrier that timed out means they never overlapped.
        assert not barrier.broken
    finally:
        dispatcher.stop()


def test_a_lane_runs_one_job_at_a_time_at_concurrency_one(tmp_path: Path) -> None:
    # Bug caught: a remote lane with concurrency 1 runs two provider requests at once.
    store = SqliteStore(tmp_path / "app_data")
    probe = _ProbeExecutor(TEST_LANE, hold_seconds=0.05)
    ids = [store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE).id for _ in range(3)]
    dispatcher = QueueDispatcher(
        [_runner(store, {TEST_FEATURE: probe, "text-to-video": _ProbeExecutor(LOCAL_LANE)}, LOCAL_LANE_SPEC),
         _runner(store, {TEST_FEATURE: probe}, TEST_LANE_SPEC)]
    )

    dispatcher.start()
    try:
        assert _wait(lambda: _failed(store, *ids))
    finally:
        dispatcher.stop()

    assert probe.max_running == 1
    assert probe.runs == ids


def test_two_workers_of_one_lane_never_run_the_same_row(tmp_path: Path) -> None:
    # Bug caught: two workers claim one row and the provider request is sent twice.
    store = SqliteStore(tmp_path / "app_data")
    # The barrier lets a job finish only when three run together, so max_running is exact.
    probe = _ProbeExecutor(TEST_LANE, barrier=threading.Barrier(3, timeout=10))
    wide_lane = LaneSpec(id=TEST_LANE, concurrency=3, uses_gpu=False, cancellable_while_running=True)
    ids = [store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE).id for _ in range(12)]
    dispatcher = QueueDispatcher(
        [_runner(store, {"text-to-video": _ProbeExecutor(LOCAL_LANE)}, LOCAL_LANE_SPEC),
         _runner(store, {TEST_FEATURE: probe}, wide_lane)]
    )

    dispatcher.start()
    try:
        assert _wait(lambda: _failed(store, *ids))
    finally:
        dispatcher.stop()

    assert sorted(probe.runs) == sorted(ids)
    assert len(set(probe.runs)) == len(ids)
    assert probe.max_running == 3


def test_wake_reaches_the_runner_of_the_lane_that_got_the_row(tmp_path: Path) -> None:
    # Bug caught: a new remote row waits for the idle poll because only the local runner is woken.
    store = SqliteStore(tmp_path / "app_data")
    probe = _ProbeExecutor(TEST_LANE)
    dispatcher = QueueDispatcher(
        [_runner(store, {"text-to-video": _ProbeExecutor(LOCAL_LANE)}, LOCAL_LANE_SPEC, idle_wait_seconds=30),
         _runner(store, {TEST_FEATURE: probe}, TEST_LANE_SPEC, idle_wait_seconds=30)]
    )
    dispatcher.start()
    try:
        time.sleep(0.2)  # Both runners are now idle and wait on their wake event.
        job = store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)
        dispatcher.wake()

        assert _wait(lambda: _failed(store, job.id), timeout=3)
    finally:
        dispatcher.stop()


def test_stop_waits_for_all_lanes_against_one_deadline(tmp_path: Path) -> None:
    # Bug caught: stop joins the lanes one after the other, so shutdown takes N times the limit.
    store = SqliteStore(tmp_path / "app_data")
    local_probe = _ProbeExecutor(LOCAL_LANE)
    remote_probe = _ProbeExecutor(TEST_LANE)
    for probe in (local_probe, remote_probe):
        probe.release.clear()  # The job ignores a stop and ends only when released.
    store.insert_generation("text-to-video", _spec())
    store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)
    slow_remote = LaneSpec(id=TEST_LANE, concurrency=1, uses_gpu=False, cancellable_while_running=True)
    dispatcher = QueueDispatcher(
        [_runner(store, {"text-to-video": local_probe}, LOCAL_LANE_SPEC),
         _runner(store, {TEST_FEATURE: remote_probe}, slow_remote)],
        stop_join_timeout_seconds=1.0,
    )
    dispatcher.start()
    assert _wait(lambda: local_probe.running == 1 and remote_probe.running == 1)

    started = time.monotonic()
    dispatcher.stop()
    elapsed = time.monotonic() - started

    local_probe.release.set()
    remote_probe.release.set()
    assert elapsed < 1.0 * 1.8


class _ProviderWaitExecutor(_ProbeExecutor):
    """A provider job: it waits on its session like the remote client does."""

    def execute(self, generation: GenerationRecord, outputs: tuple[OutputAllocation, ...], session) -> ExecutionOutcome:
        del outputs
        with self._lock:
            self.running += 1
            self.runs.append(generation.id)
        try:
            while not session.wait_cancelled(0.01):
                pass
            raise GenerationCancelledError()
        finally:
            with self._lock:
                self.running -= 1


def test_stop_cancels_a_provider_job_so_its_request_is_freed(tmp_path: Path) -> None:
    # Bug caught: the app quits with a remote request open. The job never sees a cancel and the provider keeps billing it.
    store = SqliteStore(tmp_path / "app_data")
    probe = _ProviderWaitExecutor(TEST_LANE)
    store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)
    dispatcher = QueueDispatcher(
        [_runner(store, {"text-to-video": _ProbeExecutor(LOCAL_LANE)}, LOCAL_LANE_SPEC),
         _runner(store, {TEST_FEATURE: probe}, TEST_LANE_SPEC)],
    )
    dispatcher.start()
    assert _wait(lambda: probe.running == 1)

    dispatcher.stop()

    # The job saw the cancel and returned. Boot recovery fails the row it left running.
    assert probe.running == 0


class _SubmitThenEndExecutor(_ProbeExecutor):
    """A provider job: the provider accepts it, then the job ends the given way."""

    def __init__(self, lane: str, end: str) -> None:
        super().__init__(lane)
        self.end = end
        self.submitted = threading.Event()

    def execute(self, generation: GenerationRecord, outputs: tuple[OutputAllocation, ...], session) -> ExecutionOutcome:
        del outputs
        assert session.set_provider_job_id("req-77") is True
        self.submitted.set()
        if self.end == "fail":
            raise RuntimeError("lost the provider")
        while not session.wait_cancelled(0.01):
            pass
        raise GenerationCancelledError()


def _lane_with_cancel(cancels: list[str]) -> LaneSpec:
    return LaneSpec(
        id=TEST_LANE,
        concurrency=1,
        uses_gpu=False,
        cancellable_while_running=True,
        cancel_orphan=cancels.append,
    )


def test_a_user_cancel_asks_the_provider_to_drop_the_job_and_clears_its_id(tmp_path: Path) -> None:
    # Bug caught: the row becomes cancelled, the executor stops waiting, and the provider keeps the job.
    store = SqliteStore(tmp_path / "app_data")
    executor = _SubmitThenEndExecutor(TEST_LANE, "cancel")
    cancels: list[str] = []
    job = store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)
    runner = _runner(store, {TEST_FEATURE: executor}, _lane_with_cancel(cancels))
    runner.start()
    try:
        assert executor.submitted.wait(5.0)
        cancelling = store.request_generation_cancellation(job.id)
        runner.interrupt(cancelling.id, cancelling.attempt_count)

        assert _wait(lambda: cancels == ["req-77"])
        assert _wait(lambda: (g := store.get_generation(job.id)) is not None and g.provider_job_id is None)
    finally:
        runner.stop()
    final = store.get_generation(job.id)
    assert final is not None and final.status == "cancelled"


def test_a_job_that_fails_after_the_provider_accepted_it_is_cancelled_at_the_provider(tmp_path: Path) -> None:
    # Bug caught: an executor error ends the row as failed while the provider still runs and bills the job.
    store = SqliteStore(tmp_path / "app_data")
    executor = _SubmitThenEndExecutor(TEST_LANE, "fail")
    cancels: list[str] = []
    job = store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)
    runner = _runner(store, {TEST_FEATURE: executor}, _lane_with_cancel(cancels))
    runner.start()
    try:
        assert _wait(lambda: cancels == ["req-77"])
    finally:
        runner.stop()
    final = store.get_generation(job.id)
    assert final is not None and final.status == "failed"


def _failed_with_id(store: SqliteStore, provider_id: str) -> GenerationRecord:
    store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)
    claimed = store.claim_next_queued(TEST_LANE)
    assert claimed is not None
    store.set_provider_job_id(claimed.id, claimed.attempt_count, provider_id)
    store.mark_failed(claimed.id, "OUTPUT_UNREADABLE", attempt_count=claimed.attempt_count)
    return claimed


def test_a_retry_before_the_runner_reads_the_id_does_not_lose_the_provider_job(tmp_path: Path) -> None:
    # Bug caught: the user retries between the failure and the provider cancel, and the old id is gone.
    store = SqliteStore(tmp_path / "app_data")
    cancels: list[str] = []
    runner = _runner(store, {}, _lane_with_cancel(cancels))
    failed = _failed_with_id(store, "req-1")
    store.retry_generation(failed.id)

    runner._release_provider_job(failed)

    assert _wait(lambda: cancels == ["req-1"])
    assert _wait(lambda: store.list_provider_orphans() == [])


def test_a_failed_cancel_of_a_replaced_attempt_keeps_its_id_for_the_next_boot(tmp_path: Path) -> None:
    # Bug caught: a provider that is down makes the retry drop the id for good.
    store = SqliteStore(tmp_path / "app_data")
    calls: list[str] = []

    def provider_down(provider_job_id: str) -> None:
        calls.append(provider_job_id)
        raise RuntimeError("provider down")

    lane = LaneSpec(
        id=TEST_LANE, concurrency=1, uses_gpu=False, cancellable_while_running=True,
        cancel_orphan=provider_down,
    )
    runner = _runner(store, {}, lane)
    failed = _failed_with_id(store, "req-1")
    store.retry_generation(failed.id)

    runner._release_provider_job(failed)

    assert _wait(lambda: calls == ["req-1"])
    assert [o.provider_job_id for o in store.list_provider_orphans()] == ["req-1"]


def test_a_replaced_attempt_of_another_lane_is_left_to_the_boot(tmp_path: Path) -> None:
    # Bug caught: a lane cancels a job of another lane with the wrong provider call.
    store = SqliteStore(tmp_path / "app_data")
    cancels: list[str] = []
    runner = _runner(store, {}, _lane_with_cancel(cancels))
    failed = _failed_with_id(store, "req-1")
    store.retry_generation(failed.id, lane=LOCAL_LANE)
    displaced = DisplacedProviderJobs(
        tmp_path / "app_data" / DISPLACED_PROVIDER_JOBS_FILENAME
    )
    displaced.remove(failed.id, "req-1")
    displaced.add(
        ProviderOrphan(generation_id=failed.id, lane="other-provider", provider_job_id="req-1")
    )

    runner._release_provider_job(failed)

    assert cancels == []
    assert [o.provider_job_id for o in store.list_provider_orphans()] == ["req-1"]


def test_a_job_claimed_while_the_lane_stops_is_cancelled(tmp_path: Path) -> None:
    # Bug caught: a job that registers after request_stop took its snapshot runs past the stop.
    store = SqliteStore(tmp_path / "app_data")
    store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)
    sessions: list[LaneJobSession] = []
    holder: dict[str, QueueRunner] = {}

    def factory(job: GenerationRecord) -> LaneJobSession:
        # The stop lands between the claim and the registration of the job.
        holder["runner"].request_stop()
        session = LaneJobSession(job.id, job.attempt_count, store.set_provider_job_id)
        sessions.append(session)
        return session

    runner = QueueRunner(
        db=store,
        generation=CountingLaneSlot(1),
        executors=ExecutorRegistry({TEST_FEATURE: _ProviderWaitExecutor(TEST_LANE)}),  # type: ignore[arg-type]
        lane=TEST_LANE_SPEC,
        session_factory=factory,
        idle_wait_seconds=0.05,
    )
    holder["runner"] = runner
    runner.start()
    try:
        assert _wait(lambda: len(sessions) == 1)
        assert _wait(lambda: sessions[0].wait_cancelled(0))
    finally:
        runner.stop()
