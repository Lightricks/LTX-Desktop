"""Lane routing in the store and the registry.

Each test names the bug it catches.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from PIL import Image
from pydantic import JsonValue

from api_types import QueueProgress
from services.displaced_provider_jobs import (
    DISPLACED_PROVIDER_JOBS_FILENAME,
    DisplacedProviderJobs,
)
from services.generation_queue.control import QueueLiveState
from services.generation_queue.lanes import LOCAL_LANE_SPEC, LaneRegistry, LaneSpec
from _routes._errors import HTTPError
from services.generation_queue.orphans import cancel_provider_orphans, forget_in
from services.generation_queue.registry import ExecutorRegistry
from services.records import LOCAL_LANE, OutputSpec, ProviderOrphan, UnavailableError
from services.sqlite_store import SqliteStore
from tests.fakes.lanes import TEST_FEATURE, TEST_LANE, TEST_LANE_SPEC


def _spec() -> dict[str, JsonValue]:
    return {"params": {"prompt": "fox"}, "inputs": {}}


class _TestLaneExecutor:
    """Only the part of the executor the registry reads."""

    def lane_for(self, spec) -> str:
        del spec
        return TEST_LANE


class _PlainExecutor:
    pass


def test_a_runner_never_claims_a_row_of_another_lane(tmp_path: Path) -> None:
    # Bug caught: the local runner claims a remote job and runs it on the GPU slot.
    store = SqliteStore(tmp_path / "app_data")
    remote_job = store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)

    assert store.claim_next_queued(LOCAL_LANE) is None
    claimed = store.claim_next_queued(TEST_LANE)
    assert claimed is not None
    assert claimed.id == remote_job.id
    assert claimed.lane == TEST_LANE


def test_has_queued_only_sees_its_own_lane(tmp_path: Path) -> None:
    # Bug caught: an idle local runner spins on a queued remote row it can never claim.
    store = SqliteStore(tmp_path / "app_data")
    store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)

    assert store.has_queued(TEST_LANE)
    assert not store.has_queued(LOCAL_LANE)


def test_each_lane_keeps_queue_order_on_its_own(tmp_path: Path) -> None:
    # Bug caught: a lane skips its oldest row because another lane holds an older one.
    store = SqliteStore(tmp_path / "app_data")
    local_first = store.insert_generation("text-to-video", _spec())
    remote_first = store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)
    remote_second = store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)

    remote_claim = store.claim_next_queued(TEST_LANE)
    local_claim = store.claim_next_queued(LOCAL_LANE)

    assert remote_claim is not None and remote_claim.id == remote_first.id
    assert local_claim is not None and local_claim.id == local_first.id
    next_remote = store.claim_next_queued(TEST_LANE)
    assert next_remote is not None and next_remote.id == remote_second.id


def test_retry_keeps_the_lane(tmp_path: Path) -> None:
    # Bug caught: a retried remote job falls back to the local lane.
    store = SqliteStore(tmp_path / "app_data")
    job = store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)
    claimed = store.claim_next_queued(TEST_LANE)
    assert claimed is not None
    store.mark_failed(job.id, "OUTPUT_UNREADABLE", attempt_count=claimed.attempt_count)

    retried = store.retry_generation(job.id)

    assert retried.lane == TEST_LANE
    assert store.has_queued(TEST_LANE)


def test_unregistered_lane_rows_move_to_local_at_boot(tmp_path: Path) -> None:
    # Bug caught: after the parallel lanes are turned off, queued remote rows wait forever.
    store = SqliteStore(tmp_path / "app_data")
    stranded = store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)
    kept = store.insert_generation("text-to-video", _spec())

    moved = store.reassign_unknown_lanes([LOCAL_LANE])

    assert moved == 1
    reloaded = store.get_generation(stranded.id)
    assert reloaded is not None and reloaded.lane == LOCAL_LANE
    # The moved row keeps its place in the queue order.
    claimed = store.claim_next_queued(LOCAL_LANE)
    assert claimed is not None and claimed.id == stranded.id
    next_claim = store.claim_next_queued(LOCAL_LANE)
    assert next_claim is not None and next_claim.id == kept.id


def test_boot_reassignment_leaves_finished_rows_alone(tmp_path: Path) -> None:
    # Bug caught: a boot with the kill switch rewrites the lane of every old row for good.
    store = SqliteStore(tmp_path / "app_data")
    job = store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)
    claimed = store.claim_next_queued(TEST_LANE)
    assert claimed is not None
    store.mark_failed(job.id, "OUTPUT_UNREADABLE", attempt_count=claimed.attempt_count)

    assert store.reassign_unknown_lanes([LOCAL_LANE]) == 0
    reloaded = store.get_generation(job.id)
    assert reloaded is not None and reloaded.lane == TEST_LANE


def test_retry_can_move_the_job_to_the_lane_that_is_valid_now(tmp_path: Path) -> None:
    # Bug caught: a failed remote job retried after a boot with lanes off waits for a lane with no runner.
    store = SqliteStore(tmp_path / "app_data")
    job = store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)
    claimed = store.claim_next_queued(TEST_LANE)
    assert claimed is not None
    store.mark_failed(job.id, "OUTPUT_UNREADABLE", attempt_count=claimed.attempt_count)

    retried = store.retry_generation(job.id, lane=LOCAL_LANE)

    assert retried.lane == LOCAL_LANE
    assert store.has_queued(LOCAL_LANE) and not store.has_queued(TEST_LANE)


def test_registered_lane_rows_stay_put_at_boot(tmp_path: Path) -> None:
    # Bug caught: the boot reassignment moves rows of a live lane to local.
    store = SqliteStore(tmp_path / "app_data")
    job = store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)

    assert store.reassign_unknown_lanes([LOCAL_LANE, TEST_LANE]) == 0
    reloaded = store.get_generation(job.id)
    assert reloaded is not None and reloaded.lane == TEST_LANE


def test_executor_without_lane_for_runs_on_local() -> None:
    # Bug caught: an executor that does not know about lanes gets no lane at all.
    registry = ExecutorRegistry(
        {"text-to-video": _PlainExecutor()},  # type: ignore[dict-item]
        LaneRegistry([LOCAL_LANE_SPEC, TEST_LANE_SPEC]),
    )

    assert registry.lane_for("text-to-video", _spec()) == LOCAL_LANE


def test_lane_of_an_unregistered_provider_resolves_to_local() -> None:
    # Bug caught: a job is inserted on a lane that has no runner (kill switch on).
    registry = ExecutorRegistry(
        {TEST_FEATURE: _TestLaneExecutor()},  # type: ignore[dict-item]
        LaneRegistry([LOCAL_LANE_SPEC]),
    )

    assert registry.lane_for(TEST_FEATURE, _spec()) == LOCAL_LANE


def test_executor_lane_is_used_when_registered() -> None:
    # Bug caught: the executor choice is ignored, so every job stays local.
    registry = ExecutorRegistry(
        {TEST_FEATURE: _TestLaneExecutor()},  # type: ignore[dict-item]
        LaneRegistry([LOCAL_LANE_SPEC, TEST_LANE_SPEC]),
    )

    assert registry.lane_for(TEST_FEATURE, _spec()) == TEST_LANE


def test_lane_registry_needs_the_local_lane() -> None:
    # Bug caught: a registry without local leaves unknown rows with no fallback.
    with pytest.raises(ValueError, match="local lane"):
        LaneRegistry([TEST_LANE_SPEC])


def test_lane_needs_at_least_one_worker() -> None:
    # Bug caught: a lane with zero workers accepts jobs and never runs them.
    with pytest.raises(ValueError, match="concurrency"):
        LaneSpec(id="x", concurrency=0, uses_gpu=False, cancellable_while_running=True)


class _LiveStateByJob:
    def __init__(self, states: dict[str, QueueLiveState]) -> None:
        self._states = states

    def wake(self) -> None:
        return None

    def interrupt(self, generation_id: str, attempt_count: int) -> None:
        del generation_id, attempt_count

    def live_state(self, generation_id: str) -> QueueLiveState | None:
        return self._states.get(generation_id)


def test_queue_snapshot_lists_every_running_job_with_its_own_live_state(
    client, test_state
) -> None:
    # Bug caught: the snapshot shows one running job, or shows the local job's progress
    # and cancel support on the provider job.
    test_state.generation_queue.stop()
    store = test_state.queued_generations._db
    local_job = store.insert_generation("text-to-video", _spec())
    remote_job = store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)
    assert store.claim_next_queued(LOCAL_LANE) is not None
    assert store.claim_next_queued(TEST_LANE) is not None
    test_state.queued_generations._queue_control = _LiveStateByJob(
        {
            local_job.id: QueueLiveState(
                progress=QueueProgress(
                    phase="inference",
                    progress=40,
                    currentStep=4,
                    totalSteps=10,
                    remainingSeconds=95,
                ),
                api_status=None,
                cancellable=True,
            ),
            remote_job.id: QueueLiveState(
                progress=None, api_status="processing", cancellable=False
            ),
        }
    )

    body = client.get("/api/generation-queue").json()

    by_id = {entry["generation"]["id"]: entry for entry in body["active"]}
    assert set(by_id) == {local_job.id, remote_job.id}
    assert by_id[local_job.id]["generation"]["lane"] == LOCAL_LANE
    assert by_id[local_job.id]["progress"]["progress"] == 40
    assert by_id[local_job.id]["progress"]["remainingSeconds"] == 95
    assert by_id[local_job.id]["cancellable"] is True
    assert by_id[local_job.id]["api_status"] is None
    assert by_id[remote_job.id]["generation"]["lane"] == TEST_LANE
    assert by_id[remote_job.id]["progress"] is None
    assert by_id[remote_job.id]["cancellable"] is False
    assert by_id[remote_job.id]["api_status"] == "processing"


def test_queued_rows_are_always_cancellable(client, test_state) -> None:
    # Bug caught: a queued row inherits "not cancellable" from a running provider job.
    test_state.generation_queue.stop()
    store = test_state.queued_generations._db
    store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)

    body = client.get("/api/generation-queue").json()

    assert body["active"] == []
    assert body["queued"][0]["cancellable"] is True
    assert body["queued"][0]["generation"]["lane"] == TEST_LANE


def test_retry_of_a_job_with_an_unknown_feature_still_leaves_an_unregistered_lane(test_state) -> None:
    # Bug caught: the lane lookup fails, the stored lane is kept, and no runner serves it.
    store = test_state.queued_generations._db
    job = store.insert_generation("removed-feature", _spec(), lane=TEST_LANE)
    claimed = store.claim_next_queued(TEST_LANE)
    assert claimed is not None
    store.mark_failed(job.id, "OUTPUT_UNREADABLE", attempt_count=claimed.attempt_count)

    test_state.queued_generations.retry_generation(job.id)

    reloaded = store.get_generation(job.id)
    assert reloaded is not None and reloaded.lane == LOCAL_LANE


def _claim_remote(store: SqliteStore):
    job = store.insert_generation(TEST_FEATURE, _spec(), lane=TEST_LANE)
    claimed = store.claim_next_queued(TEST_LANE)
    assert claimed is not None and claimed.id == job.id
    return claimed


def test_provider_job_id_is_stored_for_a_running_job_only(tmp_path: Path) -> None:
    # Bug caught: a late provider id lands on a queued or finished row and is later cancelled by mistake.
    store = SqliteStore(tmp_path / "app_data")
    queued = store.insert_generation("text-to-video", _spec())
    assert store.set_provider_job_id(queued.id, queued.attempt_count, "req-1") is False

    running = _claim_remote(store)
    assert store.set_provider_job_id(running.id, running.attempt_count, "req-2") is True
    reloaded = store.get_generation(running.id)
    assert reloaded is not None and reloaded.provider_job_id == "req-2"

    store.mark_failed(running.id, "OUTPUT_UNREADABLE", attempt_count=running.attempt_count)
    assert store.set_provider_job_id(running.id, running.attempt_count, "req-3") is False


def test_an_empty_provider_job_id_is_refused(tmp_path: Path) -> None:
    # Bug caught: an empty id is stored, the boot orphan list cannot be read, and the app crashes at every start.
    store = SqliteStore(tmp_path / "app_data")
    running = _claim_remote(store)

    with pytest.raises(ValueError, match="empty"):
        store.set_provider_job_id(running.id, running.attempt_count, "")

    assert store.list_provider_orphans() == []


def test_the_orphan_list_skips_an_empty_id_already_in_the_store(tmp_path: Path) -> None:
    # Bug caught: a row with an empty id (written by an older build) crashes the boot.
    store = SqliteStore(tmp_path / "app_data")
    running = _claim_remote(store)
    conn = store._connect()
    conn.execute("UPDATE generations SET provider_job_id = '' WHERE id = ?", (running.id,))
    conn.commit()
    conn.close()

    assert store.list_provider_orphans() == []


def test_a_store_error_while_moving_lanes_reads_as_unavailable(tmp_path: Path, monkeypatch) -> None:
    # Bug caught: a locked or full database crashes the boot instead of switching to the unavailable store.
    store = SqliteStore(tmp_path / "app_data")

    def broken_connect():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "_connect", broken_connect)

    with pytest.raises(UnavailableError):
        store.reassign_unknown_lanes([LOCAL_LANE])


def _fail_with_id(store: SqliteStore, provider_id: str):
    claimed = _claim_remote(store)
    store.set_provider_job_id(claimed.id, claimed.attempt_count, provider_id)
    store.mark_failed(claimed.id, "OUTPUT_UNREADABLE", attempt_count=claimed.attempt_count)
    return claimed


def test_a_retry_shows_the_newest_attempt_only_but_keeps_the_old_id_to_cancel(tmp_path: Path) -> None:
    # Bug caught: a retry shows the id of the first attempt, or drops an id that nobody cancelled.
    store = SqliteStore(tmp_path / "app_data")
    failed = _fail_with_id(store, "req-1")

    retried = store.retry_generation(failed.id)

    assert retried.provider_job_id is None
    assert [(o.generation_id, o.lane, o.provider_job_id) for o in store.list_provider_orphans()] == [
        (failed.id, TEST_LANE, "req-1")
    ]


def test_every_attempt_that_failed_keeps_its_own_id_until_its_cancel_succeeds(tmp_path: Path) -> None:
    # Bug caught: a second retry overwrites the id that the first retry saved.
    store = SqliteStore(tmp_path / "app_data")
    first = _fail_with_id(store, "req-1")
    store.retry_generation(first.id)
    second = store.claim_next_queued(TEST_LANE)
    assert second is not None
    store.set_provider_job_id(second.id, second.attempt_count, "req-2")
    store.mark_failed(second.id, "OUTPUT_UNREADABLE", attempt_count=second.attempt_count)
    store.retry_generation(second.id)

    assert sorted(o.provider_job_id for o in store.list_provider_orphans()) == ["req-1", "req-2"]

    assert store.clear_provider_job_id(first.id, "req-1") is True
    assert [o.provider_job_id for o in store.list_provider_orphans()] == ["req-2"]
    assert store.clear_provider_job_id(first.id, "req-1") is False


def test_an_id_in_the_file_and_on_the_row_is_listed_once(tmp_path: Path) -> None:
    # Bug caught: a crash between the file write and the commit of a retry cancels one job twice.
    app_data = tmp_path / "app_data"
    store = SqliteStore(app_data)
    failed = _fail_with_id(store, "req-1")
    DisplacedProviderJobs(app_data / DISPLACED_PROVIDER_JOBS_FILENAME).add(
        ProviderOrphan(generation_id=failed.id, lane=TEST_LANE, provider_job_id="req-1")
    )

    assert [o.provider_job_id for o in store.list_provider_orphans()] == ["req-1"]
    assert store.clear_provider_job_id(failed.id, "req-1") is True
    assert store.list_provider_orphans() == []


def test_a_retry_that_cannot_keep_the_old_id_fails_and_leaves_the_job_failed(tmp_path: Path) -> None:
    # Bug caught: a retry drops the id of a paid job because the file cannot be written.
    app_data = tmp_path / "app_data"
    store = SqliteStore(app_data)
    failed = _fail_with_id(store, "req-1")
    (app_data / DISPLACED_PROVIDER_JOBS_FILENAME).mkdir()

    with pytest.raises(UnavailableError):
        store.retry_generation(failed.id)

    row = store.get_generation(failed.id)
    assert row is not None
    assert (row.status, row.provider_job_id) == ("failed", "req-1")


def test_a_retry_of_a_job_with_no_id_adds_no_orphan(tmp_path: Path) -> None:
    # Bug caught: a retry of a local or early failure lists an empty or made-up id at boot.
    store = SqliteStore(tmp_path / "app_data")
    claimed = _claim_remote(store)
    store.mark_failed(claimed.id, "OUTPUT_UNREADABLE", attempt_count=claimed.attempt_count)

    store.retry_generation(claimed.id)

    assert store.list_provider_orphans() == []
    assert store.list_displaced_provider_jobs(claimed.id) == []


def test_a_deleted_failed_job_keeps_its_id_for_the_boot_cancel(tmp_path: Path) -> None:
    # Bug caught: the user deletes a failed card whose provider cancel failed, and the job runs on.
    store = SqliteStore(tmp_path / "app_data")
    failed = _fail_with_id(store, "req-1")

    assert store.delete_generation(failed.id) is True

    assert [o.provider_job_id for o in store.list_provider_orphans()] == ["req-1"]
    assert store.clear_provider_job_id(failed.id, "req-1") is True
    assert store.list_provider_orphans() == []


def test_orphans_are_rows_that_hold_a_provider_job_id_unless_the_job_succeeded(
    tmp_path: Path,
) -> None:
    # Bug caught: a failed or cancelled job drops out of the list while the provider still runs it.
    store = SqliteStore(tmp_path / "app_data")
    running = _claim_remote(store)
    store.set_provider_job_id(running.id, running.attempt_count, "req-running")
    _claim_remote(store)  # Running, but the provider never accepted it.
    failed = _claim_remote(store)
    store.set_provider_job_id(failed.id, failed.attempt_count, "req-failed")
    store.mark_failed(failed.id, "OUTPUT_UNREADABLE", attempt_count=failed.attempt_count)
    cancelled = _claim_remote(store)
    store.set_provider_job_id(cancelled.id, cancelled.attempt_count, "req-cancelled")
    store.request_generation_cancellation(cancelled.id)
    store.mark_cancelled(cancelled.id, [], attempt_count=cancelled.attempt_count)
    done = _claim_remote(store)
    store.set_provider_job_id(done.id, done.attempt_count, "req-done")
    asset_id, dest = store.allocate_output_path("image", "image/png")
    Image.new("RGB", (8, 8)).save(dest)
    store.mark_succeeded(
        done.id,
        [OutputSpec(asset_id=asset_id, dest_path=dest, ordinal=0, mime_type="image/png", name="out.png")],
        attempt_count=done.attempt_count,
    )

    orphans = store.list_provider_orphans()

    assert sorted(o.provider_job_id for o in orphans) == ["req-cancelled", "req-failed", "req-running"]
    assert {o.lane for o in orphans} == {TEST_LANE}


def test_a_late_provider_job_id_of_an_old_attempt_does_not_land_on_the_retry(tmp_path: Path) -> None:
    # Bug caught: a provider thread that outlives the stop stamps its id on the running row of a retry.
    store = SqliteStore(tmp_path / "app_data")
    first = _claim_remote(store)
    store.mark_failed(first.id, "OUTPUT_UNREADABLE", attempt_count=first.attempt_count)
    store.retry_generation(first.id)
    retry = store.claim_next_queued(TEST_LANE)
    assert retry is not None and retry.attempt_count != first.attempt_count

    assert store.set_provider_job_id(first.id, first.attempt_count, "late-id") is False

    reloaded = store.get_generation(first.id)
    assert reloaded is not None and reloaded.provider_job_id is None


def test_clearing_a_provider_job_id_leaves_a_newer_id_alone(tmp_path: Path) -> None:
    # Bug caught: a slow cancel of an old provider job wipes the id of the job that replaced it.
    store = SqliteStore(tmp_path / "app_data")
    claimed = _claim_remote(store)
    store.set_provider_job_id(claimed.id, claimed.attempt_count, "req-new")

    assert store.clear_provider_job_id(claimed.id, "req-old") is False
    assert store.clear_provider_job_id(claimed.id, "req-new") is True

    reloaded = store.get_generation(claimed.id)
    assert reloaded is not None and reloaded.provider_job_id is None


def test_boot_recovery_fails_the_running_rows_and_hands_every_orphan_to_its_lane(
    test_state, monkeypatch
) -> None:
    # Bug caught: a job that was running at the provider is failed locally and nobody cancels it.
    handler = test_state.queued_generations
    store = handler._db
    claimed = _claim_remote(store)
    store.set_provider_job_id(claimed.id, claimed.attempt_count, "req-1")
    seen: list[list[str]] = []
    monkeypatch.setattr(
        "handlers.queued_generation_handler.cancel_provider_orphans_in_background",
        lambda orphans, spec_for, on_cancelled: seen.append([o.provider_job_id for o in orphans]),
    )

    handler.recover_on_boot()

    assert seen == [["req-1"]]
    reloaded = store.get_generation(claimed.id)
    assert reloaded is not None and reloaded.status == "failed"
    assert reloaded.provider_job_id == "req-1"


def test_a_job_on_a_lane_that_cannot_cancel_is_refused_a_cancel_while_it_runs(
    test_state,
) -> None:
    # Bug caught: a cancel is accepted for a billed provider job, and the finished output is deleted.
    handler = test_state.queued_generations
    store = handler._db
    claimed = _claim_remote(store)
    billed = LaneSpec(
        id=TEST_LANE, concurrency=1, uses_gpu=False, cancellable_while_running=False
    )
    handler._executor_registry = ExecutorRegistry(
        {}, LaneRegistry([LOCAL_LANE_SPEC, billed])
    )

    with pytest.raises(HTTPError) as refused:
        handler.cancel_generation(claimed.id)

    assert refused.value.status_code == 409
    reloaded = store.get_generation(claimed.id)
    assert reloaded is not None and reloaded.status == "running"


class _Cancels:
    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[str] = []
        self._fail_on = fail_on

    def __call__(self, provider_job_id: str) -> None:
        self.calls.append(provider_job_id)
        if provider_job_id == self._fail_on:
            raise RuntimeError("provider down")


def test_each_orphan_is_cancelled_by_its_own_lane_and_one_failure_does_not_stop_the_rest(
    tmp_path: Path,
) -> None:
    # Bug caught: one failed cancel leaves every later orphan running and billed.
    store = SqliteStore(tmp_path / "app_data")
    for provider_id in ("req-1", "req-2"):
        claimed = _claim_remote(store)
        store.set_provider_job_id(claimed.id, claimed.attempt_count, provider_id)
    cancels = _Cancels(fail_on="req-1")
    lane = LaneSpec(
        id=TEST_LANE,
        concurrency=1,
        uses_gpu=False,
        cancellable_while_running=True,
        cancel_orphan=cancels,
    )

    done = cancel_provider_orphans(
        store.list_provider_orphans(),
        LaneRegistry([LOCAL_LANE_SPEC, lane]).spec,
        forget_in(store),
    )

    assert cancels.calls == ["req-1", "req-2"]
    assert done == 1


def test_a_failed_provider_cancel_keeps_the_id_for_the_next_boot_and_a_good_one_clears_it(
    tmp_path: Path,
) -> None:
    # Bug caught: a failed provider cancel drops the id, so a billed job outlives every restart.
    store = SqliteStore(tmp_path / "app_data")
    claimed = _claim_remote(store)
    store.set_provider_job_id(claimed.id, claimed.attempt_count, "req-1")
    store.fail_running_on_boot()
    provider = _Cancels(fail_on="req-1")
    lane = LaneSpec(
        id=TEST_LANE, concurrency=1, uses_gpu=False, cancellable_while_running=True,
        cancel_orphan=provider,
    )
    spec_for = LaneRegistry([LOCAL_LANE_SPEC, lane]).spec

    cancel_provider_orphans(store.list_provider_orphans(), spec_for, forget_in(store))
    assert [o.provider_job_id for o in store.list_provider_orphans()] == ["req-1"]

    provider._fail_on = None
    cancel_provider_orphans(store.list_provider_orphans(), spec_for, forget_in(store))
    assert store.list_provider_orphans() == []


def test_orphans_of_a_lane_without_a_cancel_are_skipped_and_keep_their_id(tmp_path: Path) -> None:
    # Bug caught: a removed provider lane raises at boot, or its job id is dropped before the lane returns.
    store = SqliteStore(tmp_path / "app_data")
    claimed = _claim_remote(store)
    store.set_provider_job_id(claimed.id, claimed.attempt_count, "req-1")
    store.fail_running_on_boot()
    orphans = store.list_provider_orphans()
    forget = forget_in(store)

    assert cancel_provider_orphans(orphans, LaneRegistry([LOCAL_LANE_SPEC, TEST_LANE_SPEC]).spec, forget) == 0
    assert cancel_provider_orphans(orphans, LaneRegistry([LOCAL_LANE_SPEC]).spec, forget) == 0
    assert [o.provider_job_id for o in store.list_provider_orphans()] == ["req-1"]


def test_queue_snapshot_shows_the_provider_job_id(client, test_state) -> None:
    # Bug caught: the id is stored but never reaches the API, so support cannot find the provider job.
    test_state.generation_queue.stop()
    store = test_state.queued_generations._db
    claimed = _claim_remote(store)
    store.set_provider_job_id(claimed.id, claimed.attempt_count, "req-1")

    body = client.get("/api/generation-queue").json()

    assert body["active"][0]["generation"]["provider_job_id"] == "req-1"
