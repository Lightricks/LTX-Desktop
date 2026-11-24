"""The GPU lease: a job of another lane borrows the GPU only when it is fully idle.

Each test names the bug it catches.
"""

from __future__ import annotations

import threading
import time

import pytest

from services.records import LOCAL_LANE


def _lease(test_state, *, local_lane_idle: bool = True):
    return test_state.generation.try_lease_idle_gpu(local_lane_idle=lambda: local_lane_idle)


def test_the_lease_is_granted_on_an_idle_machine_and_released_after(test_state) -> None:
    # Bug caught: the lease never releases and the next local job cannot start.
    generation = test_state.generation

    with _lease(test_state) as leased:
        assert leased
        assert generation.try_reserve_generation_start() is False

    assert generation.try_reserve_generation_start() is True
    generation.release_generation_start_reservation()


def test_the_lease_fails_while_a_local_job_holds_the_slot(test_state) -> None:
    # Bug caught: a fal rewrite loads Gemma while a local job runs.
    generation = test_state.generation
    assert generation.try_reserve_generation_start(persistent=True)

    with _lease(test_state) as leased:
        assert not leased

    # The denied lease must not release the reservation of the job that holds it.
    assert generation.try_reserve_generation_start() is False
    generation.release_generation_start_reservation()


def test_the_idle_check_runs_without_the_state_lock(test_state) -> None:
    # Bug caught: the SQLite read of the idle check stalls every handler that needs the lock.
    generation = test_state.generation
    free_for_others: list[bool] = []

    def idle_check() -> bool:
        def probe() -> None:
            acquired = generation.lock.acquire(blocking=False)
            free_for_others.append(acquired)
            if acquired:
                generation.lock.release()

        thread = threading.Thread(target=probe)
        thread.start()
        thread.join()
        return True

    with generation.try_lease_idle_gpu(local_lane_idle=idle_check) as leased:
        assert leased

    assert free_for_others == [True]


def test_the_lease_fails_while_the_local_lane_has_a_queued_row(test_state) -> None:
    # Bug caught: a fal rewrite takes the GPU between two local jobs and evicts the warm model.
    store = test_state.queued_generations._db
    store.insert_generation("text-to-video", {"params": {"prompt": "fox"}, "inputs": {}})

    assert test_state.generation_queue.lane_is_idle(LOCAL_LANE) is False
    with test_state.generation.try_lease_idle_gpu(
        local_lane_idle=lambda: test_state.generation_queue.lane_is_idle(LOCAL_LANE)
    ) as leased:
        assert not leased


def test_the_lease_fails_while_the_queue_yields_to_another_waiter(test_state) -> None:
    # Bug caught: the lease starves the assistant, which waits for the GPU through queue_yields.
    with test_state.generation.queue_yields():
        with _lease(test_state) as leased:
            assert not leased


def test_the_lease_fails_when_the_local_lane_reports_busy(test_state) -> None:
    # Bug caught: the lease ignores a running local job that has not taken the slot yet.
    with _lease(test_state, local_lane_idle=False) as leased:
        assert not leased
    assert test_state.generation.try_reserve_generation_start() is True
    test_state.generation.release_generation_start_reservation()


def test_the_lease_does_not_expire_during_a_long_rewrite(test_state, monkeypatch) -> None:
    # Bug caught: a timed reservation expires mid-rewrite and a local job starts on the same GPU.
    generation = test_state.generation
    with _lease(test_state) as leased:
        assert leased
        assert generation.state.generation_starting_since is None
        monkeypatch.setattr(time, "monotonic", lambda: 1e12)
        assert generation.try_reserve_generation_start() is False


def test_a_denied_lease_never_blocks_the_caller(test_state) -> None:
    # Bug caught: the lease waits for the slot and the fal job stalls behind a local job.
    generation = test_state.generation
    assert generation.try_reserve_generation_start(persistent=True)
    finished = threading.Event()

    def lease() -> None:
        with _lease(test_state):
            pass
        finished.set()

    threading.Thread(target=lease, daemon=True).start()

    assert finished.wait(timeout=2)
    generation.release_generation_start_reservation()


@pytest.mark.parametrize("busy", [False, True])
def test_a_granted_lease_evicts_the_diffusion_cache_and_a_denied_one_does_not(
    test_state, monkeypatch, busy: bool
) -> None:
    # Bug caught: a cancelled local video leaves its transformer cached, and Gemma loads next to it.
    evictions: list[int] = []
    monkeypatch.setattr(
        "handlers.generation_handler.diffusion_stage_cache.evict", lambda: evictions.append(1)
    )

    with _lease(test_state, local_lane_idle=not busy) as leased:
        assert leased is (not busy)

    assert len(evictions) == (0 if busy else 1)


def test_the_lease_is_released_when_the_body_raises(test_state) -> None:
    # Bug caught: a failed rewrite leaves the reservation held and no local job can start.
    generation = test_state.generation

    with pytest.raises(RuntimeError, match="rewrite failed"):
        with _lease(test_state) as leased:
            assert leased
            raise RuntimeError("rewrite failed")

    assert generation.try_reserve_generation_start() is True
    generation.release_generation_start_reservation()


def test_the_lease_is_released_when_the_cache_eviction_fails(test_state, monkeypatch) -> None:
    # Bug caught: evict() raises after the reservation was taken and the GPU stays reserved.
    def broken_evict() -> None:
        raise RuntimeError("evict failed")

    monkeypatch.setattr("handlers.generation_handler.diffusion_stage_cache.evict", broken_evict)

    with pytest.raises(RuntimeError, match="evict failed"):
        with _lease(test_state):
            pass

    assert test_state.generation.try_reserve_generation_start() is True
    test_state.generation.release_generation_start_reservation()
