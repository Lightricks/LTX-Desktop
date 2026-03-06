"""Tests for the persistent job queue."""

from __future__ import annotations

from pathlib import Path

from state.job_queue import JobQueue, QueueJob


def test_submit_job_assigns_id_and_status(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    job = queue.submit(
        job_type="video",
        model="seedance-1.5-pro",
        params={"prompt": "hello"},
        slot="api",
    )
    assert job.id
    assert job.status == "queued"
    assert job.slot == "api"
    assert job.progress == 0


def test_get_all_jobs_returns_ordered(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    j1 = queue.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")
    j2 = queue.submit(job_type="image", model="z-image-turbo", params={}, slot="gpu")
    jobs = queue.get_all_jobs()
    assert [j.id for j in jobs] == [j1.id, j2.id]


def test_next_queued_for_slot(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    queue.submit(job_type="video", model="seedance-1.5-pro", params={}, slot="api")
    queue.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")

    gpu_job = queue.next_queued_for_slot("gpu")
    assert gpu_job is not None
    assert gpu_job.slot == "gpu"

    api_job = queue.next_queued_for_slot("api")
    assert api_job is not None
    assert api_job.slot == "api"


def test_update_job_status(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    job = queue.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")
    queue.update_job(job.id, status="running", progress=50, phase="inference")
    updated = queue.get_job(job.id)
    assert updated is not None
    assert updated.status == "running"
    assert updated.progress == 50
    assert updated.phase == "inference"


def test_cancel_job(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    job = queue.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")
    queue.cancel_job(job.id)
    updated = queue.get_job(job.id)
    assert updated is not None
    assert updated.status == "cancelled"


def test_clear_finished_jobs(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    j1 = queue.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")
    j2 = queue.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")
    queue.update_job(j1.id, status="complete")
    queue.clear_finished()
    remaining = queue.get_all_jobs()
    assert len(remaining) == 1
    assert remaining[0].id == j2.id


def test_persistence_survives_reload(tmp_path: Path) -> None:
    path = tmp_path / "queue.json"
    queue1 = JobQueue(persistence_path=path)
    job = queue1.submit(job_type="video", model="ltx-fast", params={"prompt": "test"}, slot="gpu")

    queue2 = JobQueue(persistence_path=path)
    loaded = queue2.get_job(job.id)
    assert loaded is not None
    assert loaded.params == {"prompt": "test"}


def test_running_jobs_reset_to_queued_on_load(tmp_path: Path) -> None:
    path = tmp_path / "queue.json"
    queue1 = JobQueue(persistence_path=path)
    job = queue1.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")
    queue1.update_job(job.id, status="running")

    queue2 = JobQueue(persistence_path=path)
    loaded = queue2.get_job(job.id)
    assert loaded is not None
    assert loaded.status == "queued"
