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


def test_submit_job_with_batch_fields(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    job = queue.submit(
        job_type="image",
        model="zit",
        params={"prompt": "a cat"},
        slot="gpu",
        batch_id="batch_001",
        batch_index=3,
        depends_on="job_abc",
        tags=["batch:batch_001", "sweep:lora_weight"],
    )
    assert job.batch_id == "batch_001"
    assert job.batch_index == 3
    assert job.depends_on == "job_abc"
    assert job.tags == ["batch:batch_001", "sweep:lora_weight"]

    # Verify persistence round-trip
    queue2 = JobQueue(persistence_path=tmp_path / "queue.json")
    loaded = queue2.get_job(job.id)
    assert loaded is not None
    assert loaded.batch_id == "batch_001"
    assert loaded.batch_index == 3
    assert loaded.depends_on == "job_abc"
    assert loaded.tags == ["batch:batch_001", "sweep:lora_weight"]


def test_jobs_for_batch(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    queue.submit(job_type="image", model="zit", params={}, slot="gpu", batch_id="b1", batch_index=0)
    queue.submit(job_type="image", model="zit", params={}, slot="gpu", batch_id="b1", batch_index=1)
    queue.submit(job_type="image", model="zit", params={}, slot="gpu", batch_id="b2", batch_index=0)
    queue.submit(job_type="video", model="fast", params={}, slot="gpu")  # No batch

    batch_jobs = queue.jobs_for_batch("b1")
    assert len(batch_jobs) == 2
    assert all(j.batch_id == "b1" for j in batch_jobs)
    assert [j.batch_index for j in batch_jobs] == [0, 1]


def test_active_batch_ids(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    queue.submit(job_type="image", model="zit", params={}, slot="gpu", batch_id="b1")
    queue.submit(job_type="image", model="zit", params={}, slot="gpu", batch_id="b2")
    queue.submit(job_type="video", model="fast", params={}, slot="gpu")

    ids = queue.active_batch_ids()
    assert set(ids) == {"b1", "b2"}


def test_active_batch_ids_excludes_fully_resolved(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    job = queue.submit(job_type="image", model="zit", params={}, slot="gpu", batch_id="b1")
    queue.update_job(job.id, status="complete", result_paths=["/out.png"])

    ids = queue.active_batch_ids()
    assert ids == []  # b1 is fully resolved


def test_running_jobs_reset_to_queued_on_load(tmp_path: Path) -> None:
    path = tmp_path / "queue.json"
    queue1 = JobQueue(persistence_path=path)
    job = queue1.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")
    queue1.update_job(job.id, status="running")

    queue2 = JobQueue(persistence_path=path)
    loaded = queue2.get_job(job.id)
    assert loaded is not None
    assert loaded.status == "error"
    assert loaded.error == "Interrupted by app restart"
