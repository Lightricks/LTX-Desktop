"""Tests for the queue worker."""

from __future__ import annotations

from pathlib import Path

from state.job_queue import JobQueue, QueueJob
from handlers.queue_worker import QueueWorker


class FakeJobExecutor:
    def __init__(self) -> None:
        self.executed_jobs: list[QueueJob] = []
        self.raise_on_execute: Exception | None = None

    def execute(self, job: QueueJob) -> list[str]:
        self.executed_jobs.append(job)
        if self.raise_on_execute is not None:
            raise self.raise_on_execute
        return ["/fake/output.mp4"]


def test_worker_processes_gpu_job(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    job = queue.submit(job_type="video", model="ltx-fast", params={"prompt": "test"}, slot="gpu")

    executor = FakeJobExecutor()
    worker = QueueWorker(queue=queue, gpu_executor=executor, api_executor=FakeJobExecutor())
    worker.tick()

    assert len(executor.executed_jobs) == 1
    assert executor.executed_jobs[0].id == job.id
    updated = queue.get_job(job.id)
    assert updated is not None
    assert updated.status == "complete"
    assert updated.result_paths == ["/fake/output.mp4"]


def test_worker_processes_api_job(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    job = queue.submit(job_type="video", model="seedance-1.5-pro", params={"prompt": "test"}, slot="api")

    api_executor = FakeJobExecutor()
    worker = QueueWorker(queue=queue, gpu_executor=FakeJobExecutor(), api_executor=api_executor)
    worker.tick()

    assert len(api_executor.executed_jobs) == 1
    updated = queue.get_job(job.id)
    assert updated is not None
    assert updated.status == "complete"


def test_worker_handles_execution_error(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    job = queue.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")

    executor = FakeJobExecutor()
    executor.raise_on_execute = RuntimeError("GPU exploded")
    worker = QueueWorker(queue=queue, gpu_executor=executor, api_executor=FakeJobExecutor())
    worker.tick()

    updated = queue.get_job(job.id)
    assert updated is not None
    assert updated.status == "error"
    assert updated.error == "GPU exploded"


def test_worker_runs_gpu_and_api_in_parallel(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    gpu_job = queue.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")
    api_job = queue.submit(job_type="video", model="seedance-1.5-pro", params={}, slot="api")

    gpu_executor = FakeJobExecutor()
    api_executor = FakeJobExecutor()
    worker = QueueWorker(queue=queue, gpu_executor=gpu_executor, api_executor=api_executor)
    worker.tick()

    assert len(gpu_executor.executed_jobs) == 1
    assert len(api_executor.executed_jobs) == 1


def test_worker_skips_cancelled_job(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    job = queue.submit(job_type="video", model="ltx-fast", params={}, slot="gpu")
    queue.cancel_job(job.id)

    executor = FakeJobExecutor()
    worker = QueueWorker(queue=queue, gpu_executor=executor, api_executor=FakeJobExecutor())
    worker.tick()

    assert len(executor.executed_jobs) == 0
