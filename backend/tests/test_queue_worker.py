"""Tests for the queue worker."""

from __future__ import annotations

from pathlib import Path

from state.job_queue import JobQueue, QueueJob
from handlers.queue_worker import QueueWorker


class FakeJobExecutor:
    def __init__(self, result_paths: list[str] | None = None) -> None:
        self.executed_jobs: list[QueueJob] = []
        self.raise_on_execute: Exception | None = None
        self._result_paths = result_paths if result_paths is not None else ["/fake/output.mp4"]

    def execute(self, job: QueueJob) -> list[str]:
        self.executed_jobs.append(job)
        if self.raise_on_execute is not None:
            raise self.raise_on_execute
        return list(self._result_paths)


class FakeEnhanceHandler:
    def __init__(self, result: str = "") -> None:
        self._result = result

    def enhance_i2v_motion(self, image_path: str) -> str:
        return self._result


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

    # tick() dispatches to a thread — wait for it to complete
    import time
    for _ in range(50):
        updated = queue.get_job(job.id)
        if updated is not None and updated.status == "error":
            break
        time.sleep(0.05)

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


# --- Task 7: Dependency checking ---


def test_worker_skips_job_with_unresolved_dependency(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    parent = queue.submit(job_type="image", model="zit", params={"prompt": "a cat"}, slot="gpu")
    child = queue.submit(
        job_type="video", model="fast", params={}, slot="gpu",
        depends_on=parent.id,
        auto_params={"imagePath": "$dep.result_paths[0]"},
    )

    executor = FakeJobExecutor(result_paths=[])
    worker = QueueWorker(queue=queue, gpu_executor=executor, api_executor=executor)
    worker.tick()

    import time
    time.sleep(0.1)  # Let thread finish

    # Parent was picked up and executed, child should still be queued
    # (parent completes but child wasn't dispatched in same tick)
    assert len(executor.executed_jobs) == 1
    assert executor.executed_jobs[0].id == parent.id
    c = queue.get_job(child.id)
    assert c is not None and c.status == "queued"


def test_worker_dispatches_dependent_job_after_parent_completes(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    parent = queue.submit(job_type="image", model="zit", params={"prompt": "a cat"}, slot="gpu")
    child = queue.submit(
        job_type="video", model="fast", params={}, slot="gpu",
        depends_on=parent.id,
        auto_params={"imagePath": "$dep.result_paths[0]"},
    )

    # Simulate parent completing
    queue.update_job(parent.id, status="complete", result_paths=["/out/cat.png"])

    executor = FakeJobExecutor(result_paths=["/out/video.mp4"])
    worker = QueueWorker(queue=queue, gpu_executor=executor, api_executor=executor)
    worker.tick()

    # Child should now be dispatched
    c = queue.get_job(child.id)
    assert c is not None and c.status in ("running", "complete")


def test_worker_fails_dependent_job_when_parent_errors(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    parent = queue.submit(job_type="image", model="zit", params={"prompt": "a cat"}, slot="gpu")
    child = queue.submit(
        job_type="video", model="fast", params={}, slot="gpu",
        depends_on=parent.id,
    )

    # Simulate parent failing
    queue.update_job(parent.id, status="error", error="GPU OOM")

    executor = FakeJobExecutor(result_paths=[])
    worker = QueueWorker(queue=queue, gpu_executor=executor, api_executor=executor)
    worker.tick()

    child_job = queue.get_job(child.id)
    assert child_job is not None
    assert child_job.status == "error"
    assert "Upstream job" in (child_job.error or "")


# --- Task 8: Batch completion detection ---


def test_worker_detects_batch_completion(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    j1 = queue.submit(job_type="image", model="zit", params={}, slot="gpu", batch_id="b1", batch_index=0)
    j2 = queue.submit(job_type="image", model="zit", params={}, slot="gpu", batch_id="b1", batch_index=1)

    completed_batches: list[str] = []

    def on_batch_complete(batch_id: str, jobs: list[QueueJob]) -> None:
        completed_batches.append(batch_id)

    executor = FakeJobExecutor(result_paths=["/out.png"])
    worker = QueueWorker(queue=queue, gpu_executor=executor, api_executor=executor, on_batch_complete=on_batch_complete)

    # Complete both jobs
    queue.update_job(j1.id, status="complete", result_paths=["/out/1.png"])
    queue.update_job(j2.id, status="complete", result_paths=["/out/2.png"])

    worker.tick()

    assert completed_batches == ["b1"]

    # Second tick should NOT re-notify
    worker.tick()
    assert completed_batches == ["b1"]


# --- Task 11: I2V auto-prompt ---


def test_worker_generates_i2v_prompt_for_auto_prompt_job(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")

    # Parent image job already complete
    parent = queue.submit(job_type="image", model="zit", params={"prompt": "a landscape"}, slot="gpu", batch_id="b1")
    queue.update_job(parent.id, status="complete", result_paths=["/out/landscape.png"])

    # Child video job with auto_prompt
    child = queue.submit(
        job_type="video", model="fast", params={"duration": "4"}, slot="gpu",
        batch_id="b1", depends_on=parent.id,
        auto_params={"imagePath": "$dep.result_paths[0]", "auto_prompt": "true"},
    )

    fake_enhance = FakeEnhanceHandler(result="The camera pans across mountains.")
    executor = FakeJobExecutor(result_paths=["/out/video.mp4"])
    worker = QueueWorker(
        queue=queue, gpu_executor=executor, api_executor=executor,
        enhance_handler=fake_enhance,
    )
    worker.tick()

    # Verify the child job got the auto-generated prompt
    child_job = queue.get_job(child.id)
    assert child_job is not None
    assert child_job.params.get("prompt") == "The camera pans across mountains."
    assert child_job.params.get("imagePath") == "/out/landscape.png"
