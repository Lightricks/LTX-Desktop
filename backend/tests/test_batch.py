"""Tests for batch generation handler and API types."""

from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from api_types import (
    BatchJobItem,
    BatchReport,
    BatchSubmitRequest,
    BatchSubmitResponse,
    BatchStatusResponse,
    PipelineDefinition,
    PipelineStep,
    SweepAxis,
    SweepDefinition,
)
from handlers.batch_handler import BatchHandler
from state.job_queue import JobQueue


# --- Task 3: API types ---


def test_batch_submit_request_list_mode() -> None:
    req = BatchSubmitRequest(
        mode="list",
        target="local",
        jobs=[
            BatchJobItem(type="image", model="zit", params={"prompt": "a cat"}),
            BatchJobItem(type="image", model="zit", params={"prompt": "a dog"}),
        ],
    )
    assert req.mode == "list"
    assert len(req.jobs) == 2  # type: ignore[arg-type]


def test_batch_submit_request_sweep_mode() -> None:
    req = BatchSubmitRequest(
        mode="sweep",
        target="cloud",
        sweep=SweepDefinition(
            base_type="image",
            base_model="zit",
            base_params={"prompt": "a cat", "width": 1024, "height": 1024},
            axes=[
                SweepAxis(param="loraWeight", values=[0.5, 0.75, 1.0]),
                SweepAxis(param="prompt", values=["a cat", "a dog"], mode="search_replace", search="a cat"),
            ],
        ),
    )
    assert req.sweep is not None
    assert len(req.sweep.axes) == 2


def test_batch_submit_request_pipeline_mode() -> None:
    req = BatchSubmitRequest(
        mode="pipeline",
        target="local",
        pipeline=PipelineDefinition(
            steps=[
                PipelineStep(type="image", model="zit", params={"prompt": "a landscape"}),
                PipelineStep(type="video", model="fast", params={}, auto_prompt=True),
            ],
        ),
    )
    assert req.pipeline is not None
    assert req.pipeline.steps[1].auto_prompt is True


def test_batch_report_model() -> None:
    report = BatchReport(
        batch_id="abc123",
        total=10,
        succeeded=8,
        failed=2,
        cancelled=0,
        duration_seconds=120.5,
        avg_job_seconds=12.05,
        result_paths=["/out/1.png", "/out/2.png"],
        failed_indices=[3, 7],
        sweep_axes=["loraWeight"],
    )
    assert report.succeeded + report.failed == report.total


# --- Task 4: BatchHandler list mode ---


def test_batch_handler_expand_list(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    handler = BatchHandler()

    request = BatchSubmitRequest(
        mode="list",
        target="local",
        jobs=[
            BatchJobItem(type="image", model="zit", params={"prompt": "a cat"}),
            BatchJobItem(type="video", model="fast", params={"prompt": "a dog running"}),
        ],
    )
    response = handler.submit_batch(request, queue)

    assert response.total_jobs == 2
    assert len(response.job_ids) == 2

    jobs = queue.jobs_for_batch(response.batch_id)
    assert len(jobs) == 2
    assert jobs[0].type == "image"
    assert jobs[0].slot == "gpu"
    assert jobs[0].batch_index == 0
    assert f"batch:{response.batch_id}" in jobs[0].tags
    assert jobs[1].type == "video"
    assert jobs[1].batch_index == 1


def test_batch_handler_list_cloud_target(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    handler = BatchHandler()

    request = BatchSubmitRequest(
        mode="list",
        target="cloud",
        jobs=[BatchJobItem(type="image", model="zit", params={"prompt": "a cat"})],
    )
    response = handler.submit_batch(request, queue)
    job = queue.get_job(response.job_ids[0])
    assert job is not None
    assert job.slot == "api"


# --- Task 5: BatchHandler sweep mode ---


def test_batch_handler_sweep_single_axis(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    handler = BatchHandler()

    request = BatchSubmitRequest(
        mode="sweep",
        target="local",
        sweep=SweepDefinition(
            base_type="image",
            base_model="zit",
            base_params={"prompt": "a cat", "width": 1024, "height": 1024},
            axes=[SweepAxis(param="loraWeight", values=[0.5, 0.75, 1.0])],
        ),
    )
    response = handler.submit_batch(request, queue)

    assert response.total_jobs == 3
    jobs = queue.jobs_for_batch(response.batch_id)
    assert jobs[0].params["loraWeight"] == 0.5
    assert jobs[1].params["loraWeight"] == 0.75
    assert jobs[2].params["loraWeight"] == 1.0
    assert all(j.params["prompt"] == "a cat" for j in jobs)
    assert "sweep:loraWeight" in jobs[0].tags


def test_batch_handler_sweep_two_axes_cartesian(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    handler = BatchHandler()

    request = BatchSubmitRequest(
        mode="sweep",
        target="local",
        sweep=SweepDefinition(
            base_type="image",
            base_model="zit",
            base_params={"prompt": "a cat", "width": 1024},
            axes=[
                SweepAxis(param="loraWeight", values=[0.5, 1.0]),
                SweepAxis(param="numSteps", values=[4, 8]),
            ],
        ),
    )
    response = handler.submit_batch(request, queue)

    assert response.total_jobs == 4  # 2 x 2 cartesian product
    jobs = queue.jobs_for_batch(response.batch_id)
    combos = [(j.params["loraWeight"], j.params["numSteps"]) for j in jobs]
    assert (0.5, 4) in combos
    assert (0.5, 8) in combos
    assert (1.0, 4) in combos
    assert (1.0, 8) in combos


def test_batch_handler_sweep_search_replace(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    handler = BatchHandler()

    request = BatchSubmitRequest(
        mode="sweep",
        target="local",
        sweep=SweepDefinition(
            base_type="image",
            base_model="zit",
            base_params={"prompt": "a cute cat in a garden"},
            axes=[
                SweepAxis(
                    param="prompt",
                    values=["cat", "dog", "horse"],
                    mode="search_replace",
                    search="cat",
                ),
            ],
        ),
    )
    response = handler.submit_batch(request, queue)

    assert response.total_jobs == 3
    jobs = queue.jobs_for_batch(response.batch_id)
    assert jobs[0].params["prompt"] == "a cute cat in a garden"
    assert jobs[1].params["prompt"] == "a cute dog in a garden"
    assert jobs[2].params["prompt"] == "a cute horse in a garden"


# --- Task 6: BatchHandler pipeline mode ---


def test_batch_handler_pipeline_creates_chained_jobs(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    handler = BatchHandler()

    request = BatchSubmitRequest(
        mode="pipeline",
        target="local",
        pipeline=PipelineDefinition(
            steps=[
                PipelineStep(type="image", model="zit", params={"prompt": "a landscape"}),
                PipelineStep(type="video", model="fast", params={"duration": "4"}, auto_prompt=True),
            ],
        ),
    )
    response = handler.submit_batch(request, queue)

    assert response.total_jobs == 2
    jobs = queue.jobs_for_batch(response.batch_id)
    img_job = jobs[0]
    vid_job = jobs[1]

    assert img_job.type == "image"
    assert img_job.depends_on is None
    assert vid_job.type == "video"
    assert vid_job.depends_on == img_job.id
    assert vid_job.auto_params == {"imagePath": "$dep.result_paths[0]", "auto_prompt": "true"}


def test_batch_handler_pipeline_three_steps(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    handler = BatchHandler()

    request = BatchSubmitRequest(
        mode="pipeline",
        target="local",
        pipeline=PipelineDefinition(
            steps=[
                PipelineStep(type="image", model="zit", params={"prompt": "frame 1"}),
                PipelineStep(type="video", model="fast", params={}, auto_prompt=True),
                PipelineStep(type="video", model="pro", params={}, auto_prompt=False),
            ],
        ),
    )
    response = handler.submit_batch(request, queue)

    assert response.total_jobs == 3
    jobs = queue.jobs_for_batch(response.batch_id)
    assert jobs[0].depends_on is None
    assert jobs[1].depends_on == jobs[0].id
    assert jobs[2].depends_on == jobs[1].id
    assert jobs[2].auto_params == {"imagePath": "$dep.result_paths[0]"}  # No auto_prompt


# --- Task 9: Integration tests ---


def test_batch_submit_and_status_integration(client: TestClient) -> None:
    resp = client.post("/api/queue/submit-batch", json={
        "mode": "list",
        "target": "local",
        "jobs": [
            {"type": "image", "model": "zit", "params": {"prompt": "cat"}},
            {"type": "image", "model": "zit", "params": {"prompt": "dog"}},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    batch_id = data["batch_id"]
    assert data["total_jobs"] == 2

    resp = client.get(f"/api/queue/batch/{batch_id}/status")
    assert resp.status_code == 200
    status = resp.json()
    assert status["batch_id"] == batch_id
    assert status["total"] == 2
    assert status["queued"] == 2
    assert status["report"] is None


def test_batch_cancel_integration(client: TestClient) -> None:
    resp = client.post("/api/queue/submit-batch", json={
        "mode": "list",
        "target": "local",
        "jobs": [
            {"type": "image", "model": "zit", "params": {"prompt": "cat"}},
            {"type": "image", "model": "zit", "params": {"prompt": "dog"}},
        ],
    })
    batch_id = resp.json()["batch_id"]

    resp = client.post(f"/api/queue/batch/{batch_id}/cancel")
    assert resp.status_code == 200

    resp = client.get(f"/api/queue/batch/{batch_id}/status")
    status = resp.json()
    assert status["cancelled"] == 2


def test_batch_retry_failed_integration(client: TestClient) -> None:
    resp = client.post("/api/queue/submit-batch", json={
        "mode": "list",
        "target": "local",
        "jobs": [
            {"type": "image", "model": "zit", "params": {"prompt": "cat"}},
        ],
    })
    batch_id = resp.json()["batch_id"]

    resp = client.post(f"/api/queue/batch/{batch_id}/retry-failed")
    assert resp.status_code == 200


def test_queue_status_includes_batch_fields(client: TestClient) -> None:
    resp = client.post("/api/queue/submit-batch", json={
        "mode": "list",
        "target": "local",
        "jobs": [{"type": "image", "model": "zit", "params": {"prompt": "cat"}}],
    })
    batch_id = resp.json()["batch_id"]

    resp = client.get("/api/queue/status")
    data = resp.json()
    job = data["jobs"][0]
    assert job["batch_id"] == batch_id
    assert job["batch_index"] == 0
    assert f"batch:{batch_id}" in job["tags"]
