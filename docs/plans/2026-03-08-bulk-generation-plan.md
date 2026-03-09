# Bulk Generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add bulk image generation (with/without LoRAs), bulk video generation, LoRA weight sweeps, CSV/JSON import, grid sweep builder, image-to-video pipeline chaining with auto-prompt generation, and batch completion notifications.

**Architecture:** Extends the existing job queue with `batch_id`, `depends_on`, and `auto_params` fields. New `BatchHandler` expands batch definitions server-side into individual `QueueJob` entries. `QueueWorker` gains dependency checking. Frontend adds a batch builder modal with three tabs (list, import, grid). Existing gallery gains batch tag filtering.

**Tech Stack:** Python/FastAPI (backend), React/TypeScript (frontend), existing JobQueue persistence, Gemini API for i2v prompt generation.

**Design doc:** `docs/plans/2026-03-08-bulk-generation-design.md`

---

## Task 1: Extend QueueJob Dataclass with Batch Fields

**Files:**
- Modify: `backend/state/job_queue.py:13-26` (QueueJob dataclass)
- Modify: `backend/state/job_queue.py:106-123` (persistence _save/_load)
- Test: `backend/tests/test_job_queue.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_job_queue.py — add new test

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
    loaded = queue2.get(job.id)
    assert loaded is not None
    assert loaded.batch_id == "batch_001"
    assert loaded.batch_index == 3
    assert loaded.depends_on == "job_abc"
    assert loaded.tags == ["batch:batch_001", "sweep:lora_weight"]
```

**Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_job_queue.py::test_submit_job_with_batch_fields -v --tb=short`
Expected: FAIL — `submit()` doesn't accept `batch_id`, `batch_index`, `depends_on`, `tags` params

**Step 3: Add batch fields to QueueJob dataclass**

In `backend/state/job_queue.py`, extend the `QueueJob` dataclass (line 13-26):

```python
@dataclass
class QueueJob:
    id: str
    type: Literal["video", "image"]
    model: str
    params: dict[str, Any]
    status: Literal["queued", "running", "complete", "error", "cancelled"]
    slot: Literal["gpu", "api"]
    progress: int = 0
    phase: str = "queued"
    result_paths: list[str] = field(default_factory=list)
    error: str | None = None
    created_at: str = ""
    # Batch fields
    batch_id: str | None = None
    batch_index: int = 0
    depends_on: str | None = None
    auto_params: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
```

**Step 4: Update `submit()` to accept new fields**

In `backend/state/job_queue.py`, update the `submit()` method signature to accept optional batch fields and pass them through to the QueueJob constructor.

**Step 5: Update `_load()` for backwards compatibility**

In the `_load()` method, when deserializing old jobs that lack batch fields, provide defaults:

```python
batch_id=d.get("batch_id"),
batch_index=d.get("batch_index", 0),
depends_on=d.get("depends_on"),
auto_params=d.get("auto_params", {}),
tags=d.get("tags", []),
```

**Step 6: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_job_queue.py::test_submit_job_with_batch_fields -v --tb=short`
Expected: PASS

**Step 7: Run all existing queue tests to verify no regressions**

Run: `cd backend && uv run pytest tests/test_job_queue.py -v --tb=short`
Expected: All PASS

**Step 8: Commit**

```bash
git add backend/state/job_queue.py backend/tests/test_job_queue.py
git commit -m "feat: add batch_id, depends_on, tags fields to QueueJob"
```

---

## Task 2: Add JobQueue Helper Methods for Batches

**Files:**
- Modify: `backend/state/job_queue.py`
- Test: `backend/tests/test_job_queue.py`

**Step 1: Write the failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_job_queue.py::test_jobs_for_batch tests/test_job_queue.py::test_active_batch_ids tests/test_job_queue.py::test_active_batch_ids_excludes_fully_resolved -v --tb=short`
Expected: FAIL — methods don't exist

**Step 3: Implement helper methods**

Add to `JobQueue` class:

```python
def jobs_for_batch(self, batch_id: str) -> list[QueueJob]:
    return sorted(
        [j for j in self._jobs if j.batch_id == batch_id],
        key=lambda j: j.batch_index,
    )

def active_batch_ids(self) -> list[str]:
    batch_ids: set[str] = set()
    for job in self._jobs:
        if job.batch_id and job.status in ("queued", "running"):
            batch_ids.add(job.batch_id)
    return sorted(batch_ids)

def get(self, job_id: str) -> QueueJob | None:
    for job in self._jobs:
        if job.id == job_id:
            return job
    return None
```

Note: Check if `get()` already exists. If so, skip adding it.

**Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_job_queue.py -v --tb=short`
Expected: All PASS

**Step 5: Commit**

```bash
git add backend/state/job_queue.py backend/tests/test_job_queue.py
git commit -m "feat: add jobs_for_batch, active_batch_ids, get helpers to JobQueue"
```

---

## Task 3: Add Batch API Types

**Files:**
- Modify: `backend/api_types.py:321-324` (after existing queue types)
- Test: `backend/tests/test_batch.py` (NEW)

**Step 1: Write the failing test**

```python
# backend/tests/test_batch.py

from backend.api_types import (
    BatchJobItem,
    BatchSubmitRequest,
    BatchSubmitResponse,
    BatchStatusResponse,
    BatchReport,
    SweepAxis,
    SweepDefinition,
    PipelineStep,
    PipelineDefinition,
)


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
    assert len(req.jobs) == 2


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
```

**Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_batch.py -v --tb=short`
Expected: FAIL — imports don't exist

**Step 3: Add batch types to api_types.py**

Add after the existing `QueueSubmitRequest` (around line 324):

```python
# --- Batch Generation Types ---

class BatchJobItem(BaseModel):
    type: Literal["video", "image"]
    model: str
    params: dict[str, object] = {}

class SweepAxis(BaseModel):
    param: str
    values: list[object]
    mode: Literal["replace", "search_replace"] = "replace"
    search: str | None = None

class SweepDefinition(BaseModel):
    base_type: Literal["video", "image"]
    base_model: str
    base_params: dict[str, object] = {}
    axes: list[SweepAxis]

class PipelineStep(BaseModel):
    type: Literal["video", "image"]
    model: str
    params: dict[str, object] = {}
    auto_prompt: bool = False

class PipelineDefinition(BaseModel):
    steps: list[PipelineStep]

class BatchSubmitRequest(BaseModel):
    mode: Literal["list", "sweep", "pipeline"]
    target: Literal["local", "cloud"]
    jobs: list[BatchJobItem] | None = None
    sweep: SweepDefinition | None = None
    pipeline: PipelineDefinition | None = None

class BatchSubmitResponse(BaseModel):
    batch_id: str
    job_ids: list[str]
    total_jobs: int

class BatchReport(BaseModel):
    batch_id: str
    total: int
    succeeded: int
    failed: int
    cancelled: int
    duration_seconds: float
    avg_job_seconds: float
    result_paths: list[str]
    failed_indices: list[int]
    sweep_axes: list[str] | None = None

class BatchStatusResponse(BaseModel):
    batch_id: str
    total: int
    completed: int
    failed: int
    running: int
    queued: int
    jobs: list[QueueJobResponse]
    report: BatchReport | None = None
```

**Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_batch.py -v --tb=short`
Expected: All PASS

**Step 5: Commit**

```bash
git add backend/api_types.py backend/tests/test_batch.py
git commit -m "feat: add batch generation API types"
```

---

## Task 4: Implement BatchHandler — List Mode

**Files:**
- Create: `backend/handlers/batch_handler.py`
- Test: `backend/tests/test_batch.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_batch.py — add

from backend.handlers.batch_handler import BatchHandler
from backend.state.job_queue import JobQueue


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
    assert jobs[0].tags == [f"batch:{response.batch_id}"]
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
    job = queue.get(response.job_ids[0])
    assert job is not None
    assert job.slot == "api"
```

**Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_batch.py::test_batch_handler_expand_list tests/test_batch.py::test_batch_handler_list_cloud_target -v --tb=short`
Expected: FAIL — module doesn't exist

**Step 3: Implement BatchHandler with list mode**

Create `backend/handlers/batch_handler.py`:

```python
from __future__ import annotations

import uuid

from backend.api_types import (
    BatchJobItem,
    BatchSubmitRequest,
    BatchSubmitResponse,
)
from backend.state.job_queue import JobQueue


class BatchHandler:
    def submit_batch(self, request: BatchSubmitRequest, queue: JobQueue) -> BatchSubmitResponse:
        batch_id = uuid.uuid4().hex[:8]
        slot = "api" if request.target == "cloud" else "gpu"

        if request.mode == "list":
            jobs = self._expand_list(request.jobs or [], batch_id, slot)
        elif request.mode == "sweep":
            raise NotImplementedError("sweep mode not yet implemented")
        elif request.mode == "pipeline":
            raise NotImplementedError("pipeline mode not yet implemented")
        else:
            raise ValueError(f"Unknown batch mode: {request.mode}")

        job_ids: list[str] = []
        for job_def in jobs:
            job = queue.submit(
                job_type=job_def["type"],
                model=job_def["model"],
                params=job_def["params"],
                slot=slot,
                batch_id=batch_id,
                batch_index=job_def["batch_index"],
                tags=[f"batch:{batch_id}"],
            )
            job_ids.append(job.id)

        return BatchSubmitResponse(batch_id=batch_id, job_ids=job_ids, total_jobs=len(job_ids))

    def _expand_list(
        self, items: list[BatchJobItem], batch_id: str, slot: str
    ) -> list[dict[str, object]]:
        return [
            {
                "type": item.type,
                "model": item.model,
                "params": item.params,
                "batch_index": i,
            }
            for i, item in enumerate(items)
        ]
```

**Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_batch.py -v --tb=short`
Expected: All PASS

**Step 5: Commit**

```bash
git add backend/handlers/batch_handler.py backend/tests/test_batch.py
git commit -m "feat: implement BatchHandler with list mode expansion"
```

---

## Task 5: Implement BatchHandler — Sweep Mode

**Files:**
- Modify: `backend/handlers/batch_handler.py`
- Test: `backend/tests/test_batch.py`

**Step 1: Write the failing tests**

```python
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
    # All share base params
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
```

**Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_batch.py::test_batch_handler_sweep_single_axis tests/test_batch.py::test_batch_handler_sweep_two_axes_cartesian tests/test_batch.py::test_batch_handler_sweep_search_replace -v --tb=short`
Expected: FAIL — NotImplementedError

**Step 3: Implement sweep expansion**

Add to `BatchHandler`:

```python
import itertools

def _expand_sweep(
    self, sweep: SweepDefinition, batch_id: str, slot: str
) -> list[dict[str, object]]:
    # Build value lists per axis
    axis_values: list[list[tuple[str, object]]] = []
    for axis in sweep.axes:
        pairs: list[tuple[str, object]] = []
        for val in axis.values:
            pairs.append((axis.param, val))
        axis_values.append(pairs)

    # Cartesian product
    combos = list(itertools.product(*axis_values))
    results: list[dict[str, object]] = []
    for i, combo in enumerate(combos):
        params = dict(sweep.base_params)
        for param_name, value in combo:
            axis_def = next(a for a in sweep.axes if a.param == param_name)
            if axis_def.mode == "search_replace" and axis_def.search and param_name in params:
                current = str(params[param_name])
                params[param_name] = current.replace(axis_def.search, str(value))
            else:
                params[param_name] = value
        results.append({
            "type": sweep.base_type,
            "model": sweep.base_model,
            "params": params,
            "batch_index": i,
        })

    return results
```

Update `submit_batch()` to pass sweep tags and call `_expand_sweep`. Add sweep axis names to tags: `[f"batch:{batch_id}"] + [f"sweep:{a.param}" for a in sweep.axes]`.

**Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_batch.py -v --tb=short`
Expected: All PASS

**Step 5: Commit**

```bash
git add backend/handlers/batch_handler.py backend/tests/test_batch.py
git commit -m "feat: implement sweep mode with cartesian product and search/replace"
```

---

## Task 6: Implement BatchHandler — Pipeline Mode

**Files:**
- Modify: `backend/handlers/batch_handler.py`
- Test: `backend/tests/test_batch.py`

**Step 1: Write the failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_batch.py::test_batch_handler_pipeline_creates_chained_jobs tests/test_batch.py::test_batch_handler_pipeline_three_steps -v --tb=short`
Expected: FAIL — NotImplementedError

**Step 3: Implement pipeline expansion**

Add to `BatchHandler`:

```python
def _expand_pipeline(
    self, pipeline: PipelineDefinition, batch_id: str, slot: str
) -> list[dict[str, object]]:
    """Create chained jobs. Each step depends_on the previous step's job ID."""
    results: list[dict[str, object]] = []
    prev_job_id: str | None = None

    for i, step in enumerate(pipeline.steps):
        job_id = uuid.uuid4().hex[:8]
        auto_params: dict[str, str] = {}
        if prev_job_id is not None:
            auto_params["imagePath"] = "$dep.result_paths[0]"
            if step.auto_prompt:
                auto_params["auto_prompt"] = "true"

        results.append({
            "type": step.type,
            "model": step.model,
            "params": dict(step.params),
            "batch_index": i,
            "job_id": job_id,
            "depends_on": prev_job_id,
            "auto_params": auto_params,
        })
        prev_job_id = job_id

    return results
```

Update `submit_batch()` to pass `depends_on`, `auto_params`, and pre-generated `job_id` through to `queue.submit()`. This requires `queue.submit()` to accept an optional `job_id` override (modify `JobQueue.submit()` to accept `job_id: str | None = None` and use it if provided instead of generating a new one).

**Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_batch.py -v --tb=short`
Expected: All PASS

**Step 5: Commit**

```bash
git add backend/handlers/batch_handler.py backend/state/job_queue.py backend/tests/test_batch.py
git commit -m "feat: implement pipeline mode with depends_on chaining"
```

---

## Task 7: QueueWorker Dependency Checking

**Files:**
- Modify: `backend/handlers/queue_worker.py:34-62` (tick method)
- Test: `backend/tests/test_queue_worker.py`

**Step 1: Write the failing tests**

```python
# backend/tests/test_queue_worker.py — add

def test_worker_skips_job_with_unresolved_dependency(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    parent = queue.submit(job_type="image", model="zit", params={"prompt": "a cat"}, slot="gpu")
    child = queue.submit(
        job_type="video", model="fast", params={}, slot="gpu",
        depends_on=parent.id,
        auto_params={"imagePath": "$dep.result_paths[0]"},
    )

    executor = FakeExecutor(result_paths=[])
    worker = QueueWorker(queue=queue, gpu_executor=executor, api_executor=executor)
    worker.tick()

    # Parent should be running, child still queued (dependency not met)
    assert queue.get(parent.id).status == "running"
    assert queue.get(child.id).status == "queued"


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

    executor = FakeExecutor(result_paths=["/out/video.mp4"])
    worker = QueueWorker(queue=queue, gpu_executor=executor, api_executor=executor)
    worker.tick()

    # Child should now be dispatched with resolved params
    assert queue.get(child.id).status in ("running", "complete")


def test_worker_fails_dependent_job_when_parent_errors(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    parent = queue.submit(job_type="image", model="zit", params={"prompt": "a cat"}, slot="gpu")
    child = queue.submit(
        job_type="video", model="fast", params={}, slot="gpu",
        depends_on=parent.id,
    )

    # Simulate parent failing
    queue.update_job(parent.id, status="error", error="GPU OOM")

    executor = FakeExecutor(result_paths=[])
    worker = QueueWorker(queue=queue, gpu_executor=executor, api_executor=executor)
    worker.tick()

    child_job = queue.get(child.id)
    assert child_job.status == "error"
    assert "Upstream job" in child_job.error
```

Note: You may need to create a `FakeExecutor` class in the test file or in `tests/fakes/` that implements the `JobExecutor` protocol. It should store result_paths and return them from `execute()`.

**Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_queue_worker.py::test_worker_skips_job_with_unresolved_dependency tests/test_queue_worker.py::test_worker_dispatches_dependent_job_after_parent_completes tests/test_queue_worker.py::test_worker_fails_dependent_job_when_parent_errors -v --tb=short`
Expected: FAIL

**Step 3: Modify QueueWorker to check dependencies**

In `backend/handlers/queue_worker.py`, modify the job selection logic in `tick()`. Replace direct `queue.next_queued_for_slot(slot)` with a new method that checks `depends_on`:

```python
def _next_ready_job(self, slot: str) -> QueueJob | None:
    for job in self._queue.queued_jobs_for_slot(slot):
        if job.depends_on is None:
            return job
        dep = self._queue.get(job.depends_on)
        if dep is None:
            return job  # Dependency missing, run anyway
        if dep.status == "complete":
            self._resolve_auto_params(job, dep)
            return job
        if dep.status in ("error", "cancelled"):
            self._queue.update_job(
                job.id,
                status="error",
                error=f"Upstream job {dep.id} failed: {dep.error or dep.status}",
            )
            continue
        # dep still queued/running — skip this job for now
        continue
    return None

def _resolve_auto_params(self, job: QueueJob, dep: QueueJob) -> None:
    for key, template in job.auto_params.items():
        if key == "auto_prompt":
            continue  # Handled by i2v prompt generation later
        if template == "$dep.result_paths[0]" and dep.result_paths:
            job.params[key] = dep.result_paths[0]
```

Note: `queued_jobs_for_slot(slot)` may need to be added to `JobQueue` — it returns all jobs with `status == "queued"` and matching `slot`, ordered by creation.

**Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_queue_worker.py -v --tb=short`
Expected: All PASS

**Step 5: Run all backend tests**

Run: `cd backend && uv run pytest -v --tb=short`
Expected: All PASS

**Step 6: Commit**

```bash
git add backend/handlers/queue_worker.py backend/state/job_queue.py backend/tests/test_queue_worker.py
git commit -m "feat: add dependency checking to QueueWorker dispatch"
```

---

## Task 8: Batch Completion Detection

**Files:**
- Modify: `backend/handlers/queue_worker.py`
- Test: `backend/tests/test_queue_worker.py`

**Step 1: Write the failing test**

```python
def test_worker_detects_batch_completion(tmp_path: Path) -> None:
    queue = JobQueue(persistence_path=tmp_path / "queue.json")
    j1 = queue.submit(job_type="image", model="zit", params={}, slot="gpu", batch_id="b1", batch_index=0)
    j2 = queue.submit(job_type="image", model="zit", params={}, slot="gpu", batch_id="b1", batch_index=1)

    completed_batches: list[str] = []
    def on_batch_complete(batch_id: str, jobs: list[QueueJob]) -> None:
        completed_batches.append(batch_id)

    executor = FakeExecutor(result_paths=["/out.png"])
    worker = QueueWorker(queue=queue, gpu_executor=executor, api_executor=executor, on_batch_complete=on_batch_complete)

    # Complete both jobs
    queue.update_job(j1.id, status="complete", result_paths=["/out/1.png"])
    queue.update_job(j2.id, status="complete", result_paths=["/out/2.png"])

    worker.tick()

    assert completed_batches == ["b1"]

    # Second tick should NOT re-notify
    worker.tick()
    assert completed_batches == ["b1"]
```

**Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_queue_worker.py::test_worker_detects_batch_completion -v --tb=short`
Expected: FAIL — on_batch_complete param doesn't exist

**Step 3: Add batch completion checking to QueueWorker**

Add `on_batch_complete` callback to `__init__`, track `_notified_batches: set[str]`, and call `_check_batch_completions()` at end of `tick()`:

```python
def _check_batch_completions(self) -> None:
    for batch_id in self._queue.active_batch_ids():
        # active_batch_ids only returns batches with queued/running jobs, so skip
        pass
    # Instead, check ALL batch ids that have jobs
    seen: set[str] = set()
    for job in self._queue.all_jobs():
        if job.batch_id and job.batch_id not in self._notified_batches:
            seen.add(job.batch_id)
    for batch_id in seen:
        jobs = self._queue.jobs_for_batch(batch_id)
        if all(j.status in ("complete", "error", "cancelled") for j in jobs):
            self._notified_batches.add(batch_id)
            if self._on_batch_complete:
                self._on_batch_complete(batch_id, jobs)
```

Note: Need to add `all_jobs()` method to JobQueue if not present (returns `list(self._jobs)`).

**Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_queue_worker.py -v --tb=short`
Expected: All PASS

**Step 5: Commit**

```bash
git add backend/handlers/queue_worker.py backend/state/job_queue.py backend/tests/test_queue_worker.py
git commit -m "feat: add batch completion detection to QueueWorker"
```

---

## Task 9: Batch Status & Report Endpoint

**Files:**
- Modify: `backend/handlers/batch_handler.py`
- Create: `backend/_routes/batch.py`
- Modify: `backend/app_factory.py:79-96` (router registration)
- Modify: `backend/app_handler.py:247-278` (add BatchHandler)
- Test: `backend/tests/test_batch.py`

**Step 1: Write the failing integration test**

```python
# backend/tests/test_batch.py — add integration tests using TestClient

import pytest
from starlette.testclient import TestClient


def test_batch_submit_and_status_integration(client: TestClient) -> None:
    # Submit a batch
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

    # Check batch status
    resp = client.get(f"/api/queue/batch/{batch_id}/status")
    assert resp.status_code == 200
    status = resp.json()
    assert status["batch_id"] == batch_id
    assert status["total"] == 2
    assert status["queued"] == 2
    assert status["report"] is None  # Not yet complete


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
    job_id = resp.json()["job_ids"][0]

    # Simulate failure via direct queue manipulation (in real test, use handler)
    # This needs the test_state fixture to access the queue
    # For now, test the route exists and returns 200
    resp = client.post(f"/api/queue/batch/{batch_id}/retry-failed")
    assert resp.status_code == 200
```

Note: These tests require the `client` fixture from `conftest.py`. You may need to add `batch_handler` to `AppHandler.__init__()` and register the batch router in `app_factory.py`.

**Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_batch.py::test_batch_submit_and_status_integration -v --tb=short`
Expected: FAIL — route doesn't exist

**Step 3: Add batch_status and batch report methods to BatchHandler**

```python
def get_batch_status(self, batch_id: str, queue: JobQueue) -> BatchStatusResponse:
    jobs = queue.jobs_for_batch(batch_id)
    if not jobs:
        raise HTTPError(status_code=404, detail=f"Batch {batch_id} not found")

    completed = sum(1 for j in jobs if j.status == "complete")
    failed = sum(1 for j in jobs if j.status == "error")
    running = sum(1 for j in jobs if j.status == "running")
    cancelled = sum(1 for j in jobs if j.status == "cancelled")
    queued = sum(1 for j in jobs if j.status == "queued")

    report = None
    if queued == 0 and running == 0:
        report = self._build_report(batch_id, jobs)

    return BatchStatusResponse(
        batch_id=batch_id,
        total=len(jobs),
        completed=completed,
        failed=failed,
        running=running,
        queued=queued,
        jobs=[self._job_to_response(j) for j in jobs],
        report=report,
    )

def cancel_batch(self, batch_id: str, queue: JobQueue) -> None:
    for job in queue.jobs_for_batch(batch_id):
        if job.status == "queued":
            queue.update_job(job.id, status="cancelled")

def retry_failed(self, batch_id: str, queue: JobQueue) -> BatchSubmitResponse:
    failed_jobs = [j for j in queue.jobs_for_batch(batch_id) if j.status == "error"]
    new_ids: list[str] = []
    for job in failed_jobs:
        new_job = queue.submit(
            job_type=job.type, model=job.model, params=job.params,
            slot=job.slot, batch_id=batch_id, batch_index=job.batch_index,
            tags=job.tags,
        )
        new_ids.append(new_job.id)
    return BatchSubmitResponse(batch_id=batch_id, job_ids=new_ids, total_jobs=len(new_ids))
```

**Step 4: Create batch routes**

Create `backend/_routes/batch.py`:

```python
from fastapi import APIRouter, Depends
from backend.api_types import BatchSubmitRequest, BatchSubmitResponse, BatchStatusResponse
from backend.state.deps import get_state_service

batch_router = APIRouter(prefix="/api/queue", tags=["batch"])

@batch_router.post("/submit-batch", response_model=BatchSubmitResponse)
def submit_batch(request: BatchSubmitRequest, handler=Depends(get_state_service)):
    return handler.batch.submit_batch(request, handler.job_queue)

@batch_router.get("/batch/{batch_id}/status", response_model=BatchStatusResponse)
def batch_status(batch_id: str, handler=Depends(get_state_service)):
    return handler.batch.get_batch_status(batch_id, handler.job_queue)

@batch_router.post("/batch/{batch_id}/cancel")
def batch_cancel(batch_id: str, handler=Depends(get_state_service)):
    handler.batch.cancel_batch(batch_id, handler.job_queue)
    return {"status": "cancelled"}

@batch_router.post("/batch/{batch_id}/retry-failed", response_model=BatchSubmitResponse)
def batch_retry(batch_id: str, handler=Depends(get_state_service)):
    return handler.batch.retry_failed(batch_id, handler.job_queue)
```

**Step 5: Wire BatchHandler into AppHandler**

In `backend/app_handler.py`, add `from backend.handlers.batch_handler import BatchHandler` and `self.batch = BatchHandler()` after the other handler instantiations (around line 267).

**Step 6: Register batch router in app_factory.py**

In `backend/app_factory.py`, add `from backend._routes.batch import batch_router` and `app.include_router(batch_router)` alongside the other router registrations (around line 89).

**Step 7: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_batch.py -v --tb=short`
Expected: All PASS

**Step 8: Run all backend tests**

Run: `cd backend && uv run pytest -v --tb=short`
Expected: All PASS

**Step 9: Commit**

```bash
git add backend/_routes/batch.py backend/handlers/batch_handler.py backend/app_handler.py backend/app_factory.py backend/tests/test_batch.py
git commit -m "feat: add batch routes — submit, status, cancel, retry-failed"
```

---

## Task 10: I2V Motion Prompt Generation

**Files:**
- Modify: `backend/handlers/enhance_prompt_handler.py:102-141`
- Test: `backend/tests/test_enhance_prompt.py`

**Step 1: Write the failing test**

```python
# backend/tests/test_enhance_prompt.py — add

def test_enhance_prompt_i2v_motion_mode(client: TestClient, test_state: AppHandler) -> None:
    test_state.app_state.app_settings.gemini_api_key = "test-key"

    # Queue a fake Gemini response for image caption + motion prompt
    fake_http = test_state.services.http_client  # FakeHTTPClient
    # First call: image caption response
    fake_http.queue_post_response(FakeResponse(
        status_code=200,
        json_payload={
            "candidates": [{"content": {"parts": [{"text": "A serene mountain landscape at golden hour with a lake in the foreground."}]}}]
        },
    ))
    # Second call: motion prompt generation
    fake_http.queue_post_response(FakeResponse(
        status_code=200,
        json_payload={
            "candidates": [{"content": {"parts": [{"text": "The camera slowly pans across the mountain range as golden light shifts. Gentle ripples spread across the lake surface while distant birds glide overhead."}]}}]
        },
    ))

    resp = client.post("/api/prompt/enhance", json={
        "prompt": "",
        "level": "i2v_motion",
        "image_path": "/path/to/landscape.png",
    })
    assert resp.status_code == 200
    result = resp.json()
    assert "camera" in result["enhanced_prompt"].lower() or "pan" in result["enhanced_prompt"].lower()
```

Note: Adjust based on actual enhance prompt route path and response schema. Check `backend/_routes/` for the enhance prompt route.

**Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_enhance_prompt.py::test_enhance_prompt_i2v_motion_mode -v --tb=short`
Expected: FAIL — `i2v_motion` level not recognized

**Step 3: Add i2v_motion mode to EnhancePromptHandler**

In `backend/handlers/enhance_prompt_handler.py`, add a new branch in the enhance method for `level == "i2v_motion"`. Add a dedicated method `_enhance_i2v_motion()`:

```python
I2V_MOTION_SYSTEM_PROMPT = """You are an expert cinematographer. Given a description of a still image, generate a video motion prompt that describes what happens next.

Rules:
- Do NOT describe what the image already shows — describe what HAPPENS NEXT
- Structure: subject motion → camera movement → environmental dynamics
- Use specific cinematography terms: dolly, pan, tilt, orbit, truck, rack focus, crane
- Add motion intensity qualifiers: subtle, steady, dramatic, gentle
- Animate empty regions (sky, water, foliage) to prevent frozen areas
- Use present tense, single flowing paragraph, 4-6 sentences
- Focus on plausible, physics-grounded motion"""

async def _enhance_i2v_motion(self, image_path: str) -> str:
    # Step 1: Caption the image
    caption = await self._caption_image(image_path)
    # Step 2: Generate motion prompt from caption
    motion_prompt = await self._generate_motion_prompt(caption)
    return motion_prompt
```

The implementation depends on the existing Gemini calling pattern (lines 102-141). Follow the same HTTP client pattern but with the i2v system prompt.

**Step 4: Update EnhancePromptRequest to accept image_path**

In `api_types.py`, add `image_path: str | None = None` to the enhance prompt request model.

**Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_enhance_prompt.py -v --tb=short`
Expected: All PASS

**Step 6: Commit**

```bash
git add backend/handlers/enhance_prompt_handler.py backend/api_types.py backend/tests/test_enhance_prompt.py
git commit -m "feat: add i2v_motion prompt generation mode"
```

---

## Task 11: Wire I2V Auto-Prompt into QueueWorker

**Files:**
- Modify: `backend/handlers/queue_worker.py`
- Test: `backend/tests/test_queue_worker.py`

**Step 1: Write the failing test**

```python
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

    # Create a fake enhance handler that returns a canned motion prompt
    fake_enhance = FakeEnhanceHandler(result="The camera pans across mountains.")
    executor = FakeExecutor(result_paths=["/out/video.mp4"])
    worker = QueueWorker(
        queue=queue, gpu_executor=executor, api_executor=executor,
        enhance_handler=fake_enhance,
    )
    worker.tick()

    # Verify the child job got the auto-generated prompt
    child_job = queue.get(child.id)
    assert child_job.params.get("prompt") == "The camera pans across mountains."
    assert child_job.params.get("imagePath") == "/out/landscape.png"
```

Note: `FakeEnhanceHandler` is a simple class with an `enhance_i2v_motion(image_path)` method returning a canned string.

**Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_queue_worker.py::test_worker_generates_i2v_prompt_for_auto_prompt_job -v --tb=short`
Expected: FAIL

**Step 3: Add auto-prompt resolution to QueueWorker**

In `_resolve_auto_params()`, add handling for `auto_prompt`:

```python
def _resolve_auto_params(self, job: QueueJob, dep: QueueJob) -> None:
    for key, template in list(job.auto_params.items()):
        if template == "$dep.result_paths[0]" and dep.result_paths:
            job.params[key] = dep.result_paths[0]

    if job.auto_params.get("auto_prompt") == "true" and self._enhance_handler:
        image_path = job.params.get("imagePath", dep.result_paths[0] if dep.result_paths else "")
        if image_path:
            motion_prompt = self._enhance_handler.enhance_i2v_motion(str(image_path))
            job.params["prompt"] = motion_prompt
```

Add `enhance_handler` as an optional `__init__` parameter.

**Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_queue_worker.py -v --tb=short`
Expected: All PASS

**Step 5: Commit**

```bash
git add backend/handlers/queue_worker.py backend/tests/test_queue_worker.py
git commit -m "feat: wire i2v auto-prompt generation into QueueWorker"
```

---

## Task 12: Add batchSoundEnabled Setting

**Files:**
- Modify: `backend/state/app_settings.py:62-91`
- Test: `backend/tests/test_settings.py`

**Step 1: Write the failing test**

```python
def test_batch_sound_enabled_default(client: TestClient) -> None:
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert data["batchSoundEnabled"] is True  # Default on
```

**Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_settings.py::test_batch_sound_enabled_default -v --tb=short`
Expected: FAIL — field doesn't exist

**Step 3: Add field to AppSettings**

In `backend/state/app_settings.py`, add to `AppSettings` class:

```python
batch_sound_enabled: bool = True
```

Add corresponding field to `SettingsResponse`:

```python
batch_sound_enabled: bool
```

Update `to_settings_response()` to include it.

**Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_settings.py -v --tb=short`
Expected: All PASS

**Step 5: Commit**

```bash
git add backend/state/app_settings.py backend/tests/test_settings.py
git commit -m "feat: add batchSoundEnabled setting (default true)"
```

---

## Task 13: Extend QueueJobResponse with Batch Fields

**Files:**
- Modify: `backend/api_types.py:240-251`
- Modify: `backend/_routes/queue.py:29-41` (status mapping)
- Test: `backend/tests/test_batch.py`

**Step 1: Write the failing test**

```python
def test_queue_status_includes_batch_fields(client: TestClient) -> None:
    # Submit a batch
    resp = client.post("/api/queue/submit-batch", json={
        "mode": "list",
        "target": "local",
        "jobs": [{"type": "image", "model": "zit", "params": {"prompt": "cat"}}],
    })
    batch_id = resp.json()["batch_id"]

    # Check regular queue status
    resp = client.get("/api/queue/status")
    data = resp.json()
    job = data["jobs"][0]
    assert job["batch_id"] == batch_id
    assert job["batch_index"] == 0
    assert "batch:" in job["tags"][0]
```

**Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_batch.py::test_queue_status_includes_batch_fields -v --tb=short`
Expected: FAIL — fields not in response

**Step 3: Add batch fields to QueueJobResponse**

In `backend/api_types.py`, extend `QueueJobResponse`:

```python
class QueueJobResponse(BaseModel):
    id: str
    type: str
    model: str
    params: dict[str, object] = {}
    status: str
    slot: str
    progress: int
    phase: str
    result_paths: list[str] = []
    error: str | None = None
    created_at: str = ""
    # Batch fields
    batch_id: str | None = None
    batch_index: int = 0
    tags: list[str] = []
```

Update the queue status route mapping in `backend/_routes/queue.py` to include the new fields when converting `QueueJob` to `QueueJobResponse`.

**Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_batch.py -v --tb=short`
Expected: All PASS

**Step 5: Commit**

```bash
git add backend/api_types.py backend/_routes/queue.py backend/tests/test_batch.py
git commit -m "feat: include batch fields in QueueJobResponse"
```

---

## Task 14: Frontend — Batch Types and API Client

**Files:**
- Modify: `frontend/hooks/use-generation.ts:5-17` (QueueJob interface)
- Create: `frontend/lib/batch-api.ts`
- Create: `frontend/types/batch.ts`

**Step 1: Add batch types**

Create `frontend/types/batch.ts`:

```typescript
export interface BatchJobItem {
  type: 'video' | 'image'
  model: string
  params: Record<string, unknown>
}

export interface SweepAxis {
  param: string
  values: unknown[]
  mode: 'replace' | 'search_replace'
  search?: string
}

export interface SweepDefinition {
  base_type: 'video' | 'image'
  base_model: string
  base_params: Record<string, unknown>
  axes: SweepAxis[]
}

export interface PipelineStep {
  type: 'video' | 'image'
  model: string
  params: Record<string, unknown>
  auto_prompt: boolean
}

export interface PipelineDefinition {
  steps: PipelineStep[]
}

export interface BatchSubmitRequest {
  mode: 'list' | 'sweep' | 'pipeline'
  target: 'local' | 'cloud'
  jobs?: BatchJobItem[]
  sweep?: SweepDefinition
  pipeline?: PipelineDefinition
}

export interface BatchSubmitResponse {
  batch_id: string
  job_ids: string[]
  total_jobs: number
}

export interface BatchReport {
  batch_id: string
  total: number
  succeeded: number
  failed: number
  cancelled: number
  duration_seconds: number
  avg_job_seconds: number
  result_paths: string[]
  failed_indices: number[]
  sweep_axes: string[] | null
}

export interface BatchStatusResponse {
  batch_id: string
  total: number
  completed: number
  failed: number
  running: number
  queued: number
  jobs: QueueJob[]
  report: BatchReport | null
}
```

**Step 2: Create batch API client**

Create `frontend/lib/batch-api.ts`:

```typescript
import type { BatchSubmitRequest, BatchSubmitResponse, BatchStatusResponse } from '@/types/batch'

const getBaseUrl = async (): Promise<string> => {
  if (window.electronAPI) {
    return await window.electronAPI.getBackendUrl()
  }
  return 'http://localhost:8000'
}

export async function submitBatch(request: BatchSubmitRequest): Promise<BatchSubmitResponse> {
  const base = await getBaseUrl()
  const resp = await fetch(`${base}/api/queue/submit-batch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!resp.ok) throw new Error(`Batch submit failed: ${resp.status}`)
  return resp.json()
}

export async function getBatchStatus(batchId: string): Promise<BatchStatusResponse> {
  const base = await getBaseUrl()
  const resp = await fetch(`${base}/api/queue/batch/${batchId}/status`)
  if (!resp.ok) throw new Error(`Batch status failed: ${resp.status}`)
  return resp.json()
}

export async function cancelBatch(batchId: string): Promise<void> {
  const base = await getBaseUrl()
  await fetch(`${base}/api/queue/batch/${batchId}/cancel`, { method: 'POST' })
}

export async function retryFailedBatch(batchId: string): Promise<BatchSubmitResponse> {
  const base = await getBaseUrl()
  const resp = await fetch(`${base}/api/queue/batch/${batchId}/retry-failed`, { method: 'POST' })
  if (!resp.ok) throw new Error(`Batch retry failed: ${resp.status}`)
  return resp.json()
}
```

**Step 3: Update QueueJob interface in use-generation.ts**

Add batch fields to the existing `QueueJob` interface:

```typescript
export interface QueueJob {
  // ... existing fields
  batch_id: string | null
  batch_index: number
  tags: string[]
}
```

**Step 4: Commit**

```bash
git add frontend/types/batch.ts frontend/lib/batch-api.ts frontend/hooks/use-generation.ts
git commit -m "feat: add frontend batch types and API client"
```

---

## Task 15: Frontend — useBatch Hook

**Files:**
- Create: `frontend/hooks/use-batch.ts`

**Step 1: Create the hook**

```typescript
import { useState, useRef, useCallback, useEffect } from 'react'
import type { BatchSubmitRequest, BatchStatusResponse, BatchReport } from '@/types/batch'
import { submitBatch, getBatchStatus, cancelBatch, retryFailedBatch } from '@/lib/batch-api'

export interface UseBatchReturn {
  activeBatchId: string | null
  batchStatus: BatchStatusResponse | null
  batchReport: BatchReport | null
  isRunning: boolean
  submit: (request: BatchSubmitRequest) => Promise<void>
  cancel: () => Promise<void>
  retryFailed: () => Promise<void>
  reset: () => void
}

export function useBatch(): UseBatchReturn {
  const [activeBatchId, setActiveBatchId] = useState<string | null>(null)
  const [batchStatus, setBatchStatus] = useState<BatchStatusResponse | null>(null)
  const [batchReport, setBatchReport] = useState<BatchReport | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const startPolling = useCallback((batchId: string) => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const status = await getBatchStatus(batchId)
        setBatchStatus(status)
        if (status.report) {
          setBatchReport(status.report)
          stopPolling()
          // Play completion sound
          playCompletionSound()
        }
      } catch {
        // Ignore polling errors
      }
    }, 1000) // Poll every 1s for batches (less aggressive than single job)
  }, [stopPolling])

  const submit = useCallback(async (request: BatchSubmitRequest) => {
    const response = await submitBatch(request)
    setActiveBatchId(response.batch_id)
    setBatchReport(null)
    startPolling(response.batch_id)
  }, [startPolling])

  const cancel = useCallback(async () => {
    if (activeBatchId) {
      await cancelBatch(activeBatchId)
    }
  }, [activeBatchId])

  const retryFailed = useCallback(async () => {
    if (activeBatchId) {
      await retryFailedBatch(activeBatchId)
      startPolling(activeBatchId)
    }
  }, [activeBatchId, startPolling])

  const reset = useCallback(() => {
    stopPolling()
    setActiveBatchId(null)
    setBatchStatus(null)
    setBatchReport(null)
  }, [stopPolling])

  useEffect(() => stopPolling, [stopPolling])

  const isRunning = batchStatus !== null && batchStatus.report === null

  return { activeBatchId, batchStatus, batchReport, isRunning, submit, cancel, retryFailed, reset }
}

function playCompletionSound(): void {
  try {
    const audio = new Audio('/sounds/batch-complete.mp3')
    audio.volume = 0.5
    audio.play().catch(() => {}) // Ignore autoplay restrictions
  } catch {
    // Sound not critical
  }
}
```

**Step 2: Commit**

```bash
git add frontend/hooks/use-batch.ts
git commit -m "feat: add useBatch hook with polling and completion sound"
```

---

## Task 16: Frontend — Batch Builder Modal (List Tab)

**Files:**
- Create: `frontend/components/BatchBuilderModal.tsx`
- Modify: `frontend/views/GenSpace.tsx` (add batch builder button)

**Step 1: Create BatchBuilderModal with List tab**

Create `frontend/components/BatchBuilderModal.tsx` with:
- Modal overlay with three tabs: List, Import, Grid
- List tab: table with prompt, type, model, LoRA fields
- Add Row / Delete Row / Duplicate Row buttons
- Per-batch target selector (Local / Cloud)
- Pipeline toggle: "Also generate video from each image"
- Submit button that calls `useBatch().submit()`

Keep the component focused — the List tab only. Import and Grid tabs render "Coming soon" placeholders for now (implemented in Tasks 17-18).

Follow the existing modal patterns in the codebase (check `frontend/components/SettingsModal.tsx` for the pattern).

Use Tailwind classes matching the app's existing dark theme. Refer to the global design standards in the CLAUDE.md for OKLCH colors and component patterns.

**Step 2: Add "Batch" button to GenSpace**

In `frontend/views/GenSpace.tsx`, add a button near the generate button that opens the BatchBuilderModal. Icon: grid/layers icon from Lucide.

**Step 3: Commit**

```bash
git add frontend/components/BatchBuilderModal.tsx frontend/views/GenSpace.tsx
git commit -m "feat: add batch builder modal with list tab"
```

---

## Task 17: Frontend — Batch Builder Import Tab

**Files:**
- Modify: `frontend/components/BatchBuilderModal.tsx`
- Create: `frontend/lib/batch-import.ts`

**Step 1: Create CSV/JSON parser**

Create `frontend/lib/batch-import.ts`:

```typescript
import type { BatchJobItem } from '@/types/batch'

export function parseCSV(text: string): BatchJobItem[] {
  const lines = text.trim().split('\n')
  if (lines.length < 2) return []
  const headers = lines[0].split(',').map(h => h.trim().toLowerCase())
  const promptIdx = headers.indexOf('prompt')
  if (promptIdx === -1) throw new Error('CSV must have a "prompt" column')

  return lines.slice(1).map(line => {
    const cols = parseCSVLine(line)
    const params: Record<string, unknown> = {}
    headers.forEach((h, i) => {
      if (h !== 'type' && h !== 'model' && cols[i]?.trim()) {
        params[h] = inferType(cols[i].trim())
      }
    })
    return {
      type: (cols[headers.indexOf('type')]?.trim() as 'video' | 'image') || 'image',
      model: cols[headers.indexOf('model')]?.trim() || 'zit',
      params,
    }
  })
}

export function parseJSON(text: string): BatchJobItem[] {
  const data = JSON.parse(text)
  const defaults = data.defaults || {}
  return (data.jobs || []).map((job: Record<string, unknown>) => ({
    type: job.type || defaults.type || 'image',
    model: job.model || defaults.model || 'zit',
    params: { ...defaults, ...job.params, prompt: job.prompt || '' },
  }))
}
```

Include a `parseCSVLine` helper that handles quoted fields with commas. Include an `inferType` helper that converts numeric strings to numbers.

**Step 2: Wire into Import tab**

Add textarea for pasting, file upload button, preview table, validation error display.

**Step 3: Commit**

```bash
git add frontend/lib/batch-import.ts frontend/components/BatchBuilderModal.tsx
git commit -m "feat: add CSV/JSON import tab to batch builder"
```

---

## Task 18: Frontend — Batch Builder Grid Tab (Sweeps)

**Files:**
- Modify: `frontend/components/BatchBuilderModal.tsx`

**Step 1: Build Grid tab UI**

Add to the Grid tab:
- Base prompt + settings section (inherits from current GenSpace settings)
- Up to 3 axis rows, each with:
  - Param selector dropdown (loraWeight, loraPath, prompt, numSteps, seed, cameraMotion, model)
  - Values input field (comma-separated, or range syntax `start-end:count`)
  - Remove axis button
- "Add Axis" button
- Live preview: "{X} x {Y} x {Z} = {total} jobs" with estimated time
- Range parser: `0.3-1.0:8` → `[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]`

**Step 2: Wire to sweep mode submission**

When user clicks Generate, build a `BatchSubmitRequest` with `mode: "sweep"` and the configured axes. Call `useBatch().submit()`.

**Step 3: Commit**

```bash
git add frontend/components/BatchBuilderModal.tsx
git commit -m "feat: add grid sweep builder tab to batch builder"
```

---

## Task 19: Frontend — Batch Queue Panel

**Files:**
- Modify: Wherever the queue/progress panel is rendered (likely in `GenSpace.tsx` or a queue component)

**Step 1: Add batch grouping to queue display**

When rendering the jobs list from `useGeneration().jobs`, group jobs by `batch_id`:
- Non-batch jobs render as today
- Batch jobs render under a collapsible header: "Batch {batch_id}: {completed}/{total} complete"
- Header shows aggregate progress bar
- Expand to see individual job rows
- Per-batch actions: Cancel Batch, Retry Failed (visible when batch has errors)

**Step 2: Add completion toast**

When `useBatch().batchReport` becomes non-null, show a toast notification:
- "Batch complete — {succeeded}/{total} succeeded ({failed} failed) in {duration}"
- "View Results" button that navigates to Gallery with `?batch={batch_id}` filter

**Step 3: Commit**

```bash
git add frontend/components/ frontend/views/GenSpace.tsx
git commit -m "feat: add batch grouping and completion toast to queue panel"
```

---

## Task 20: Frontend — Gallery Batch Filtering

**Files:**
- Modify: `frontend/views/Gallery.tsx`

**Step 1: Add batch tag filter**

Read `tags` from gallery items (already available via queue job results). Add a filter dropdown or pill selector that shows available batch tags. When selected, filter gallery items to those matching the tag.

Support URL param `?batch={batch_id}` for direct linking from completion toast.

**Step 2: Commit**

```bash
git add frontend/views/Gallery.tsx
git commit -m "feat: add batch tag filtering to gallery"
```

---

## Task 21: Add Completion Sound Asset

**Files:**
- Create: `public/sounds/batch-complete.mp3`

**Step 1: Source or generate a short completion chime**

Use a royalty-free chime sound (~1 second, pleasant, not jarring). Place at `public/sounds/batch-complete.mp3`.

If generating: use a simple ascending two-note chime (C5 → E5, ~0.8s, sine wave with decay).

**Step 2: Commit**

```bash
git add public/sounds/batch-complete.mp3
git commit -m "feat: add batch completion sound asset"
```

---

## Task 22: Typecheck and Full Test Pass

**Files:** All modified files

**Step 1: Run Python typecheck**

Run: `cd backend && uv run pyright`
Expected: No new errors. Fix any type issues introduced.

**Step 2: Run all backend tests**

Run: `cd backend && uv sync --frozen --extra test --extra dev && uv run pytest -v --tb=short`
Expected: All PASS

**Step 3: Run TypeScript typecheck**

Run: `pnpm typecheck:ts`
Expected: No errors. Fix any TS issues.

**Step 4: Run frontend build**

Run: `pnpm build:frontend`
Expected: Build succeeds.

**Step 5: Fix any issues found, commit**

```bash
git add -A
git commit -m "chore: fix typecheck and test issues from bulk generation"
```

---

## Summary

| Task | Component | Commits |
|------|-----------|---------|
| 1-2 | QueueJob batch fields + helpers | 2 |
| 3 | Batch API types | 1 |
| 4-6 | BatchHandler (list, sweep, pipeline) | 3 |
| 7-8 | QueueWorker dependency + completion | 2 |
| 9 | Batch routes + wiring | 1 |
| 10-11 | I2V auto-prompt generation | 2 |
| 12 | batchSoundEnabled setting | 1 |
| 13 | QueueJobResponse batch fields | 1 |
| 14-15 | Frontend types, API client, hook | 2 |
| 16-18 | Batch builder modal (3 tabs) | 3 |
| 19 | Queue panel batch grouping | 1 |
| 20 | Gallery batch filtering | 1 |
| 21 | Sound asset | 1 |
| 22 | Typecheck + test pass | 1 |
| **Total** | | **22 commits** |
