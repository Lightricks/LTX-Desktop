# Bulk Generation Design

**Date:** 2026-03-08
**Status:** Approved

## Overview

Add bulk image generation (with/without LoRAs), bulk video generation, and image-to-video pipeline chaining to LTX Desktop. Supports local GPU and cloud API execution. Three input modes: manual list, CSV/JSON import, and grid sweep builder.

## Architecture

Extends the existing job queue system. A batch is a group of `QueueJob` entries sharing a `batch_id`. Server-side expansion keeps frontend thin.

```
Frontend BatchBuilder → POST /api/queue/submit-batch → BatchHandler.submit_batch()
  → expands to N QueueJobs (list | sweep cartesian product | pipeline chain)
  → QueueWorker.tick() dispatches per slot with dependency checking
  → BatchReport generated on completion → sound + toast notification
```

## Data Model

### QueueJob Extensions

New fields on existing `QueueJob` dataclass:

| Field | Type | Purpose |
|-------|------|---------|
| `batch_id` | `str \| None` | Groups jobs from same batch |
| `batch_index` | `int` | Position within batch (grid ordering) |
| `depends_on` | `str \| None` | Job ID that must complete first |
| `auto_params` | `dict[str, str]` | Template refs resolved at dispatch, e.g. `{"imagePath": "$dep.result_paths[0]"}` |
| `tags` | `list[str]` | Gallery filtering, e.g. `["batch:abc123", "sweep:lora_weight"]` |

### API Types

```python
class BatchSubmitRequest(BaseModel):
    mode: Literal["list", "sweep", "pipeline"]
    target: Literal["local", "cloud"]
    jobs: list[BatchJobItem] | None = None          # mode: list
    sweep: SweepDefinition | None = None             # mode: sweep
    pipeline: PipelineDefinition | None = None       # mode: pipeline

class BatchJobItem(BaseModel):
    type: Literal["video", "image"]
    model: str
    params: dict[str, object] = {}

class SweepDefinition(BaseModel):
    base_type: Literal["video", "image"]
    base_model: str
    base_params: dict[str, object] = {}
    axes: list[SweepAxis]  # 1-3 axes

class SweepAxis(BaseModel):
    param: str                                       # e.g. "loraWeight", "prompt", "loraPath"
    values: list[object]                             # e.g. [0.5, 0.75, 1.0]
    mode: Literal["replace", "search_replace"] = "replace"
    search: str | None = None                        # For search_replace mode

class PipelineDefinition(BaseModel):
    steps: list[PipelineStep]

class PipelineStep(BaseModel):
    type: Literal["video", "image"]
    model: str
    params: dict[str, object] = {}
    auto_prompt: bool = False                        # Generate i2v motion prompt from previous step's image

class BatchSubmitResponse(BaseModel):
    batch_id: str
    job_ids: list[str]
    total_jobs: int

class BatchStatusResponse(BaseModel):
    batch_id: str
    total: int
    completed: int
    failed: int
    running: int
    queued: int
    jobs: list[QueueJobResponse]
    report: BatchReport | None = None                # Populated when batch fully resolved

class BatchReport(BaseModel):
    batch_id: str
    total: int
    succeeded: int
    failed: int
    cancelled: int
    duration_seconds: float
    avg_job_seconds: float
    result_paths: list[str]                          # Ordered by batch_index
    failed_indices: list[int]                        # Grid gaps
    sweep_axes: list[str] | None = None
```

## Backend: BatchHandler

New handler at `backend/handlers/batch_handler.py`.

```python
class BatchHandler:
    def submit_batch(self, request, queue) -> BatchSubmitResponse:
        batch_id = generate_id()
        slot = "api" if request.target == "cloud" else "gpu"
        match request.mode:
            case "list":    jobs = self._expand_list(request.jobs, batch_id, slot)
            case "sweep":   jobs = self._expand_sweep(request.sweep, batch_id, slot)
            case "pipeline": jobs = self._expand_pipeline(request.pipeline, batch_id, slot)
        for job in jobs:
            queue.submit(job)
        return BatchSubmitResponse(batch_id=batch_id, job_ids=[j.id for j in jobs], total_jobs=len(jobs))
```

- `_expand_list`: Maps each `BatchJobItem` to a `QueueJob` with shared `batch_id`.
- `_expand_sweep`: Computes cartesian product of axes. Each combination becomes a job with `batch_index` encoding grid position.
- `_expand_pipeline`: Creates chained jobs where step N has `depends_on` pointing to step N-1. For multi-image pipelines, each image spawns its own chain.

## Backend: QueueWorker Changes

### Dependency Checking

In `_next_job_for_slot()`, before dispatching a job:

```python
if job.depends_on:
    dep = queue.get(job.depends_on)
    if dep.status == "complete":
        resolve_auto_params(job, dep)  # Substitute $dep.result_paths[0] etc.
        return job  # Ready to run
    elif dep.status == "error":
        job.status = "error"
        job.error = f"Upstream job {dep.id} failed"
        continue  # Skip, check next
    else:
        continue  # Not ready yet, skip
else:
    return job  # No dependency, dispatch normally
```

### Batch Completion Detection

After each tick, check if any batch just became fully resolved:

```python
def _check_batch_completions(self):
    for batch_id in queue.active_batch_ids():
        jobs = queue.jobs_for_batch(batch_id)
        if all(j.status in ("complete", "error", "cancelled") for j in jobs):
            if batch_id not in self._notified_batches:
                self._notified_batches.add(batch_id)
                self._emit_batch_complete(batch_id, jobs)
```

### Failure Handling

- Individual job failure does NOT cancel the batch.
- Remaining jobs continue processing.
- `POST /api/queue/batch/{batch_id}/retry-failed` re-queues errored jobs with same params + fresh IDs.
- `POST /api/queue/batch/{batch_id}/cancel` cancels all `queued` jobs, aborts `running` if possible.

## Backend: I2V Auto-Prompt Generation

When a pipeline step has `auto_prompt: true`, the system generates a motion-focused video prompt from the previous step's image output.

### Enhancement Mode

New `i2v_motion` level in `EnhancePromptHandler`:

```python
class EnhancePromptRequest(BaseModel):
    prompt: str
    level: Literal["standard", "creative", "i2v_motion"] = "standard"
    image_path: str | None = None  # Required for i2v_motion
```

### I2V Motion System Prompt

> You are an expert cinematographer. Given an image, generate a video motion prompt. Rules:
> - Do NOT describe what the image already shows — describe what HAPPENS NEXT
> - Structure: subject motion, then camera movement, then environmental dynamics
> - Use specific cinematography terms: dolly, pan, tilt, orbit, truck, rack focus
> - Add motion intensity qualifiers: subtle, steady, dramatic
> - Animate empty regions (sky, water, foliage) to prevent frozen areas
> - Use present tense, single flowing paragraph, 4-6 sentences

### Two-Stage Flow

1. Caption the image via Gemini vision (or Palette enhance endpoint).
2. Feed caption + i2v_motion system prompt to LLM.
3. Inject generated motion prompt into the video job's params.

Executed in `QueueWorker._resolve_auto_params()` when `auto_params["auto_prompt"]` is set.

## Backend: Routes

New route file `backend/_routes/batch.py`:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/queue/submit-batch` | Submit a batch definition |
| `GET` | `/api/queue/batch/{batch_id}/status` | Batch aggregate status + report |
| `POST` | `/api/queue/batch/{batch_id}/retry-failed` | Re-queue failed jobs |
| `POST` | `/api/queue/batch/{batch_id}/cancel` | Cancel remaining jobs |

## Batch Completion: Sound + Report

### Sound

Frontend plays a short completion chime when polling detects a batch fully resolved. Respects `batchSoundEnabled` in app settings (default: on). Also triggers an Electron native notification if the app is in background.

### Report

`BatchReport` is populated on the `BatchStatusResponse` when all jobs are resolved. Contains success/fail counts, wall-clock duration, average job time, all result paths ordered by `batch_index`, and failed indices for grid gap display.

### Toast

Frontend shows: **"Batch complete — 47/50 succeeded (3 failed) in 12m 34s"** with a "View Results" button that filters gallery to `batch:{batch_id}`.

## Frontend: Batch Builder UI

New modal or panel accessible from GenSpace. Three tabs:

### Tab 1: Manual List

Table with rows: prompt, type (image/video), model, LoRA, LoRA weight. "Add Row" button, duplicate, delete, drag-to-reorder. Each row inherits current GenSpace settings as defaults.

### Tab 2: CSV/JSON Import

File picker or paste textarea. Preview table with validation (highlight errors). Supported CSV columns: `prompt, type, model, lora_path, lora_weight, width, height, duration, fps, camera_motion, seed`. JSON supports `defaults` block + `jobs` array.

### Tab 3: Grid Builder (Sweeps)

Base prompt + settings at top. Up to 3 axis selectors: pick param, enter values (comma-separated or range syntax `0.3-1.0:8`). Live preview showing grid dimensions and total job count.

### Pipeline Toggle

Checkbox on any batch: "Also generate video from each image." When checked, each image job gets a chained video job with `auto_prompt: true`. User sets video model, duration, fps for the chained step.

### Batch Queue Panel

Existing queue panel gains batch grouping: collapsible headers showing "Batch: 12/50 complete." Per-batch actions: cancel, retry failed.

## Gallery Integration

Minimal changes. Jobs in a batch get tags (`batch:{batch_id}`, `sweep:{param}`). Gallery gains a filter dropdown for batch tags. Results sorted by `batch_index` to preserve grid order.

## Per-Batch Execution Target

User picks "Run locally" or "Run on cloud" per batch. Maps to slot assignment: `target: "local"` → `slot: "gpu"`, `target: "cloud"` → `slot: "api"`. Cloud requires Palette API key or Replicate API key.

## Testing Strategy

- Unit tests for sweep expansion (cartesian product correctness).
- Unit tests for dependency resolution in QueueWorker.
- Integration tests for batch submit → poll → completion flow using fake services.
- Test partial failure: job 3/10 fails, jobs 4-10 continue, report shows 9/10 with 1 failed.
- Test pipeline: image job completes → video job auto-dispatches with resolved params.
- Test i2v auto-prompt: mock enhance handler, verify motion prompt injected.
- CSV/JSON parsing tests with edge cases (empty fields, invalid values, missing headers).

## Key Decisions

1. **Server-side expansion** over client-side — testable, atomic, consistent.
2. **`depends_on` single-parent** over full DAG — sufficient for i2v chains without engine complexity.
3. **Tags for gallery** over separate batch gallery view — minimal frontend changes, uses existing gallery.
4. **Per-batch target** over per-job target — simpler UX, avoids confusing mixed-slot batches.
5. **Sound + toast + report** on completion — easy wins for UX polish.
