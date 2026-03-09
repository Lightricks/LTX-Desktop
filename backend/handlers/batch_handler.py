"""Batch generation handler — expands batch definitions into individual queue jobs."""

from __future__ import annotations

import itertools
import uuid
from typing import Any

from api_types import (
    BatchJobItem,
    BatchSubmitRequest,
    BatchSubmitResponse,
    BatchStatusResponse,
    BatchReport,
    QueueJobResponse,
    SweepDefinition,
    PipelineDefinition,
)
from state.job_queue import JobQueue, QueueJob


class BatchHandler:
    def submit_batch(self, request: BatchSubmitRequest, queue: JobQueue) -> BatchSubmitResponse:
        batch_id = uuid.uuid4().hex[:8]
        slot: str = "api" if request.target == "cloud" else "gpu"

        if request.mode == "list":
            job_defs = self._expand_list(request.jobs or [], batch_id, slot)
            tags = [f"batch:{batch_id}"]
        elif request.mode == "sweep":
            if request.sweep is None:
                raise ValueError("sweep mode requires sweep definition")
            job_defs = self._expand_sweep(request.sweep, batch_id, slot)
            tags = [f"batch:{batch_id}"] + [f"sweep:{a.param}" for a in request.sweep.axes]
        elif request.mode == "pipeline":
            if request.pipeline is None:
                raise ValueError("pipeline mode requires pipeline definition")
            return self._submit_pipeline(request.pipeline, batch_id, slot, queue)
        else:
            raise ValueError(f"Unknown batch mode: {request.mode}")

        job_ids: list[str] = []
        for job_def in job_defs:
            job = queue.submit(
                job_type=str(job_def["type"]),
                model=str(job_def["model"]),
                params=dict(job_def["params"]),  # type: ignore[arg-type]
                slot=slot,
                batch_id=batch_id,
                batch_index=int(job_def["batch_index"]),  # type: ignore[arg-type]
                tags=list(tags),
            )
            job_ids.append(job.id)

        return BatchSubmitResponse(batch_id=batch_id, job_ids=job_ids, total_jobs=len(job_ids))

    def get_batch_status(self, batch_id: str, queue: JobQueue) -> BatchStatusResponse:
        jobs = queue.jobs_for_batch(batch_id)
        if not jobs:
            raise ValueError(f"Batch {batch_id} not found")

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
            cancelled=cancelled,
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
                job_type=job.type,
                model=job.model,
                params=job.params,
                slot=job.slot,
                batch_id=batch_id,
                batch_index=job.batch_index,
                tags=job.tags,
            )
            new_ids.append(new_job.id)
        return BatchSubmitResponse(batch_id=batch_id, job_ids=new_ids, total_jobs=len(new_ids))

    def _expand_list(
        self, items: list[BatchJobItem], batch_id: str, slot: str
    ) -> list[dict[str, object]]:
        return [
            {
                "type": item.type,
                "model": item.model,
                "params": dict(item.params),
                "batch_index": i,
            }
            for i, item in enumerate(items)
        ]

    def _expand_sweep(
        self, sweep: SweepDefinition, batch_id: str, slot: str
    ) -> list[dict[str, object]]:
        axis_values: list[list[tuple[str, object]]] = []
        for axis in sweep.axes:
            pairs: list[tuple[str, object]] = []
            for val in axis.values:
                pairs.append((axis.param, val))
            axis_values.append(pairs)

        combos = list(itertools.product(*axis_values))
        results: list[dict[str, object]] = []
        for i, combo in enumerate(combos):
            params: dict[str, Any] = dict(sweep.base_params)
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

    def _submit_pipeline(
        self, pipeline: PipelineDefinition, batch_id: str, slot: str, queue: JobQueue
    ) -> BatchSubmitResponse:
        job_ids: list[str] = []
        prev_job_id: str | None = None
        tags = [f"batch:{batch_id}"]

        for i, step in enumerate(pipeline.steps):
            auto_params: dict[str, str] = {}
            depends_on: str | None = None
            if prev_job_id is not None:
                depends_on = prev_job_id
                auto_params["imagePath"] = "$dep.result_paths[0]"
                if step.auto_prompt:
                    auto_params["auto_prompt"] = "true"

            job = queue.submit(
                job_type=step.type,
                model=step.model,
                params=dict(step.params),
                slot=slot,
                batch_id=batch_id,
                batch_index=i,
                depends_on=depends_on,
                auto_params=auto_params,
                tags=list(tags),
            )
            job_ids.append(job.id)
            prev_job_id = job.id

        return BatchSubmitResponse(batch_id=batch_id, job_ids=job_ids, total_jobs=len(job_ids))

    def _build_report(self, batch_id: str, jobs: list[QueueJob]) -> BatchReport:
        succeeded = sum(1 for j in jobs if j.status == "complete")
        failed = sum(1 for j in jobs if j.status == "error")
        cancelled = sum(1 for j in jobs if j.status == "cancelled")
        result_paths: list[str] = []
        failed_indices: list[int] = []
        for j in jobs:
            result_paths.extend(j.result_paths)
            if j.status == "error":
                failed_indices.append(j.batch_index)

        sweep_axes: list[str] | None = None
        sweep_tags = [t for t in (jobs[0].tags if jobs else []) if t.startswith("sweep:")]
        if sweep_tags:
            sweep_axes = [t.split(":", 1)[1] for t in sweep_tags]

        return BatchReport(
            batch_id=batch_id,
            total=len(jobs),
            succeeded=succeeded,
            failed=failed,
            cancelled=cancelled,
            duration_seconds=0.0,
            avg_job_seconds=0.0,
            result_paths=result_paths,
            failed_indices=failed_indices,
            sweep_axes=sweep_axes,
        )

    def _job_to_response(self, job: QueueJob) -> QueueJobResponse:
        return QueueJobResponse(
            id=job.id,
            type=job.type,
            model=job.model,
            params=job.params,
            status=job.status,
            slot=job.slot,
            progress=job.progress,
            phase=job.phase,
            result_paths=job.result_paths,
            error=job.error,
            created_at=job.created_at,
            batch_id=job.batch_id,
            batch_index=job.batch_index,
            tags=job.tags,
        )
