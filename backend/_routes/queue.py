"""Route handlers for /api/queue/*."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import QueueSubmitRequest, QueueSubmitResponse, QueueStatusResponse, QueueJobResponse
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api/queue", tags=["queue"])


@router.post("/submit", response_model=QueueSubmitResponse)
def route_queue_submit(
    req: QueueSubmitRequest,
    handler: AppHandler = Depends(get_state_service),
) -> QueueSubmitResponse:
    slot = handler.determine_slot(req.model)
    job = handler.job_queue.submit(
        job_type=req.type,
        model=req.model,
        params={k: v for k, v in req.params.items()},
        slot=slot,
    )
    return QueueSubmitResponse(id=job.id, status=job.status)


@router.get("/status", response_model=QueueStatusResponse)
def route_queue_status(
    handler: AppHandler = Depends(get_state_service),
) -> QueueStatusResponse:
    jobs = handler.job_queue.get_all_jobs()
    return QueueStatusResponse(jobs=[
        QueueJobResponse(
            id=j.id, type=j.type, model=j.model, params={k: v for k, v in j.params.items()},
            status=j.status, slot=j.slot, progress=j.progress, phase=j.phase,
            result_paths=j.result_paths, error=j.error, created_at=j.created_at,
            batch_id=j.batch_id, batch_index=j.batch_index, tags=j.tags,
        )
        for j in jobs
    ])


@router.post("/cancel/{job_id}", response_model=QueueSubmitResponse)
def route_queue_cancel(
    job_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> QueueSubmitResponse:
    handler.job_queue.cancel_job(job_id)
    # Also cancel via GenerationHandler so running pipelines stop
    handler.generation.cancel_generation()
    job = handler.job_queue.get_job(job_id)
    status = job.status if job else "not_found"
    return QueueSubmitResponse(id=job_id, status=status)


@router.post("/clear", response_model=QueueStatusResponse)
def route_queue_clear(
    handler: AppHandler = Depends(get_state_service),
) -> QueueStatusResponse:
    handler.job_queue.clear_finished()
    return route_queue_status(handler)
