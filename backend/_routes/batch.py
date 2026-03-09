"""Route handlers for /api/queue/batch/* — batch generation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import BatchSubmitRequest, BatchSubmitResponse, BatchStatusResponse
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api/queue", tags=["batch"])


@router.post("/submit-batch", response_model=BatchSubmitResponse)
def route_batch_submit(
    req: BatchSubmitRequest,
    handler: AppHandler = Depends(get_state_service),
) -> BatchSubmitResponse:
    return handler.batch.submit_batch(req, handler.job_queue)


@router.get("/batch/{batch_id}/status", response_model=BatchStatusResponse)
def route_batch_status(
    batch_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> BatchStatusResponse:
    return handler.batch.get_batch_status(batch_id, handler.job_queue)


@router.post("/batch/{batch_id}/cancel")
def route_batch_cancel(
    batch_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, str]:
    handler.batch.cancel_batch(batch_id, handler.job_queue)
    return {"status": "cancelled"}


@router.post("/batch/{batch_id}/retry-failed", response_model=BatchSubmitResponse)
def route_batch_retry(
    batch_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> BatchSubmitResponse:
    return handler.batch.retry_failed(batch_id, handler.job_queue)
