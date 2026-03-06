"""Route for receiving jobs from Director's Palette."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import ReceiveJobRequest, ReceiveJobResponse
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api/sync", tags=["sync"])


@router.post("/receive-job", response_model=ReceiveJobResponse)
def route_receive_job(
    req: ReceiveJobRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ReceiveJobResponse:
    return handler.receive_job_handler.receive_job(req)
