"""Route handlers for /health and /api/gpu-info."""

from __future__ import annotations

import os
import signal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from api_types import GpuInfoResponse, HealthResponse
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def route_health(handler: AppHandler = Depends(get_state_service)) -> HealthResponse:
    return handler.health.get_health()


@router.get("/api/gpu-info", response_model=GpuInfoResponse)
def route_gpu_info(handler: AppHandler = Depends(get_state_service)) -> GpuInfoResponse:
    return handler.health.get_gpu_info()


def _shutdown_process() -> None:
    os.kill(os.getpid(), signal.SIGTERM)


@router.post("/api/system/shutdown")
def route_shutdown(
    background_tasks: BackgroundTasks,
    request: Request,
    handler: AppHandler = Depends(get_state_service),
) -> dict[str, str]:
    if handler.config.deployment_mode == "standalone":
        raise HTTPException(status_code=403, detail="Shutdown is disabled in standalone mode")
    client_host = request.client.host if request.client else None
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail="Forbidden")

    background_tasks.add_task(_shutdown_process)
    return {"status": "shutting_down"}
