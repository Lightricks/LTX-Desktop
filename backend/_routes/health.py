"""Route handlers for /health and /api/gpu-info."""

from __future__ import annotations

import os
import signal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from api_types import GpuInfoResponse, HealthResponse, MpsMemoryResponse, RuntimeTelemetryResponse
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def route_health(handler: AppHandler = Depends(get_state_service)) -> HealthResponse:
    return handler.health.get_health()


@router.get("/api/gpu-info", response_model=GpuInfoResponse)
def route_gpu_info(handler: AppHandler = Depends(get_state_service)) -> GpuInfoResponse:
    return handler.health.get_gpu_info()


@router.get("/api/gpu-info/mps", response_model=MpsMemoryResponse)
def route_mps_memory(handler: AppHandler = Depends(get_state_service)) -> MpsMemoryResponse:
    return handler.health.get_mps_memory()


@router.get("/api/runtime-telemetry", response_model=RuntimeTelemetryResponse)
def route_runtime_telemetry(handler: AppHandler = Depends(get_state_service)) -> RuntimeTelemetryResponse:
    return handler.health.get_runtime_telemetry()


def _shutdown_process() -> None:
    os.kill(os.getpid(), signal.SIGTERM)


@router.post("/api/system/shutdown")
def route_shutdown(background_tasks: BackgroundTasks, request: Request) -> dict[str, str]:
    client_host = request.client.host if request.client else None
    if client_host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail="Forbidden")

    background_tasks.add_task(_shutdown_process)
    return {"status": "shutting_down"}
