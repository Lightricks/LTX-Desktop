"""Local Desktop management of the app-owned Radeon video engine."""

from typing import cast
from fastapi import APIRouter, HTTPException, Request
from _routes._admin_guard import guard_admin_permission
from services.amd_comfy_engine import (
    ComfyEngineStatus,
    EngineAction,
    ManagedComfyEngine,
)

router = APIRouter(prefix="/api/comfy/engine", tags=["models"])


def _service(request: Request) -> ManagedComfyEngine:
    if not request.client or request.client.host not in {
        "127.0.0.1",
        "::1",
        "localhost",
    }:
        raise HTTPException(status_code=403, detail="Local desktop access required")
    return cast(ManagedComfyEngine, request.app.state.managed_comfy)


@router.get("", response_model=ComfyEngineStatus)
def engine_status(request: Request) -> ComfyEngineStatus:
    return _service(request).status()


@router.post("/{action}", response_model=ComfyEngineStatus)
def manage_comfy_engine(action: EngineAction, request: Request) -> ComfyEngineStatus:
    guard_admin_permission(request)
    return _service(request).action(action)
