"""Local Desktop controls for the optional Kandinsky ComfyUI pack."""

from typing import cast

from fastapi import APIRouter, HTTPException, Query, Request

from _routes._admin_guard import guard_admin_permission
from services.amd_kandinsky_node import KandinskyNodeSetup, KandinskyNodeSetupStatus

router = APIRouter(prefix="/api/amd/comfy/kandinsky-node", tags=["models"])


def _setup(request: Request) -> KandinskyNodeSetup:
    if not request.client or request.client.host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(status_code=403, detail="Local desktop access required")
    return cast(KandinskyNodeSetup, request.app.state.kandinsky_node_setup)


@router.get("", response_model=KandinskyNodeSetupStatus)
def kandinsky_node_status(request: Request) -> KandinskyNodeSetupStatus:
    return _setup(request).run()


@router.post("/install", response_model=KandinskyNodeSetupStatus)
def install_kandinsky_node(request: Request, managed_only: bool = Query(False, alias="managedOnly")) -> KandinskyNodeSetupStatus:
    guard_admin_permission(request)
    return _setup(request).run("install", managed_only=managed_only)


@router.post("/restart", response_model=KandinskyNodeSetupStatus)
def restart_kandinsky_node(request: Request, managed_only: bool = Query(False, alias="managedOnly")) -> KandinskyNodeSetupStatus:
    guard_admin_permission(request)
    return _setup(request).run("restart", managed_only=managed_only)
