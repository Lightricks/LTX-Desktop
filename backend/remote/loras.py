"""Remote LoRA-catalog surface: list + download for the recipe preflight.

Mirrors the plain-LoRA endpoints of ``_routes.lora_catalog``. IC-LoRA downloads
stay on the desktop. The phone only lists IC-LoRAs so a Home feature can tell
whether the weights are already installed. The response models carry no desktop
filesystem paths (only the HF ``repo_id`` and variant filenames), so they are
safe to hand to a paired session.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api_types import (
    ActiveLoraDownloadResponse,
    CatalogDownloadStartResponse,
    IcLoraListResponse,
    LoraDownloadProgressResponse,
    LoraDownloadRequest,
    LoraListResponse,
)
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api")


@router.get("/ic-loras", response_model=IcLoraListResponse, tags=["ic-loras"])
def route_list_ic_loras(
    handler: AppHandler = Depends(get_state_service),
) -> IcLoraListResponse:
    return handler.catalog.list_ic_loras()


@router.get("/loras", response_model=LoraListResponse, tags=["loras"])
def route_list_loras(
    handler: AppHandler = Depends(get_state_service),
) -> LoraListResponse:
    return handler.catalog.list_loras()


@router.post(
    "/loras/download",
    response_model=CatalogDownloadStartResponse,
    tags=["loras"],
)
def route_lora_download(
    req: LoraDownloadRequest,
    handler: AppHandler = Depends(get_state_service),
) -> CatalogDownloadStartResponse:
    return handler.catalog.start_lora_download(
        req.lora_id, use_hf_auth=req.use_hf_auth, variant_id=req.variant_id
    )


@router.get(
    "/loras/download/progress",
    response_model=LoraDownloadProgressResponse,
    tags=["loras"],
)
def route_lora_download_progress(
    sessionId: str = Query(...),
    handler: AppHandler = Depends(get_state_service),
) -> LoraDownloadProgressResponse:
    return handler.catalog.get_lora_download_progress(sessionId)


@router.get(
    "/loras/download/active",
    response_model=ActiveLoraDownloadResponse,
    tags=["loras"],
)
def route_lora_download_active(
    handler: AppHandler = Depends(get_state_service),
) -> ActiveLoraDownloadResponse:
    return handler.catalog.get_active_lora_download()
