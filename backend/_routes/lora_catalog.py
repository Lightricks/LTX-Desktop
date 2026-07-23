from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api_types import (
    CatalogDownloadStartResponse,
    IcLoraDownloadProgressResponse,
    IcLoraDownloadRequest,
    IcLoraListResponse,
    LoraDownloadProgressResponse,
    LoraDownloadRequest,
    LoraListResponse,
)
from app_handler import AppHandler
from state import get_state_service

# Plain LoRAs and IC-LoRAs share one handler (handler.catalog) and one catalog file, so
# their endpoints live together. Both kinds keep their own paths + tags (and isolated
# download sessions in the handler). The "/api" prefix lives on the router (as in models.py)
# so a future prefix change follows automatically.
router = APIRouter(prefix="/api")


# --- Plain LoRAs ---
@router.get("/loras", response_model=LoraListResponse, tags=["loras"])
def route_list_loras(handler: AppHandler = Depends(get_state_service)) -> LoraListResponse:
    return handler.catalog.list_loras()


@router.post("/loras/download", response_model=CatalogDownloadStartResponse, tags=["loras"])
def route_lora_download(
    req: LoraDownloadRequest,
    handler: AppHandler = Depends(get_state_service),
) -> CatalogDownloadStartResponse:
    return handler.catalog.start_lora_download(
        req.lora_id, use_hf_auth=req.use_hf_auth, variant_id=req.variant_id
    )


@router.get("/loras/download/progress", response_model=LoraDownloadProgressResponse, tags=["loras"])
def route_lora_download_progress(
    sessionId: str = Query(...),
    handler: AppHandler = Depends(get_state_service),
) -> LoraDownloadProgressResponse:
    return handler.catalog.get_lora_download_progress(sessionId)


# --- IC-LoRAs ---
@router.get("/ic-loras", response_model=IcLoraListResponse, tags=["ic-loras"])
def route_list_ic_loras(handler: AppHandler = Depends(get_state_service)) -> IcLoraListResponse:
    return handler.catalog.list_ic_loras()


@router.post("/ic-loras/download", response_model=CatalogDownloadStartResponse, tags=["ic-loras"])
def route_ic_lora_download(
    req: IcLoraDownloadRequest,
    handler: AppHandler = Depends(get_state_service),
) -> CatalogDownloadStartResponse:
    return handler.catalog.start_ic_lora_download(
        req.ic_lora_id, use_hf_auth=req.use_hf_auth, variant_id=req.variant_id
    )


@router.get("/ic-loras/download/progress", response_model=IcLoraDownloadProgressResponse, tags=["ic-loras"])
def route_ic_lora_download_progress(
    sessionId: str = Query(...),
    handler: AppHandler = Depends(get_state_service),
) -> IcLoraDownloadProgressResponse:
    return handler.catalog.get_ic_lora_download_progress(sessionId)
