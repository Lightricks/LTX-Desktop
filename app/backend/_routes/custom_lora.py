from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query

from api_types import (
    ActiveCustomLoraImportResponse,
    CustomLoraImportProgressResponse,
    CustomLoraImportRequest,
    CustomLoraImportStartResponse,
    CustomLoraLinkInfoRequest,
    CustomLoraLinkInfoResponse,
    CustomLoraListResponse,
    CustomLoraRestoreRequest,
    StatusResponse,
)
from app_handler import AppHandler
from state import get_state_service

# Desktop only: adding and deleting touch the user's disk. The phone gets the read-only list
# from remote/loras.py.
router = APIRouter(prefix="/api")


@router.get("/custom-loras", response_model=CustomLoraListResponse, tags=["custom-loras"])
def route_list_custom_loras(
    handler: AppHandler = Depends(get_state_service),
) -> CustomLoraListResponse:
    return handler.custom_loras.list_loras()


@router.post("/custom-loras", response_model=CustomLoraImportStartResponse, tags=["custom-loras"])
def route_import_custom_lora(
    req: CustomLoraImportRequest,
    handler: AppHandler = Depends(get_state_service),
) -> CustomLoraImportStartResponse:
    return handler.custom_loras.start_import(req)


@router.post(
    "/custom-loras/link-info", response_model=CustomLoraLinkInfoResponse, tags=["custom-loras"]
)
def route_custom_lora_link_info(
    req: CustomLoraLinkInfoRequest,
    handler: AppHandler = Depends(get_state_service),
) -> CustomLoraLinkInfoResponse:
    return handler.custom_loras.link_info(req.url)


@router.get(
    "/custom-loras/import/progress",
    response_model=CustomLoraImportProgressResponse,
    tags=["custom-loras"],
)
def route_custom_lora_import_progress(
    sessionId: str = Query(...),
    handler: AppHandler = Depends(get_state_service),
) -> CustomLoraImportProgressResponse:
    return handler.custom_loras.get_import_progress(sessionId)


@router.get(
    "/custom-loras/import/active",
    response_model=ActiveCustomLoraImportResponse,
    tags=["custom-loras"],
)
def route_custom_lora_import_active(
    handler: AppHandler = Depends(get_state_service),
) -> ActiveCustomLoraImportResponse:
    return handler.custom_loras.get_active_import()


@router.post(
    "/custom-loras/{custom_lora_id}/weights",
    response_model=CustomLoraImportStartResponse,
    tags=["custom-loras"],
)
def route_restore_custom_lora_weights(
    req: CustomLoraRestoreRequest,
    custom_lora_id: str = Path(pattern=r"^[a-z0-9-]+$", max_length=64),
    handler: AppHandler = Depends(get_state_service),
) -> CustomLoraImportStartResponse:
    return handler.custom_loras.restore_weights(custom_lora_id, req)


@router.delete(
    "/custom-loras/{custom_lora_id}/weights", response_model=StatusResponse, tags=["custom-loras"]
)
def route_delete_custom_lora_weights(
    custom_lora_id: str = Path(pattern=r"^[a-z0-9-]+$", max_length=64),
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    handler.custom_loras.delete_weights(custom_lora_id)
    return StatusResponse(status="ok")


@router.delete("/custom-loras/{custom_lora_id}", response_model=StatusResponse, tags=["custom-loras"])
def route_delete_custom_lora(
    custom_lora_id: str = Path(pattern=r"^[a-z0-9-]+$", max_length=64),
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    handler.custom_loras.delete(custom_lora_id)
    return StatusResponse(status="ok")
