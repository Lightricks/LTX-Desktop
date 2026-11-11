"""Remote LoRA-catalog surface: list + download for the recipe preflight, plus the custom LoRA list.

Mirrors the plain-LoRA endpoints of ``_routes.lora_catalog``. IC-LoRA downloads
stay on the desktop. The phone only lists IC-LoRAs so a Home feature can tell
whether the weights are already installed. The response models carry no desktop
filesystem paths (only the HF ``repo_id`` and variant filenames), so they are
safe to hand to a paired session.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query
from fastapi.responses import FileResponse

from _routes._errors import HTTPError
from api_types import (
    ActiveLoraDownloadResponse,
    CatalogDownloadStartResponse,
    CustomLoraItem,
    CustomLoraListResponse,
    IcLoraListResponse,
    LoraDownloadProgressResponse,
    LoraDownloadRequest,
    LoraListResponse,
)
from app_handler import AppHandler
from remote.deps import RemoteContext, get_remote_context
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


def _cover_for_phone(item: CustomLoraItem, remote: RemoteContext) -> CustomLoraItem:
    """Swap the desktop file path for a signed URL. A phone cannot read ``file://``."""
    if item.cover_url is None:
        return item
    return item.model_copy(
        update={
            "cover_url": remote.signer.sign(
                f"/api/custom-loras/{item.id}/cover",
                asset_id=item.id,
                device_id=remote.device.id,
            )
        }
    )


@router.get("/custom-loras", response_model=CustomLoraListResponse, tags=["custom-loras"])
def route_list_custom_loras(
    handler: AppHandler = Depends(get_state_service),
    remote: RemoteContext = Depends(get_remote_context),
) -> CustomLoraListResponse:
    # Read-only. Adding or deleting a custom LoRA stays on the desktop.
    # The desktop list puts a file path in cover_url. Replace it before the phone sees the list.
    listed = handler.custom_loras.list_loras()
    return CustomLoraListResponse(loras=[_cover_for_phone(item, remote) for item in listed.loras])


@router.get("/custom-loras/{custom_lora_id}/cover", tags=["custom-loras"])
def route_custom_lora_cover(
    custom_lora_id: str = Path(pattern=r"^[a-z0-9-]+$", max_length=64),
    handler: AppHandler = Depends(get_state_service),
) -> FileResponse:
    # The auth middleware already checked the signed URL. This only serves the cover file.
    found = handler.custom_loras.cover_file(custom_lora_id)
    if found is None:
        raise HTTPError(404, "CUSTOM_LORA_COVER_NOT_FOUND")
    path, media_type = found
    return FileResponse(path, media_type=media_type, content_disposition_type="inline")
