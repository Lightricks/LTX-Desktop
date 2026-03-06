"""Route handlers for /api/gallery/local endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import GalleryListResponse, StatusResponse
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api/gallery", tags=["gallery"])


@router.get("/local", response_model=GalleryListResponse)
def route_list_local_assets(
    page: int = 1,
    per_page: int = 50,
    type: str = "all",
    handler: AppHandler = Depends(get_state_service),
) -> GalleryListResponse:
    return handler.gallery.list_local_assets(
        page=page,
        per_page=per_page,
        asset_type=type,
    )


@router.delete("/local/{asset_id}", response_model=StatusResponse)
def route_delete_local_asset(
    asset_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    handler.gallery.delete_local_asset(asset_id)
    return StatusResponse(status="ok")
