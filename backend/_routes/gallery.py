"""Route handlers for /api/gallery/local endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from api_types import GalleryListResponse, StatusResponse
from app_handler import AppHandler
from _routes._errors import HTTPError
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


@router.get("/local/file/{filename:path}")
def route_serve_local_file(
    filename: str,
    handler: AppHandler = Depends(get_state_service),
) -> FileResponse:
    file_path = handler.config.outputs_dir / Path(filename).name
    if not file_path.is_file():
        raise HTTPError(404, f"File not found: {filename}")
    return FileResponse(path=str(file_path))


@router.delete("/local/{asset_id}", response_model=StatusResponse)
def route_delete_local_asset(
    asset_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    handler.gallery.delete_local_asset(asset_id)
    return StatusResponse(status="ok")
