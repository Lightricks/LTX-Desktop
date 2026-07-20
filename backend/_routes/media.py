"""Opaque media upload and generated artifact routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

from api_types import MediaRef, MediaType, StatusResponse
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api", tags=["media"])


@router.post("/media", response_model=MediaRef)
def route_upload_media(
    file: Annotated[UploadFile, File(...)],
    media_type: Annotated[MediaType, Form(...)],
    handler: AppHandler = Depends(get_state_service),
) -> MediaRef:
    return handler.media.upload(
        file.file,
        filename=file.filename or "media",
        media_type=media_type,
    )


@router.delete("/media/{media_id}", response_model=StatusResponse)
def route_delete_media(
    media_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    handler.media.delete_upload(media_id)
    return StatusResponse(status="ok")


@router.get("/artifacts/{artifact_id}", response_class=FileResponse)
def route_download_artifact(
    artifact_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> FileResponse:
    artifact = handler.media.resolve_artifact(artifact_id)
    return FileResponse(
        path=artifact.path,
        media_type=artifact.content_type,
        filename=artifact.filename,
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.delete("/artifacts/{artifact_id}", response_model=StatusResponse)
def route_delete_artifact(
    artifact_id: str,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    handler.media.delete_artifact(artifact_id)
    return StatusResponse(status="ok")
