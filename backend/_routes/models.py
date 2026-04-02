"""Route handlers for /api/models, /api/models/status, /api/models/download/*."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from api_types import (
    DownloadProgressResponse,
    ModelDownloadRequest,
    ModelDownloadStartResponse,
    ModelInfo,
    ModelsStatusResponse,
    RequiredModelsResponse,
    TextEncoderAlreadyDownloadedResponse,
    TextEncoderDownloadStartedResponse,
    TextEncoderDownloadResponse,
)
from _routes._errors import HTTPError
from state import get_state_service
from app_handler import AppHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models", response_model=list[ModelInfo])
def route_models_list(handler: AppHandler = Depends(get_state_service)) -> list[ModelInfo]:
    return handler.models.get_models_list()


@router.get("/models/status", response_model=ModelsStatusResponse)
def route_models_status(handler: AppHandler = Depends(get_state_service)) -> ModelsStatusResponse:
    return handler.models.get_models_status()


@router.get("/models/download/progress", response_model=DownloadProgressResponse)
def route_download_progress(
    sessionId: str = Query(...),
    handler: AppHandler = Depends(get_state_service),
) -> DownloadProgressResponse:
    try:
        return handler.downloads.get_download_progress(sessionId)
    except ValueError as exc:
        raise HTTPError(404, str(exc))


@router.get("/models/required-models", response_model=RequiredModelsResponse)
def route_required_models(
    skipTextEncoder: bool = Query(default=False),
    handler: AppHandler = Depends(get_state_service),
) -> RequiredModelsResponse:
    return RequiredModelsResponse(
        modelTypes=handler.models.get_required_model_types(skip_text_encoder=skipTextEncoder),
    )


@router.post("/models/download", response_model=ModelDownloadStartResponse)
def route_model_download(
    req: ModelDownloadRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ModelDownloadStartResponse:
    if handler.downloads.is_download_running():
        raise HTTPError(409, "Download already in progress")

    session_id = handler.downloads.start_model_download(model_types=req.modelTypes)
    if session_id:
        return ModelDownloadStartResponse(
            status="started",
            message="Model download started",
            sessionId=session_id,
        )

    raise HTTPError(400, "Failed to start download")


@router.post("/text-encoder/download", response_model=TextEncoderDownloadResponse)
def route_text_encoder_download(handler: AppHandler = Depends(get_state_service)) -> TextEncoderDownloadResponse:
    if handler.downloads.is_download_running():
        raise HTTPError(409, "Download already in progress")

    files = handler.models.refresh_available_files()
    if files["text_encoder"] is not None:
        return TextEncoderAlreadyDownloadedResponse(status="already_downloaded", message="Text encoder already downloaded")

    session_id = handler.downloads.start_text_encoder_download()
    if session_id:
        return TextEncoderDownloadStartedResponse(status="started", message="Text encoder download started", sessionId=session_id)

    raise HTTPError(400, "Failed to start download")
