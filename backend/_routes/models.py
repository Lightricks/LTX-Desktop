"""Route handlers for checkpoint recommendation and download APIs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from _routes._errors import HTTPError
from api_types import (
    ActiveDownloadResponse,
    CheckModelAccessRequest,
    CheckModelAccessResponse,
    DescribeCheckpointsRequest,
    DescribeCheckpointsResponse,
    DownloadProgressResponse,
    ImageGenRecommendationResponse,
    InstalledModelsResponse,
    LtxIcLoraRecommendationResponse,
    LtxModelVersionsResponse,
    LtxRecommendationResponse,
    ModelDeleteRequest,
    ModelDownloadRequest,
    ModelDownloadStartResponse,
    SetActiveLtxModelRequest,
    StatusResponse,
    TextEncoderRecommendationResponse,
)
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models", response_model=InstalledModelsResponse)
def route_list_models(
    model_type: str | None = Query(default=None, alias="type"),
    handler: AppHandler = Depends(get_state_service),
) -> InstalledModelsResponse:
    """Return installed model files. ?type=lora -> regular LoRAs only;
    ?type=ic-lora -> IC-LoRAs only; no type -> all installed models."""
    return handler.models.list_installed_models(model_type=model_type)


@router.get("/models/ltx-recommendation", response_model=LtxRecommendationResponse)
def route_ltx_recommendation(handler: AppHandler = Depends(get_state_service)) -> LtxRecommendationResponse:
    return handler.models.get_ltx_recommendation()


@router.get("/models/img-gen-recommendation", response_model=ImageGenRecommendationResponse)
def route_img_gen_recommendation(handler: AppHandler = Depends(get_state_service)) -> ImageGenRecommendationResponse:
    return handler.models.get_img_gen_recommendation()


@router.get("/models/ltx-ic-lora-recommendation", response_model=LtxIcLoraRecommendationResponse)
def route_ltx_ic_lora_recommendation(
    handler: AppHandler = Depends(get_state_service),
) -> LtxIcLoraRecommendationResponse:
    return handler.models.get_ltx_ic_lora_recommendation()


@router.get("/models/text-encoder-recommendation", response_model=TextEncoderRecommendationResponse)
def route_text_encoder_recommendation(
    handler: AppHandler = Depends(get_state_service),
) -> TextEncoderRecommendationResponse:
    return handler.models.get_text_encoder_recommendation()


@router.post("/models/describe", response_model=DescribeCheckpointsResponse)
def route_describe_checkpoints(
    req: DescribeCheckpointsRequest,
    handler: AppHandler = Depends(get_state_service),
) -> DescribeCheckpointsResponse:
    return handler.models.describe_checkpoints(req.cp_ids)


@router.get("/models/download/active", response_model=ActiveDownloadResponse)
def route_active_download(handler: AppHandler = Depends(get_state_service)) -> ActiveDownloadResponse:
    return handler.downloads.get_active_download()


@router.get("/models/download/progress", response_model=DownloadProgressResponse)
def route_download_progress(
    sessionId: str = Query(...),
    handler: AppHandler = Depends(get_state_service),
) -> DownloadProgressResponse:
    try:
        return handler.downloads.get_download_progress(sessionId)
    except ValueError as exc:
        raise HTTPError(404, "UNKNOWN_DOWNLOAD_SESSION") from exc


@router.post("/models/check-access", response_model=CheckModelAccessResponse)
def route_check_model_access(
    req: CheckModelAccessRequest,
    handler: AppHandler = Depends(get_state_service),
) -> CheckModelAccessResponse:
    return handler.downloads.check_model_access(req.cp_ids)


@router.post("/models/download", response_model=ModelDownloadStartResponse)
def route_model_download(
    req: ModelDownloadRequest,
    handler: AppHandler = Depends(get_state_service),
) -> ModelDownloadStartResponse:
    session_id = handler.downloads.start_model_download(
        download_type=req.type,
        cp_ids=req.cp_ids,
    )
    return ModelDownloadStartResponse(
        status="started",
        message="Model download started",
        sessionId=session_id,
    )


@router.get("/models/ltx-versions", response_model=LtxModelVersionsResponse)
def route_ltx_versions(handler: AppHandler = Depends(get_state_service)) -> LtxModelVersionsResponse:
    return handler.models.list_ltx_versions()


@router.post("/models/active-ltx-model", response_model=StatusResponse)
def route_set_active_ltx_model(
    req: SetActiveLtxModelRequest,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    handler.models.set_active_ltx_model(req.model_id)
    return StatusResponse(status="ok")


@router.delete("/models/delete", response_model=StatusResponse)
def route_model_delete(
    req: ModelDeleteRequest,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    if handler.downloads.is_download_running():
        raise HTTPError(409, "DOWNLOAD_ALREADY_RUNNING")
    handler.models.delete_checkpoints(req.cp_ids)
    return StatusResponse(status="ok")
