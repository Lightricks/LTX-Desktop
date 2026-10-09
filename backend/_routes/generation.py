"""Route handlers for /api/generate, /api/generate/cancel, /api/generation/progress."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from _routes._analytics import (
    desktop_generation_details,
    run_tracked_generation,
    video_execution,
)
from api_model_specs import video_generation_feature
from api_types import (
    CancelResponse,
    GenerateVideoModelsSpecsResponse,
    GenerateVideoRequest,
    GenerateVideoResponse,
    GenerationProgressResponse,
    LtxInsufficientFundsErrorResponse,
)
from services.services_utils import JSONValue
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api", tags=["generation"])
specs_router = APIRouter(prefix="/api", tags=["generation"])


@router.post(
    "/generate",
    response_model=GenerateVideoResponse,
    responses={
        402: {
            "model": LtxInsufficientFundsErrorResponse,
            "description": "LTX API credits are insufficient for the requested generation",
        },
    },
)
def route_generate(
    req: GenerateVideoRequest,
    handler: AppHandler = Depends(get_state_service),
) -> GenerateVideoResponse:
    """POST /api/generate — video generation from JSON body."""
    feature = video_generation_feature(
        audio_path=req.audioPath,
        image_path=req.imagePath,
        last_image_path=req.lastImagePath,
        keyframes=req.keyframes,
    )
    catalog_ids: list[JSONValue] = [
        lora.catalogId for lora in req.loras if lora.catalogId is not None
    ]
    execution = video_execution(
        force_api_generations=handler.config.force_api_generations,
        settings=handler.state.app_settings,
    )
    metadata: dict[str, JSONValue] = {
        "model": req.model,
        "resolution": req.resolution,
        "fps": req.fps,
        "prompt_provenance": handler.prompt_enhancement.preview_prompt_provenance(
            req.prompt,
            provenance=req.promptProvenance,
            explore_generation=False,
            force_api=execution == "ltx_api",
        ),
    }
    if req.duration is not None:
        metadata["duration_sec"] = req.duration
    if catalog_ids:
        metadata["lora_catalog_ids"] = catalog_ids
    if any(lora.catalogId is None for lora in req.loras):
        metadata["has_custom_lora"] = True
    details = desktop_generation_details(
        surface="genspace",
        feature=feature,
        kind="video",
        execution=execution,
        **metadata,
    )
    return run_tracked_generation(
        handler.analytics,
        details,
        lambda: handler.video_generation.generate(req),
    )


@specs_router.get("/generate/models-specs", response_model=GenerateVideoModelsSpecsResponse)
def route_generate_model_specs(
    handler: AppHandler = Depends(get_state_service),
) -> GenerateVideoModelsSpecsResponse:
    """GET /api/generate/models-specs."""
    return handler.video_generation.get_model_specs()


@router.post("/generate/cancel", response_model=CancelResponse)
def route_generate_cancel(handler: AppHandler = Depends(get_state_service)) -> CancelResponse:
    """POST /api/generate/cancel."""
    return handler.generation.cancel_generation()


@router.get("/generation/progress", response_model=GenerationProgressResponse)
def route_generation_progress(handler: AppHandler = Depends(get_state_service)) -> GenerationProgressResponse:
    """GET /api/generation/progress."""
    return handler.generation.get_generation_progress()
