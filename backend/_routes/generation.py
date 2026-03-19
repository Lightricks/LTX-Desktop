"""Route handlers for /api/generate, /api/generate/cancel, /api/generation/progress."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import (
    CancelResponse,
    GenerateLongVideoRequest,
    GenerateLongVideoResponse,
    GenerateVideoRequest,
    GenerateVideoResponse,
    GenerationProgressResponse,
)
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api", tags=["generation"])


@router.post("/generate", response_model=GenerateVideoResponse)
def route_generate(
    req: GenerateVideoRequest,
    handler: AppHandler = Depends(get_state_service),
) -> GenerateVideoResponse:
    """POST /api/generate — video generation from JSON body."""
    return handler.video_generation.generate(req)


@router.post("/generate/long", response_model=GenerateLongVideoResponse)
def route_generate_long(
    req: GenerateLongVideoRequest,
    handler: AppHandler = Depends(get_state_service),
) -> GenerateLongVideoResponse:
    """POST /api/generate/long — chain-extend I2V to target duration."""
    try:
        video_path = handler.video_generation.generate_long_video(
            prompt=req.prompt,
            image_path=req.imagePath,
            target_duration=req.targetDuration,
            resolution=req.resolution,
            aspect_ratio=req.aspectRatio,
            fps=req.fps,
            segment_duration=req.segmentDuration,
            camera_motion=req.cameraMotion,
            lora_path=req.loraPath,
            lora_weight=req.loraWeight,
        )
        segments = max(1, (req.targetDuration + req.segmentDuration - 1) // req.segmentDuration)
        return GenerateLongVideoResponse(
            status="complete", video_path=video_path,
            segments=segments, total_duration=req.targetDuration,
        )
    except Exception as e:
        if "cancelled" in str(e).lower():
            return GenerateLongVideoResponse(status="cancelled")
        raise


@router.post("/generate/cancel", response_model=CancelResponse)
def route_generate_cancel(handler: AppHandler = Depends(get_state_service)) -> CancelResponse:
    """POST /api/generate/cancel."""
    return handler.generation.cancel_generation()


@router.get("/generation/progress", response_model=GenerationProgressResponse)
def route_generation_progress(handler: AppHandler = Depends(get_state_service)) -> GenerationProgressResponse:
    """GET /api/generation/progress."""
    return handler.generation.get_generation_progress()
