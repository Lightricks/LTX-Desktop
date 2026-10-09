"""Route handlers for /api/generate-image."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from _routes._analytics import (
    desktop_generation_details,
    image_execution,
    pixel_resolution,
    run_tracked_generation,
)
from api_types import GenerateImageRequest, GenerateImageResponse
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api", tags=["image"])


@router.post("/generate-image", response_model=GenerateImageResponse)
def route_generate_image(
    req: GenerateImageRequest,
    handler: AppHandler = Depends(get_state_service),
) -> GenerateImageResponse:
    """POST /api/generate-image."""
    details = desktop_generation_details(
        surface="genspace",
        feature="image-to-image" if req.imagePath else "text-to-image",
        kind="image",
        execution=image_execution(
            force_api_generations=handler.config.force_api_generations,
            settings=handler.state.app_settings,
        ),
        resolution=pixel_resolution(req.width, req.height),
        output_count=req.numImages,
    )
    return run_tracked_generation(
        handler.analytics,
        details,
        lambda: handler.image_generation.generate(req),
    )

