"""Route handler for POST /api/extend."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from _routes._analytics import (
    desktop_generation_details,
    pixel_resolution,
    run_tracked_generation,
    video_execution,
)
from api_types import ExtendRequest, ExtendResponse
from services.analytics import unenhanced_prompt_provenance
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api", tags=["extend"])


@router.post("/extend", response_model=ExtendResponse)
def route_extend(req: ExtendRequest, handler: AppHandler = Depends(get_state_service)) -> ExtendResponse:
    details = desktop_generation_details(
        surface="genspace",
        feature="extend",
        kind="video",
        execution=video_execution(
            force_api_generations=handler.config.force_api_generations,
            settings=handler.state.app_settings,
        ),
        model=req.model,
        duration_sec=req.duration,
        prompt_provenance=unenhanced_prompt_provenance(req.prompt_provenance),
        **(
            {"resolution": pixel_resolution(req.resolution.width, req.resolution.height)}
            if req.resolution is not None
            else {}
        ),
    )
    return run_tracked_generation(
        handler.analytics,
        details,
        lambda: handler.extend.run(req),
    )
