"""Route handlers for /api/ic-lora/* endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from _routes._analytics import (
    desktop_generation_details,
    pixel_resolution,
    run_tracked_generation,
)
from api_types import (
    IcLoraExtractRequest,
    IcLoraExtractResponse,
    IcLoraGenerateRequest,
    IcLoraGenerateResponse,
)
from services.services_utils import JSONValue
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api/ic-lora", tags=["ic-lora"])


@router.post("/extract-conditioning", response_model=IcLoraExtractResponse)
def route_ic_lora_extract(
    req: IcLoraExtractRequest,
    handler: AppHandler = Depends(get_state_service),
) -> IcLoraExtractResponse:
    return handler.ic_lora.extract_conditioning(req)


@router.post("/generate", response_model=IcLoraGenerateResponse)
def route_ic_lora_generate(
    req: IcLoraGenerateRequest,
    handler: AppHandler = Depends(get_state_service),
) -> IcLoraGenerateResponse:
    metadata: dict[str, JSONValue] = {}
    if req.resolution is not None:
        metadata["resolution"] = pixel_resolution(
            req.resolution.width, req.resolution.height
        )
    if req.fps_override is not None:
        metadata["fps"] = req.fps_override
    duration = req.control_values.get("duration")
    if isinstance(duration, int):
        metadata["duration_sec"] = duration
    metadata["prompt_provenance"] = handler.prompt_enhancement.preview_prompt_provenance(
        req.prompt,
        provenance=req.prompt_provenance,
        explore_generation=False,
    )
    if req.ic_lora_id is not None:
        metadata["lora_catalog_ids"] = [req.ic_lora_id]
    elif req.custom_lora_ref is not None:
        metadata["has_custom_lora"] = True
    details = desktop_generation_details(
        surface="genspace",
        feature="ic-lora",
        kind="video",
        execution="local",
        **metadata,
    )
    return run_tracked_generation(
        handler.analytics,
        details,
        lambda: handler.ic_lora.generate(req),
    )
