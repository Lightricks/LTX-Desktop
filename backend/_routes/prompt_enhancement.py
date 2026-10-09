"""Route handlers for /api/enhance-prompt and /api/prompt-enhancer."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import (
    EnhancePromptRequest,
    EnhancePromptResponse,
    PromptEnhancerStatusResponse,
)
from app_handler import AppHandler
from services.prompt_enhancement.resolve_enhance_assets import resolve_enhance_image_assets
from state import get_state_service

router = APIRouter(prefix="/api", tags=["prompt-enhancement"])


@router.get("/prompt-enhancer", response_model=PromptEnhancerStatusResponse)
def route_prompt_enhancer(
    handler: AppHandler = Depends(get_state_service),
) -> PromptEnhancerStatusResponse:
    """GET /api/prompt-enhancer."""
    return handler.prompt_enhancement.status(prefer_gemini=False)


@router.post("/enhance-prompt", response_model=EnhancePromptResponse)
def route_enhance_prompt(
    req: EnhancePromptRequest,
    handler: AppHandler = Depends(get_state_service),
) -> EnhancePromptResponse:
    """POST /api/enhance-prompt."""
    return handler.prompt_enhancement.enhance(
        resolve_enhance_image_assets(handler.assets, req)
    )
