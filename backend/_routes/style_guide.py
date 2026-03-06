"""Route for style guide grid generation."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import GenerateStyleGuideRequest, GenerateStyleGuideResponse
from app_handler import AppHandler
from state import get_state_service

router = APIRouter(prefix="/api/style-guide", tags=["style-guide"])


@router.post("/generate", response_model=GenerateStyleGuideResponse)
def route_generate_style_guide(
    req: GenerateStyleGuideRequest,
    handler: AppHandler = Depends(get_state_service),
) -> GenerateStyleGuideResponse:
    return handler.style_guide.generate(req)
