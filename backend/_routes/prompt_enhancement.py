"""Route handlers for /api/enhance-prompt."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import EnhancePromptRequest, EnhancePromptResponse
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api", tags=["prompt-enhancement"])


@router.post("/enhance-prompt", response_model=EnhancePromptResponse)
def route_enhance_prompt(
    req: EnhancePromptRequest,
    handler: AppHandler = Depends(get_state_service),
) -> EnhancePromptResponse:
    """POST /api/enhance-prompt."""
    return handler.prompt_enhancement.enhance(req)
