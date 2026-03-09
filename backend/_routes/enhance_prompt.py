"""Prompt enhancement route."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app_handler import AppHandler
from state import get_state_service

router = APIRouter(tags=["enhance"])


class EnhancePromptRequest(BaseModel):
    prompt: str = ""
    mode: str = "text-to-video"
    model: str = "ltx-fast"
    imagePath: str | None = None


@router.post("/api/enhance-prompt")
def enhance_prompt(
    req: EnhancePromptRequest,
    handler: AppHandler = Depends(get_state_service),
):
    return handler.enhance_prompt.enhance(
        req.prompt, req.mode, req.model, image_path=req.imagePath,
    )
