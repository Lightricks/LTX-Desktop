"""Remote Explore Enhance: status + prompt rewrite. Asset ids only — no paths."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from _routes._errors import HTTPError
from api_types import (
    EnhancePromptRequest,
    EnhancePromptResponse,
    PromptEnhancerStatusResponse,
)
from app_handler import AppHandler
from services.prompt_enhancement.resolve_enhance_assets import resolve_enhance_image_assets
from state import get_state_service

router = APIRouter(prefix="/api", tags=["prompt-enhancement"])


class RemoteEnhancePromptRequest(BaseModel):
    """Phone Enhance body. Filesystem paths stay off this port."""

    model_config = ConfigDict(strict=True, extra="forbid")
    prompt: str
    loraCatalogIds: list[str] = Field(default_factory=list)
    imageAssetId: str | None = None
    lastImageAssetId: str | None = None
    provider: Literal["local", "api"] | None = None
    mediaType: Literal["video"] = "video"

    @field_validator("imageAssetId", "lastImageAssetId", mode="before")
    @classmethod
    def _asset_id_must_be_present_when_set(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("Asset id cannot be blank")
        return stripped

    @model_validator(mode="after")
    def _last_frame_needs_first(self) -> RemoteEnhancePromptRequest:
        if self.lastImageAssetId and not self.imageAssetId:
            raise ValueError("Last frame requires a first-frame image")
        if not self.prompt.strip() and not self.imageAssetId:
            raise ValueError("Prompt is required unless an image is provided")
        return self


@router.get("/prompt-enhancer", response_model=PromptEnhancerStatusResponse)
def route_remote_prompt_enhancer(
    handler: AppHandler = Depends(get_state_service),
) -> PromptEnhancerStatusResponse:
    return handler.prompt_enhancement.status(prefer_gemini=True)


def _looks_like_filesystem_path(detail: str) -> bool:
    stripped = detail.strip()
    if not stripped:
        return False
    if Path(stripped).is_absolute():
        return True
    return "/" in stripped or "\\" in stripped


def _sanitize_enhance_http_error(exc: HTTPError) -> HTTPError:
    """Keep resolved image paths out of Remote error bodies."""
    if _looks_like_filesystem_path(exc.detail):
        return HTTPError(exc.status_code, "ASSET_UNAVAILABLE")
    return exc


@router.post("/enhance-prompt", response_model=EnhancePromptResponse)
def route_remote_enhance_prompt(
    req: RemoteEnhancePromptRequest,
    handler: AppHandler = Depends(get_state_service),
) -> EnhancePromptResponse:
    provider = req.provider
    if provider is None:
        provider = handler.prompt_enhancement.status(
            prefer_gemini=True
        ).defaultProvider
    desktop_req = EnhancePromptRequest(
        prompt=req.prompt,
        loraCatalogIds=req.loraCatalogIds,
        imageAssetId=req.imageAssetId,
        lastImageAssetId=req.lastImageAssetId,
        provider=provider,
        mediaType=req.mediaType,
    )
    try:
        return handler.prompt_enhancement.enhance(
            resolve_enhance_image_assets(handler.assets, desktop_req)
        )
    except HTTPError as exc:
        raise _sanitize_enhance_http_error(exc) from exc
