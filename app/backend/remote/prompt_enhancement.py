"""Remote Explore Enhance: status + prompt rewrite. Asset ids only — no paths."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from _routes._errors import HTTPError
from api_types import (
    IC_LORA_SHEET_MAX_IMAGES,
    LOCAL_MULTI_KEYFRAME_MAX_COUNT,
    EnhancePromptRequest,
    KeyframeRef,
    EnhancePromptResponse,
    LTXVideoGenAspectRatio,
    LTXVideoGenResolution,
    PromptEnhancerStatusResponse,
    ReplacedWindow,
)
from app_handler import AppHandler
from runtime_config.nag_limits import NEGATIVE_PROMPT_MAX_LENGTH
from services.prompt_enhancement.resolve_enhance_assets import resolve_enhance_media_assets
from state import get_state_service

router = APIRouter(prefix="/api", tags=["prompt-enhancement"])


class RemoteEnhancePromptRequest(BaseModel):
    """Phone Enhance body. Filesystem paths stay off this port."""

    model_config = ConfigDict(strict=True, extra="forbid")
    prompt: str
    negativePrompt: str | None = Field(default=None, max_length=NEGATIVE_PROMPT_MAX_LENGTH)
    loraCatalogIds: list[str] = Field(default_factory=list)
    # The saved custom LoRA. The id is checked against a strict pattern where it is read.
    customLoraIds: list[str] = Field(default_factory=list, max_length=1)
    icLoraId: str | None = None
    imageAssetId: str | None = None
    lastImageAssetId: str | None = None
    # The panel images of an image-list IC-LoRA (Ingredients), in order.
    imageAssetIds: list[str] = Field(default_factory=list[str], max_length=IC_LORA_SHEET_MAX_IMAGES)
    # The stills of a Keyframes clip. `duration` and `fps` give the rewrite their clock times.
    keyframeAssets: list[KeyframeRef] = Field(
        default_factory=list[KeyframeRef], max_length=LOCAL_MULTI_KEYFRAME_MAX_COUNT
    )
    duration: int | None = Field(default=None, gt=0)
    fps: int | None = Field(default=None, gt=0)
    aspectRatio: LTXVideoGenAspectRatio | None = None
    resolution: LTXVideoGenResolution | None = None
    videoAssetId: str | None = None
    replacedWindow: ReplacedWindow | None = None
    provider: Literal["local", "api"] | None = None
    mediaType: Literal["video", "image"] = "video"

    @field_validator("imageAssetId", "lastImageAssetId", "videoAssetId", mode="before")
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

    @field_validator("imageAssetIds")
    @classmethod
    def _asset_ids_must_be_present(cls, value: list[str]) -> list[str]:
        if any(not asset_id.strip() for asset_id in value):
            raise ValueError("Asset id cannot be blank")
        return [asset_id.strip() for asset_id in value]

    @model_validator(mode="after")
    def _last_frame_needs_first(self) -> RemoteEnhancePromptRequest:
        if self.lastImageAssetId and not self.imageAssetId:
            raise ValueError("Last frame requires a first-frame image")
        if self.videoAssetId and self.lastImageAssetId:
            raise ValueError("A video cannot be combined with a last-frame image")
        if self.videoAssetId and self.imageAssetId and self.icLoraId is None:
            raise ValueError("A video combines with an image only for an IC-LoRA look image")
        if (
            not self.prompt.strip()
            and not self.imageAssetId
            and not self.imageAssetIds
            and not self.keyframeAssets
            and not self.videoAssetId
        ):
            raise ValueError("Prompt is required unless an image or video is provided")
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
    try:
        desktop_req = EnhancePromptRequest(
            prompt=req.prompt,
            negativePrompt=req.negativePrompt,
            loraCatalogIds=req.loraCatalogIds,
            customLoraIds=req.customLoraIds,
            icLoraId=req.icLoraId,
            imageAssetId=req.imageAssetId,
            lastImageAssetId=req.lastImageAssetId,
            imageAssetIds=req.imageAssetIds,
            keyframeAssets=req.keyframeAssets,
            duration=req.duration,
            fps=req.fps,
            aspectRatio=req.aspectRatio,
            resolution=req.resolution,
            videoAssetId=req.videoAssetId,
            replacedWindow=req.replacedWindow,
            provider=provider,
            mediaType=req.mediaType,
        )
    except ValidationError as exc:
        # The Remote model does not repeat the selection rules. A bad mix is the client's error.
        reason = str(exc.errors()[0]["msg"]).removeprefix("Value error, ")
        raise HTTPError(422, reason) from exc
    try:
        return handler.prompt_enhancement.enhance(
            resolve_enhance_media_assets(handler.assets, desktop_req)
        )
    except HTTPError as exc:
        raise _sanitize_enhance_http_error(exc) from exc
