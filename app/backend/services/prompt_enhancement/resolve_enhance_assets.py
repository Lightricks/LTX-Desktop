"""Resolve Explore asset ids to filesystem paths for EnhancePromptRequest."""

from __future__ import annotations

from typing import Literal

from _routes._errors import HTTPError
from api_types import EnhancePromptRequest, KeyframeInput
from handlers.asset_handler import AssetHandler


def resolve_enhance_media_assets(
    assets: AssetHandler, req: EnhancePromptRequest
) -> EnhancePromptRequest:
    """Fill imagePath / lastImagePath / videoPath / imagePaths / keyframes from asset ids. No-op when ids are absent."""
    updates: dict[str, object] = {}
    if req.imageAssetId:
        updates["imagePath"] = _asset_path(assets, req.imageAssetId, "image")
        updates["imageAssetId"] = None
    if req.lastImageAssetId:
        updates["lastImagePath"] = _asset_path(assets, req.lastImageAssetId, "image")
        updates["lastImageAssetId"] = None
    if req.imageAssetIds:
        updates["imagePaths"] = [_asset_path(assets, asset_id, "image") for asset_id in req.imageAssetIds]
        updates["imageAssetIds"] = []
    if req.keyframeAssets:
        updates["keyframes"] = [
            KeyframeInput(
                imagePath=_asset_path(assets, keyframe.assetId, "image"),
                frameIndex=keyframe.frameIndex,
                strength=keyframe.strength,
            )
            for keyframe in req.keyframeAssets
        ]
        updates["keyframeAssets"] = []
    if req.videoAssetId:
        updates["videoPath"] = _asset_path(assets, req.videoAssetId, "video")
        updates["videoAssetId"] = None
    if not updates:
        return req
    return req.model_copy(update=updates)


def _asset_path(assets: AssetHandler, asset_id: str, kind: Literal["image", "video"]) -> str:
    asset = assets.get_asset(asset_id)
    if asset.media_kind != kind:
        raise HTTPError(400, f"ASSET_NOT_{kind.upper()}")
    return asset.path
