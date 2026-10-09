"""Resolve Explore asset ids to filesystem paths for EnhancePromptRequest."""

from __future__ import annotations

from _routes._errors import HTTPError
from api_types import EnhancePromptRequest
from handlers.asset_handler import AssetHandler


def resolve_enhance_image_assets(
    assets: AssetHandler, req: EnhancePromptRequest
) -> EnhancePromptRequest:
    """Fill imagePath / lastImagePath from asset ids. No-op when ids are absent."""
    updates: dict[str, str | None] = {}
    if req.imageAssetId:
        updates["imagePath"] = _image_asset_path(assets, req.imageAssetId)
        updates["imageAssetId"] = None
    if req.lastImageAssetId:
        updates["lastImagePath"] = _image_asset_path(assets, req.lastImageAssetId)
        updates["lastImageAssetId"] = None
    if not updates:
        return req
    return req.model_copy(update=updates)


def _image_asset_path(assets: AssetHandler, asset_id: str) -> str:
    asset = assets.get_asset(asset_id)
    if asset.media_kind != "image":
        raise HTTPError(400, "ASSET_NOT_IMAGE")
    return asset.path
