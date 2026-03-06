"""Gallery handler for listing and managing local generated assets."""

from __future__ import annotations

import hashlib
import logging
import math
import os
from pathlib import Path

from _routes._errors import HTTPError
from api_types import GalleryAsset, GalleryAssetType, GalleryListResponse

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".webm"})
ALL_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

# Known filename prefixes that indicate which model produced the file.
_MODEL_PREFIXES: list[tuple[str, str]] = [
    ("zit_image_", "zit"),
    ("api_image_", "api"),
    ("ltx_fast_", "ltx-fast"),
    ("ltx_", "ltx"),
    ("api_video_", "api-video"),
    ("seedance_", "seedance"),
    ("nano_banana_", "nano-banana"),
    ("ic_lora_", "ic-lora"),
    ("retake_", "retake"),
]


def _parse_model_name(filename: str) -> str | None:
    """Extract model name from known filename prefixes."""
    lower = filename.lower()
    for prefix, model in _MODEL_PREFIXES:
        if lower.startswith(prefix):
            return model
    return None


def _classify_file(ext: str) -> GalleryAssetType | None:
    """Return 'image' or 'video' based on extension, or None if unsupported."""
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def _asset_id(filepath: Path) -> str:
    """Produce a stable, short identifier from the file path."""
    return hashlib.sha256(str(filepath).encode()).hexdigest()[:16]


class GalleryHandler:
    """Scans an output directory for generated image/video files."""

    def __init__(self, outputs_dir: Path) -> None:
        self._outputs_dir = outputs_dir

    def list_local_assets(
        self,
        page: int = 1,
        per_page: int = 50,
        asset_type: str = "all",
    ) -> GalleryListResponse:
        """List generated assets with pagination and optional type filtering."""
        if page < 1:
            page = 1
        if per_page < 1:
            per_page = 50

        allowed_exts: frozenset[str]
        if asset_type == "image":
            allowed_exts = IMAGE_EXTENSIONS
        elif asset_type == "video":
            allowed_exts = VIDEO_EXTENSIONS
        else:
            allowed_exts = ALL_EXTENSIONS

        assets: list[GalleryAsset] = []

        if self._outputs_dir.is_dir():
            for entry in self._outputs_dir.iterdir():
                if not entry.is_file():
                    continue
                ext = entry.suffix.lower()
                if ext not in allowed_exts:
                    continue

                file_type = _classify_file(ext)
                if file_type is None:
                    continue

                stat = entry.stat()
                # Use st_ctime_ns for cross-platform compatibility (st_ctime
                # property is deprecated on Python 3.12+).
                created_at_secs = stat.st_ctime_ns / 1_000_000_000
                assets.append(
                    GalleryAsset(
                        id=_asset_id(entry),
                        filename=entry.name,
                        path=str(entry),
                        url=f"/api/gallery/local/file/{entry.name}",
                        type=file_type,
                        size_bytes=stat.st_size,
                        created_at=str(created_at_secs),
                        model_name=_parse_model_name(entry.name),
                    )
                )

        # Sort by created_at descending (newest first).
        assets.sort(key=lambda a: a.created_at, reverse=True)

        total = len(assets)
        total_pages = max(1, math.ceil(total / per_page))
        start = (page - 1) * per_page
        end = start + per_page

        return GalleryListResponse(
            items=assets[start:end],
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
        )

    def delete_local_asset(self, asset_id: str) -> None:
        """Delete a local asset file by its ID."""
        if not self._outputs_dir.is_dir():
            raise HTTPError(404, f"Asset {asset_id} not found")

        for entry in self._outputs_dir.iterdir():
            if not entry.is_file():
                continue
            ext = entry.suffix.lower()
            if ext not in ALL_EXTENSIONS:
                continue
            if _asset_id(entry) == asset_id:
                try:
                    os.remove(entry)
                except OSError as exc:
                    raise HTTPError(500, f"Failed to delete asset: {exc}") from exc
                return

        raise HTTPError(404, f"Asset {asset_id} not found")
