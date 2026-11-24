from __future__ import annotations

from typing import TypeVar

from api_types import IcLoraCatalogItem, LoraCatalogItem, MediaSpec

# Demo clips for catalog cards live under this CDN prefix. Catalog JSON stores relative
# paths (e.g. "fpv-motion_web.mp4", "ltx-team/DayToNight-light.mp4"); list APIs join the
# base before returning so the UI always gets an absolute URL.
LORA_CATALOG_MEDIA_BASE_URL = "https://storage.googleapis.com/ltx-desktop-artifacts/assets/"

_T = TypeVar("_T", LoraCatalogItem, IcLoraCatalogItem)


def resolve_media_url(path: str | None, *, base_url: str = LORA_CATALOG_MEDIA_BASE_URL) -> str | None:
    """Join a relative catalog media path to the CDN base; leave absolute URLs alone."""
    if path is None or path == "":
        return path
    if path.startswith(("http://", "https://")):
        return path
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def with_resolved_media(item: _T, *, base_url: str = LORA_CATALOG_MEDIA_BASE_URL) -> _T:
    """Return a copy of ``item`` with media URLs resolved for serving to the UI."""
    media = item.media
    if media is None:
        return item
    thumbnail = resolve_media_url(media.thumbnail, base_url=base_url)
    demo_video = resolve_media_url(media.demo_video, base_url=base_url)
    if thumbnail == media.thumbnail and demo_video == media.demo_video:
        return item
    return item.model_copy(update={"media": MediaSpec(thumbnail=thumbnail, demo_video=demo_video)})
