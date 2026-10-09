from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Protocol, cast

import av
from PIL import Image

from api_types import MediaKind
from server_utils.oriented_image import open_oriented_image
from services.ffmpeg import FFMPEG_PROTOCOL_WHITELIST, run_ffmpeg

logger = logging.getLogger(__name__)
_MAX_EDGE = 512
_JPEG_QUALITY = 85
# A light-theme transparency grid, close to the design system one (--checkerboard-base /
# --checkerboard-square). The sizes differ from the CSS, and the dark theme has other colors.
_GRID_BASE = (255, 255, 255)
_GRID_SQUARE = (232, 232, 232)
_GRID_SIZE = 16
_ALPHA_FRAME_TIMEOUT_SECONDS = 30.0


class ThumbnailWriter(Protocol):
    def write(self, source_path: str, media_kind: MediaKind) -> str | None:
        """JPEG path beside the media, or None (audio / failure). Never raises."""
        ...


class PillowAvThumbnailWriter:
    def write(self, source_path: str, media_kind: MediaKind) -> str | None:
        try:
            if media_kind == "audio":
                return None
            source = Path(source_path)
            dest = source.with_name(f"{source.stem}-thumb.jpg")
            image = _load_rgb(source, media_kind)
            if image is None:
                return None
            image.thumbnail((_MAX_EDGE, _MAX_EDGE))
            image.save(dest, "JPEG", quality=_JPEG_QUALITY)
            return str(dest)
        except Exception:
            logger.warning("thumbnail write failed for %s", source_path, exc_info=True)
            return None


def _load_rgb(source: Path, media_kind: MediaKind) -> Image.Image | None:
    if media_kind == "image":
        image = open_oriented_image(source)
        if _has_transparency(image):
            # Scale first, so the grid is not drawn at the size of a large import.
            rgba = image.convert("RGBA")
            rgba.thumbnail((_MAX_EDGE, _MAX_EDGE))
            return _over_grid(rgba)
        return image.convert("RGB")
    if media_kind == "video":
        container = av.open(str(source))
        try:
            if _is_vp9_alpha(container):
                return _load_alpha_over_grid(source)
            for frame in container.decode(video=0):
                to_image = getattr(frame, "to_image")
                pil_image = cast(Image.Image, to_image())
                return pil_image.convert("RGB")
        finally:
            container.close()
        return None
    return None


def _has_transparency(image: Image.Image) -> bool:
    """An alpha band, or a palette (GIF, PNG8) with a transparent index."""
    return image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info


def _is_vp9_alpha(container: object) -> bool:
    """A VP9 WebM with an alpha plane, which Matroska flags with ``alpha_mode``."""
    streams = cast(list[object], getattr(getattr(container, "streams", None), "video", []))
    if not streams:
        return False
    stream = streams[0]
    codec_name = getattr(getattr(stream, "codec_context", None), "name", None)
    metadata = cast(dict[str, str], getattr(stream, "metadata", {}))
    return codec_name == "vp9" and metadata.get("alpha_mode") == "1"


def _load_alpha_over_grid(source: Path) -> Image.Image:
    """First frame of a VP9 alpha WebM, over the transparency grid.

    PyAV's native VP9 decoder drops the alpha plane, so a cutout would show its
    source frame. libvpx-vp9 keeps alpha. A JPEG has none, so the grid shows through.
    The frame is scaled before it is written, so a large import stays cheap.
    """
    fit = f"scale='min({_MAX_EDGE},iw)':'min({_MAX_EDGE},ih)':force_original_aspect_ratio=decrease"
    with tempfile.TemporaryDirectory() as tmp:
        frame_path = Path(tmp) / "frame.png"
        run_ffmpeg(
            [
                "-y",
                "-c:v",
                "libvpx-vp9",
                "-protocol_whitelist",
                FFMPEG_PROTOCOL_WHITELIST,
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                fit,
                str(frame_path),
            ],
            timeout=_ALPHA_FRAME_TIMEOUT_SECONDS,
        )
        with Image.open(frame_path) as frame:
            rgba = frame.convert("RGBA")
    return _over_grid(rgba)


def _over_grid(rgba: Image.Image) -> Image.Image:
    """JPEG has no alpha, so the transparency grid shows where the image is clear."""
    grid = Image.new("RGB", rgba.size, _GRID_BASE)
    for top in range(0, rgba.height, _GRID_SIZE):
        for left in range(0, rgba.width, _GRID_SIZE):
            if (top // _GRID_SIZE + left // _GRID_SIZE) % 2 == 1:
                grid.paste(_GRID_SQUARE, (left, top, left + _GRID_SIZE, top + _GRID_SIZE))
    grid.paste(rgba, mask=rgba.getchannel("A"))
    return grid
