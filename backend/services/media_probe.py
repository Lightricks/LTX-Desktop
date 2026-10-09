from __future__ import annotations

import logging
import mimetypes
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Literal, assert_never, cast

import av
from PIL import Image

from server_utils.oriented_image import open_oriented_image
from services.records import MediaError
from api_types import (
    AudioAssetMetadata,
    AudioMeta,
    AssetMetadata,
    ImageAssetMetadata,
    ImageMeta,
    MediaKind,
    VideoAssetMetadata,
    VideoMeta,
)

if TYPE_CHECKING:
    import numpy as np

ALLOWED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
ALLOWED_VIDEO_SUFFIXES = frozenset({".mp4", ".webm", ".mov"})
ALLOWED_AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".m4a"})
ALLOWED_INGEST_SUFFIXES = (
    ALLOWED_IMAGE_SUFFIXES | ALLOWED_VIDEO_SUFFIXES | ALLOWED_AUDIO_SUFFIXES
)

MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_AUDIO_BYTES = 100 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000

_EXT_BY_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
}

_IMAGE_FORMAT_MIME = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    # A GIF is only made here, by a cutout recipe. Its suffix is not an allowed upload.
    "GIF": "image/gif",
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AvProbe:
    width: int
    height: int
    duration_ms: int
    audio_stream_count: int
    size_bytes: int
    fps: float | None = None


def extension_for_mime(mime_type: str) -> str:
    mapped = _EXT_BY_MIME.get(mime_type)
    if mapped is not None:
        return mapped
    guessed = mimetypes.guess_extension(mime_type)
    if guessed is not None:
        return guessed
    return ".bin"


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected int, got {type(value)!r}")
    return value


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError(f"expected float, got {type(value)!r}")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Fraction):
        return float(value)
    raise TypeError(f"expected float, got {type(value)!r}")


def _duration_ms(container: object, stream: object) -> int:
    container_duration = getattr(container, "duration", None)
    if container_duration is not None:
        seconds = _as_float(container_duration) / 1_000_000
        if seconds > 0:
            return round(seconds * 1000)
    stream_duration = getattr(stream, "duration", None)
    time_base = getattr(stream, "time_base", None)
    if stream_duration is not None and time_base is not None:
        seconds = _as_float(stream_duration) * _as_float(time_base)
        if seconds > 0:
            return round(seconds * 1000)
    raise ValueError("failed to compute media duration")


def _tagged_rotation(raw: object) -> int | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return int(round(float(raw))) % 360
    except ValueError:
        return None


def _side_data_rotation(container: object, stream: object) -> int | None:
    """Display-matrix rotation from the first decoded frame, or None when it cannot be read."""
    decode = getattr(container, "decode", None)
    if not callable(decode):
        return None
    try:
        frame = next(cast(Iterator[object], decode(stream)), None)
    except Exception:
        return None
    if frame is None:
        return None
    rotation = getattr(frame, "rotation", None)
    if isinstance(rotation, bool) or not isinstance(rotation, int):
        return None
    return rotation % 360


def _display_size(
    stream: object, container: object, width: int, height: int
) -> tuple[int, int]:
    """Coded size swapped for a 90° or 270° display rotation, matching oriented images."""
    metadata = getattr(stream, "metadata", None)
    tags = cast("dict[str, object]", metadata) if isinstance(metadata, dict) else {}
    rotation = _tagged_rotation(tags.get("rotate"))
    if rotation is None:
        rotation = _side_data_rotation(container, stream)
    if rotation in (90, 270):
        return height, width
    return width, height


def _probe_av(path: Path, *, required_type: Literal["video", "audio"]) -> _AvProbe:
    try:
        container: object = av.open(str(path))  # type: ignore[reportUnknownMemberType]
    except Exception as exc:
        logger.warning("unreadable media file %s: %s", path, exc)
        raise MediaError("UNREADABLE_MEDIA", "unreadable media file") from exc
    try:
        streams_obj = getattr(container, "streams")
        streams = list(cast(list[object], streams_obj))
        matched: object | None = None
        audio_stream_count = 0
        for stream in streams:
            stream_type = str(getattr(stream, "type", ""))
            if stream_type == "audio":
                audio_stream_count += 1
            if matched is None and stream_type == required_type:
                matched = stream
        fps: float | None = None
        if matched is None:
            logger.warning("no %s track in %s", required_type, path)
            raise MediaError("UNREADABLE_MEDIA", f"no {required_type} track")
        codec_context = getattr(matched, "codec_context", None)
        if required_type == "video":
            width = _as_int(getattr(codec_context, "width", 0))
            height = _as_int(getattr(codec_context, "height", 0))
            if width <= 0 or height <= 0:
                logger.warning(
                    "invalid video dimensions for %s: %sx%s", path, width, height
                )
                raise MediaError(
                    "UNREADABLE_MEDIA",
                    f"invalid video dimensions: {width}x{height}",
                )
            width, height = _display_size(matched, container, width, height)
            fps = _stream_fps(matched)
        elif required_type == "audio":
            width = 0
            height = 0
        else:
            assert_never(required_type)
        duration_ms = _duration_ms(container, matched)
        if duration_ms < 1:
            logger.warning("invalid %s duration for %s", required_type, path)
            raise MediaError(
                "UNREADABLE_MEDIA", f"invalid {required_type} duration"
            )
        return _AvProbe(
            width=width,
            height=height,
            duration_ms=duration_ms,
            audio_stream_count=audio_stream_count,
            size_bytes=path.stat().st_size,
            fps=fps,
        )
    finally:
        close = getattr(container, "close", None)
        if callable(close):
            close()


def _stream_fps(stream: object) -> float | None:
    """None when the container has no usable rate. Ingest stores that and continues."""
    rate = getattr(stream, "average_rate", None)
    if rate is None:
        return None
    try:
        fps = _as_float(rate)
    except TypeError:
        return None
    if fps <= 0:
        return None
    return fps


def video_fps(path: Path) -> float:
    """Frame rate of the first video stream. Ingest stores this on the asset."""
    try:
        container: object = av.open(str(path))  # type: ignore[reportUnknownMemberType]
    except Exception as exc:
        logger.warning("unreadable media file %s: %s", path, exc)
        raise MediaError("UNREADABLE_MEDIA", "unreadable media file") from exc
    try:
        streams_obj = getattr(container, "streams")
        streams = list(cast(list[object], streams_obj))
        matched: object | None = None
        for stream in streams:
            if str(getattr(stream, "type", "")) == "video":
                matched = stream
                break
        if matched is None:
            raise MediaError("UNREADABLE_MEDIA", "no video track")
        fps = _stream_fps(matched)
        if fps is None:
            raise MediaError("UNREADABLE_MEDIA", "missing frame rate")
        return fps
    finally:
        close = getattr(container, "close", None)
        if callable(close):
            close()


def video_frame_count(path: Path) -> int:
    """Frames in the first video stream. Counts packets, so a container with no stored count works."""
    try:
        container = av.open(str(path))  # type: ignore[reportUnknownMemberType]
    except Exception as exc:
        logger.warning("unreadable media file %s: %s", path, exc)
        raise MediaError("UNREADABLE_MEDIA", "unreadable media file") from exc
    try:
        if not container.streams.video:
            raise MediaError("UNREADABLE_MEDIA", "no video track")
        stream = container.streams.video[0]
        return sum(1 for packet in container.demux(stream) if packet.size > 0)  # type: ignore[reportUnknownMemberType, reportUnknownVariableType]
    finally:
        container.close()


def validate_ingest_source(source: Path) -> None:
    suffix = source.suffix.lower()
    if suffix not in ALLOWED_INGEST_SUFFIXES:
        logger.warning("unsupported media type for %s", source)
        raise MediaError("UNSUPPORTED_MEDIA", "unsupported media type")
    if suffix in ALLOWED_IMAGE_SUFFIXES:
        limit = MAX_IMAGE_BYTES
    elif suffix in ALLOWED_AUDIO_SUFFIXES:
        limit = MAX_AUDIO_BYTES
    else:
        return
    try:
        size_bytes = source.stat().st_size
    except OSError as exc:
        logger.warning("unreadable media file %s: %s", source, exc)
        raise MediaError("UNREADABLE_MEDIA", "unreadable media file") from exc
    if size_bytes > limit:
        logger.warning(
            "file too large %s: %s bytes exceeds %s", source, size_bytes, limit
        )
        raise MediaError("FILE_TOO_LARGE", "file too large")


def probe_file(path: Path) -> tuple[MediaKind, str, AssetMetadata]:
    try:
        return _probe_file(path)
    except MediaError:
        raise
    except Exception as exc:
        logger.warning("unreadable media file %s: %s", path, exc)
        raise MediaError("UNREADABLE_MEDIA", "unreadable media file") from exc


def _probe_file(path: Path) -> tuple[MediaKind, str, AssetMetadata]:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    suffix = path.suffix.lower()
    if mime.startswith("image/") or suffix in ALLOWED_IMAGE_SUFFIXES:
        # Header only. exif_transpose decodes the bitmap, and a rotation does
        # not change the pixel count, so reject oversized images before that.
        with Image.open(path) as raw:
            width, height = raw.size
            fmt = str(raw.format or "").upper()
        mapped = _IMAGE_FORMAT_MIME.get(fmt)
        if mapped is None:
            logger.warning("unsupported image format %s for %s", fmt, path)
            raise MediaError("UNREADABLE_MEDIA", "unsupported image format")
        if width <= 0 or height <= 0 or (width * height) > MAX_IMAGE_PIXELS:
            logger.warning(
                "image dimensions too large %s: %sx%s", path, width, height
            )
            raise MediaError(
                "IMAGE_DIMENSIONS_TOO_LARGE",
                "image dimensions too large",
            )
        image = open_oriented_image(path)
        width, height = image.size
        return (
            "image",
            mapped,
            ImageAssetMetadata(
                mediaType="image",
                metadata=ImageMeta(width=width, height=height),
            ),
        )
    if mime.startswith("video/") or suffix in ALLOWED_VIDEO_SUFFIXES:
        mime = mime if mime.startswith("video/") else "video/mp4"
        probed = _probe_av(path, required_type="video")
        return (
            "video",
            mime,
            VideoAssetMetadata(
                mediaType="video",
                metadata=VideoMeta(
                    width=probed.width,
                    height=probed.height,
                    durationMs=probed.duration_ms,
                    sizeBytes=probed.size_bytes,
                    audioStreamCount=probed.audio_stream_count,
                    fps=probed.fps,
                ),
            ),
        )
    if mime.startswith("audio/") or suffix in ALLOWED_AUDIO_SUFFIXES:
        mime = mime if mime.startswith("audio/") else "audio/wav"
        probed = _probe_av(path, required_type="audio")
        bitrate = round((probed.size_bytes * 8) / (probed.duration_ms / 1000))
        if bitrate < 1:
            logger.warning("invalid audio bitrate for %s", path)
            raise MediaError("UNREADABLE_MEDIA", "invalid audio bitrate")
        return (
            "audio",
            mime,
            AudioAssetMetadata(
                mediaType="audio",
                metadata=AudioMeta(
                    durationMs=probed.duration_ms,
                    bitrate=bitrate,
                ),
            ),
        )
    logger.warning("unsupported media type for %s", path)
    raise MediaError("UNSUPPORTED_MEDIA", "unsupported media type")


def media_has_audio(path: str) -> bool:
    with av.open(path) as container:
        return any(stream.type == "audio" for stream in container.streams)


def _frame_timestamp(frame: object, time_base: object) -> float | None:
    timestamp = cast(float | None, getattr(frame, "time", None))
    if timestamp is None:
        pts = getattr(frame, "pts", None)
        if pts is not None and time_base is not None:
            timestamp = float(pts * time_base)
    return timestamp


def sample_gray_frames(
    path: str, start: float, count: int, width: int, height: int
) -> list[np.ndarray]:
    import numpy as np

    try:
        container = av.open(path)
    except av.FFmpegError:
        return []
    try:
        stream = container.streams.video[0]
        time_base = stream.time_base
        if start > 0 and time_base is not None:
            try:
                container.seek(int(start / time_base), stream=stream)
            except av.FFmpegError:
                pass
        frames: list[np.ndarray] = []
        for frame in container.decode(video=0):
            timestamp = _frame_timestamp(frame, time_base)
            if timestamp is None or timestamp + 1e-3 < start:
                continue
            if len(frames) >= count:
                break
            gray = frame.reformat(width=width, height=height, format="gray")
            frames.append(gray.to_ndarray().astype(np.float32))
        return frames
    finally:
        container.close()


def sample_rgb_mean_std(
    path: str,
    time_range: tuple[float, float],
    fps: float,
    width: int,
    height: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    import numpy as np

    start, end = time_range
    chunks: list[np.ndarray] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        if start > 0 and time_base is not None:
            try:
                container.seek(int(start / time_base), stream=stream)
            except av.FFmpegError:
                pass
        for frame in container.decode(video=0):
            timestamp = _frame_timestamp(frame, time_base)
            if timestamp is None or timestamp + 1.0 / fps < start:
                continue
            if timestamp >= end:
                break
            pixels = frame.to_ndarray(format="rgb24")
            if pixels.shape[1] != width or pixels.shape[0] != height:
                pixels = np.asarray(
                    Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS)
                )
            chunks.append(pixels.reshape(-1, 3).astype(np.float64))
    if not chunks:
        return None
    stacked = np.concatenate(chunks, axis=0)
    mean_vals = stacked.mean(axis=0)
    std_vals = stacked.std(axis=0)
    return (
        (float(mean_vals[0]), float(mean_vals[1]), float(mean_vals[2])),
        (float(std_vals[0]), float(std_vals[1]), float(std_vals[2])),
    )

