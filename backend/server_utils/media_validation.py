"""Media validation helpers for filesystem path inputs.

These helpers are intentionally handler-oriented (raise HTTPError) so endpoints
return HTTP 400 for invalid user-supplied paths instead of leaking exceptions.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from _routes._errors import HTTPError
from services.video_processor.video_processor import VideoProcessor

_MAX_IMAGE_BYTES = 50 * 1024 * 1024
_MAX_AUDIO_BYTES = 100 * 1024 * 1024
_MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024
_MAX_IMAGE_PIXELS = 50_000_000

_ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "GIF", "BMP", "TIFF"}


def normalize_optional_path(value: str | None) -> str | None:
    """Treat None/empty/whitespace-only values as 'not provided'."""

    if value is None:
        return None
    if value.strip() == "":
        return None
    return value


def _assert_is_file(file_path: Path, *, kind: str, raw_path: str) -> None:
    try:
        is_file = file_path.is_file()
    except OSError:
        is_file = False
    if not is_file:
        raise HTTPError(400, f"{kind} file not found: {raw_path}")


def _assert_max_bytes(file_path: Path, *, limit_bytes: int, error_detail: str) -> None:
    try:
        size_bytes = file_path.stat().st_size
    except OSError:
        raise HTTPError(400, error_detail) from None
    if size_bytes > limit_bytes:
        raise HTTPError(400, error_detail)


def validate_image_file(path: str) -> Path:
    """Validate that `path` points to a supported image file on disk."""

    try:
        file_path = Path(path)
    except Exception:
        raise HTTPError(400, f"Image file not found: {path}") from None

    _assert_is_file(file_path, kind="Image", raw_path=path)
    _assert_max_bytes(file_path, limit_bytes=_MAX_IMAGE_BYTES, error_detail=f"Image file too large: {path}")

    try:
        with Image.open(file_path) as img:
            fmt = str(img.format or "").upper()
            width, height = img.size
            img.verify()
    except Exception:
        raise HTTPError(400, f"Invalid image file: {path}") from None

    if fmt not in _ALLOWED_IMAGE_FORMATS:
        raise HTTPError(400, f"Invalid image file: {path}")

    if width <= 0 or height <= 0 or (width * height) > _MAX_IMAGE_PIXELS:
        raise HTTPError(400, f"Image dimensions too large: {path}")

    return file_path


def _read_header(file_path: Path, *, num_bytes: int = 64) -> bytes:
    try:
        with file_path.open("rb") as handle:
            return handle.read(num_bytes)
    except OSError:
        return b""


def _sniff_audio(header: bytes, ext: str) -> bool:
    if len(header) < 4:
        return False

    is_wav = len(header) >= 12 and header[0:4] == b"RIFF" and header[8:12] == b"WAVE"
    is_flac = header[0:4] == b"fLaC"
    is_ogg = header[0:4] == b"OggS"

    is_mp3_id3 = header[0:3] == b"ID3"
    has_mp3_frame_sync = len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0

    is_aac_adif = header[0:4] == b"ADIF"
    has_aac_adts_sync = len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xF6) == 0xF0

    has_mp4_ftyp = len(header) >= 8 and header[4:8] == b"ftyp"

    ext = ext.lower()
    if ext == ".wav":
        return is_wav
    if ext == ".flac":
        return is_flac
    if ext == ".ogg":
        return is_ogg
    if ext == ".mp3":
        return is_mp3_id3 or has_mp3_frame_sync
    if ext == ".aac":
        return is_aac_adif or has_aac_adts_sync
    if ext == ".m4a":
        return has_mp4_ftyp

    # Unknown extension: only accept unambiguous signatures (avoid classifying MP4 containers).
    return is_wav or is_flac or is_ogg or is_mp3_id3


def validate_audio_file(path: str) -> Path:
    """Validate that `path` points to a supported audio file on disk."""

    try:
        file_path = Path(path)
    except Exception:
        raise HTTPError(400, f"Audio file not found: {path}") from None

    _assert_is_file(file_path, kind="Audio", raw_path=path)
    _assert_max_bytes(file_path, limit_bytes=_MAX_AUDIO_BYTES, error_detail=f"Audio file too large: {path}")

    header = _read_header(file_path)
    if not _sniff_audio(header, file_path.suffix):
        raise HTTPError(400, f"Invalid audio file: {path}")

    return file_path


def validate_video_file(path: str, video_processor: VideoProcessor) -> Path:
    """Validate a readable video using the injected VideoProcessor service."""

    try:
        file_path = Path(path)
    except Exception:
        raise HTTPError(400, f"Video file not found: {path}") from None
    _assert_is_file(file_path, kind="Video", raw_path=path)
    _assert_max_bytes(file_path, limit_bytes=_MAX_VIDEO_BYTES, error_detail=f"Video file too large: {path}")

    try:
        capture = video_processor.open_video(str(file_path))
        try:
            if not bool(capture.isOpened()):
                raise ValueError("not opened")
            info = video_processor.get_video_info(capture)
            if (
                float(info["fps"]) <= 0
                or int(info["frame_count"]) <= 0
                or int(info["width"]) <= 0
                or int(info["height"]) <= 0
                or video_processor.read_frame(capture, frame_idx=0) is None
            ):
                raise ValueError("invalid metadata")
        finally:
            video_processor.release(capture)
    except Exception:
        raise HTTPError(400, f"Invalid video file: {path}") from None
    return file_path
