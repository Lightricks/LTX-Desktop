"""Bake EXIF orientation into image pixels.

Phone cameras store the sensor buffer sideways and record the display rotation
in the EXIF Orientation tag. ``Image.open`` leaves that tag unapplied, so every
decode of a user photo goes through this module. The stored file is left as-is.
"""

from __future__ import annotations

import io
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from server_utils.media_validation import image_mime_type

_ORIENTATION = 0x0112
# High enough that a phone JPEG stays a photo, not a multi-tens-of-MB PNG.
_JPEG_QUALITY = 95


def open_oriented_image(path: str | Path) -> Image.Image:
    """Return a loaded image whose pixels match the EXIF display orientation.

    The result is detached from the file. Images with no orientation tag, or
    orientation 1, keep their pixels. The original file format is preserved on
    the returned image so callers can still reject unsupported types.
    """
    with Image.open(path) as raw:
        source_format = raw.format
        loaded = ImageOps.exif_transpose(raw).copy()
    if loaded.format is None:
        loaded.format = source_format
    return loaded


def open_oriented_rgb(path: str | Path) -> Image.Image:
    return open_oriented_image(path).convert("RGB")


def oriented_image_payload(path: str | Path) -> tuple[bytes, str]:
    """Bytes and MIME type for an API that will not apply EXIF orientation.

    Untagged and already-upright images are returned unchanged. A sideways JPEG
    is re-encoded as a JPEG with the rotation baked in and the orientation tag
    omitted. Any other sideways image is re-encoded as PNG for the same reason.
    """
    source = Path(path)
    if _exif_orientation(source) in (0, 1):
        return source.read_bytes(), image_mime_type(str(source))
    encoded = _encode_oriented(source)
    return encoded.data, encoded.mime


@contextmanager
def oriented_image_file(path: str | Path) -> Iterator[Path]:
    """Yield a path whose pixels are display-oriented.

    The original path is yielded when no transpose is required. Otherwise a
    temporary file is written and removed when the context exits.
    """
    source = Path(path)
    if _exif_orientation(source) in (0, 1):
        yield source
        return
    encoded = _encode_oriented(source)
    fd, name = tempfile.mkstemp(suffix=encoded.suffix)
    os.close(fd)
    dest = Path(name)
    try:
        dest.write_bytes(encoded.data)
        yield dest
    finally:
        dest.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _Encoded:
    data: bytes
    mime: str
    suffix: str


def _encode_oriented(source: Path) -> _Encoded:
    image = open_oriented_image(source)
    buffer = io.BytesIO()
    if str(image.format or "").upper() == "JPEG":
        image.convert("RGB").save(buffer, format="JPEG", quality=_JPEG_QUALITY, exif=b"")
        return _Encoded(buffer.getvalue(), "image/jpeg", ".jpg")
    image.save(buffer, format="PNG")
    return _Encoded(buffer.getvalue(), "image/png", ".png")


def _exif_orientation(path: Path) -> int:
    with Image.open(path) as image:
        value = image.getexif().get(_ORIENTATION, 1)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1
