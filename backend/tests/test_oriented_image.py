"""EXIF orientation is baked in wherever a user photo is decoded."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from PIL import Image
from services.media_probe import probe_file
from services.records import MediaError
from services.services_utils import compute_edit_dimensions
from services.thumbnails import PillowAvThumbnailWriter
from services.video_processor.video_processor_impl import VideoProcessorImpl
from server_utils.oriented_image import (
    open_oriented_image,
    open_oriented_rgb,
    oriented_image_file,
    oriented_image_payload,
)
from tests.fakes import FakeResponse

_RAW_SIZE = (32, 16)
# Pillow's EXIF map: 6 is 90° clockwise (ROTATE_270), 8 is 90° counter-clockwise.
_TRANSPOSE = {
    6: Image.Transpose.ROTATE_270,
    8: Image.Transpose.ROTATE_90,
}


def _marked_image(size: tuple[int, int] = _RAW_SIZE) -> Image.Image:
    image = Image.new("RGB", size, (10, 20, 30))
    image.putpixel((0, 0), (255, 0, 0))
    image.putpixel((size[0] - 1, 0), (0, 255, 0))
    image.putpixel((0, size[1] - 1), (0, 0, 255))
    image.putpixel((size[0] - 1, size[1] - 1), (255, 255, 0))
    return image


def _save_oriented(path: Path, orientation: int, *, fmt: str = "JPEG") -> None:
    image = _marked_image()
    exif = image.getexif()
    exif[0x0112] = orientation
    image.save(path, format=fmt, exif=exif, quality=95)


def _display_rgb(path: Path, orientation: int) -> Image.Image:
    with Image.open(path) as raw:
        method = _TRANSPOSE.get(orientation)
        image = raw if method is None else raw.transpose(method)
        return image.convert("RGB")


def test_open_oriented_image_rotates_phone_orientations(tmp_path: Path) -> None:
    for orientation in (6, 8):
        path = tmp_path / f"phone-{orientation}.jpg"
        _save_oriented(path, orientation)
        expected = _display_rgb(path, orientation)
        loaded = open_oriented_rgb(path)
        assert loaded.size == expected.size
        assert loaded.tobytes() == expected.tobytes()
        assert loaded.getexif().get(0x0112) in (None, 1)


def test_open_oriented_image_leaves_upright_pixels_and_format(tmp_path: Path) -> None:
    path = tmp_path / "upright.png"
    _marked_image().save(path, format="PNG")
    loaded = open_oriented_image(path)
    assert loaded.format == "PNG"
    assert loaded.size == _RAW_SIZE
    assert loaded.tobytes() == _marked_image().tobytes()


def test_oriented_payload_keeps_original_bytes_when_upright(tmp_path: Path) -> None:
    path = tmp_path / "plain.jpg"
    _marked_image().save(path, format="JPEG", quality=95)
    raw, mime = oriented_image_payload(path)
    assert mime == "image/jpeg"
    assert raw == path.read_bytes()


def test_oriented_payload_reencodes_sideways_jpeg_as_jpeg(tmp_path: Path) -> None:
    path = tmp_path / "sideways.jpg"
    _save_oriented(path, 6)
    raw, mime = oriented_image_payload(path)
    assert mime == "image/jpeg"
    with Image.open(io.BytesIO(raw)) as sent:
        assert sent.format == "JPEG"
        assert sent.size == _display_rgb(path, 6).size
        assert sent.getexif().get(0x0112, 1) in (0, 1)
    assert path.read_bytes()[:2] == b"\xff\xd8"


def test_oriented_payload_reencodes_sideways_png_as_png(tmp_path: Path) -> None:
    path = tmp_path / "sideways.png"
    _save_oriented(path, 6, fmt="PNG")
    raw, mime = oriented_image_payload(path)
    assert mime == "image/png"
    with Image.open(io.BytesIO(raw)) as sent:
        assert sent.format == "PNG"
        assert sent.size == _display_rgb(path, 6).size
        assert sent.tobytes() == _display_rgb(path, 6).tobytes()


def test_oriented_image_file_removes_temp_jpeg(tmp_path: Path) -> None:
    path = tmp_path / "sideways.jpg"
    _save_oriented(path, 6)
    with oriented_image_file(path) as upload_path:
        assert upload_path.suffix == ".jpg"
        assert upload_path != path
        with Image.open(upload_path) as sent:
            assert sent.format == "JPEG"
            assert sent.getexif().get(0x0112, 1) in (0, 1)
        held = upload_path
    assert not held.exists()

    upright = tmp_path / "upright.png"
    _marked_image().save(upright, format="PNG")
    with oriented_image_file(upright) as upload_path:
        assert upload_path == upright


def test_prepare_image_crops_the_upright_frame(test_state, tmp_path: Path) -> None:
    path = tmp_path / "start.jpg"
    _save_oriented(path, 6)
    expected = _display_rgb(path, 6)
    prepared = test_state.video_generation._prepare_image(str(path), expected.width, expected.height)
    assert prepared.size == expected.size
    assert prepared.tobytes() == expected.tobytes()


def test_probe_rejects_oversized_image_before_decode(tmp_path: Path) -> None:
    import services.media_probe as media_probe

    path = tmp_path / "wide.png"
    _marked_image().save(path, format="PNG")
    original_limit = media_probe.MAX_IMAGE_PIXELS
    original_open = media_probe.open_oriented_image

    def _decode_should_not_run(_path: Path) -> Image.Image:
        raise AssertionError("oversized image was decoded")

    media_probe.MAX_IMAGE_PIXELS = 1
    media_probe.open_oriented_image = _decode_should_not_run
    try:
        with pytest.raises(MediaError) as caught:
            probe_file(path)
        assert caught.value.code == "IMAGE_DIMENSIONS_TOO_LARGE"
    finally:
        media_probe.MAX_IMAGE_PIXELS = original_limit
        media_probe.open_oriented_image = original_open


def test_probe_reports_upright_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "asset.jpg"
    _save_oriented(path, 6)
    kind, mime, metadata = probe_file(path)
    assert kind == "image"
    assert mime == "image/jpeg"
    assert metadata.mediaType == "image"
    assert metadata.metadata.width == _RAW_SIZE[1]
    assert metadata.metadata.height == _RAW_SIZE[0]


def test_thumbnail_is_upright(tmp_path: Path) -> None:
    path = tmp_path / "asset.jpg"
    _save_oriented(path, 8)
    written = PillowAvThumbnailWriter().write(str(path), "image")
    assert written is not None
    with Image.open(written) as thumb:
        assert thumb.size == _display_rgb(path, 8).size


def test_ic_lora_probe_uses_upright_dimensions(test_state, tmp_path: Path) -> None:
    path = tmp_path / "control.jpg"
    _save_oriented(path, 6)
    width, height, frames = test_state.ic_lora._probe_ic_lora_source(str(path), "image", 9)
    assert (width, height, frames) == (_RAW_SIZE[1], _RAW_SIZE[0], 9)


def test_video_processor_reads_upright_bgr(tmp_path: Path) -> None:
    path = tmp_path / "control.jpg"
    _save_oriented(path, 6)
    frame = VideoProcessorImpl().read_image(str(path))
    expected = _display_rgb(path, 6)
    assert frame.shape == (expected.height, expected.width, 3)
    assert tuple(int(channel) for channel in frame[0, 0]) == expected.getpixel((0, 0))[::-1]


def test_image_edit_resizes_the_upright_source(client, test_state, fake_services, tmp_path: Path) -> None:
    test_state.config.local_generations_mode = "unsupported"
    test_state.state.app_settings.fal_api_key = "fal-key"
    path = tmp_path / "edit.jpg"
    _save_oriented(path, 6)
    expected = _display_rgb(path, 6)

    response = client.post(
        "/api/generate-image",
        json={"prompt": "x", "imagePath": str(path), "strength": 0.6},
    )
    assert response.status_code == 200
    sent = fake_services.zit_api_client.image_to_image_calls[0]["image_bytes"]
    with Image.open(io.BytesIO(sent)) as image:
        assert image.size == compute_edit_dimensions(expected.width, expected.height)


def test_gemini_enhance_sends_upright_image(client, test_state, tmp_path: Path) -> None:
    test_state.state.app_settings.gemini_api_key = "key"
    test_state.http.queue(
        "post",
        FakeResponse(
            status_code=200,
            json_payload={"candidates": [{"content": {"parts": [{"text": "enhanced"}]}}]},
        ),
    )
    path = tmp_path / "frame.jpg"
    _save_oriented(path, 8)

    response = client.post(
        "/api/enhance-prompt",
        json={"prompt": "a cat", "provider": "api", "imagePath": str(path)},
    )
    assert response.status_code == 200
    parts = test_state.http.calls[-1].json_payload["contents"][0]["parts"]
    inline = next(part["inlineData"] for part in parts if "inlineData" in part)
    with Image.open(io.BytesIO(base64.b64decode(inline["data"]))) as sent:
        assert sent.size == _display_rgb(path, 8).size


def test_forced_api_uploads_upright_start_and_end_frames(
    client, test_state, fake_services, tmp_path: Path
) -> None:
    test_state.config.local_generations_mode = "unsupported"
    test_state.state.app_settings.ltx_api_key = "api-key"
    start = tmp_path / "start.jpg"
    end = tmp_path / "end.jpg"
    _save_oriented(start, 6)
    _save_oriented(end, 8)

    response = client.post(
        "/api/generate",
        json={
            "prompt": "Animate this frame",
            "resolution": "2160p",
            "model": "pro",
            "duration": 8,
            "fps": 25,
            "audio": False,
            "cameraMotion": "jib_up",
            "imagePath": str(start),
            "lastImagePath": str(end),
        },
    )
    assert response.status_code == 200, response.text
    uploads = fake_services.ltx_api_client.upload_file_calls
    assert len(uploads) == 2
    expected_sizes = (_display_rgb(start, 6).size, _display_rgb(end, 8).size)
    for upload, expected_size in zip(uploads, expected_sizes, strict=True):
        with Image.open(io.BytesIO(upload["contents"])) as sent:
            assert sent.format == "JPEG"
            assert sent.size == expected_size
            assert sent.getexif().get(0x0112, 1) in (0, 1)
        assert not Path(upload["file_path"]).exists()
    assert start.exists() and end.exists()
