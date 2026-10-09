from pathlib import Path

import pytest
from PIL import Image
from services.thumbnails import PillowAvThumbnailWriter


def _png(path: Path, size: tuple[int, int] = (64, 48)) -> Path:
    Image.new("RGB", size, color=(10, 20, 30)).save(path)
    return path


def test_image_write_returns_jpeg_beside_source(tmp_path: Path) -> None:
    source = _png(tmp_path / "a.png")
    written = PillowAvThumbnailWriter().write(str(source), "image")
    assert written is not None
    thumb = Path(written)
    assert thumb.suffix.lower() == ".jpg"
    assert thumb.parent == source.parent
    with Image.open(thumb) as im:
        assert im.format == "JPEG"


def _half_clear_image(path: Path) -> Path:
    """Red on the left. The right is clear, with green under it as ffmpeg may leave it."""
    image = Image.new("RGBA", (64, 48), (0, 255, 0, 0))
    image.paste((200, 40, 40, 255), (0, 0, 32, 48))
    image.save(path)
    return path


@pytest.mark.parametrize("suffix", [".gif", ".png"])
def test_an_image_thumbnail_shows_the_grid_where_the_image_is_clear(
    tmp_path: Path, suffix: str
) -> None:
    source = _half_clear_image(tmp_path / f"cutout{suffix}")
    written = PillowAvThumbnailWriter().write(str(source), "image")
    assert written is not None
    with Image.open(written) as thumb:
        opaque = thumb.getpixel((8, 24))
        clear = thumb.getpixel((56, 24))
    assert opaque[0] > 150 and opaque[1] < 90  # the red subject
    assert all(channel > 200 for channel in clear)  # the grid, not the green under it


def test_audio_write_returns_none(tmp_path: Path) -> None:
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    assert PillowAvThumbnailWriter().write(str(wav), "audio") is None


def test_write_swallows_errors(tmp_path: Path) -> None:
    missing = tmp_path / "gone.png"
    assert PillowAvThumbnailWriter().write(str(missing), "image") is None


def test_webm_cutout_thumbnail_shows_the_grid_where_the_clip_is_transparent(
    tmp_path: Path,
) -> None:
    """A cutout keeps its source color under alpha 0. The thumbnail must not show it."""
    source = tmp_path / "cutout.webm"
    _bake_half_transparent_webm(source)
    written = PillowAvThumbnailWriter().write(str(source), "video")
    assert written is not None
    with Image.open(written) as thumb:
        width, height = thumb.size
        opaque = thumb.getpixel((8, height // 2))
        transparent = thumb.getpixel((width - 8, height // 2))
    assert opaque[0] > 150 and opaque[1] < 90  # the red subject
    assert all(channel > 200 for channel in transparent)  # the grid, not the red source


def test_a_plain_webm_keeps_the_generic_decoder(tmp_path: Path) -> None:
    """A VP8 WebM is valid input. Only a VP9 alpha stream needs libvpx-vp9."""
    import numpy as np
    import imageio.v2 as imageio

    source = tmp_path / "plain.webm"
    writer = imageio.get_writer(str(source), fps=10, codec="libvpx", macro_block_size=None)
    for _ in range(5):
        writer.append_data(np.full((64, 96, 3), 120, dtype=np.uint8))
    writer.close()
    written = PillowAvThumbnailWriter().write(str(source), "video")
    assert written is not None


def test_a_large_alpha_webm_gets_a_capped_thumbnail(tmp_path: Path) -> None:
    source = tmp_path / "cutout.webm"
    _bake_half_transparent_webm(source)
    written = PillowAvThumbnailWriter().write(str(source), "video")
    assert written is not None
    with Image.open(written) as thumb:
        assert max(thumb.size) <= 512


def _bake_half_transparent_webm(dest: Path) -> None:
    import numpy as np
    from services.features.video.alpha_cutout import bake_alpha_cutout
    import imageio.v2 as imageio

    def mp4(path: Path, frame: "np.ndarray") -> Path:
        writer = imageio.get_writer(str(path), fps=24, codec="libx264", macro_block_size=None)
        for _ in range(9):
            writer.append_data(frame)
        writer.close()
        return path

    source = np.zeros((64, 96, 3), dtype=np.uint8)
    source[:, :] = (200, 40, 40)
    matte = np.zeros((64, 96, 3), dtype=np.uint8)
    matte[:, :48] = 255
    bake_alpha_cutout(mp4(dest.with_name("s.mp4"), source), mp4(dest.with_name("m.mp4"), matte), dest)
