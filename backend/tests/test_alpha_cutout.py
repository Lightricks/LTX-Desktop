"""AlphaGen cutout: the matte and the source become one WebM with a real alpha channel."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image

from api_types import (
    IcLoraRecipeStoredParams,
    LoraEntry,
    VideoAssetMetadata,
    VideoMeta,
)
from services.features.ic_lora_recipes import IC_LORA_RECIPES
from services.features.video.alpha_cutout import (
    bake_alpha_cutout,
    bake_alpha_gif,
    bake_alpha_gif_if_possible,
)
from services.features.video.ic_lora_recipe import IcLoraRecipeExecutor
from services.media_probe import video_frame_count
from services.generation_queue.types import OutputAllocation
from services.records import AssetRecord, GenerationRecord

_SOURCE_RGB = (200, 40, 40)
_MATTE_SIZE = (48, 64)  # height, width


def _write_mp4(path: Path, frame: np.ndarray, *, frames: int = 9) -> Path:
    writer = imageio.get_writer(str(path), fps=24, codec="libx264", macro_block_size=None)
    for _ in range(frames):
        writer.append_data(frame)
    writer.close()
    return path


def _source(path: Path) -> Path:
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    frame[:, :] = _SOURCE_RGB
    return _write_mp4(path, frame)


def _matte(path: Path) -> Path:
    """White left half (subject), black right half (background)."""
    frame = np.zeros((*_MATTE_SIZE, 3), dtype=np.uint8)
    frame[:, : _MATTE_SIZE[1] // 2] = 255
    return _write_mp4(path, frame)


def _first_frame_rgba(webm: Path) -> np.ndarray:
    """Decode with libvpx-vp9. The native decoder drops the alpha plane."""
    raw = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-c:v", "libvpx-vp9",
            "-i", str(webm),
            "-frames:v", "1",
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(*_MATTE_SIZE, 4)


def _stream_info(webm: Path) -> str:
    return subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(webm)],
        capture_output=True,
        text=True,
        check=False,
    ).stderr


def _assert_cutout(webm: Path) -> None:
    assert "alpha_mode" in _stream_info(webm)
    rgba = _first_frame_rgba(webm)
    subject = rgba[_MATTE_SIZE[0] // 2, 8]
    background = rgba[_MATTE_SIZE[0] // 2, _MATTE_SIZE[1] - 8]
    assert subject[3] == 255
    assert background[3] == 0
    # The color is the source color, scaled to the matte size.
    assert all(abs(int(a) - b) < 12 for a, b in zip(subject[:3], _SOURCE_RGB))


def test_bake_uses_the_matte_as_alpha_and_the_matte_size(tmp_path: Path) -> None:
    dest = tmp_path / "cutout.webm"
    bake_alpha_cutout(_source(tmp_path / "source.mp4"), _matte(tmp_path / "matte.mp4"), dest)
    _assert_cutout(dest)


def test_bake_keeps_soft_alpha(tmp_path: Path) -> None:
    """A faint matte value stays faint. A second range expansion would turn 26 into about 12."""
    frame = np.zeros((*_MATTE_SIZE, 3), dtype=np.uint8)
    frame[:, : _MATTE_SIZE[1] // 2] = 128  # half transparent
    frame[:, _MATTE_SIZE[1] // 2 :] = 26  # barely visible
    dest = tmp_path / "cutout.webm"
    bake_alpha_cutout(_source(tmp_path / "source.mp4"), _write_mp4(tmp_path / "matte.mp4", frame), dest)

    rgba = _first_frame_rgba(dest)
    half = int(rgba[_MATTE_SIZE[0] // 2, 8][3])
    faint = int(rgba[_MATTE_SIZE[0] // 2, _MATTE_SIZE[1] - 8][3])
    assert abs(half - 128) <= 8
    assert 15 <= faint <= 40


def _gif_frame(gif: Path, index: int = 0) -> Image.Image:
    with Image.open(gif) as image:
        image.seek(index)
        return image.convert("RGBA")


def _assert_gif_cutout(gif: Path) -> None:
    with Image.open(gif) as image:
        assert image.format == "GIF"
        assert image.n_frames > 1
        assert image.info.get("loop") == 0
    frame = _gif_frame(gif)
    subject = frame.getpixel((8, _MATTE_SIZE[0] // 2))
    background = frame.getpixel((_MATTE_SIZE[1] - 8, _MATTE_SIZE[0] // 2))
    assert subject[3] == 255
    assert background[3] == 0
    assert all(abs(int(a) - b) < 24 for a, b in zip(subject[:3], _SOURCE_RGB))


def test_the_gif_is_the_cutout_with_clear_pixels_and_the_matte_size(tmp_path: Path) -> None:
    dest = tmp_path / "cutout.gif"
    bake_alpha_gif(_source(tmp_path / "source.mp4"), _matte(tmp_path / "matte.mp4"), dest)
    _assert_gif_cutout(dest)
    assert _gif_frame(dest).size == (_MATTE_SIZE[1], _MATTE_SIZE[0])


def test_the_gif_fits_in_720_and_never_grows(tmp_path: Path) -> None:
    large = np.zeros((540, 960, 3), dtype=np.uint8)
    large[:, :480] = 255
    dest = tmp_path / "large.gif"
    bake_alpha_gif(
        _source(tmp_path / "source.mp4"), _write_mp4(tmp_path / "matte.mp4", large, frames=3), dest
    )
    assert _gif_frame(dest).size == (720, 405)


def test_the_gif_is_stored_as_an_image_asset(tmp_path: Path) -> None:
    from services.media_probe import probe_file

    dest = tmp_path / "cutout.gif"
    bake_alpha_gif(_source(tmp_path / "source.mp4"), _matte(tmp_path / "matte.mp4"), dest)
    kind, mime, metadata = probe_file(dest)
    assert (kind, mime) == ("image", "image/gif")
    assert metadata.mediaType == "image"


def test_a_gif_is_not_an_allowed_upload(tmp_path: Path) -> None:
    from services.media_probe import validate_ingest_source
    from services.records import MediaError

    gif = tmp_path / "upload.gif"
    Image.new("RGB", (4, 4)).save(gif)
    with pytest.raises(MediaError):
        validate_ingest_source(gif)


def _decoded_frames(webm: Path) -> int:
    run = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-c:v", "libvpx-vp9", "-i", str(webm), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(re.findall(r"frame= *(\d+)", run.stderr)[-1])


@pytest.mark.parametrize("frames", [9, 20, 150])
def test_the_bake_keeps_every_frame_of_the_matte(tmp_path: Path, frames: int) -> None:
    """The alpha merge ended one frame early on two streams of the same length."""
    frame = np.full((60, 80, 3), 120, dtype=np.uint8)
    dest = tmp_path / "cutout.webm"
    bake_alpha_cutout(
        _write_mp4(tmp_path / "source.mp4", frame, frames=frames),
        _write_mp4(tmp_path / "matte.mp4", np.full((48, 64, 3), 255, dtype=np.uint8), frames=frames),
        dest,
    )

    assert _decoded_frames(dest) == frames


def test_the_bake_keeps_every_frame_when_the_matte_audio_ends_early(tmp_path: Path) -> None:
    """Source audio can end before the video. The audio ends. The video does not."""
    frames = 48
    silent = _write_mp4(tmp_path / "silent.mp4", np.full((48, 64, 3), 255, dtype=np.uint8), frames=frames)
    matte = tmp_path / "matte.mp4"
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y", "-i", str(silent),
            "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac",
            str(matte),
        ],
        capture_output=True,
        check=True,
    )
    dest = tmp_path / "cutout.webm"

    bake_alpha_cutout(
        _write_mp4(tmp_path / "source.mp4", np.full((60, 80, 3), 120, dtype=np.uint8), frames=frames),
        matte,
        dest,
    )

    assert _decoded_frames(dest) == frames


class _MatteWriter:
    """Stands in for the model. It writes a matte where the real one would."""

    def __init__(self, matte: Path) -> None:
        self._matte = matte

    def generate_local_reserved(self, _request: object, **kwargs: object) -> object:
        shutil.copyfile(self._matte, Path(str(kwargs["output_path"])))
        return object()


def _allocations(tmp_path: Path, executor: IcLoraRecipeExecutor, generation: GenerationRecord):
    return tuple(
        OutputAllocation(plan=plan, asset_id=plan.slot, dest_path=str(tmp_path / plan.name))
        for plan in executor.plan_outputs(generation)
    )


class _GridModel:
    """Stands in for the model, with its frame snap. The matte has the shape of the guide clip.

    The pipeline snaps the frame count DOWN to 8k+1. The subject is the left half.
    """

    def generate_local_reserved(self, request: object, **kwargs: object) -> object:
        guide = imageio.get_reader(request.input_path)  # type: ignore[attr-defined]
        try:
            frames = guide.count_frames()
            height, width = np.asarray(guide.get_data(0)).shape[:2]
        finally:
            guide.close()
        matte = np.zeros((height, width, 3), dtype=np.uint8)
        matte[:, : width // 2] = 255
        _write_mp4(Path(str(kwargs["output_path"])), matte, frames=(frames - 1) // 8 * 8 + 1)
        return object()


def _run_alpha_gen(test_state, tmp_path: Path, model: object, source: Path, *, frames: int):
    asset = AssetRecord(
        id="clip",
        media_kind="video",
        origin="uploaded",
        path=str(source),
        mime_type="video/mp4",
        name="clip.mp4",
        metadata=VideoAssetMetadata(
            mediaType="video",
            metadata=VideoMeta(
                width=80,
                height=60,
                durationMs=round(frames / 24 * 1000),
                sizeBytes=source.stat().st_size,
                audioStreamCount=0,
                fps=24,
            ),
        ),
        created_at=0,
    )
    params = IcLoraRecipeStoredParams(
        prompt="",
        model="ltx-2.5-fast",
        resolution="540p",
        audioMode="off",
        fps=24,
        loras=[LoraEntry(ref="", scale=1.0, catalogId="alpha-gen")],
    )
    generation = GenerationRecord(
        id="gen",
        feature="alpha-gen",
        contract_version=1,
        status="running",
        spec={
            "params": params.model_dump(mode="json"),
            "inputs": {"video": {"assetId": asset.id}},
        },
        created_at=0,
        queued_at=0,
        attempt_count=1,
        started_at=0,
        outputs=(),
    )

    class _Clip:
        def get_asset(self, asset_id: str) -> AssetRecord | None:
            return asset if asset_id == asset.id else None

    executor = IcLoraRecipeExecutor(
        model,
        _Clip(),
        lambda: test_state.config.default_models_dir,
        catalog_id="alpha-gen",
        cutout=IC_LORA_RECIPES["alpha-gen"].cutout,
    )
    outputs = _allocations(tmp_path, executor, generation)
    assert [(o.plan.slot, o.plan.mime_type) for o in outputs] == [
        ("output", "video/webm"),
        ("matte", "video/mp4"),
        ("gif", "image/gif"),
    ]
    executor.execute(generation, outputs)
    return outputs


def test_an_alpha_gen_run_stores_the_cutout_the_matte_and_the_gif_in_that_order(
    test_state, create_fake_model_files, tmp_path: Path
) -> None:
    create_fake_model_files()
    source = _source(tmp_path / "clip.mp4")

    outputs = _run_alpha_gen(
        test_state, tmp_path, _MatteWriter(_matte(tmp_path / "model-matte.mp4")), source, frames=9
    )

    _assert_cutout(Path(outputs[0].dest_path))
    assert Path(outputs[1].dest_path).exists()
    _assert_gif_cutout(Path(outputs[2].dest_path))
    assert not list(tmp_path.glob("*-crop.mp4"))
    assert not list(tmp_path.glob("*-fit.mp4"))


@pytest.mark.parametrize("frames", [145, 150])
def test_an_alpha_gen_run_keeps_every_frame_of_the_clip(
    test_state, create_fake_model_files, tmp_path: Path, frames: int
) -> None:
    """The pipeline snaps down to 8k+1. A 150 frame clip lost 5 frames before the clip was padded."""
    create_fake_model_files()
    source = _write_mp4(tmp_path / "clip.mp4", np.full((60, 80, 3), 120, dtype=np.uint8), frames=frames)

    outputs = _run_alpha_gen(test_state, tmp_path, _GridModel(), source, frames=frames)

    assert video_frame_count(Path(outputs[1].dest_path)) == frames
    assert video_frame_count(Path(outputs[0].dest_path)) == frames


def test_only_alpha_gen_is_a_cutout_recipe() -> None:
    assert {recipe_id for recipe_id, recipe in IC_LORA_RECIPES.items() if recipe.cutout} == {
        "alpha-gen"
    }


def test_a_cancel_during_the_bake_stops_ffmpeg(tmp_path: Path) -> None:
    """The flag is polled while ffmpeg runs, so a cancel does not wait for the encode."""
    from services.ffmpeg import FfmpegCancelledError, run_ffmpeg

    started = time.monotonic()
    with pytest.raises(FfmpegCancelledError):
        run_ffmpeg(
            ["-f", "lavfi", "-i", "testsrc=duration=600:size=1280x720:rate=30", "-f", "null", "-"],
            should_cancel=lambda: True,
        )
    assert time.monotonic() - started < 10


def test_a_cancelled_bake_is_a_generation_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ffmpeg helper has its own cancel error. The bake maps it to the queue's."""
    from services.ffmpeg import FfmpegCancelledError
    from services.generation_interrupt import GenerationCancelledError

    def cancelled(*_args: object, **_kwargs: object) -> None:
        raise FfmpegCancelledError()

    monkeypatch.setattr("services.features.video.alpha_cutout.run_ffmpeg", cancelled)
    with pytest.raises(GenerationCancelledError):
        bake_alpha_cutout(
            _source(tmp_path / "source.mp4"), _matte(tmp_path / "matte.mp4"), tmp_path / "out.webm"
        )


def test_a_failed_bake_is_reported_as_a_bake_failure(tmp_path: Path) -> None:
    from services.records import MediaError

    with pytest.raises(MediaError, match="alpha cutout failed"):
        bake_alpha_cutout(tmp_path / "missing.mp4", _matte(tmp_path / "matte.mp4"), tmp_path / "out.webm")


def test_a_cancelled_gif_bake_is_a_generation_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from services.ffmpeg import FfmpegCancelledError
    from services.generation_interrupt import GenerationCancelledError

    def cancelled(*_args: object, **_kwargs: object) -> None:
        raise FfmpegCancelledError()

    monkeypatch.setattr("services.features.video.alpha_cutout.run_ffmpeg", cancelled)
    with pytest.raises(GenerationCancelledError):
        bake_alpha_gif(
            _source(tmp_path / "source.mp4"), _matte(tmp_path / "matte.mp4"), tmp_path / "out.gif"
        )


def test_a_failed_gif_bake_is_reported_as_a_bake_failure(tmp_path: Path) -> None:
    from services.records import MediaError

    with pytest.raises(MediaError, match="alpha gif failed"):
        bake_alpha_gif(tmp_path / "missing.mp4", _matte(tmp_path / "matte.mp4"), tmp_path / "out.gif")


def test_a_gif_that_fails_leaves_no_file_and_does_not_raise(tmp_path: Path) -> None:
    dest = tmp_path / "out.gif"
    dest.write_bytes(b"partial")

    bake_alpha_gif_if_possible(tmp_path / "missing.mp4", _matte(tmp_path / "matte.mp4"), dest)

    assert not dest.exists()


def test_a_gif_that_ffmpeg_wrote_but_nothing_can_read_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def write_garbage(*_args: object, **_kwargs: object) -> None:
        (tmp_path / "out.gif").write_bytes(b"not a gif")

    monkeypatch.setattr("services.features.video.alpha_cutout.run_ffmpeg", write_garbage)
    bake_alpha_gif_if_possible(
        _source(tmp_path / "source.mp4"), _matte(tmp_path / "matte.mp4"), tmp_path / "out.gif"
    )
    assert not (tmp_path / "out.gif").exists()


def test_a_cancel_during_the_gif_still_cancels_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from services.ffmpeg import FfmpegCancelledError
    from services.generation_interrupt import GenerationCancelledError

    def cancelled(*_args: object, **_kwargs: object) -> None:
        raise FfmpegCancelledError()

    monkeypatch.setattr("services.features.video.alpha_cutout.run_ffmpeg", cancelled)
    with pytest.raises(GenerationCancelledError):
        bake_alpha_gif_if_possible(
            _source(tmp_path / "source.mp4"), _matte(tmp_path / "matte.mp4"), tmp_path / "out.gif"
        )


def test_the_gif_is_the_only_optional_output() -> None:
    from services.features.video.alpha_cutout import CUTOUT_OUTPUT_PLANS

    assert [(plan.slot, plan.optional) for plan in CUTOUT_OUTPUT_PLANS] == [
        ("output", False),
        ("matte", False),
        ("gif", True),
    ]
