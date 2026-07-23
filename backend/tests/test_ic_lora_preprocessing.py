from pathlib import Path

import numpy as np
import pytest
from api_types import PreprocessingStep
from ic_lora_preprocessing import (
    MediaArtifact,
    OutpaintParams,
    PreprocessingContext,
    run_preprocessing,
)
from tests.fakes.services import FakeCapture, FakeVideoProcessor


def _ctx(num_frames: int, tmp_path: Path) -> PreprocessingContext:
    return PreprocessingContext(
        num_frames=num_frames,
        outputs_dir=tmp_path,
        video_processor=FakeVideoProcessor(),
    )


def test_image_to_frames_produces_video_artifact(tmp_path: Path):
    steps = [PreprocessingStep(utility="image_to_frames", params={"fps": 24})]
    inp = MediaArtifact(path=str(tmp_path / "in.png"), kind="image")
    out = run_preprocessing(steps, inp, _ctx(num_frames=25, tmp_path=tmp_path))
    assert out.kind == "video"
    assert out.frame_count == 25


def test_pipeline_must_end_in_video(tmp_path: Path):
    # An image input with no steps still ends as an image -> rejected.
    inp_img = MediaArtifact(path="x.png", kind="image")
    with pytest.raises(ValueError):
        run_preprocessing([], inp_img, _ctx(8, tmp_path))


def test_unknown_utility_raises(tmp_path: Path):
    inp = MediaArtifact(path="x.png", kind="image")
    with pytest.raises(KeyError):
        run_preprocessing([PreprocessingStep(utility="nope", params={})], inp, _ctx(8, tmp_path))


def test_outpaint_canvas_rejects_unreadable_source_dims(tmp_path: Path):
    src = str(tmp_path / "bad.mp4")
    vp = FakeVideoProcessor()
    vp.videos[src] = FakeCapture(frames=[], width=0, height=0, fps=24)
    ctx = PreprocessingContext(
        num_frames=9,
        outputs_dir=tmp_path,
        video_processor=vp,
        outpaint=OutpaintParams(left=10, right=10, top=0, bottom=0),
    )
    with pytest.raises(ValueError):
        run_preprocessing(
            [PreprocessingStep(utility="outpaint_canvas", params={})],
            MediaArtifact(path=src, kind="video"),
            ctx,
        )


def _outpaint_ctx(vp: FakeVideoProcessor, tmp_path: Path, n: int, pads: OutpaintParams) -> PreprocessingContext:
    return PreprocessingContext(
        num_frames=n, outputs_dir=tmp_path, video_processor=vp, outpaint=pads
    )


def test_outpaint_canvas_pads_green_and_emits_mask(tmp_path: Path):
    src_w, src_h, n = 100, 100, 9
    src = str(tmp_path / "in.mp4")
    vp = FakeVideoProcessor()
    # 0xAA fill so the kept region is distinguishable from the green pad.
    vp.videos[src] = FakeCapture(
        frames=[np.full((src_h, src_w, 3), 0xAA, dtype=np.uint8) for _ in range(n)],
        width=src_w,
        height=src_h,
        fps=24,
    )
    # Asymmetric pads: +50px on the left only → 150×100, source occupies x in [50, 150).
    ctx = _outpaint_ctx(vp, tmp_path, n, OutpaintParams(left=50, right=0, top=0, bottom=0))
    out = run_preprocessing(
        [PreprocessingStep(utility="outpaint_canvas", params={})],
        MediaArtifact(path=src, kind="video"),
        ctx,
    )
    assert out.kind == "video"
    assert out.frame_count == n
    assert out.mask_path is not None

    # Conditioning gets one frame per source frame; the static mask is written once.
    cond_writer, mask_writer = vp.writers[-2], vp.writers[-1]
    assert len(cond_writer.frames) == n and len(mask_writer.frames) == 1
    cond, mask = cond_writer.frames[0], mask_writer.frames[0]
    assert cond.shape == (100, 150, 3)  # source + (left=50) → width 150
    # Left pad is chroma green (BGR); the kept region carries the source pixels.
    assert tuple(int(v) for v in cond[0, 0]) == (0, 255, 0)
    assert int(cond[50, 75, 0]) == 0xAA  # x=75 ≥ 50 → inside source
    # Mask: kept region white (preserve reference), pad black (generate).
    assert int(mask[50, 75, 0]) == 255
    assert int(mask[0, 0, 0]) == 0


def test_outpaint_canvas_rejects_pad_over_cap(tmp_path: Path):
    src_w, src_h, n = 100, 100, 9
    src = str(tmp_path / "in.mp4")
    vp = FakeVideoProcessor()
    vp.videos[src] = FakeCapture(
        frames=[np.zeros((src_h, src_w, 3), dtype=np.uint8) for _ in range(n)],
        width=src_w,
        height=src_h,
        fps=24,
    )
    # left=101 > src_w=100 → exceeds the +100% cap.
    ctx = _outpaint_ctx(vp, tmp_path, n, OutpaintParams(left=101, right=0, top=0, bottom=0))
    with pytest.raises(ValueError):
        run_preprocessing(
            [PreprocessingStep(utility="outpaint_canvas", params={})],
            MediaArtifact(path=src, kind="video"),
            ctx,
        )
