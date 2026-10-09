"""v1 IC-LoRA local envelope: 1080p-class spatial. Duration is the token budget."""

from __future__ import annotations

from api_types import TargetResolution
from frame_math import compute_num_frames
from runtime_config.ic_lora_local_envelope import (
    IC_LORA_SOURCE_TOO_LARGE,
    ic_lora_output_canvas,
    ic_lora_v1_canvas_envelope_error,
    ic_lora_v1_envelope_error,
    stage1_size,
)


def test_stage1_is_half_canvas() -> None:
    # skip_stage_2 renders at canvas//2; source-dim jobs pass 2×source as canvas.
    assert stage1_size(1920, 1080) == (960, 540)


def test_540p_20s_fits_spatially() -> None:
    assert ic_lora_v1_envelope_error(960, 540, duration_seconds=20.0) is None
    assert ic_lora_v1_envelope_error(1024, 576, duration_seconds=20.0) is None


def test_skip_stage_2_explicit_resolution_is_not_the_768_bucket() -> None:
    explicit = ic_lora_output_canvas(
        skip_stage_2=True,
        resolution_factor=1.5,
        input_width=960,
        input_height=540,
        resolution=TargetResolution(width=1024, height=576),
    )
    assert explicit == (2048, 1152, 1.0)
    bucket = ic_lora_output_canvas(
        skip_stage_2=True,
        resolution_factor=1.5,
        input_width=960,
        input_height=540,
        resolution=None,
    )
    assert bucket[0] == 768
    assert bucket[2] == 1.5


def test_1080p_cell_fits_and_larger_is_over() -> None:
    for width, height in ((1280, 704), (1920, 1088)):
        assert ic_lora_v1_envelope_error(width, height, duration_seconds=5.0) is None
    for width, height in ((2560, 1440), (3840, 2160)):
        err = ic_lora_v1_envelope_error(width, height, duration_seconds=5.0)
        assert err is not None
        assert err.code == IC_LORA_SOURCE_TOO_LARGE


def test_540p_longer_than_20s_is_still_spatial_ok() -> None:
    # Duration is no longer a hard 20s cap; the job budget decides stream vs 422.
    assert ic_lora_v1_envelope_error(960, 540, duration_seconds=20.1) is None


def test_portrait_540p_fits() -> None:
    assert ic_lora_v1_envelope_error(576, 1024, duration_seconds=8.0) is None


def test_snapped_20s_canvas_fits() -> None:
    # compute_num_frames(20, 24) == 481, slightly over 20.0s of wall time.
    frames = compute_num_frames(20, 24)
    assert ic_lora_v1_canvas_envelope_error(1920, 1080, frame_count=frames, fps=24.0) is None


def test_over_snapped_20s_canvas_is_still_spatial_ok() -> None:
    frames = compute_num_frames(20, 24) + 8
    assert ic_lora_v1_canvas_envelope_error(1920, 1080, frame_count=frames, fps=24.0) is None


def _stage_2_ic_lora_canvas(
    width: int, height: int, resolution: TargetResolution | None = None
) -> tuple[int, int]:
    canvas_w, canvas_h, factor = ic_lora_output_canvas(
        skip_stage_2=False,
        resolution_factor=1.0,
        input_width=width,
        input_height=height,
        resolution=resolution,
        stage_2_ic_lora=True,
    )
    assert factor == 1.0
    return canvas_w, canvas_h


def test_stage_2_canvas_of_a_1250x704_source_at_720p_keeps_the_source_height() -> None:
    # The 128 grid would give 1152x640. The 64 grid keeps 1216x704, 2.7% off the source aspect.
    canvas = _stage_2_ic_lora_canvas(1250, 704, TargetResolution(width=1280, height=720))
    assert canvas == (1216, 704)


def test_stage_2_canvas_never_upscales_a_540p_source() -> None:
    # 960x540 on the 64 grid is 896x512 (1.4% off aspect). 960x576 would exceed the source.
    for resolution in (None, TargetResolution(width=1920, height=1080)):
        width, height = _stage_2_ic_lora_canvas(960, 540, resolution)
        assert (width, height) == (896, 512)
        assert width <= 960 and height <= 540
        assert width % 64 == 0 and height % 64 == 0


def test_stage_2_canvas_of_a_2160p_source_without_a_resolution_is_the_1080p_cell() -> None:
    assert _stage_2_ic_lora_canvas(3840, 2160) == (1920, 1088)


def test_stage_2_canvas_of_a_1080p_source_at_1080p_stays_inside_the_source_height() -> None:
    # 1920x1088 would be 0.7% off aspect but its height is above the 1080 source.
    assert _stage_2_ic_lora_canvas(1920, 1080, TargetResolution(width=1920, height=1080)) == (1856, 1024)


def test_stage_2_canvas_of_a_source_below_one_cell_is_the_64x64_minimum() -> None:
    assert _stage_2_ic_lora_canvas(40, 30) == (64, 64)


def test_stage_2_canvas_of_a_portrait_source_on_the_grid_is_unchanged() -> None:
    assert _stage_2_ic_lora_canvas(704, 1280) == (704, 1280)
    assert _stage_2_ic_lora_canvas(704, 1280, TargetResolution(width=720, height=1280)) == (704, 1280)
