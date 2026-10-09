"""IC-LoRA chunking rule: single-stage, above the former cap, past one window."""

from __future__ import annotations

from runtime_config.ic_lora_chunking import (
    IC_LORA_CHUNK_PIXEL_FRAMES,
    ic_lora_chunk_pixel_frames,
)
from runtime_config.ic_lora_stage_mode import IcLoraStageMode
from runtime_config.ic_lora_tiling import IcLoraTiling


def test_large_single_stage_long_clip_chunks() -> None:
    window = ic_lora_chunk_pixel_frames(1920, 1088, 193, stage_mode=IcLoraStageMode.SINGLE_STAGE)
    assert window == IC_LORA_CHUNK_PIXEL_FRAMES


def test_clip_within_one_window_does_not_chunk() -> None:
    assert ic_lora_chunk_pixel_frames(1920, 1088, 97, stage_mode=IcLoraStageMode.SINGLE_STAGE) is None


def test_stage_1_lora_never_chunks() -> None:
    assert ic_lora_chunk_pixel_frames(1920, 1088, 193, stage_mode=IcLoraStageMode.STAGE_1_LORA) is None


def test_job_at_or_below_the_former_cap_does_not_chunk() -> None:
    assert ic_lora_chunk_pixel_frames(1024, 576, 241, stage_mode=IcLoraStageMode.SINGLE_STAGE) is None


def test_stage_2_ic_lora_sizes_the_stage_2_canvas() -> None:
    # Stage 1 of a 1920x1088 canvas is 960x544.
    window = ic_lora_chunk_pixel_frames(960, 544, 217, stage_mode=IcLoraStageMode.STAGE_2_LORA)
    assert window == IC_LORA_CHUNK_PIXEL_FRAMES


def test_stage_2_ic_lora_below_the_token_limit_does_not_chunk() -> None:
    # Stage 1 of a 1280x704 canvas is 640x352.
    assert ic_lora_chunk_pixel_frames(640, 352, 217, stage_mode=IcLoraStageMode.STAGE_2_LORA) is None


def test_off_grid_123_frames_decide_like_121_frames() -> None:
    # Both counts are 16 latent frames. At 1920x1088 that is 16 x 2040 = 32640 tokens,
    # under the 32940 limit. A raw 123 would count 16.25 latent frames: 33150 tokens.
    for frames in (121, 123):
        assert ic_lora_chunk_pixel_frames(1920, 1088, frames, stage_mode=IcLoraStageMode.SINGLE_STAGE) is None
    # 193 and 195 are both 25 latent frames, so both chunk.
    for frames in (193, 195):
        window = ic_lora_chunk_pixel_frames(1920, 1088, frames, stage_mode=IcLoraStageMode.SINGLE_STAGE)
        assert window == IC_LORA_CHUNK_PIXEL_FRAMES


def _restore_tiling(window_frames: int | None = 97) -> IcLoraTiling:
    return IcLoraTiling(long_side=960, short_side=544, window_frames=window_frames)


def test_fixed_window_keeps_exactly_one_window_whole() -> None:
    window = ic_lora_chunk_pixel_frames(
        1920, 1088, 97, stage_mode=IcLoraStageMode.SINGLE_STAGE, tiling=_restore_tiling()
    )
    assert window is None


def test_fixed_window_chunks_the_next_frame_grid_count() -> None:
    # 105 is the first grid count above 97. 98 to 104 snap down to 97.
    for frames in (98, 104):
        assert (
            ic_lora_chunk_pixel_frames(
                1920, 1088, frames, stage_mode=IcLoraStageMode.SINGLE_STAGE, tiling=_restore_tiling()
            )
            is None
        )
    window = ic_lora_chunk_pixel_frames(
        1920, 1088, 105, stage_mode=IcLoraStageMode.SINGLE_STAGE, tiling=_restore_tiling()
    )
    assert window == 97


def test_fixed_window_chunks_a_small_canvas_that_the_size_rule_leaves_whole() -> None:
    # 1024x576 is at the former cap, so the size rule never chunks it.
    assert ic_lora_chunk_pixel_frames(1024, 576, 241, stage_mode=IcLoraStageMode.SINGLE_STAGE) is None
    window = ic_lora_chunk_pixel_frames(
        1024, 576, 241, stage_mode=IcLoraStageMode.SINGLE_STAGE, tiling=_restore_tiling()
    )
    assert window == 97


def test_fixed_window_overrides_the_default_window_length() -> None:
    window = ic_lora_chunk_pixel_frames(
        1920, 1088, 241, stage_mode=IcLoraStageMode.SINGLE_STAGE, tiling=_restore_tiling(window_frames=49)
    )
    assert window == 49
    assert window != IC_LORA_CHUNK_PIXEL_FRAMES


def test_tiling_without_a_window_leaves_the_choice_to_the_size_rule() -> None:
    tiling = IcLoraTiling(long_side=1024, short_side=576)
    assert ic_lora_chunk_pixel_frames(1920, 1088, 241, stage_mode=IcLoraStageMode.SINGLE_STAGE, tiling=tiling) is None
