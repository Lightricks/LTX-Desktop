"""Token/VRAM job budget: 5090-calibrated curve plus IC-LoRA wrapper."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from api_types import (
    REFERENCE_STILL_FRAME_IDX,
    DownloadSpec,
    DownloadVariant,
    IcLoraCatalogItem,
    IcLoraControl,
    IcLoraSettings,
    InputSpec,
    InstructionSection,
    PreprocessingStep,
)
from frame_math import compute_num_frames
import pytest
from pytest import approx
from runtime_config.ic_lora_chunking import IC_LORA_CHUNK_PIXEL_FRAMES
from runtime_config.ic_lora_job_budget import (
    REFINE_DECODE_FLOOR_MESSAGE,
    IcLoraDurationOutcome,
    IcLoraJobLoad,
    IcLoraJobReject,
    catalog_output_fps,
    catalog_preview_stage1,
    decide_ic_lora_durations,
    decide_ic_lora_job,
    dropped_duration_message_and_code,
    ic_lora_sequence_tokens,
    two_stage_exceeds_priced,
    with_budgeted_duration_controls,
)
from runtime_config.budget_admission import (
    Admission,
    Admitted,
    Refused,
    budget_readout_for_reject,
    compared_available_gib,
    is_tight_fit,
    load_headroom_gib,
    reject_over_gib,
    reject_over_ram_gb,
    reject_risk_tier,
    reject_shortfall_kind,
    risk_for_reject,
    unmeasured_refusal_message,
)
from runtime_config.budget_decision import IcLoraJobDecision, VideoJobDecision
from runtime_config.ic_lora_local_envelope import effective_denoise_stage1, ic_lora_output_canvas
from runtime_config.ltx_capabilities import local_caps, pixels_for
from runtime_config.model_download_specs import get_ltx_model_spec
from runtime_config.ic_lora_stage_mode import IcLoraStageMode
from runtime_config.ic_lora_tiling import IcLoraTiling
from runtime_config.video_job_budget import (
    LOCAL_GENERATION_RAM_SHORTFALL,
    LOCAL_GENERATION_RAM_SHORTFALL_MESSAGE,
    MPS_SEQUENCE_OVER_MESSAGE,
    VIDEO_JOB_TOO_LARGE,
    VIDEO_JOB_TOO_LARGE_MESSAGE,
    HardwareProfile,
    LocalJobFilter,
    MemoryBudget,
    FastDurationOutcome,
    VideoJobLoad,
    VideoJobReject,
    advertised_fast_durations,
    _decide_fast_pass,  # pyright: ignore[reportPrivateUsage]
    _mac_edit_stream_required_gib,  # pyright: ignore[reportPrivateUsage]
    budget_route,
    decide_edit_job,
    decide_video_job,
    estimated_giB,
    fits,
    mac_disk_stream_host_gib,
    max_edit_frames_that_load,
    memory_gb_for_job,
    price_alternative_pass,
    reject_message_and_code,
    video_tokens,
)
from services.retake_pipeline.window import (
    duration_to_extend_frames,
    window_for_extend,
    window_for_retake,
)

_GIB = 1024**3


def _priced_video_job(*args: Any, **kwargs: Any) -> VideoJobDecision:
    """The mode-blind verdict: these tests pin pricing, not admission."""
    return decide_video_job(*args, **kwargs).priced.decision


def _priced_edit_job(*args: Any, **kwargs: Any) -> VideoJobDecision:
    return decide_edit_job(*args, **kwargs).priced.decision


def _priced_ic_lora_job(*args: Any, **kwargs: Any) -> IcLoraJobDecision:
    return decide_ic_lora_job(*args, **kwargs).priced.decision


def _refused_reasons(outcome: FastDurationOutcome | IcLoraDurationOutcome) -> dict[int, str]:
    return {seconds: refused.reject.reason for seconds, refused in outcome.refused.items()}


def _untested_basis(admission: Admission) -> str | None:
    """The basis of an untested refusal with priced figures, else None."""
    if not isinstance(admission, Refused) or admission.reject.reason != "unmeasured_cell":
        return None
    reject = admission.reject
    assert isinstance(reject, VideoJobReject)
    assert reject.stream_gib > 0
    return reject.unmeasured_basis
_FAST_2_5_MODEL_ID = "ltx-2.5-22b-distilled"
_MAC_HOST_GIB = mac_disk_stream_host_gib(_FAST_2_5_MODEL_ID)


def _cap(cap_gib: float) -> float:
    """The MPS pool that admits what a measured allocator cap admitted.

    Caps bound MPS memory only; the loader's host buffers ride on top of them.
    """
    return cap_gib + _MAC_HOST_GIB
_WINDOWS_2_3_MODEL_ID = "ltx-2.3-22b-distilled-1.1"


def _source_dim_stage1(width: int, height: int) -> tuple[int, int]:
    canvas_w, canvas_h, factor = ic_lora_output_canvas(
        skip_stage_2=True,
        resolution_factor=0,
        input_width=width,
        input_height=height,
        resolution=None,
    )
    return effective_denoise_stage1(
        canvas_w, canvas_h, skip_stage_2=True, resolution_factor=factor
    )


def test_video_tokens_5s_540p_matches_calibration() -> None:
    frames = compute_num_frames(5, 24)
    tokens = video_tokens(576, 1024, frames)
    assert frames == 121
    assert tokens == 16 * 18 * 32
    assert ic_lora_sequence_tokens(1024, 576, frames) == tokens * 2


def test_fits_31gb_5s_540p_full_and_20s_not_full() -> None:
    seq_5 = ic_lora_sequence_tokens(1024, 576, compute_num_frames(5, 24))
    seq_20 = ic_lora_sequence_tokens(1024, 576, compute_num_frames(20, 24))
    assert fits(seq_5, "full_models_loading", 31)
    assert not fits(seq_20, "full_models_loading", 31)
    assert fits(seq_20, "streaming_models_loading", 31)


def test_decide_31gb_5s_540p_full() -> None:
    w, h = _source_dim_stage1(1024, 576)
    decision = _priced_ic_lora_job(
        w, h, compute_num_frames(5, 24),
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_job_filter(memory_gb=31, process_mode="full_models_loading"),
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.mode == "full_models_loading"


def test_decide_31gb_8s_and_10s_540p_stream() -> None:
    w, h = _source_dim_stage1(1024, 576)
    for seconds in (8, 10):
        decision = _priced_ic_lora_job(
            w, h, compute_num_frames(seconds, 24),
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_job_filter(memory_gb=31, process_mode="full_models_loading"),
        )
        assert isinstance(decision, IcLoraJobLoad)
        assert decision.mode == "streaming_models_loading"


def test_decide_chunked_1080p_prices_one_window() -> None:
    w, h = _source_dim_stage1(1920, 1080)
    frames = compute_num_frames(8, 24)
    chunked = _priced_ic_lora_job(
        w, h, frames, job_filter=_job_filter(memory_gb=31, process_mode="full_models_loading"), stage_mode=IcLoraStageMode.SINGLE_STAGE
    )
    assert isinstance(chunked, IcLoraJobLoad)
    assert chunked.mode == "streaming_models_loading"
    window = _priced_ic_lora_job(
        w, h, 97, job_filter=_job_filter(memory_gb=31, process_mode="full_models_loading"), stage_mode=IcLoraStageMode.SINGLE_STAGE
    )
    assert isinstance(window, IcLoraJobLoad)
    assert chunked.seq == window.seq


def test_decide_stage_2_ic_lora_1080p_chunks_and_fits() -> None:
    decision = _priced_ic_lora_job(
        960,
        544,
        compute_num_frames(9, 24),
        job_filter=_job_filter(memory_gb=32, process_mode="full_models_loading"),
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_decide_1080p_single_stage_5s_on_22gb_chunks_and_loads_instead_of_rejecting() -> None:
    # Whole clip: 2 x 16 x 2025 = 64800 tokens, stream estimate 23.5 GiB, over 22.
    # One 97-frame window: 2 x 13 x 2025 = 52650 tokens, stream estimate 19.7 GiB.
    decision = _priced_ic_lora_job(
        1920, 1080, compute_num_frames(5, 24), job_filter=_job_filter(memory_gb=22, process_mode="full_models_loading"),
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.mode == "streaming_models_loading"
    assert decision.chunk_pixel_frames == 97
    assert decision.seq == 52650.0


def test_decide_off_grid_123_frames_chunks_like_121_frames() -> None:
    # Both counts are 16 latent frames, so both need the same window on a 22 GB card.
    def window(frames: int) -> int | None:
        decision = _priced_ic_lora_job(
            1920, 1088, frames, job_filter=_job_filter(memory_gb=22, process_mode="full_models_loading"),
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
        )
        assert isinstance(decision, IcLoraJobLoad)
        return decision.chunk_pixel_frames

    assert window(121) == 97
    assert window(123) == 97


def test_decide_a_longer_clip_is_never_accepted_after_a_shorter_clip_is_rejected() -> None:
    # Chunking prices one window, so a long clip must not look cheaper than a short one
    # that fits in no window. A card at the limit accepts up to some length and rejects after.
    cases = (
        (1920, 1080, IcLoraStageMode.SINGLE_STAGE, 16),
        (1920, 1080, IcLoraStageMode.SINGLE_STAGE, 22),
        (960, 544, IcLoraStageMode.STAGE_2_LORA, 16),
        (960, 544, IcLoraStageMode.STAGE_2_LORA, 22),
        (960, 544, IcLoraStageMode.STAGE_1_LORA, 22),
    )
    for width, height, mode, memory_gb in cases:
        accepted = [
            isinstance(
                _priced_ic_lora_job(
                    width, height, compute_num_frames(seconds, 24),
                    job_filter=_job_filter(memory_gb=memory_gb, process_mode="full_models_loading"),
                    stage_mode=mode,
                ),
                IcLoraJobLoad,
            )
            for seconds in range(1, 21)
        ]
        first_reject = accepted.index(False) if False in accepted else len(accepted)
        assert not any(accepted[first_reject:]), (width, height, mode, memory_gb)


def test_decide_does_not_chunk_two_stage_or_small_jobs() -> None:
    frames = compute_num_frames(8, 24)
    # A chunked job prices one 97-frame window, so an unchunked job prices above it.
    for width, height, mode in (
        (960, 512, IcLoraStageMode.STAGE_1_LORA),
        (1024, 576, IcLoraStageMode.SINGLE_STAGE),
    ):
        priced = _priced_ic_lora_job(
            width, height, frames, job_filter=_job_filter(memory_gb=31, process_mode="full_models_loading"), stage_mode=mode
        )
        window = _priced_ic_lora_job(
            width, height, 97, job_filter=_job_filter(memory_gb=31, process_mode="full_models_loading"), stage_mode=mode
        )
        assert priced.seq > window.seq


def test_decide_31gb_20s_540p_streams() -> None:
    w, h = _source_dim_stage1(1024, 576)
    decision = _priced_ic_lora_job(
        w, h, compute_num_frames(20, 24),
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_job_filter(memory_gb=31, process_mode="full_models_loading"),
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_decide_31gb_5s_2160p_spatial_reject() -> None:
    w, h = _source_dim_stage1(3840, 2160)
    decision = _priced_ic_lora_job(
        w, h, compute_num_frames(5, 24),
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_job_filter(memory_gb=31, process_mode="full_models_loading"),
    )
    assert isinstance(decision, IcLoraJobReject)
    assert decision.reason == "spatial"


def test_decide_31gb_5s_2160p_two_stage_stage_1_lora_is_a_spatial_reject() -> None:
    # A 3840x2160 source gives a 3840x2048 canvas, so stage 1 is 1920x1024. That is inside
    # the 1920x1088 cell, but stage 2 runs at 3840x2048 and is over it.
    decision = _priced_ic_lora_job(
        1920, 1024, compute_num_frames(5, 24), job_filter=_job_filter(memory_gb=32, process_mode="full_models_loading"),
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
    )
    assert isinstance(decision, IcLoraJobReject)
    assert decision.reason == "spatial"


def test_stage_1_lora_prices_the_stage_2_canvas_when_it_is_heavier_than_stage_1() -> None:
    # Stage 1 960x544, 121 frames: 16 latent frames of 17x30 cells, times 2 for the
    # reference = 16320 tokens. Stage 2 at 1920x1088 has no reference: 16 x 34 x 60 = 32640.
    decision = _priced_ic_lora_job(
        960, 544, 121, job_filter=_job_filter(memory_gb=32, process_mode="full_models_loading"),
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.seq == 32640.0


def test_a_reference_still_adds_one_latent_frame_of_tokens_to_the_stage_2_lora_price() -> None:
    # Stage 1 640x352 gives a 1280x704 canvas: 22 x 40 = 880 cells per latent frame.
    # 121 frames are 16 latent frames, doubled for the reference: 2 x 16 x 880 = 28160.
    # One still adds one latent frame of the canvas: 880.
    def price(stills: int) -> float:
        decision = _priced_ic_lora_job(
            640, 352, 121, job_filter=_job_filter(memory_gb=32, process_mode="full_models_loading"),
            stage_mode=IcLoraStageMode.STAGE_2_LORA, stills=stills,
        )
        assert isinstance(decision, IcLoraJobLoad)
        return decision.seq

    assert price(0) == 28160.0
    assert price(1) == 29040.0


def test_decide_ic_lora_unsupported_beats_spatial() -> None:
    w, h = _source_dim_stage1(1280, 720)
    decision = _priced_ic_lora_job(
        w, h, compute_num_frames(5, 24),
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_job_filter(memory_gb=31, process_mode="unsupported"),
    )
    assert isinstance(decision, IcLoraJobReject)
    assert decision.reason == "unsupported"


def test_decide_ic_lora_unsupported_process_rejects() -> None:
    w, h = _source_dim_stage1(1024, 576)
    decision = _priced_ic_lora_job(
        w, h, compute_num_frames(5, 24),
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_job_filter(memory_gb=31, process_mode="unsupported"),
    )
    assert isinstance(decision, IcLoraJobReject)
    assert decision.reason == "unsupported"


def test_decide_24gb_stream_process_never_full() -> None:
    w, h = _source_dim_stage1(1024, 576)
    five = _priced_ic_lora_job(
        w,
        h,
        compute_num_frames(5, 24),
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_job_filter(memory_gb=24, process_mode="streaming_models_loading"),
    )
    twenty = _priced_ic_lora_job(
        w,
        h,
        compute_num_frames(20, 24),
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_job_filter(memory_gb=24, process_mode="streaming_models_loading"),
    )
    assert isinstance(five, IcLoraJobLoad)
    assert five.mode == "streaming_models_loading"
    assert isinstance(twenty, VideoJobReject)
    assert twenty.reason == "stream_over"


def test_windows_tall_20s_never_asks_below_the_line_over_i1() -> None:
    """Portrait 576x960 @ 20s (65,880 seq): the line asks 25.93 over I1's native 24.81."""
    frames = compute_num_frames(20, 24)
    short = _priced_ic_lora_job(
        576, 960, frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(31, 64, free_gib=25.93),
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        576, 960, frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(31, 64, free_gib=25.94),
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"


def test_windows_tall_10s_never_asks_below_the_line_over_i2() -> None:
    """Portrait 576x960 @ 10s (33,480 seq): the line asks 16.38 over I2's native 15.36."""
    frames = compute_num_frames(10, 24)
    short = _priced_ic_lora_job(
        576, 960, frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(31, 64, free_gib=16.37),
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        576, 960, frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(31, 64, free_gib=16.38),
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"


def test_two_stage_job_is_not_priced_on_the_single_stage_line() -> None:
    """960x512 x 241 frames: two-stage asks 24.06, single-stage 15.28 — 11.0 GiB
    free must refuse the two-stage job."""
    frames = compute_num_frames(10, 24)
    assert frames == 241
    refused = _priced_ic_lora_job(
        960, 512, frames,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_windows_measured_filter(31, 64, free_gib=11.0),
    )
    assert isinstance(refused, VideoJobReject)
    assert refused.reason == "stream_over"


def test_single_stage_job_is_not_priced_on_the_two_stage_line() -> None:
    """960x512 x 241: single-stage asks 15.28 (2x stage-1 tokens) and must load at
    15.29 free — on the two-stage count (4x) it would ask 24.06 and refuse."""
    frames = compute_num_frames(10, 24)
    loads = _priced_ic_lora_job(
        960, 512, frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(31, 64, free_gib=15.29),
    )
    assert isinstance(loads, IcLoraJobLoad)
    assert loads.mode == "streaming_models_loading"


def test_stage_1_lora_is_priced_on_four_times_its_stage_1_tokens() -> None:
    """960x512 x 241: stage 2 runs at 1920x1024 without the reference, 59,520 seq,
    asking 24.06. On the single-stage count (2x) it would ask 15.28 and admit."""
    frames = compute_num_frames(10, 24)
    short = _priced_ic_lora_job(
        960, 512, frames,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_windows_measured_filter(31, 64, free_gib=24.05),
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        960, 512, frames,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_windows_measured_filter(31, 64, free_gib=24.06),
    )
    assert isinstance(fits, IcLoraJobLoad)


def test_catalog_preview_prices_two_stage_when_default_settings_say_two_stage() -> None:
    """A two-stage recipe previews the 1080p stage-1 (960x512), where 5s asks
    15.56 — at 12.0 free it must drop with stream_over, while the single-stage
    twin (11.60) keeps 5s."""
    job_filter = _windows_measured_filter(31, 64, free_gib=12.0)
    two_stage = with_budgeted_duration_controls(
        _catalog_item_two_stage(), job_filter=job_filter
    )
    duration = next(c for c in two_stage.controls if c.id == "duration")
    assert duration.dropped_options is not None
    assert [(d.value, d.reason) for d in duration.dropped_options] == [
        (5, "stream_over"),
        (8, "stream_over"),
        (10, "stream_over"),
        (20, "stream_over"),
    ]
    single_stage = with_budgeted_duration_controls(
        _catalog_item_two_stage().model_copy(
            update={
                "default_settings": IcLoraSettings(
                    skip_stage_2=True, resolution_factor=1.5
                )
            }
        ),
        job_filter=job_filter,
    )
    control = next(c for c in single_stage.controls if c.id == "duration")
    assert control.options == [5]


def test_two_stage_t1_cell_never_asks_below_its_native_peak() -> None:
    """Two-stage 960x512 @ 5s (30,720 seq): the line asks 15.56 over T1's native 13.65."""
    frames = compute_num_frames(5, 24)
    short = _priced_ic_lora_job(
        960, 512, frames,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_windows_measured_filter(31, 64, free_gib=15.56),
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        960, 512, frames,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_windows_measured_filter(31, 64, free_gib=15.57),
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"


def test_two_stage_ask_exceeds_single_stage_ask_by_the_measured_stage_2_delta() -> None:
    """At 960x512 x 121 the two asks must separate by >= the 1.73 GiB measured
    stage-2 increment — a too-flat stage-2 slope would collapse them."""
    frames = compute_num_frames(5, 24)
    job_filter = _windows_measured_filter(31, 64, free_gib=31.0)
    two = _priced_ic_lora_job(
        960, 512, frames, stage_mode=IcLoraStageMode.STAGE_1_LORA, job_filter=job_filter
    )
    one = _priced_ic_lora_job(
        960, 512, frames, stage_mode=IcLoraStageMode.SINGLE_STAGE, job_filter=job_filter
    )
    assert isinstance(two, IcLoraJobLoad)
    assert isinstance(one, IcLoraJobLoad)
    assert two.stream_gib - one.stream_gib >= 1.73


def test_two_stage_floor_binds_below_the_crossover() -> None:
    """640x320 @ 1s (3,200 seq): the unclamped line asks 7.44 — the 8.00 floor
    must bind, not the bare line."""
    frames = compute_num_frames(1, 24)
    short = _priced_ic_lora_job(
        640, 320, frames,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_windows_measured_filter(31, 64, free_gib=7.99),
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        640, 320, frames,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_windows_measured_filter(31, 64, free_gib=8.0),
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"


def test_two_stage_at_the_priced_ceiling_is_offered_not_fenced() -> None:
    """D4's exact 14,880 tokens (960x512 x 241) is the cell the ceiling was
    derived from — an off-by-one must not fence it."""
    assert compute_num_frames(10, 24) == 241
    assert not two_stage_exceeds_priced(
        stage_mode=IcLoraStageMode.STAGE_1_LORA, stage1_width=960, stage1_height=512, frames=241
    )
    outcome = decide_ic_lora_durations(
        [10],
        stage1_width=960,
        stage1_height=512,
        fps=24,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_windows_measured_filter(31, 64, free_gib=31.0),
    )
    assert outcome.kept == [10]


def test_two_stage_one_token_past_the_priced_ceiling_is_untested() -> None:
    """512x647 x 361 frames is exactly 14,881 tokens — one past the ceiling."""
    assert compute_num_frames(15, 24) == 361
    assert video_tokens(647, 512, 361) == 14881
    assert two_stage_exceeds_priced(
        stage_mode=IcLoraStageMode.STAGE_1_LORA, stage1_width=512, stage1_height=647, frames=361
    )
    outcome = decide_ic_lora_durations(
        [15],
        stage1_width=512,
        stage1_height=647,
        fps=24,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_windows_measured_filter(31, 64, free_gib=31.0),
    )
    assert outcome.kept == []
    assert _untested_basis(outcome.refused[15]) == "two_stage_range"


def test_two_stage_fence_is_token_based_not_duration_based() -> None:
    """D2 is a 20 s two-stage job at 12,200 tokens that measurement prices at
    20.90 — a leftover 5-second rule would refuse it."""
    assert compute_num_frames(20, 24) == 481
    assert not two_stage_exceeds_priced(
        stage_mode=IcLoraStageMode.STAGE_1_LORA, stage1_width=640, stage1_height=320, frames=481
    )
    priced = decide_ic_lora_durations(
        [20],
        stage1_width=640,
        stage1_height=320,
        fps=24,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_windows_measured_filter(31, 64, free_gib=20.90),
    )
    assert priced.kept == [20]
    short = decide_ic_lora_durations(
        [20],
        stage1_width=640,
        stage1_height=320,
        fps=24,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_windows_measured_filter(31, 64, free_gib=20.89),
    )
    assert _refused_reasons(short) == {20: "stream_over"}


def test_two_stage_t3_cell_never_asks_below_its_native_peak() -> None:
    """640x320 @ 5s (12,800 seq) reserved 10.23 natively; the line asks 10.28."""
    frames = compute_num_frames(5, 24)
    short = _priced_ic_lora_job(
        640, 320, frames,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_windows_measured_filter(31, 64, free_gib=10.27),
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        640, 320, frames,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_windows_measured_filter(31, 64, free_gib=10.28),
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"


def test_two_stage_idle_10gb_refuses_3200t_cell() -> None:
    """Same 3,200-token cell asks 10.28 — must refuse on an idle 10 GB card
    (free 9.9)."""
    frames = compute_num_frames(5, 24)
    decision = _priced_ic_lora_job(
        640, 320, frames,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_windows_measured_filter(10, 64, free_gib=9.9),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"


def test_two_stage_12gb_still_streams_3200t_cell() -> None:
    """The refit must not strand 12 GB cards: 10.28 still fits idle 12 GB free."""
    frames = compute_num_frames(5, 24)
    decision = _priced_ic_lora_job(
        640, 320, frames,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_windows_measured_filter(12, 64, free_gib=11.9),
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_over_cap_two_stage_refuses_spatial_not_with_the_fence_advice() -> None:
    """A job past the token fence whose output is over the 1080p cap at any
    length refuses spatial — "try a shorter clip" would be wrong."""
    assert two_stage_exceeds_priced(
        stage_mode=IcLoraStageMode.STAGE_1_LORA, stage1_width=576, stage1_height=960, frames=481
    )
    outcome = decide_ic_lora_durations(
        [20],
        stage1_width=576,
        stage1_height=960,
        fps=24,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_windows_measured_filter(31, 64, free_gib=31.0),
    )
    assert outcome.kept == []
    assert _refused_reasons(outcome) == {20: "spatial"}


def test_two_stage_refusal_message_does_not_claim_a_five_second_rule() -> None:
    """User-visible wire copy: the refusal must not cite the retired 5s rule."""
    message, code = dropped_duration_message_and_code(
        VideoJobReject(
            reason="unmeasured_cell",
            seq=0.0,
            full_gib=0.0,
            stream_gib=99.9,
            unmeasured_basis="two_stage_range",
        ),
        _windows_measured_filter(31, 64, free_gib=12.0),
    )
    assert code == VIDEO_JOB_TOO_LARGE
    assert message == unmeasured_refusal_message("two_stage_range")
    assert "5 second" not in message


def test_spatial_cap_corner_is_the_1080p_output() -> None:
    """1920x1088 output prices (either way round); one grid step wider refuses."""
    job_filter = _windows_measured_filter(31, 64, free_gib=31.0)
    for width, height in ((1920, 1088), (1088, 1920)):
        prices = _priced_ic_lora_job(
            width, height, 97, stage_mode=IcLoraStageMode.SINGLE_STAGE, job_filter=job_filter
        )
        assert isinstance(prices, IcLoraJobLoad), (width, height)
    refuses = _priced_ic_lora_job(
        1952, 1088, 97, stage_mode=IcLoraStageMode.SINGLE_STAGE, job_filter=job_filter
    )
    assert isinstance(refuses, IcLoraJobReject)
    assert refuses.reason == "spatial"


def _mac_iclora_filter(pool_gib: float | None, *, memory_gb: float | None = None) -> LocalJobFilter:
    """Darwin filter with an exact sampled MPS pool for the Mac IC-LoRA gate.

    ``memory_gb`` is the coarse available-RAM figure list-time filtering keys
    its early-return on; decide-level tests leave it None.
    """
    return _job_filter(
        process_mode="streaming_models_loading",
        memory_gb=memory_gb,
        **_mac_profile(pool_gib=pool_gib),
    )


def test_mac_growth_ask_covers_the_t3_pass_bottom() -> None:
    """Portrait 576x960 @ 10s (16,740 tokens): T3 moved the edge to
    (15.46, 15.96] where the 1.3 intercept asked 15.33 — the refit line
    asks 15.97, so the measured 15.46 fail must refuse."""
    frames = compute_num_frames(10, 24)
    short = _priced_ic_lora_job(
        576, 960, frames, stage_mode=IcLoraStageMode.SINGLE_STAGE, job_filter=_mac_iclora_filter(_cap(15.46))
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        576, 960, frames, stage_mode=IcLoraStageMode.SINGLE_STAGE, job_filter=_mac_iclora_filter(_cap(15.97))
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"


def test_mac_growth_ask_covers_the_p20v_pass_bottom() -> None:
    """Portrait 576x960 @ 20s (32,940 tokens): P20V brackets (25.35, 25.85]
    where the T4 line asked 24.85 — the refit asks 25.86, so a 25.85 pool
    must refuse (a carried 5.486e-4 slope would admit it)."""
    frames = compute_num_frames(20, 24)
    short = _priced_ic_lora_job(
        576, 960, frames, stage_mode=IcLoraStageMode.SINGLE_STAGE, job_filter=_mac_iclora_filter(_cap(25.85))
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        576, 960, frames, stage_mode=IcLoraStageMode.SINGLE_STAGE, job_filter=_mac_iclora_filter(_cap(25.87))
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"


def test_mac_ingredients_adder_covers_the_xm1_pass_top() -> None:
    """Portrait 576x960 @ 10s with ingredients: XM1 tops at 16.46 where the
    video line asks 15.97 — pricing the adapter on the line admits a
    measured-fail pool, so 16.46 must refuse and 16.47 load."""
    frames = compute_num_frames(10, 24)
    short = _priced_ic_lora_job(
        576, 960, frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_mac_iclora_filter(_cap(16.46)),
        ic_lora_id="ingredients",
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        576, 960, frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_mac_iclora_filter(_cap(16.47)),
        ic_lora_id="ingredients",
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"


def test_mac_ingredients_adder_covers_the_p20i_pass_top() -> None:
    """Portrait 576x960 @ 20s with ingredients: P20I tops at 26.35, +0.50 over
    the fitted 25.86 exactly as XM1 — the adder is token-flat, so 26.35 must
    refuse and 26.36 load (a token-scaled term misses one end)."""
    frames = compute_num_frames(20, 24)
    short = _priced_ic_lora_job(
        576, 960, frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_mac_iclora_filter(_cap(26.35)),
        ic_lora_id="ingredients",
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        576, 960, frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_mac_iclora_filter(_cap(26.36)),
        ic_lora_id="ingredients",
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"


def test_mac_image_adder_ignores_non_ingredients_ids() -> None:
    """The +0.50 keys on the ingredients id only: image-fed day-to-night tops
    at 15.46 (XO1) and heavy greenscreen-avatar passes 14.96 (XG) — a feed-
    or size-keyed adder refuses these runnable cells at a 15.97 pool."""
    frames = compute_num_frames(10, 24)
    for ic_lora_id in ("day-to-night", "greenscreen-avatar", None):
        decision = _priced_ic_lora_job(
            576, 960, frames,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_mac_iclora_filter(_cap(15.97)),
            ic_lora_id=ic_lora_id,
        )
        assert isinstance(decision, IcLoraJobLoad)


def _mac_ingredients_list_item() -> IcLoraCatalogItem:
    """Ingredients-shaped catalog entry: adapter id, single-stage defaults, duration control."""
    return IcLoraCatalogItem(
        id="ingredients",
        name="Ingredients",
        description="d",
        download=DownloadSpec(
            repo_id="r",
            variants=[DownloadVariant(id="default", label="Default", filename="f.safetensors", size_bytes=10, base_model="LTX-2.3")],
        ),
        requires_hf_login=False,
        input=InputSpec(kind="image"),
        preprocessing=[PreprocessingStep(utility="image_to_frames", params={"fps": 24})],
        instructions=[InstructionSection(kind="summary", title="What it does", body="b")],
        default_settings=IcLoraSettings(skip_stage_2=True, resolution_factor=1.5),
        controls=[
            IcLoraControl(
                id="duration",
                label="Duration",
                kind="int",
                default=5,
                options=[5, 6, 8, 10, 20],
            )
        ],
    )


def test_mac_ingredients_list_time_matches_generate_in_the_adder_gap() -> None:
    """Ingredients 20s in a (25.86, 26.36] pool: list-time priced the video
    line only, so it offered 20s that generate (with the +0.50 adder) 422s —
    the offer must equal the verdict on both sides of the 26.36 edge."""
    frames = compute_num_frames(20, 24)
    generate_short = _priced_ic_lora_job(
        576, 960, frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_mac_iclora_filter(_cap(26.0)),
        ic_lora_id="ingredients",
    )
    assert isinstance(generate_short, VideoJobReject)
    listed_short = with_budgeted_duration_controls(
        _mac_ingredients_list_item(),
        job_filter=_mac_iclora_filter(_cap(26.0), memory_gb=48),
    )
    short_control = next(c for c in listed_short.controls if c.id == "duration")
    assert short_control.options == [5, 6, 8, 10]
    assert short_control.dropped_options is not None
    assert [(d.value, d.reason) for d in short_control.dropped_options] == [(20, "stream_over")]

    generate_fits = _priced_ic_lora_job(
        576, 960, frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_mac_iclora_filter(_cap(26.36)),
        ic_lora_id="ingredients",
    )
    assert isinstance(generate_fits, IcLoraJobLoad)
    listed_fits = with_budgeted_duration_controls(
        _mac_ingredients_list_item(),
        job_filter=_mac_iclora_filter(_cap(26.36), memory_gb=48),
    )
    fits_control = next(c for c in listed_fits.controls if c.id == "duration")
    assert fits_control.options == [5, 6, 8, 10, 20]
    assert fits_control.dropped_options is None


def test_mac_floor_still_binds_below_the_growth_crossover() -> None:
    """At 2,304 tokens the refit line asks 7.16 — the 9.84 floor must
    still govern small cells, not become dead code under the new intercept."""
    frames = compute_num_frames(5, 24)
    short = _priced_ic_lora_job(
        576, 256, frames, stage_mode=IcLoraStageMode.SINGLE_STAGE, job_filter=_mac_iclora_filter(_cap(9.83))
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        576, 256, frames, stage_mode=IcLoraStageMode.SINGLE_STAGE, job_filter=_mac_iclora_filter(_cap(9.85))
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"


def test_mac_two_stage_outside_cell_is_priced_on_the_curve_not_fenced() -> None:
    """Two-stage shapes past the measured Mac cell are untested at any pool, but
    priced on the two-stage curve: both layouts at 960x512 here, past the
    928x512 cell. A fence returning before the curve refuses with no figures."""
    frames = compute_num_frames(5, 24)
    for mode in (IcLoraStageMode.STAGE_1_LORA, IcLoraStageMode.STAGE_2_LORA):
        admission = decide_ic_lora_job(
            960, 512, frames, stage_mode=mode, job_filter=_mac_iclora_filter(100)
        )
        assert _untested_basis(admission) == "mac_two_stage_cell", mode


def test_mac_two_stage_past_the_token_range_names_the_range_not_the_mac_cell() -> None:
    """Mac two-stage past the priced token range is tagged with the
    platform-blind range first, the Mac cell below it."""
    outcome = decide_ic_lora_durations(
        [5, 10],
        stage1_width=960,
        stage1_height=544,
        fps=24,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_mac_iclora_filter(100),
    )
    assert outcome.kept == []
    assert _untested_basis(outcome.refused[5]) == "mac_two_stage_cell"
    assert _untested_basis(outcome.refused[10]) == "two_stage_range"


def test_mac_layout_to_render_540p_5s_edges() -> None:
    """PhaseMN-A1-shaped L2R (540p 5s: 448x256 x 121, 1 still) edges (11.5, 12.0]
    — refusing it as unmeasured bricks 540p two-stage where it fits.
    The 12.00 refusal must be stream_over: the fence would hide a chord regression."""
    frames = compute_num_frames(5, 24)
    short = _priced_ic_lora_job(
        448, 256, frames,
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_mac_iclora_filter(_cap(12.00)),
        stills=1,
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        448, 256, frames,
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_mac_iclora_filter(_cap(12.01)),
        stills=1,
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"


def test_mac_layout_to_render_540p_10s_edges() -> None:
    """PhaseL2R-T-shaped L2R (540p 10s: 448x256 x 241, 1 still) asks 14.7527 on
    the higher chord — a single-chord price under-asks it by 0.25 onto a pool
    the 720p chord refuses."""
    frames = compute_num_frames(10, 24)
    short = _priced_ic_lora_job(
        448, 256, frames,
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_mac_iclora_filter(_cap(14.75)),
        stills=1,
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        448, 256, frames,
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_mac_iclora_filter(_cap(14.76)),
        stills=1,
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"


def test_mac_layout_to_render_720p_5s_edges() -> None:
    """PhaseL2R-P5-shaped L2R (720p 5s: 640x352 x 121, 1 still) asks 15.00 unchunked
    and retries chunked at 97f (13.67) when the pool sits between 13.67 and 15.00."""
    frames = compute_num_frames(5, 24)
    short = _priced_ic_lora_job(
        640, 352, frames,
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_mac_iclora_filter(_cap(13.67)),
        stills=1,
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    chunked = _priced_ic_lora_job(
        640, 352, frames,
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_mac_iclora_filter(_cap(14.99)),
        stills=1,
    )
    assert isinstance(chunked, IcLoraJobLoad)
    assert chunked.chunk_pixel_frames == IC_LORA_CHUNK_PIXEL_FRAMES
    fits = _priced_ic_lora_job(
        640, 352, frames,
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_mac_iclora_filter(_cap(15.00)),
        stills=1,
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"
    assert fits.chunk_pixel_frames is None


def test_mac_layout_to_render_720p_10s_retries_chunked_on_24gb_pool() -> None:
    """PhaseL2R-P10-shaped L2R (720p 10s: 640x352 x 241, 1 still) asks 23.00 unchunked
    and retries chunked at 97f (13.67) on a 24 GB Mac (~18.7 GiB pool) — suppressing
    the chunked retry rejects a job that fits in 97f windows."""
    frames = compute_num_frames(10, 24)
    short = _priced_ic_lora_job(
        640, 352, frames,
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_mac_iclora_filter(_cap(13.67)),
        stills=1,
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    chunked = _priced_ic_lora_job(
        640, 352, frames,
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_mac_iclora_filter(_cap(22.99)),
        stills=1,
    )
    assert isinstance(chunked, IcLoraJobLoad)
    assert chunked.chunk_pixel_frames == IC_LORA_CHUNK_PIXEL_FRAMES
    fits = _priced_ic_lora_job(
        640, 352, frames,
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_mac_iclora_filter(_cap(23.00)),
        stills=1,
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"
    assert fits.chunk_pixel_frames is None


def test_mac_layout_to_render_1080p_5s_edges() -> None:
    """PhaseL2R-H-shaped L2R (1080p 5s: 928x512 x 121, 1 still) asks 24.76 unchunked
    and 21.39 when retried chunked at 97f — refusing it strands 32 GB Macs to API."""
    frames = compute_num_frames(5, 24)
    short = _priced_ic_lora_job(
        928, 512, frames,
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_mac_iclora_filter(_cap(21.38)),
        stills=1,
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    chunked = _priced_ic_lora_job(
        928, 512, frames,
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_mac_iclora_filter(_cap(24.75)),
        stills=1,
    )
    assert isinstance(chunked, IcLoraJobLoad)
    assert chunked.chunk_pixel_frames == IC_LORA_CHUNK_PIXEL_FRAMES
    fits = _priced_ic_lora_job(
        928, 512, frames,
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_mac_iclora_filter(_cap(24.76)),
        stills=1,
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"
    assert fits.chunk_pixel_frames is None


def test_mac_opt_in_two_stage_540p_10s_edges() -> None:
    """PhaseO1-shaped opt-in (540p 10s: 448x256 x 241, 13,888 tokens) asks 11.83
    on the 540p chord — a fence re-narrowed to Layout to Render refuses a
    measured cell to API. The 11.83 refusal must be stream_over: the fence
    would hide a chord regression."""
    frames = compute_num_frames(10, 24)
    short = _priced_ic_lora_job(
        448, 256, frames,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_mac_iclora_filter(_cap(11.83)),
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        448, 256, frames,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_mac_iclora_filter(_cap(11.84)),
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"


def test_mac_opt_in_two_stage_540p_5s_loads() -> None:
    """PhaseO5-shaped opt-in (540p 5s: 448x256 x 121, 7,168 tokens) asks 10.58 —
    the low-end confirm of the opening. A fence re-narrowed to Layout to Render
    refuses it as unmeasured; the 10.58 refusal must be stream_over so a chord
    regression cannot hide behind the fence."""
    frames = compute_num_frames(5, 24)
    short = _priced_ic_lora_job(
        448, 256, frames,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_mac_iclora_filter(_cap(10.58)),
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        448, 256, frames,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_mac_iclora_filter(_cap(10.59)),
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"


def test_mac_opt_in_two_stage_refuses_past_the_cell() -> None:
    """540p opt-in at 15s (361 frames, 5,152 tokens) slides under the widened
    token fence at 448x256 — the 241-frame duration fence must carry the opt-in
    layout too, not just Layout to Render. An opening that drops the gate for
    the opt-in layout prices this shape on chords that never saw it."""
    assert compute_num_frames(15, 24) == 361
    admission = decide_ic_lora_job(
        448, 256, 361,
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_mac_iclora_filter(100),
    )
    assert _untested_basis(admission) == "mac_two_stage_cell"


def test_mac_two_stage_refuses_540p_15s_past_the_duration_fence() -> None:
    """540p L2R at 15s (361 frames, 5,152 tokens) slides under the widened token
    fence at 448x256 — the 241-frame duration fence must carry it."""
    assert compute_num_frames(15, 24) == 361
    admission = decide_ic_lora_job(
        448, 256, 361,
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_mac_iclora_filter(100),
        stills=1,
    )
    assert _untested_basis(admission) == "mac_two_stage_cell"


def test_mac_two_stage_loads_chunked_1080p_10s_and_720p_20s() -> None:
    """Chunked Mac L2R (K/K4: 640x352 x 481f, 7w of 97f; H10c: 928x512 x 241f,
    3w of 97f) prices flat on the per-window chord — refusing window != None
    tags measured chunked two-stage shapes untested."""
    k_short = _priced_ic_lora_job(
        640, 352, compute_num_frames(20, 24),
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_mac_iclora_filter(_cap(13.67)),
        stills=1,
    )
    assert isinstance(k_short, VideoJobReject)
    assert k_short.reason == "stream_over"
    k_fits = _priced_ic_lora_job(
        640, 352, compute_num_frames(20, 24),
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_mac_iclora_filter(_cap(13.68)),
        stills=1,
    )
    assert isinstance(k_fits, IcLoraJobLoad)
    assert k_fits.chunk_pixel_frames == IC_LORA_CHUNK_PIXEL_FRAMES

    h10c_short = _priced_ic_lora_job(
        928, 512, compute_num_frames(10, 24),
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_mac_iclora_filter(_cap(21.38)),
        stills=1,
    )
    assert isinstance(h10c_short, VideoJobReject)
    assert h10c_short.reason == "stream_over"
    h10c_fits = _priced_ic_lora_job(
        928, 512, compute_num_frames(10, 24),
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_mac_iclora_filter(_cap(21.39)),
        stills=1,
    )
    assert isinstance(h10c_fits, IcLoraJobLoad)
    assert h10c_fits.chunk_pixel_frames == IC_LORA_CHUNK_PIXEL_FRAMES


def test_two_stage_fence_ignores_single_stage() -> None:
    assert not two_stage_exceeds_priced(
        stage_mode=IcLoraStageMode.SINGLE_STAGE, stage1_width=576, stage1_height=960, frames=481
    )


def test_decide_ic_lora_job_windows_full_pick_legacy() -> None:
    """Full-vs-stream stays legacy: 5s full-loads on 31 GB Windows."""
    w, h = _source_dim_stage1(1024, 576)
    decision = _priced_ic_lora_job(
        w,
        h,
        compute_num_frames(5, 24),
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(
            31, 48, process_mode="full_models_loading"
        ),
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.mode == "full_models_loading"


def test_decide_ic_lora_job_windows_spatial_still_first() -> None:
    """A 4K source is over the 1080p cap: spatial on Windows, before the measured gate."""
    w, h = _source_dim_stage1(3840, 2160)
    decision = _priced_ic_lora_job(
        w,
        h,
        compute_num_frames(5, 24),
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(31, 48, free_gib=12.0),
    )
    assert isinstance(decision, IcLoraJobReject)
    assert decision.reason == "spatial"



_RESTORE_TILING = IcLoraTiling(long_side=960, short_side=544, window_frames=97)
_REFINE_TILING = IcLoraTiling(long_side=1024, short_side=576)


def test_windows_prices_one_window_restore_at_tile_plus_decode() -> None:
    """Restore 1080p x 97 asks 11.71 (tile 10.41 + 1.30 canvas decode): canvas
    pricing strands 16 GB cards to API for a measured cell, while bare tile
    pricing under-asks the decode peak by 1.3 into an OOM."""
    short = _priced_ic_lora_job(
        1920, 1088, 97,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(16, 64, free_gib=11.71),
        tiling=_RESTORE_TILING,
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        1920, 1088, 97,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(16, 64, free_gib=11.72),
        tiling=_RESTORE_TILING,
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"
    assert fits.chunk_pixel_frames is None
    assert fits.seq == 13_260


def test_windows_streams_one_window_tiled_on_full_mode_cards() -> None:
    """The legacy full curve on a tile seq would full-load 32 GB cards, where
    resident (~23) plus tile working set exceeds the headroom — tiled stays
    streamed. The small untiled control proves full would otherwise fire."""
    control = _priced_ic_lora_job(
        576, 256, 121,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(31, 64, free_gib=31.0, process_mode="full_models_loading"),
    )
    assert isinstance(control, IcLoraJobLoad)
    assert control.mode == "full_models_loading"
    decision = _priced_ic_lora_job(
        1920, 1088, 97,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(31, 64, free_gib=31.0, process_mode="full_models_loading"),
        tiling=_RESTORE_TILING,
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_windows_prices_one_window_refine_at_tile_plus_decode() -> None:
    """Refine 1080p x 97 asks 12.22 (tile 10.92 + 1.30): the bigger tile prices
    its own pass, and bare tile pricing under-asks its 11.71 decode peak."""
    short = _priced_ic_lora_job(
        1920, 1088, 97,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(16, 64, free_gib=12.21),
        tiling=_REFINE_TILING,
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        1920, 1088, 97,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(16, 64, free_gib=12.22),
        tiling=_REFINE_TILING,
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.seq == 14_976


def test_windows_single_tile_small_canvas_skips_decode_term() -> None:
    """Restore 540p x 97 (one tile = canvas) asks 10.18 with no decode term: its
    7.96 peak stays denoise-bound under the ask, and a flat term would strand
    12 GB cards for it."""
    short = _priced_ic_lora_job(
        960, 512, 97,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(12, 64, free_gib=10.18),
        tiling=_RESTORE_TILING,
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        960, 512, 97,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(12, 64, free_gib=10.19),
        tiling=_RESTORE_TILING,
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.seq == 12_480


def test_windows_refine_still_at_fence_falls_back_to_canvas() -> None:
    """A look still pushes Refine's 97f tile pass (14,976 + 576) over the tile-seq
    fence: uncounted stills would under-price the pass, so it prices the canvas."""
    decision = _priced_ic_lora_job(
        1920, 1088, 97,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(16, 64, free_gib=15.9),
        stills=1,
        tiling=_REFINE_TILING,
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"
    assert decision.seq == 55_080


def test_legacy_route_keeps_main_s_tile_price() -> None:
    """Linux prices the 960x544 tile (2 x 6,630 = 13,260 seq), as on main."""
    decision = _priced_ic_lora_job(
        1920, 1088, 97,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_job_filter(memory_gb=32, process_mode="streaming_models_loading"),
        tiling=_RESTORE_TILING,
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.seq == 13_260


def test_canvas_pricing_keeps_restore_s_fixed_window() -> None:
    """Restore 1080p x 241 on Windows: priced on the unified canvas-window decode
    curve (15.00 GiB over R3/R6's 14.50–14.58 GiB and C1's 12.11 GiB @ 16g cap)
    instead of the 22.15 GiB untiled canvas ask that stranded 16 GB cards."""
    short = _priced_ic_lora_job(
        1920, 1088, 241,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(16, 64, free_gib=14.99),
        tiling=_RESTORE_TILING,
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        1920, 1088, 241,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(16, 64, free_gib=15.00),
        tiling=_RESTORE_TILING,
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.chunk_pixel_frames == 97
    assert fits.seq == 13_260


def test_windows_prices_chunked_1080p_refine_on_canvas_window_decode_curve() -> None:
    """T1 (Refine 1080p x 241 in 5 windows of 73f) peaked at 12.74 GiB: falling
    back to untiled canvas pricing (18.54 GiB) when window is not None refuses
    16 GB cards a job that saturates at 12.74 GiB."""
    refine_chunked_tiling = IcLoraTiling(long_side=1024, short_side=576, window_frames=73)
    short = _priced_ic_lora_job(
        1920, 1088, 241,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(16, 64, free_gib=12.73),
        tiling=refine_chunked_tiling,
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        1920, 1088, 241,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(16, 64, free_gib=12.74),
        tiling=refine_chunked_tiling,
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.chunk_pixel_frames == 73
    assert fits.seq == 11_520


def test_windows_prices_half_res_ref_adapters_at_1_25x_video_tokens() -> None:
    """Video Guide and Motion Track use reference_downscale_factor=2 (-ref0.5),
    appending 0.25x video tokens (1.25x total): pricing them at 2.0x over-asks
    VG1 (1080p 121f, 12.80 GiB measured) at 25.76 GiB and VG3 (1080p 241f 3w,
    13.30 GiB measured) at 22.15 GiB, stranding 16 GB and 24 GB cards."""
    for adapter_id in ("video-guide", "motion-track"):
        vg1_short = _priced_ic_lora_job(
            1920, 1088, 121,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(16, 64, free_gib=13.35),
            ic_lora_id=adapter_id,
        )
        assert isinstance(vg1_short, VideoJobReject)
        assert vg1_short.reason == "stream_over"
        vg1_fits = _priced_ic_lora_job(
            1920, 1088, 121,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(16, 64, free_gib=13.37),
            ic_lora_id=adapter_id,
        )
        assert isinstance(vg1_fits, IcLoraJobLoad)
        assert vg1_fits.seq == 40_800

    vg3_short = _priced_ic_lora_job(
        1920, 1088, 241,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(16, 64, free_gib=14.60),
        ic_lora_id="video-guide",
    )
    assert isinstance(vg3_short, VideoJobReject)
    assert vg3_short.reason == "stream_over"
    vg3_fits = _priced_ic_lora_job(
        1920, 1088, 241,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(16, 64, free_gib=14.62),
        ic_lora_id="video-guide",
    )
    assert isinstance(vg3_fits, IcLoraJobLoad)
    assert vg3_fits.chunk_pixel_frames == 97
    assert vg3_fits.seq == 33_150


def test_mac_prices_restore_shaped_tiled_job_at_its_tile() -> None:
    """Restore 1080p x 97 tiled passes at 9.84 — pricing it at the 22 GB canvas
    refuses 24 GB Macs a job the tile fits."""
    short = _priced_ic_lora_job(
        1920, 1088, 97,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_mac_iclora_filter(_cap(9.83)),
        tiling=_RESTORE_TILING,
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        1920, 1088, 97,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_mac_iclora_filter(_cap(9.85)),
        tiling=_RESTORE_TILING,
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"
    assert fits.chunk_pixel_frames is None
    assert fits.seq == 13_260


def test_mac_prices_chunked_restore_at_flat_s20() -> None:
    """Restore 1080p x 241 chunked asks the flat 18.7 S20 pass: canvas-window
    pricing (21.94) refuses pools the measured pass fits, while the 9.84 tile
    price fails the same clip outright."""
    short = _priced_ic_lora_job(
        1920, 1088, 241,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_mac_iclora_filter(_cap(18.69)),
        tiling=_RESTORE_TILING,
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        1920, 1088, 241,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_mac_iclora_filter(_cap(18.71)),
        tiling=_RESTORE_TILING,
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.mode == "streaming_models_loading"
    assert fits.chunk_pixel_frames == 97
    assert fits.seq == 53_040


def test_mac_chunked_restore_20s_prices_flat() -> None:
    """Restore 1080p x 481 (the measured 7-window max) asks the flat 18.7 —
    the fence must reach the measured edge, not stop at 10 s."""
    decision = _priced_ic_lora_job(
        1920, 1088, 481,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_mac_iclora_filter(_cap(18.71)),
        tiling=_RESTORE_TILING,
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.chunk_pixel_frames == 97


def test_mac_chunked_restore_past_20s_falls_back_to_canvas_window() -> None:
    """Restore 1080p x 505 runs past the measured 7 windows: flat pricing it
    extrapolates growth nobody measured, so it prices the canvas window."""
    decision = _priced_ic_lora_job(
        1920, 1088, 505,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_mac_iclora_filter(18.70),
        tiling=_RESTORE_TILING,
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"
    assert decision.seq == 53_040


def test_mac_small_canvas_chunked_restore_keeps_window_price() -> None:
    """Restore 720p x 241 chunked asks its canvas window (12.73), not the flat
    18.7: the flat price misapplied refuses fittable mid pools."""
    decision = _priced_ic_lora_job(
        1280, 704, 241,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_mac_iclora_filter(_cap(15.0)),
        tiling=_RESTORE_TILING,
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.chunk_pixel_frames == 97
    assert decision.seq == 22_880


def test_mac_chunked_restore_still_at_fence_falls_back_to_canvas() -> None:
    """A look still pushes the 97f tile pass (13,260 + 510) over the flat fence:
    an uncounted still prices 18.7 where the true need exceeds it."""
    decision = _priced_ic_lora_job(
        1920, 1088, 241,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_mac_iclora_filter(18.70),
        stills=1,
        tiling=_RESTORE_TILING,
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"
    assert decision.seq == 55_080


def test_mac_tiled_price_stays_canvas_for_bigger_tile() -> None:
    """Refine's 1024x576 tile (7,488 tokens at 97f) was never measured — only
    Restore's 960x544 tile prices the tile."""
    decision = _priced_ic_lora_job(
        1920, 1088, 97,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_mac_iclora_filter(12.5),
        tiling=IcLoraTiling(long_side=1024, short_side=576),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"
    assert decision.seq == 53_040


def test_refine_decode_floor_refuses_long_1080p_on_huge_vram() -> None:
    """Refine 1080p x 241 unchunked refuses with its own reason on 48 GB free:
    canvas figures would admit it straight into the pipeline's DiffVAE 400."""
    decision = _priced_ic_lora_job(
        1920, 1088, 241,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(48, 128, free_gib=47.0),
        tiling=_REFINE_TILING,
    )
    assert isinstance(decision, IcLoraJobReject)
    assert decision.reason == "refine_decode_floor"


def test_refine_decode_floor_bites_on_mac_too() -> None:
    """The DiffVAE floor is a pipeline limit, not a CUDA curve — a CUDA-only
    fence lets huge-pool Macs into the same 400."""
    decision = _priced_ic_lora_job(
        1920, 1088, 241,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_mac_iclora_filter(100),
        tiling=_REFINE_TILING,
    )
    assert isinstance(decision, IcLoraJobReject)
    assert decision.reason == "refine_decode_floor"


def test_windows_prices_long_720p_refine_at_canvas_plus_decode() -> None:
    """Refine 720p x 241 unchunked asks 23.90 (canvas 22.60 + 1.30): under 720p
    the decode fits usable, but bare canvas pricing would admit 24 GB-edge pools
    its decode volume exceeds."""
    short = _priced_ic_lora_job(
        1280, 704, 241,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(31, 64, free_gib=22.70),
        tiling=_REFINE_TILING,
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_ic_lora_job(
        1280, 704, 241,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(31, 64, free_gib=23.90),
        tiling=_REFINE_TILING,
    )
    assert isinstance(fits, IcLoraJobLoad)
    assert fits.chunk_pixel_frames is None
    assert fits.seq == 54_560


def test_refine_decode_floor_refusal_carries_fixed_copy_without_figures() -> None:
    """The decode-floor drop messages fixed copy (nothing was compared) under VIDEO_JOB_TOO_LARGE."""
    message, code = dropped_duration_message_and_code(
        IcLoraJobReject(
            reason="refine_decode_floor",
            seq=0.0,
            full_gib=0.0,
            stream_gib=99.9,
        ),
        _windows_measured_filter(48, 128, free_gib=47.0),
    )
    assert (message, code) == (REFINE_DECODE_FLOOR_MESSAGE, VIDEO_JOB_TOO_LARGE)


def test_windows_layout_to_render_prices_the_look_still() -> None:
    """One still adds one latent frame of the 1920x1088 stage-2 grid (2,040 tokens)."""
    job_filter = _windows_measured_filter(31, 64, free_gib=31.0)

    def priced(stills: int) -> IcLoraJobLoad:
        decision = _priced_ic_lora_job(
            960, 544, 121,
            stage_mode=IcLoraStageMode.STAGE_2_LORA,
            job_filter=job_filter,
            stills=stills,
        )
        assert isinstance(decision, IcLoraJobLoad)
        return decision

    without, with_still = priced(0), priced(1)
    assert with_still.seq - without.seq == 2_040
    assert with_still.stream_gib - without.stream_gib == pytest.approx(2_040 * 2.95e-4)


def test_windows_prices_a_chunked_job_as_one_window() -> None:
    """1080p x 241 single-stage runs in 97-frame windows asking 22.15 — it fits a
    24 GB card; the whole clip would ask 43.81."""
    decision = _priced_ic_lora_job(
        1920, 1088, 241,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(24, 64, free_gib=23.9),
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.chunk_pixel_frames == 97


def test_windows_chunked_reject_keeps_the_priced_window() -> None:
    # The bug is a downgrade running the whole clip: the reject dropped the
    # 97-frame window, and None means one unchunked pass.
    decision = _priced_ic_lora_job(
        1920, 1088, 241,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(16, 64, free_gib=15.0),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"
    assert decision.chunk_pixel_frames == 97


def test_tight_fit_stops_when_headroom_equals_the_threshold() -> None:
    # Spec: tight is headroom under max(0.5 GiB, 5% of ask). Equality is not tight.
    ask = 10.0
    load = VideoJobLoad(
        mode="streaming_models_loading", seq=1, full_gib=20, stream_gib=ask
    )

    def tier(available: float) -> str | None:
        job_filter = _job_filter(
            process_mode="streaming_models_loading",
            system="Windows",
            model_id="ltx-2.5-22b-distilled",
            vram_bytes=int(available * _GIB),
            free_vram_bytes=int(available * _GIB),
        )
        return "tight" if is_tight_fit(load, job_filter) else None

    assert tier(ask + 0.5) is None
    assert tier(ask + 0.49) == "tight"


def test_two_stage_fence_counts_one_window_for_chunked_layout_to_render() -> None:
    """Layout to Render 1080p x 241: 15,810 stage-1 tokens over the clip, past the
    fence, but it runs in 6,630-token windows."""
    decision = _priced_ic_lora_job(
        960, 544, 241,
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
        job_filter=_windows_measured_filter(31, 64, free_gib=31.0),
        stills=1,
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.chunk_pixel_frames == 97


def test_decide_ic_lora_job_windows_ram_gate() -> None:
    """L5 floors: 64 CPU / 48 DISK (L5-48 holds); unknown fails closed (Fast mirror)."""
    w, h = _source_dim_stage1(1024, 576)
    frames = compute_num_frames(5, 24)
    short_cpu = _priced_ic_lora_job(
        w, h, frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(31, 62, free_gib=12.0)
    )
    assert isinstance(short_cpu, VideoJobReject)
    assert short_cpu.reason == "ram_shortfall"
    ok_cpu_63 = _priced_ic_lora_job(
        w, h, frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(31, 63, free_gib=12.0)
    )
    assert isinstance(ok_cpu_63, IcLoraJobLoad)
    ok_cpu = _priced_ic_lora_job(
        w, h, frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(31, 64, free_gib=12.0)
    )
    assert isinstance(ok_cpu, IcLoraJobLoad)
    short_disk = _priced_ic_lora_job(
        w,
        h,
        frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(
            31, 30, free_gib=12.0, drive_type="nvme"
        ),
    )
    assert isinstance(short_disk, VideoJobReject)
    assert short_disk.reason == "ram_shortfall"
    ok_disk_31 = _priced_ic_lora_job(
        w,
        h,
        frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(
            31, 31, free_gib=12.0, drive_type="nvme"
        ),
    )
    assert isinstance(ok_disk_31, IcLoraJobLoad)
    ok_disk = _priced_ic_lora_job(
        w,
        h,
        frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(
            31, 32, free_gib=12.0, drive_type="nvme"
        ),
    )
    assert isinstance(ok_disk, IcLoraJobLoad)
    unknown_ram = _priced_ic_lora_job(
        w, h, frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(31, None, free_gib=12.0)
    )
    assert isinstance(unknown_ram, VideoJobReject)
    assert unknown_ram.reason == "unsupported"


def test_windows_iclora_tall_20s_refuses_paging_cell_on_24gb() -> None:
    """Allocated ask 18.81 admits at free 24, but native reserved is 24.81; must refuse."""
    decision = _priced_ic_lora_job(
        576, 960, compute_num_frames(20, 24),
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(24, 64, free_gib=24.0),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"


def test_windows_iclora_wide_20s_refuses_on_10gb() -> None:
    """Floor ask 8.00 admits at free 9.99, but native reserved is 10.97; must refuse."""
    decision = _priced_ic_lora_job(
        576, 256, compute_num_frames(20, 24),
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(9.99, 64, free_gib=9.99),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"


def test_windows_iclora_small_cell_loads_on_10gb() -> None:
    """Reserved ask 8.53 must still load at free 9.95 (guards over-correction stranding idle 10GB)."""
    decision = _priced_ic_lora_job(
        576, 256, compute_num_frames(10, 24),
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(9.99, 64, free_gib=9.95),
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_windows_iclora_two_stage_d4_refuses_paging_band() -> None:
    """Allocated ask 18.01 admits at free 22, but native reserved is 22.61; must refuse."""
    decision = _priced_ic_lora_job(
        960, 512, compute_num_frames(10, 24),
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
        job_filter=_windows_measured_filter(24, 64, free_gib=22.0),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"


def test_memory_gb_for_job_uses_available_ram_on_darwin() -> None:
    assert memory_gb_for_job(vram_gb=31, available_ram_gb=48, darwin=True) == 48
    assert memory_gb_for_job(vram_gb=31, available_ram_gb=48, darwin=False) == 31


def _job_filter(**kwargs) -> LocalJobFilter:
    return LocalJobFilter(
        profile=HardwareProfile(
            system=kwargs.get("system", "Linux"),
            model_id=kwargs.get("model_id"),
            process_mode=kwargs["process_mode"],
            drive_type=kwargs.get("drive_type"),
        ),
        budget=MemoryBudget(
            total_vram_bytes=kwargs.get("vram_bytes"),
            free_vram_bytes=kwargs.get("free_vram_bytes"),
            total_ram_bytes=kwargs.get("total_ram_bytes"),
            memory_gb=kwargs.get("memory_gb"),
            mps_pool_bytes=kwargs.get("mps_pool_bytes"),
            total_only=kwargs.get("total_only", False),
        ),
    )


def test_ingredients_preview_stage1_is_576x960() -> None:
    w, h = catalog_preview_stage1(IcLoraSettings(skip_stage_2=True, resolution_factor=1.5))
    assert sorted((w, h)) == [576, 960]


def test_advertised_duration_drops_20s_on_10gb() -> None:
    w, h = catalog_preview_stage1(IcLoraSettings(skip_stage_2=True, resolution_factor=1.5))
    kept = decide_ic_lora_durations(
        [5, 6, 8, 10, 20],
        stage1_width=w,
        stage1_height=h,
        fps=24,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_job_filter(memory_gb=10, process_mode="full_models_loading"),
    ).kept
    assert 20 not in kept
    assert 5 in kept


def test_full_estimate_is_above_stream() -> None:
    seq = ic_lora_sequence_tokens(1024, 576, compute_num_frames(5, 24))
    assert estimated_giB("full_models_loading", seq) > estimated_giB("streaming_models_loading", seq)


def _video_job(width: int, height: int, seconds: int, fps: int = 24, **kwargs):
    return _priced_video_job(
        width,
        height,
        compute_num_frames(seconds, fps),
        job_filter=_job_filter(**kwargs),
    )


def _one_pass_priced(width: int, height: int, frames: int, *, job_filter: LocalJobFilter) -> VideoJobDecision:
    """The one-pass verdict: what the Fast law says before the picker tries windows."""
    return _decide_fast_pass(width, height, frames, windows=1, job_filter=job_filter)  # pyright: ignore[reportPrivateUsage]


def _one_pass_job(width: int, height: int, seconds: int, fps: int = 24, **kwargs):
    return _one_pass_priced(
        width, height, compute_num_frames(seconds, fps), job_filter=_job_filter(**kwargs)
    )


def test_decide_video_job_31gb_540p_5s_full() -> None:
    decision = _video_job(1024, 576, 5, memory_gb=31, process_mode="full_models_loading")
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "full_models_loading"


def test_decide_video_job_32gb_720p_20s_streams() -> None:
    decision = _video_job(1280, 704, 20, memory_gb=32, process_mode="full_models_loading")
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_decide_video_job_15gb_540p_5s_still_streams() -> None:
    decision = _video_job(1024, 576, 5, memory_gb=15, process_mode="streaming_models_loading")
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_decide_edit_job_31gb_720p_extend_streams() -> None:
    """5090 extend 1280×704×337 estimated 27.2 GiB and reserved the whole card."""
    decision = _priced_edit_job(
        1280,
        704,
        337,
        job_filter=_job_filter(memory_gb=31, process_mode="full_models_loading"),
    )
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "streaming_models_loading"
    reported_as_32 = _priced_edit_job(
        1280, 704, 337,
        job_filter=_job_filter(memory_gb=32, process_mode="full_models_loading"),
    )
    assert isinstance(reported_as_32, VideoJobLoad)
    assert reported_as_32.mode == "streaming_models_loading"


def _windows_measured_filter(
    vram_gib: float,
    total_ram_gib: float | None,
    *,
    free_gib: float | None = None,
    drive_type: object = None,
    process_mode: str = "streaming_models_loading",
    total_only: bool = False,
) -> LocalJobFilter:
    """Windows filter with exact bytes for the measured edit/IC-LoRA gates."""
    extra: dict[str, object] = {}
    if free_gib is not None:
        extra["free_vram_bytes"] = int(free_gib * _GIB)
    return _job_filter(
        **_windows_profile(
            vram_gib=vram_gib, total_ram_gib=total_ram_gib, drive_type=drive_type
        ),
        memory_gb=vram_gib,
        process_mode=process_mode,
        total_only=total_only,
        **extra,
    )


def test_decide_edit_job_windows_540p_floor_is_8_75() -> None:
    """Floor edge (8.5, 8.75]: fails below 8.75 free, streams at 8.75 (RAM ok)."""
    short = _priced_edit_job(
        1024,
        576,
        121,
        job_filter=_windows_measured_filter(31, 64, free_gib=8.74),
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_edit_job(
        1024, 576, 121, job_filter=_windows_measured_filter(31, 64, free_gib=8.75)
    )
    assert isinstance(fits, VideoJobLoad)
    assert fits.mode == "streaming_models_loading"


def test_decide_edit_job_windows_720p_follows_formula() -> None:
    """337 denoise frames need 15.48 (curve binds, not the 9.0 floor)."""
    short = _priced_edit_job(
        1280,
        704,
        337,
        job_filter=_windows_measured_filter(31, 64, free_gib=15.4),
    )
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    fits = _priced_edit_job(
        1280, 704, 337, job_filter=_windows_measured_filter(31, 64, free_gib=15.5)
    )
    assert isinstance(fits, VideoJobLoad)
    assert fits.mode == "streaming_models_loading"


def test_decide_edit_job_windows_1080p_417f_refuses_paging_window() -> None:
    """417f admitted at allocated ask 30.18 paged (reserved 34.70); reserved ask 34.95 refuses at free 30.3."""
    decision = _priced_edit_job(
        1920,
        1088,
        417,
        job_filter=_windows_measured_filter(31.84, 64, free_gib=30.3),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"


def test_decide_edit_job_windows_1080p_337f_loads_step0_clean_window() -> None:
    """337f ran clean on base (reserved 28.86); reserved ask 29.30 still loads at free 30.3."""
    decision = _priced_edit_job(
        1920,
        1088,
        337,
        job_filter=_windows_measured_filter(31.84, 64, free_gib=30.3),
    )
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_decide_edit_job_windows_full_pick_unchanged() -> None:
    """Full-vs-stream stays legacy: short 720p full-loads on 31 GB Windows."""
    decision = _priced_edit_job(
        1280,
        704,
        compute_num_frames(5, 24),
        job_filter=_windows_measured_filter(
            31, 48, process_mode="full_models_loading"
        ),
    )
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "full_models_loading"


def test_decide_edit_job_windows_ram_gate() -> None:
    """E3 floors: 64 CPU / 32 DISK (holds by Fast precedent); unknown fails closed."""
    base = dict(vram_gib=31)
    short_cpu = _priced_edit_job(
        1024,
        576,
        121,
        job_filter=_windows_measured_filter(
            **base, total_ram_gib=16, free_gib=12.0
        ),
    )
    assert isinstance(short_cpu, VideoJobReject)
    assert short_cpu.reason == "ram_shortfall"
    short_cpu_62 = _priced_edit_job(
        1024,
        576,
        121,
        job_filter=_windows_measured_filter(
            **base, total_ram_gib=62, free_gib=12.0
        ),
    )
    assert isinstance(short_cpu_62, VideoJobReject)
    assert short_cpu_62.reason == "ram_shortfall"
    ok_cpu_63 = _priced_edit_job(
        1024,
        576,
        121,
        job_filter=_windows_measured_filter(
            **base, total_ram_gib=63, free_gib=12.0
        ),
    )
    assert isinstance(ok_cpu_63, VideoJobLoad)
    ok_cpu_64 = _priced_edit_job(
        1024,
        576,
        121,
        job_filter=_windows_measured_filter(
            **base, total_ram_gib=64, free_gib=12.0
        ),
    )
    assert isinstance(ok_cpu_64, VideoJobLoad)
    ok_disk = _priced_edit_job(
        1024,
        576,
        121,
        job_filter=_windows_measured_filter(
            **base,
            total_ram_gib=36,
            free_gib=12.0,
            drive_type="nvme",
        ),
    )
    assert isinstance(ok_disk, VideoJobLoad)
    short_sata = _priced_edit_job(
        1024,
        576,
        121,
        job_filter=_windows_measured_filter(
            **base,
            total_ram_gib=36,
            free_gib=12.0,
            drive_type="ssd",
        ),
    )
    assert isinstance(short_sata, VideoJobReject)
    assert short_sata.reason == "ram_shortfall"
    unknown_ram = _priced_edit_job(
        1024,
        576,
        121,
        job_filter=_windows_measured_filter(
            **base, total_ram_gib=None, free_gib=12.0
        ),
    )
    assert isinstance(unknown_ram, VideoJobReject)
    assert unknown_ram.reason == "unsupported"


def test_ram_gate_admits_63_45gib_box_on_64gib_floor() -> None:
    """A real 64 GB box reports ~63.45; exact-compare must not refuse it."""
    decision = _priced_edit_job(
        1024,
        576,
        121,
        job_filter=_windows_measured_filter(31, 63.45, free_gib=12.0),
    )
    assert isinstance(decision, VideoJobLoad)


def test_ram_gate_admits_23_5gib_on_24gib_disk_floor() -> None:
    """A 24 GB NVMe box reports under nominal; streamed edit must load."""
    decision = _priced_edit_job(
        1024,
        576,
        121,
        job_filter=_windows_measured_filter(31, 23.5, free_gib=12.0, drive_type="nvme"),
    )
    assert isinstance(decision, VideoJobLoad)


def test_ram_gate_still_refuses_below_floor_minus_headroom() -> None:
    """Headroom must not swallow genuine shortfalls: 62 on a 64 floor refuses."""
    decision = _priced_edit_job(
        1024,
        576,
        121,
        job_filter=_windows_measured_filter(31, 62.0, free_gib=12.0),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "ram_shortfall"


def test_non_fp8_windows_takes_legacy_route() -> None:
    """bf16 streaming must not price on fp8-fit laws (pre-Ada cards)."""
    profile = HardwareProfile(
        system="Windows",
        model_id=_FAST_2_5_MODEL_ID,
        process_mode="streaming_models_loading",
        fp8_capable=False,
    )
    assert budget_route("fast", profile) == "legacy"
    assert budget_route("edit", profile) == "legacy"
    assert budget_route("iclora", profile) == "legacy"


def test_fp8_windows_keeps_measured_route() -> None:
    profile = HardwareProfile(
        system="Windows",
        model_id=_FAST_2_5_MODEL_ID,
        process_mode="streaming_models_loading",
        fp8_capable=True,
    )
    assert budget_route("fast", profile) == "windows_measured"
    assert budget_route("edit", profile) == "windows_measured"
    assert budget_route("iclora", profile) == "windows_measured"


def test_max_edit_frames_shrinks_1080p_on_31gb_and_keeps_720p() -> None:
    """1920×1088×505 streams past the 31 GB budget. The cap must load; 720p×505 already does."""
    job_filter = _job_filter(memory_gb=31, process_mode="full_models_loading")
    cap = max_edit_frames_that_load(1920, 1088, frame_cap=505, job_filter=job_filter)
    assert cap < 505
    assert isinstance(_priced_edit_job(1920, 1088, 505, job_filter=job_filter), VideoJobReject)
    assert isinstance(_priced_edit_job(1920, 1088, cap, job_filter=job_filter), VideoJobLoad)
    assert isinstance(_priced_edit_job(1920, 1088, cap + 1, job_filter=job_filter), VideoJobReject)
    assert (
        max_edit_frames_that_load(1280, 736, frame_cap=505, job_filter=job_filter) == 505
    )

    extend_frames = duration_to_extend_frames(4, 25)
    window = window_for_extend(
        source_frames=1000,
        extend_frames=extend_frames,
        mode="end",
        max_input_frames=cap,
    )
    assert isinstance(
        _priced_edit_job(
            1920, 1088, window.context_frames + extend_frames, job_filter=job_filter
        ),
        VideoJobLoad,
    )
    retake = window_for_retake(
        source_frames=1000,
        mask_start_frame=100,
        mask_end_frame=140,
        max_input_frames=cap,
        fps=25,
    )
    assert retake.encode_frames <= cap
    assert isinstance(
        _priced_edit_job(1920, 1088, retake.encode_frames, job_filter=job_filter),
        VideoJobLoad,
    )


def test_decide_edit_job_31gb_short_720p_stays_full() -> None:
    decision = _priced_edit_job(
        1280,
        704,
        compute_num_frames(5, 24),
        job_filter=_job_filter(memory_gb=31, process_mode="full_models_loading"),
    )
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "full_models_loading"


def _windows_5090_edit_filter() -> LocalJobFilter:
    """Idle-desktop 5090: 31.84 total, 30.1 free — the shrink-then-run box."""
    return _job_filter(
        system="Windows",
        process_mode="streaming_models_loading",
        vram_bytes=int(31.84 * _GIB),
        free_vram_bytes=int(30.1 * _GIB),
        total_ram_bytes=64 * _GIB,
    )


def test_decide_edit_job_refuses_369f_on_5090_free_short() -> None:
    """The total-fitting 369f window asks 31.6 against 30.1 free: refuse, don't page."""
    decision = _priced_edit_job(1920, 1088, 369, job_filter=_windows_5090_edit_filter())
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"


def test_decide_edit_job_loads_345f_on_5090_free_short() -> None:
    """The shrunken 345f window asks 29.9 against 30.1 free: load, don't refuse."""
    decision = _priced_edit_job(1920, 1088, 345, job_filter=_windows_5090_edit_filter())
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_fast_budget_still_full_loads_the_720p_extend_clip() -> None:
    decision = _priced_video_job(
        1280, 704, 337,
        job_filter=_job_filter(memory_gb=31, process_mode="full_models_loading"),
    )
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "full_models_loading"


def test_decide_video_job_31gb_1080p_10s_streams() -> None:
    decision = _video_job(1920, 1088, 10, memory_gb=31, process_mode="full_models_loading")
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_decide_video_job_24gb_process_never_full() -> None:
    decision = _video_job(1024, 576, 5, memory_gb=24, process_mode="streaming_models_loading")
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_decide_video_job_darwin_never_full() -> None:
    decision = _video_job(
        1024,
        576,
        5,
        memory_gb=48,
        process_mode="full_models_loading",
        system="Darwin",
    )
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_decide_video_job_darwin_m4_loads_jobs_cuda_curve_would_reject() -> None:
    """Free RAM on a 48 GB Mac is often ~20 GiB; Fast 720p/20s stream estimate is ~20."""
    for width, height, seconds in ((1280, 704, 20), (1920, 1088, 10)):
        decision = _video_job(
            width,
            height,
            seconds,
            memory_gb=20,
            process_mode="full_models_loading",
            system="Darwin",
        )
        assert isinstance(decision, VideoJobLoad)
        assert decision.mode == "streaming_models_loading"


def test_legacy_darwin_no_model_never_rejects() -> None:
    """Pre-download Darwin (no model resolved) shows the grid instead of 422ing.

    The same cell is stream_over on 8 GB elsewhere; on Darwin the legacy curve
    never compares against unified memory.
    """
    decision = _video_job(
        1920,
        1088,
        20,
        memory_gb=8,
        process_mode="streaming_models_loading",
        system="Darwin",
        model_id=None,
    )
    assert isinstance(decision, VideoJobLoad)


def test_decide_video_job_stream_over_rejects() -> None:
    decision = _one_pass_job(1280, 704, 20, memory_gb=8, process_mode="full_models_loading")
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"


def test_decide_video_job_16gb_1080p_10s_rejects() -> None:
    decision = _one_pass_job(1920, 1088, 10, memory_gb=16, process_mode="streaming_models_loading")
    assert isinstance(decision, VideoJobReject)


def test_decide_video_job_unsupported_process_rejects() -> None:
    decision = _video_job(1024, 576, 5, memory_gb=31, process_mode="unsupported")
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "unsupported"


def _fast_durations(width: int, height: int, seconds: tuple[int, ...], **kwargs) -> list[int]:
    return advertised_fast_durations(
        width,
        height,
        24,
        seconds,
        job_filter=_job_filter(**kwargs),
    )


def test_advertised_fast_cuda_16gb_drops_720p20_and_1080p10() -> None:
    kwargs = dict(memory_gb=16, process_mode="streaming_models_loading")
    assert 20 not in _fast_durations(1280, 704, (5, 6, 8, 10, 20), **kwargs)
    assert 10 not in _fast_durations(1920, 1088, (5, 10), **kwargs)
    assert _fast_durations(1024, 576, (5, 6, 8, 10, 20), **kwargs) == [5, 6, 8, 10, 20]
    assert _fast_durations(1280, 704, (5, 6, 8, 10, 20), **kwargs) == [5, 6, 8, 10]


def test_advertised_fast_cuda_31gb_keeps_720p20_and_1080p10() -> None:
    kwargs = dict(memory_gb=31, process_mode="full_models_loading")
    assert 20 in _fast_durations(1280, 704, (5, 6, 8, 10, 20), **kwargs)
    assert 10 in _fast_durations(1920, 1088, (5, 10), **kwargs)


def test_advertised_fast_cuda_24gb_keeps_1080p10() -> None:
    kept = _fast_durations(
        1920,
        1088,
        (5, 10),
        memory_gb=24,
        process_mode="streaming_models_loading",
    )
    assert kept == [5, 10]


def test_advertised_fast_darwin_20gib_keeps_ceiling() -> None:
    kwargs = dict(memory_gb=20, process_mode="streaming_models_loading", system="Darwin")
    assert 20 in _fast_durations(1280, 704, (5, 6, 8, 10, 20), **kwargs)
    assert 10 in _fast_durations(1920, 1088, (5, 10), **kwargs)


def test_advertised_fast_never_emits_1080p20() -> None:
    kept = _fast_durations(
        1920,
        1088,
        (5, 10, 20),
        memory_gb=31,
        process_mode="full_models_loading",
    )
    assert 20 not in kept


def test_advertised_fast_unsupported_is_empty() -> None:
    assert _fast_durations(1024, 576, (5, 6, 8, 10, 20), memory_gb=31, process_mode="unsupported") == []


def test_video_job_reject_errors_covers_every_reason() -> None:
    from typing import get_args

    from runtime_config.reject_reasons import VideoJobRejectReason
    from runtime_config.video_job_budget import VIDEO_JOB_REJECT_ERRORS

    assert set(VIDEO_JOB_REJECT_ERRORS) == set(get_args(VideoJobRejectReason))


def test_reject_free_short_says_close_apps_not_lower() -> None:
    message, code = reject_message_and_code(
        VideoJobReject(
            seq=0.0, full_gib=0.0, stream_gib=15.8, reason="stream_over"
        ),
        _job_filter(
            system="Windows",
            process_mode="streaming_models_loading",
            vram_bytes=16 * _GIB,
            free_vram_bytes=int(15.0 * _GIB),
        ),
    )
    assert code == VIDEO_JOB_TOO_LARGE
    assert message == (
        "This job needs 15.8 GB of GPU memory but only 15.0 GB is free "
        "right now (of 16.0 GB total). Close other heavy apps (like games, browsers, or video editors) and retry."
    )


def test_reject_home_free_short_says_lower_resolution_not_close_apps() -> None:
    """The free-short branch must honor the caller's action: Home re-pads, so retry is futile."""
    message, code = reject_message_and_code(
        VideoJobReject(
            seq=0.0, full_gib=0.0, stream_gib=31.6, reason="stream_over"
        ),
        _job_filter(
            system="Windows",
            process_mode="streaming_models_loading",
            vram_bytes=int(31.84 * _GIB),
            free_vram_bytes=int(30.1 * _GIB),
        ),
        action="Lower the resolution.",
        free_short_action="Lower the resolution.",
    )
    assert code == VIDEO_JOB_TOO_LARGE
    assert message == (
        "This job needs 31.6 GB of GPU memory but only 30.1 GB is free "
        "right now (of 31.8 GB total). Lower the resolution."
    )


def test_reject_total_short_says_lower_not_close_apps() -> None:
    message, code = reject_message_and_code(
        VideoJobReject(
            seq=0.0, full_gib=0.0, stream_gib=18.3, reason="stream_over"
        ),
        _job_filter(
            system="Windows",
            process_mode="streaming_models_loading",
            vram_bytes=16 * _GIB,
            free_vram_bytes=16 * _GIB,
        ),
    )
    assert code == VIDEO_JOB_TOO_LARGE
    assert message == (
        "This job needs 18.3 GB of GPU memory but this card has 16.0 GB. "
        "Lower the resolution, duration, frame rate, or aspect width."
    )


def test_reject_darwin_pool_message_has_no_close_apps_advice() -> None:
    message, _code = reject_message_and_code(
        VideoJobReject(
            seq=0.0, full_gib=0.0, stream_gib=24.2, reason="stream_over"
        ),
        _job_filter(
            system="Darwin",
            process_mode="streaming_models_loading",
            mps_pool_bytes=int(20.0 * _GIB),
        ),
    )
    assert message == (
        "This job needs 24.2 GB of GPU memory but this Mac allows 20.0 GB. "
        "Lower the resolution, duration, frame rate, or aspect width."
    )


def test_reject_ram_shortfall_names_required_vs_detected_gb() -> None:
    message, code = reject_message_and_code(
        VideoJobReject(
            seq=0.0,
            full_gib=0.0,
            stream_gib=8.0,
            reason="ram_shortfall",
            required_ram_bytes=48 * _GIB,
        ),
        _job_filter(
            system="Windows",
            process_mode="streaming_models_loading",
            vram_bytes=16 * _GIB,
            total_ram_bytes=32 * _GIB,
        ),
    )
    assert code == LOCAL_GENERATION_RAM_SHORTFALL
    assert message == (
        "This job needs 48 GB of system RAM but this machine has 32 GB. "
        "Close heavy programs; on desktops, more RAM is the real fix."
    )


def test_reject_ram_shortfall_unknown_figures_falls_back_to_legacy() -> None:
    message, code = reject_message_and_code(
        VideoJobReject(
            seq=0.0, full_gib=0.0, stream_gib=8.0, reason="ram_shortfall"
        ),
        _job_filter(
            system="Windows",
            process_mode="streaming_models_loading",
            vram_bytes=16 * _GIB,
            total_ram_bytes=None,
        ),
    )
    assert code == LOCAL_GENERATION_RAM_SHORTFALL
    assert message == LOCAL_GENERATION_RAM_SHORTFALL_MESSAGE


def test_reject_stream_over_coherent_mismatch_falls_back_to_legacy() -> None:
    message, code = reject_message_and_code(
        VideoJobReject(
            seq=0.0, full_gib=0.0, stream_gib=8.0, reason="stream_over"
        ),
        _job_filter(
            system="Linux",
            process_mode="streaming_models_loading",
            vram_bytes=16 * _GIB,
            free_vram_bytes=16 * _GIB,
        ),
    )
    assert code == VIDEO_JOB_TOO_LARGE
    assert message == VIDEO_JOB_TOO_LARGE_MESSAGE


def test_reject_caller_action_overrides_default_advice() -> None:
    message, _code = reject_message_and_code(
        VideoJobReject(
            seq=0.0, full_gib=0.0, stream_gib=18.3, reason="stream_over"
        ),
        _job_filter(
            system="Windows",
            process_mode="streaming_models_loading",
            vram_bytes=16 * _GIB,
            free_vram_bytes=16 * _GIB,
        ),
        action="Lower the resolution.",
    )
    assert message == (
        "This job needs 18.3 GB of GPU memory but this card has 16.0 GB. "
        "Lower the resolution."
    )


@pytest.mark.usefixtures("single_pass_only")
def test_decide_fast_durations_reports_per_cell_reject_reasons() -> None:
    from runtime_config.video_job_budget import decide_fast_durations

    over_vram = decide_fast_durations(
        1920,
        1088,
        24,
        (5, 10),
        job_filter=_job_filter(
            memory_gb=16, process_mode="streaming_models_loading", **_windows_profile(vram_gib=16)
        ),
    )
    assert over_vram.kept == [5]
    assert _refused_reasons(over_vram) == {10: "stream_over"}

    short_ram = decide_fast_durations(
        1024,
        576,
        24,
        (5,),
        job_filter=_job_filter(
            memory_gb=16,
            process_mode="streaming_models_loading",
            **_windows_profile(vram_gib=16, total_ram_gib=32),
        ),
    )
    assert short_ram.kept == []
    assert _refused_reasons(short_ram) == {5: "ram_shortfall"}


def _mac_profile(
    *,
    pool_gib: float | None,
    model_id: str = _FAST_2_5_MODEL_ID,
) -> dict[str, object]:
    """Darwin Fast (2.5 default; 2.3 rides the same pool gate) with a sampled pool."""
    return {
        "system": "Darwin",
        "model_id": model_id,
        "mps_pool_bytes": None if pool_gib is None else int(pool_gib * _GIB),
    }


@pytest.mark.parametrize(
    ("width", "height", "seconds", "low", "high"),
    [
        # (cell, pass edge): 540p/5s 9.75, 720p/10s 12.70, 1080p/5s 13.77,
        # 540p/20s 15.30, 720p/20s 21.29, 1080p/10s 24.71. The 720p/10s
        # fence cell landed on the line: no duration term. 1080p/5s ran at 13.77
        # where the model asks 14.49 — an upper bound, not a floor.
        # The +0.03 intercept lift covers the fence cell exactly and leaves three asks
        # ≤0.03 above their pass edges (safe-direction over-refusal, never
        # an inversion: every fail top stays refused).
        (1024, 576, 5, 9.5, 9.75),
        (1280, 704, 10, 12.5, 12.70),
        (1920, 1088, 5, 13.5, 14.5),
        (1024, 576, 20, 15.0, 15.35),
        (1280, 704, 20, 21.0, 21.55),
        (1920, 1088, 10, 24.5, 24.75),
    ],
)
def test_mac_fast_stream_required_matches_measured_brackets(
    width: int, height: int, seconds: int, low: float, high: float
) -> None:
    from runtime_config.video_job_budget import _mac_fast_stream_required_gib

    tokens = video_tokens(height, width, compute_num_frames(seconds, 24))
    required = _mac_fast_stream_required_gib(tokens)
    assert low < required <= high


def test_mac_25_pool_edges_gate_f1_cell() -> None:
    load = _one_pass_job(
        1280, 704, 10,
        process_mode="streaming_models_loading",
        **_mac_profile(pool_gib=_cap(12.70)),
    )
    assert isinstance(load, VideoJobLoad)
    assert load.mode == "streaming_models_loading"
    reject = _one_pass_job(
        1280, 704, 10,
        process_mode="streaming_models_loading",
        **_mac_profile(pool_gib=_cap(12.30)),
    )
    assert isinstance(reject, VideoJobReject)
    assert reject.reason == "stream_over"


def test_mac_25_540p5s_floor_edges() -> None:
    load = _video_job(
        1024, 576, 5,
        process_mode="streaming_models_loading",
        **_mac_profile(pool_gib=_cap(9.75)),
    )
    assert isinstance(load, VideoJobLoad)
    reject = _video_job(
        1024, 576, 5,
        process_mode="streaming_models_loading",
        **_mac_profile(pool_gib=_cap(9.40)),
    )
    assert isinstance(reject, VideoJobReject)
    assert reject.reason == "stream_over"


def test_mac_25_1080p10s_top_cell_edges() -> None:
    # Ask 24.74 sits 0.03 above the 24.71 pass edge (the +0.03 intercept lift
    # covers the 720p/10s fence cell): safe-direction over-refusal, never an
    # inversion — the 24.31 fail top stays refused.
    load = _one_pass_job(
        1920, 1088, 10,
        process_mode="streaming_models_loading",
        **_mac_profile(pool_gib=_cap(24.74)),
    )
    assert isinstance(load, VideoJobLoad)
    reject = _one_pass_job(
        1920, 1088, 10,
        process_mode="streaming_models_loading",
        **_mac_profile(pool_gib=_cap(24.71)),
    )
    assert isinstance(reject, VideoJobReject)
    assert reject.reason == "stream_over"
    fail_top = _one_pass_job(
        1920, 1088, 10,
        process_mode="streaming_models_loading",
        **_mac_profile(pool_gib=_cap(24.31)),
    )
    assert isinstance(fail_top, VideoJobReject)
    assert fail_top.reason == "stream_over"


def test_mac_25_unlisted_resolution_gets_finite_requirement() -> None:
    """1280x1280 was never measured: must load on a big pool, not fail closed."""
    decision = _video_job(
        1280,
        1280,
        5,
        process_mode="streaming_models_loading",
        **_mac_profile(pool_gib=32),
    )
    assert isinstance(decision, VideoJobLoad)


def test_mac_25_never_full_on_huge_pool() -> None:
    decision = _video_job(
        1024, 576, 5,
        process_mode="streaming_models_loading",
        **_mac_profile(pool_gib=100),
    )
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_mac_23_routes_through_pool_gate_instead_of_never_reject() -> None:
    """Darwin 2.3 refuses over the pool: the legacy never-reject branch was fail-open."""
    refused = _video_job(
        1920, 1088, 10,
        process_mode="streaming_models_loading",
        **_mac_profile(pool_gib=5, model_id=_WINDOWS_2_3_MODEL_ID),
    )
    assert isinstance(refused, VideoJobReject)
    assert refused.reason == "stream_over"
    loaded = _video_job(
        1920, 1088, 10,
        process_mode="streaming_models_loading",
        **_mac_profile(pool_gib=32, model_id=_WINDOWS_2_3_MODEL_ID),
    )
    assert isinstance(loaded, VideoJobLoad)
    assert loaded.mode == "streaming_models_loading"


def test_mac_fast_unknown_model_id_is_unmeasured() -> None:
    """A future model id prices as unmeasured instead of crashing or silently passing."""
    job_filter = _job_filter(
        process_mode="streaming_models_loading",
        **_mac_profile(pool_gib=32, model_id="ltx-9.9-future"),  # type: ignore[arg-type]
    )
    cautious = decide_video_job(
        1024,
        576,
        compute_num_frames(5, 24),
        job_filter=job_filter,
    )
    assert _untested_basis(cautious) == "unmeasured_model"

    brave = decide_video_job(
        1024,
        576,
        compute_num_frames(5, 24),
        job_filter=LocalJobFilter(
            profile=job_filter.profile,
            budget=job_filter.budget,
            budget_mode="brave",
        ),
    )
    assert isinstance(brave, Admitted)
    assert brave.risk is not None
    assert brave.risk.tier == "unmeasured"
    assert brave.risk.unmeasured_basis == "unmeasured_model"


def test_mac_25_unknown_pool_fails_closed() -> None:
    decision = _video_job(
        1024, 576, 5,
        process_mode="streaming_models_loading",
        **_mac_profile(pool_gib=None),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "unsupported"


def test_windows_25_ignores_pool_bytes() -> None:
    """Pool neither gates nor rescues the Windows VRAM profile."""
    loads = _one_pass_job(
        1024, 576, 5,
        process_mode="streaming_models_loading",
        mps_pool_bytes=None,
        **_windows_profile(vram_gib=16),
    )
    assert isinstance(loads, VideoJobLoad)
    rejects = _one_pass_job(
        1280, 704, 20,
        process_mode="streaming_models_loading",
        mps_pool_bytes=int(100 * _GIB),
        **_windows_profile(vram_gib=12),
    )
    assert isinstance(rejects, VideoJobReject)
    assert rejects.reason == "stream_over"


def test_mac_25_unsupported_mode_precedence() -> None:
    decision = _video_job(
        1024, 576, 5,
        process_mode="unsupported",
        **_mac_profile(pool_gib=100),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "unsupported"


def test_mac_25_pool_gates_darwin_grid() -> None:
    """A 14 GiB MPS budget runs 1080p only in short windows, and 20 s needs more than five of them."""
    kwargs = dict(process_mode="streaming_models_loading", **_mac_profile(pool_gib=_cap(14.0)))
    assert _fast_durations(1024, 576, (5, 6, 8, 10, 20), **kwargs) == [5, 6, 8, 10, 20]
    assert _fast_durations(1920, 1088, (5, 10, 20), **kwargs) == [5, 10]


def test_mac_edit_ask_also_carries_the_loader_buffers() -> None:
    """A pool that fits a 540p/5s retake's MPS ask alone must still refuse it."""
    mps_ask = _mac_edit_stream_required_gib(video_tokens(576, 1024, 121))

    def edit(pool_gib: float) -> VideoJobDecision:
        return _priced_edit_job(
            1024, 576, 121,
            job_filter=_job_filter(
                process_mode="streaming_models_loading", **_mac_profile(pool_gib=pool_gib)
            ),
        )

    short = edit(mps_ask + 0.01)
    assert isinstance(short, VideoJobReject)
    assert short.reason == "stream_over"
    assert isinstance(edit(mps_ask + _MAC_HOST_GIB + 0.01), VideoJobLoad)


def test_mac_over_figure_is_positive_when_only_the_loader_buffers_miss() -> None:
    """11 GiB fits 540p/5s's 9.75 MPS floor but not floor + buffers: the refusal must
    print the charged shortfall, not need-under-have."""
    job_filter = _job_filter(process_mode="streaming_models_loading", **_mac_profile(pool_gib=11.0))
    decision = _one_pass_priced(1024, 576, compute_num_frames(5, 24), job_filter=job_filter)
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"
    assert reject_over_gib(decision, job_filter) == approx(9.75 + _MAC_HOST_GIB - 11.0)
    readout = budget_readout_for_reject(decision, job_filter)
    assert readout is not None
    assert readout.need_gb > readout.have_gb


def test_mac_future_model_prices_the_largest_known_block() -> None:
    """A model id past the spec table pays the loader buffers instead of riding free."""

    def ask(model_id: str) -> float:
        decision = _priced_video_job(
            1024, 576, compute_num_frames(5, 24),
            job_filter=_job_filter(
                process_mode="streaming_models_loading",
                **_mac_profile(pool_gib=107.52, model_id=model_id),
            ),
        )
        return decision.stream_gib

    largest = mac_disk_stream_host_gib(_WINDOWS_2_3_MODEL_ID)
    assert largest > _MAC_HOST_GIB
    assert ask("ltx-9.9-future") == approx(9.75 + largest)
    assert ask(_FAST_2_5_MODEL_ID) == approx(9.75 + _MAC_HOST_GIB)


def _windows_profile(
    *,
    vram_gib: float | None,
    total_ram_gib: float | None = 48,
    model_id: str = _FAST_2_5_MODEL_ID,
    system: str = "Windows",
    drive_type: object = None,
) -> dict[str, object]:
    return {
        "system": system,
        "model_id": model_id,
        "drive_type": drive_type,
        "vram_bytes": None if vram_gib is None else int(vram_gib * _GIB),
        "total_ram_bytes": (
            None if total_ram_gib is None else int(total_ram_gib * _GIB)
        ),
    }


@pytest.mark.parametrize(
    ("width", "height", "seconds", "low", "high"),
    [
        # (cell, required): 540p/5s 8.00, 540p/10s 8.00, 720p/8s 8.00,
        # 720p/10s 8.78, 1080p/5s 10.20, 540p/20s 10.86, 720p/20s 15.77,
        # 1080p/10s 18.30. Bounds from T1 acceptance; 1080p/5s ran at 10.0
        # where the model asks 10.20 — an upper bound, not a floor.
        (1024, 576, 5, 7.9, 8.0),
        (1024, 576, 10, 7.9, 8.25),
        (1280, 704, 8, 7.9, 8.25),
        (1280, 704, 10, 8.5, 9.0),
        (1920, 1088, 5, 10.0, 10.5),
        (1024, 576, 20, 10.5, 11.0),
        (1280, 704, 20, 15.5, 16.0),
        (1920, 1088, 10, 18.0, 19.0),
    ],
)
def test_windows_fast_stream_required_matches_continuous_model(
    width: int, height: int, seconds: int, low: float, high: float
) -> None:
    from runtime_config.video_job_budget import _windows_fast_stream_required_gib

    tokens = video_tokens(height, width, compute_num_frames(seconds, 24))
    required = _windows_fast_stream_required_gib(tokens)
    assert low < required <= high


def test_windows_25_unlisted_resolution_gets_finite_requirement() -> None:
    """1280x1280 was never in the slack table: must load, not fail closed."""
    decision = _video_job(
        1280,
        1280,
        5,
        memory_gb=32,
        process_mode="streaming_models_loading",
        **_windows_profile(vram_gib=32),
    )
    assert isinstance(decision, VideoJobLoad)


@pytest.mark.usefixtures("single_pass_only")
def test_windows_25_16gib_grid_drops_720p20() -> None:
    """720p/20s needs 20.03 reserved (measured 19.53); 16 GB keeps the rest."""
    profile = _windows_profile(vram_gib=16)

    assert _fast_durations(
        1024, 576, (5, 6, 8, 10, 20), memory_gb=16, process_mode="streaming_models_loading", **profile
    ) == [5, 6, 8, 10, 20]
    assert _fast_durations(
        1280, 704, (5, 6, 8, 10, 20), memory_gb=16, process_mode="streaming_models_loading", **profile
    ) == [5, 6, 8, 10]
    assert _fast_durations(
        1920, 1088, (5, 10), memory_gb=16, process_mode="streaming_models_loading", **profile
    ) == [5]


@pytest.mark.usefixtures("single_pass_only")
def test_windows_25_12gib_grid_serves_limited_cells() -> None:
    profile = _windows_profile(vram_gib=12)

    assert _fast_durations(
        1024, 576, (5, 6, 8, 10, 20), memory_gb=12, process_mode="streaming_models_loading", **profile
    ) == [5, 6, 8, 10]
    assert _fast_durations(
        1280, 704, (5, 6, 8, 10, 20), memory_gb=12, process_mode="streaming_models_loading", **profile
    ) == [5, 6, 8]
    assert _fast_durations(
        1920, 1088, (5, 10), memory_gb=12, process_mode="streaming_models_loading", **profile
    ) == []


def test_windows_25_10gb_grid_serves_small_cells() -> None:
    """10 GB-class at 9.99 (free == total): 5 cells; 540p/10s (10.00) is the first refusal."""
    profile = _windows_profile(vram_gib=9.99)

    assert _fast_durations(
        1024, 576, (5, 6, 8, 10, 20), memory_gb=9, process_mode="streaming_models_loading", **profile
    ) == [5, 6, 8]
    assert _fast_durations(
        1280, 704, (5, 6, 8, 10, 20), memory_gb=9, process_mode="streaming_models_loading", **profile
    ) == [5, 6]
    assert _fast_durations(
        1920, 1088, (5, 10), memory_gb=9, process_mode="streaming_models_loading", **profile
    ) == []


def test_windows_25_supported_mode_empty_grid_when_free_below_smallest_cell() -> None:
    """The floor and the model stay distinguishable if the floor moves again.

    Total 9.2 clears the early-out floor (mode streaming), but free 7.8 sits
    below the 8.00 smallest cell, so every cell rejects as ``stream_over`` —
    the model refusing on its merits, not the floor deciding. Note the
    free/total distinction is the point: 7.8 as *total* VRAM is below the
    floor and would be ``unsupported`` mode, not an empty grid.
    """
    from runtime_config.runtime_policy import decide_local_generation_mode
    from runtime_config.video_job_budget import decide_fast_durations

    assert (
        decide_local_generation_mode(system="Windows", cuda_available=True, vram_gb=9.2)
        == "streaming_models_loading"
    )
    job_filter = _job_filter(
        memory_gb=9,
        process_mode="streaming_models_loading",
        free_vram_bytes=int(7.8 * _GIB),
        **_windows_profile(vram_gib=9.2),
    )
    for width, height, durations in (
        (1024, 576, (5, 6, 8, 10, 20)),
        (1280, 704, (5, 6, 8, 10, 20)),
        (1920, 1088, (5, 10)),
    ):
        outcome = decide_fast_durations(width, height, 24, durations, job_filter=job_filter)
        assert outcome.kept == []
        assert set(_refused_reasons(outcome).values()) == {"stream_over"}


def test_linux_12gib_keeps_generic_curve_subset() -> None:
    profile = _windows_profile(vram_gib=12, system="Linux")

    assert _fast_durations(
        1024, 576, (5, 6, 8, 10, 20), memory_gb=12, process_mode="streaming_models_loading", **profile
    ) == [5, 6, 8, 10]
    assert _fast_durations(
        1280, 704, (5, 6, 8, 10, 20), memory_gb=12, process_mode="streaming_models_loading", **profile
    ) == [5, 6, 8, 10]
    assert _fast_durations(
        1920, 1088, (5, 10), memory_gb=12, process_mode="streaming_models_loading", **profile
    ) == []


def test_windows_23_streaming_ram_floor_matches_25() -> None:
    # The bug is 2.3 skipping the Fast RAM floor and streaming on a 32 GB box.
    for model_id in (_FAST_2_5_MODEL_ID, _WINDOWS_2_3_MODEL_ID):
        short = _video_job(
            1024,
            576,
            5,
            memory_gb=16,
            process_mode="streaming_models_loading",
            **_windows_profile(vram_gib=16, total_ram_gib=32, model_id=model_id),
        )
        assert isinstance(short, VideoJobReject)
        assert short.reason == "ram_shortfall"
        fits = _video_job(
            1024,
            576,
            5,
            memory_gb=16,
            process_mode="streaming_models_loading",
            **_windows_profile(vram_gib=16, total_ram_gib=40, model_id=model_id),
        )
        assert isinstance(fits, VideoJobLoad)


@pytest.mark.usefixtures("single_pass_only")
def test_windows_23_12gib_matches_25_measured_grid() -> None:
    # The bug is Windows 2.3 staying on the rough curve and keeping a 720p/10s
    # cell that 2.5 refuses at 12 GB.
    profile = _windows_profile(vram_gib=12, model_id=_WINDOWS_2_3_MODEL_ID)

    assert _fast_durations(
        1024, 576, (5, 6, 8, 10, 20), memory_gb=12, process_mode="streaming_models_loading", **profile
    ) == [5, 6, 8, 10]
    assert _fast_durations(
        1280, 704, (5, 6, 8, 10, 20), memory_gb=12, process_mode="streaming_models_loading", **profile
    ) == [5, 6, 8]
    assert _fast_durations(
        1920, 1088, (5, 10), memory_gb=12, process_mode="streaming_models_loading", **profile
    ) == []


@pytest.mark.usefixtures("single_pass_only")
def test_windows_25_23gib_unlocks_1080p10_and_22gib_does_not() -> None:
    full = _windows_profile(vram_gib=23)
    assert _fast_durations(
        1920, 1088, (5, 10), memory_gb=23, process_mode="streaming_models_loading", **full
    ) == [5, 10]
    assert _fast_durations(
        1280, 704, (5, 6, 8, 10, 20), memory_gb=23, process_mode="streaming_models_loading", **full
    ) == [5, 6, 8, 10, 20]

    below = _windows_profile(vram_gib=22)
    assert _fast_durations(
        1920, 1088, (5, 10), memory_gb=22, process_mode="streaming_models_loading", **below
    ) == [5]


@pytest.mark.usefixtures("single_pass_only")
def test_windows_25_vram_boundaries_are_strict_per_cell() -> None:
    import math

    from runtime_config.video_job_budget import _windows_fast_reserved_required_gib

    # Absolute values are pinned by the reserved fit (native reserved points); here one byte
    # on either side of each boundary flips the cell. ceil, not int: the
    # comparison is available >= required, so the floor byte would reject.
    tokens_720p20 = video_tokens(704, 1280, compute_num_frames(20, 24))
    at_720p20_bytes = math.ceil(_windows_fast_reserved_required_gib(tokens_720p20) * _GIB)
    at_720p20 = _windows_profile(vram_gib=15.5)
    at_720p20["vram_bytes"] = at_720p20_bytes
    assert _fast_durations(
        1280, 704, (5, 6, 8, 10, 20), memory_gb=15, process_mode="streaming_models_loading", **at_720p20
    ) == [5, 6, 8, 10, 20]

    below_720p20 = _windows_profile(vram_gib=15.5)
    below_720p20["vram_bytes"] = at_720p20_bytes - 1
    assert _fast_durations(
        1280, 704, (5, 6, 8, 10, 20), memory_gb=15, process_mode="streaming_models_loading", **below_720p20
    ) == [5, 6, 8, 10]
    # Smaller cells still fit below the 720p/20s boundary.
    assert _fast_durations(
        1024, 576, (5, 6, 8, 10, 20), memory_gb=15, process_mode="streaming_models_loading", **below_720p20
    ) == [5, 6, 8, 10, 20]

    tokens_1080p10 = video_tokens(1088, 1920, compute_num_frames(10, 24))
    at_1080p10_bytes = math.ceil(_windows_fast_reserved_required_gib(tokens_1080p10) * _GIB)
    at_1080p10 = _windows_profile(vram_gib=19)
    at_1080p10["vram_bytes"] = at_1080p10_bytes
    assert _fast_durations(
        1920, 1088, (5, 10), memory_gb=18, process_mode="streaming_models_loading", **at_1080p10
    ) == [5, 10]

    below_1080p10 = _windows_profile(vram_gib=19)
    below_1080p10["vram_bytes"] = at_1080p10_bytes - 1
    assert _fast_durations(
        1920, 1088, (5, 10), memory_gb=19, process_mode="streaming_models_loading", **below_1080p10
    ) == [5]


@pytest.mark.usefixtures("single_pass_only")
def test_windows_25_unknown_vram_bytes_falls_back_to_integer_gb() -> None:
    """Bytes-probe glitch with a viable integer figure: evaluate, don't empty.

    Without the fallback this is §3.5's contradiction — process mode says
    streaming while every cell fails closed. Floor rounding only hides, so
    the fallback grid is conservative by construction.
    """
    profile = _windows_profile(vram_gib=None)
    assert _fast_durations(
        1024, 576, (5, 6, 8, 10, 20), memory_gb=16, process_mode="streaming_models_loading", **profile
    ) == [5, 6, 8, 10, 20]
    assert _fast_durations(
        1280, 704, (5, 6, 8, 10, 20), memory_gb=16, process_mode="streaming_models_loading", **profile
    ) == [5, 6, 8, 10]
    assert _fast_durations(
        1920, 1088, (5, 10), memory_gb=16, process_mode="streaming_models_loading", **profile
    ) == [5]


def test_windows_25_unknown_vram_bytes_and_gb_fails_closed() -> None:
    profile = _windows_profile(vram_gib=None)
    assert _fast_durations(
        1024, 576, (5, 6, 8, 10, 20), memory_gb=None, process_mode="streaming_models_loading", **profile
    ) == []


def test_windows_25_oversize_cell_is_stream_over_not_unsupported() -> None:
    decision = _one_pass_job(
        1920,
        1088,
        10,
        memory_gb=16,
        process_mode="streaming_models_loading",
        **_windows_profile(vram_gib=16),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"


def test_windows_25_static_cells_load_or_refuse_with_copy_on_32gib() -> None:
    from typing import Literal

    from runtime_config.video_job_budget import decide_fast_durations

    caps = local_caps("ltx-2.5-22b-distilled")
    spec = get_ltx_model_spec("ltx-2.5-22b-distilled").supported_pipelines[0][1]
    assert spec.supported_resolutions_durations is not None
    profile = _windows_profile(vram_gib=32)
    aspects: tuple[Literal["16:9", "9:16"], ...] = ("16:9", "9:16")
    refused: set[tuple[str, str, int]] = set()

    for resolution, rspec in spec.supported_resolutions_durations.items():
        durations = tuple(int(d) for d in rspec.fps_to_durations.get(24, []))
        assert durations, f"{resolution} must advertise 24fps durations"
        for aspect in aspects:
            width, height = pixels_for(caps, resolution, aspect)
            outcome = decide_fast_durations(
                width,
                height,
                24,
                durations,
                job_filter=_job_filter(
                    memory_gb=32,
                    process_mode="streaming_models_loading",
                    **profile,
                ),
            )
            assert set(outcome.kept) | set(_refused_reasons(outcome)) == set(durations)
            for duration, reason in _refused_reasons(outcome).items():
                # Measured profiles always compare a figure: the refusal must
                # carry it for picker copy, never vanish silently.
                assert reason != "unsupported"
                assert duration in outcome.refused
                assert reason != "unmeasured_cell"  # every static 1440p and 4K cell has a run
                refused.add((resolution, aspect, duration))

    # 1080p/20s needs 34.5 GiB in one pass, the only static cell above the 32 GB envelope. A card the
    # windowed 1080p run covers (RTX 5090 class) now runs it in windows, so nothing is refused.
    assert refused == set()


def test_windows_25_streaming_jobs_require_40_gib_total_ram() -> None:
    at_floor = _video_job(
        1024,
        576,
        5,
        memory_gb=15,
        process_mode="streaming_models_loading",
        **_windows_profile(vram_gib=15.5, total_ram_gib=40),
    )
    assert isinstance(at_floor, VideoJobLoad)

    # Reporting headroom: a 40 GB box reads ~39.5, which still admits.
    within_headroom = _video_job(
        1024,
        576,
        5,
        memory_gb=15,
        process_mode="streaming_models_loading",
        **_windows_profile(vram_gib=15.5, total_ram_gib=39.5),
    )
    assert isinstance(within_headroom, VideoJobLoad)

    below_floor = _video_job(
        1024,
        576,
        5,
        memory_gb=15,
        process_mode="streaming_models_loading",
        **_windows_profile(vram_gib=15.5, total_ram_gib=39 - 1 / _GIB),
    )
    assert isinstance(below_floor, VideoJobReject)
    assert below_floor.reason == "ram_shortfall"


@pytest.mark.parametrize("drive_type", ["ssd", "hdd"])
def test_windows_25_cpu_path_ram_floor_stays_40(drive_type: str) -> None:
    """Non-NVMe drives keep CPU pinning and its 40 GB floor (unknown drive: see above)."""
    at_floor = _video_job(
        1024,
        576,
        5,
        memory_gb=16,
        process_mode="streaming_models_loading",
        **_windows_profile(vram_gib=16, total_ram_gib=40, drive_type=drive_type),
    )
    assert isinstance(at_floor, VideoJobLoad)

    below_floor = _video_job(
        1024,
        576,
        5,
        memory_gb=16,
        process_mode="streaming_models_loading",
        **_windows_profile(vram_gib=16, total_ram_gib=39 - 1 / _GIB, drive_type=drive_type),
    )
    assert isinstance(below_floor, VideoJobReject)
    assert below_floor.reason == "ram_shortfall"


def test_windows_25_disk_path_ram_floor_is_24() -> None:
    """NVMe streams from disk (~5–6 GB working set), so the floor drops to 24, strict below 23."""
    at_floor = _video_job(
        1024,
        576,
        5,
        memory_gb=16,
        process_mode="streaming_models_loading",
        **_windows_profile(vram_gib=16, total_ram_gib=24, drive_type="nvme"),
    )
    assert isinstance(at_floor, VideoJobLoad)

    within_headroom = _video_job(
        1024,
        576,
        5,
        memory_gb=16,
        process_mode="streaming_models_loading",
        **_windows_profile(vram_gib=16, total_ram_gib=23.5, drive_type="nvme"),
    )
    assert isinstance(within_headroom, VideoJobLoad)

    below_floor = _video_job(
        1024,
        576,
        5,
        memory_gb=16,
        process_mode="streaming_models_loading",
        **_windows_profile(vram_gib=16, total_ram_gib=23 - 1 / _GIB, drive_type="nvme"),
    )
    assert isinstance(below_floor, VideoJobReject)
    assert below_floor.reason == "ram_shortfall"


def test_windows_25_disk_and_cpu_paths_disagree_at_36gib_ram() -> None:
    """The whole point of the DISK floor: same machine, different drive, different verdict."""
    nvme = _video_job(
        1024,
        576,
        5,
        memory_gb=16,
        process_mode="streaming_models_loading",
        **_windows_profile(vram_gib=16, total_ram_gib=36, drive_type="nvme"),
    )
    assert isinstance(nvme, VideoJobLoad)
    assert nvme.mode == "streaming_models_loading"

    sata = _video_job(
        1024,
        576,
        5,
        memory_gb=16,
        process_mode="streaming_models_loading",
        **_windows_profile(vram_gib=16, total_ram_gib=36, drive_type="ssd"),
    )
    assert isinstance(sata, VideoJobReject)
    assert sata.reason == "ram_shortfall"


def test_linux_streaming_ignores_ram_and_drive() -> None:
    """No RAM gate off the Windows Fast profile: a VRAM-over cell is stream_over, never shortfall."""
    decision = _one_pass_job(
        1280,
        704,
        20,
        memory_gb=16,
        process_mode="streaming_models_loading",
        **_windows_profile(vram_gib=16, total_ram_gib=24, system="Linux", drive_type="nvme"),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"


def test_windows_25_unknown_total_ram_fails_closed_as_unsupported() -> None:
    decision = _video_job(
        1024,
        576,
        5,
        memory_gb=15,
        process_mode="streaming_models_loading",
        **_windows_profile(vram_gib=15.5, total_ram_gib=None),
    )

    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "unsupported"


def test_windows_streaming_ram_threshold_matches_linux_cpu_offload_floor() -> None:
    from runtime_config.runtime_policy import CUDA_CPU_OFFLOAD_RAM_FLOOR_GB
    from runtime_config.video_job_budget import WINDOWS_FAST_STREAMING_RAM_BYTES

    assert WINDOWS_FAST_STREAMING_RAM_BYTES == CUDA_CPU_OFFLOAD_RAM_FLOOR_GB * _GIB


@pytest.mark.usefixtures("single_pass_only")
def test_windows_25_free_vram_restricts_advertised_cells() -> None:
    profile = _windows_profile(vram_gib=16)
    busy = dict(profile, free_vram_bytes=int(11.0 * _GIB))
    assert _fast_durations(
        1280, 704, (5, 6, 8, 10, 20), memory_gb=16, process_mode="streaming_models_loading", **busy
    ) == [5, 6]
    assert _fast_durations(
        1024, 576, (5, 6, 8, 10, 20), memory_gb=16, process_mode="streaming_models_loading", **busy
    ) == [5, 6, 8, 10]

    idle = dict(profile, free_vram_bytes=int(15.0 * _GIB))
    assert _fast_durations(
        1280, 704, (5, 6, 8, 10, 20), memory_gb=16, process_mode="streaming_models_loading", **idle
    ) == [5, 6, 8, 10]
    assert _fast_durations(
        1024, 576, (5, 6, 8, 10, 20), memory_gb=16, process_mode="streaming_models_loading", **idle
    ) == [5, 6, 8, 10, 20]


def test_windows_25_unknown_free_vram_falls_back_to_total() -> None:
    decision = _video_job(
        1280,
        704,
        10,
        memory_gb=16,
        process_mode="streaming_models_loading",
        free_vram_bytes=None,
        **_windows_profile(vram_gib=16),
    )
    assert isinstance(decision, VideoJobLoad)


def test_windows_25_cell_over_free_vram_is_stream_over() -> None:
    decision = _one_pass_job(
        1280,
        704,
        20,
        memory_gb=16,
        process_mode="streaming_models_loading",
        free_vram_bytes=int(15.0 * _GIB),
        **_windows_profile(vram_gib=16),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"


def test_windows_fast_540p20_refuses_paging_cell_on_12gb() -> None:
    """Allocated ask 10.86 admits at free 11.5, but native reserved is 14.64; must refuse."""
    decision = _one_pass_job(
        1024,
        576,
        20,
        memory_gb=12,
        process_mode="streaming_models_loading",
        free_vram_bytes=int(11.5 * _GIB),
        **_windows_profile(vram_gib=12),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"


def test_windows_fast_1080p5_refuses_paging_cell_on_12gb() -> None:
    """Allocated ask 10.20 admits at free 11.5, but native reserved is 13.61; must refuse."""
    decision = _one_pass_job(
        1920,
        1088,
        5,
        memory_gb=12,
        process_mode="streaming_models_loading",
        free_vram_bytes=int(11.5 * _GIB),
        **_windows_profile(vram_gib=12),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"


def test_windows_fast_720p5_encode_spike_refuses_at_free_9_5() -> None:
    """720p/5s encode spike hits 9.84; floor 9.90 refuses at free 9.5."""
    decision = _video_job(
        1280,
        704,
        5,
        memory_gb=9,
        process_mode="streaming_models_loading",
        free_vram_bytes=int(9.5 * _GIB),
        **_windows_profile(vram_gib=9.99),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"


def test_windows_fast_floor_keeps_idle_10gb_for_540p5() -> None:
    """Floor 9.90 must not strand idle 10GB (total 9.99): 540p/5s (R 8.95) loads at free 9.95."""
    decision = _video_job(
        1024,
        576,
        5,
        memory_gb=9,
        process_mode="streaming_models_loading",
        free_vram_bytes=int(9.95 * _GIB),
        **_windows_profile(vram_gib=9.99),
    )
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_windows_25_free_vram_cannot_expand_beyond_total() -> None:
    decision = _one_pass_job(
        1920,
        1088,
        10,
        memory_gb=15,
        process_mode="streaming_models_loading",
        free_vram_bytes=int(19.0 * _GIB),
        **_windows_profile(vram_gib=15.5),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"


def test_windows_25_full_mode_pick_ignores_free_vram() -> None:
    decision = _video_job(
        1024,
        576,
        5,
        memory_gb=32,
        process_mode="full_models_loading",
        free_vram_bytes=int(10.0 * _GIB),
        **_windows_profile(vram_gib=32),
    )
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "full_models_loading"


def test_windows_fast_full_reports_reserved_not_legacy_ask() -> None:
    """540p/5s full ask is the reserved 23.76, not the legacy 20.25: total
    25.75 streams (budget 23.75), total 25.76 full-loads (budget 23.76)."""
    streams = _video_job(
        1024,
        576,
        5,
        memory_gb=25,
        process_mode="full_models_loading",
        free_vram_bytes=int(25.75 * _GIB),
        **_windows_profile(vram_gib=25.75),
    )
    assert isinstance(streams, VideoJobLoad)
    assert streams.mode == "streaming_models_loading"
    full = _video_job(
        1024,
        576,
        5,
        memory_gb=25,
        process_mode="full_models_loading",
        free_vram_bytes=int(25.76 * _GIB),
        **_windows_profile(vram_gib=25.76),
    )
    assert isinstance(full, VideoJobLoad)
    assert full.mode == "full_models_loading"
    assert full.full_gib > 23.0


def test_windows_fast_24gb_streams_small_cells_never_full() -> None:
    """Full-mode floor is 23.70 over a 22.0 budget — nothing full-loads on
    24 GB, not even the smallest measured cells (both paged natively)."""
    for width, height in [(1024, 576), (1280, 704)]:
        decision = _video_job(
            width,
            height,
            5,
            memory_gb=24,
            process_mode="full_models_loading",
            free_vram_bytes=int(23.5 * _GIB),
            **_windows_profile(vram_gib=24),
        )
        assert isinstance(decision, VideoJobLoad)
        assert decision.mode == "streaming_models_loading"


def test_windows_fast_32gb_1080p5_streams_after_full_audit() -> None:
    """1080p/5s full ask is 31.02 (censored peak >= 29.93) — must stream on
    32 GB, where the legacy 25.96 full-loaded it into a page."""
    decision = _video_job(
        1920,
        1088,
        5,
        memory_gb=31,
        process_mode="full_models_loading",
        free_vram_bytes=int(31.0 * _GIB),
        **_windows_profile(vram_gib=31.84),
    )
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_windows_fast_32gb_720p10_stays_full_on_thin_margin() -> None:
    """720p/10s full ask 29.36 clears the 29.84 budget by 0.48 — a slope or
    rounding drift must not silently strand it to stream."""
    decision = _video_job(
        1280,
        704,
        10,
        memory_gb=31,
        process_mode="full_models_loading",
        free_vram_bytes=int(31.0 * _GIB),
        **_windows_profile(vram_gib=31.84),
    )
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "full_models_loading"


def test_windows_25_budget_accepts_non_24fps_frame_counts() -> None:
    """No 24fps hardcode: fps flows through frame counts into continuous tokens.

    A 601-frame (20s@30fps) 720p job needs ~19.3 GiB — over 16, under 24.
    The old duration-table check would have rejected it outright (601 > 481).
    """
    decision = _video_job(
        1280,
        704,
        20,
        fps=30,
        memory_gb=24,
        process_mode="streaming_models_loading",
        **_windows_profile(vram_gib=24),
    )
    assert isinstance(decision, VideoJobLoad)


@pytest.mark.usefixtures("single_pass_only")
def test_windows_25_portrait_matches_landscape_at_boundary() -> None:
    """9:16 is the same latent geometry (identical tokens, identical requirement)."""
    import math

    from runtime_config.video_job_budget import _windows_fast_reserved_required_gib

    tokens_720p20 = video_tokens(704, 1280, compute_num_frames(20, 24))
    at_bytes = math.ceil(_windows_fast_reserved_required_gib(tokens_720p20) * _GIB)
    at_cap = _windows_profile(vram_gib=15.5)
    at_cap["vram_bytes"] = at_bytes
    assert _fast_durations(
        704, 1280, (20,), memory_gb=15, process_mode="streaming_models_loading", **at_cap
    ) == [20]

    below_cap = _windows_profile(vram_gib=15.5)
    below_cap["vram_bytes"] = at_bytes - 1
    assert _fast_durations(
        704, 1280, (20,), memory_gb=15, process_mode="streaming_models_loading", **below_cap
    ) == []


def test_windows_25_full_selected_job_does_not_require_48_gib_total_ram() -> None:
    decision = _video_job(
        1024,
        576,
        5,
        memory_gb=31,
        process_mode="full_models_loading",
        **_windows_profile(vram_gib=31, total_ram_gib=32),
    )

    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "full_models_loading"


def test_linux_and_darwin_keep_existing_envelopes() -> None:
    linux = _fast_durations(
        1280,
        704,
        (5, 6, 8, 10, 20),
        memory_gb=16,
        process_mode="streaming_models_loading",
        **_windows_profile(vram_gib=15, system="Linux"),
    )
    # Darwin leg pins the routed envelope: 2.3 takes the pool gate like 2.5,
    # and an unprobed pool fails closed (the legacy never-reject branch is
    # gone for known model ids).
    darwin = _fast_durations(
        1280,
        704,
        (5, 6, 8, 10, 20),
        memory_gb=20,
        process_mode="streaming_models_loading",
        **_windows_profile(
            vram_gib=None, system="Darwin", model_id=_WINDOWS_2_3_MODEL_ID
        ),
    )

    assert linux == [5, 6, 8, 10]
    assert darwin == []


@pytest.mark.usefixtures("single_pass_only")
def test_windows_23_keeps_existing_generic_curve() -> None:
    kept = _fast_durations(
        1280,
        704,
        (5, 6, 8, 10, 20),
        memory_gb=16,
        process_mode="streaming_models_loading",
        **_windows_profile(
            vram_gib=14,
            model_id=_WINDOWS_2_3_MODEL_ID,
        ),
    )

    assert kept == [5, 6, 8, 10]


def test_catalog_preview_two_stage_is_1080p_source_not_768_bucket() -> None:
    two_stage = catalog_preview_stage1(IcLoraSettings(skip_stage_2=False, resolution_factor=1.5))
    # 1920×1088 snapped /128 → 1920×1024 canvas, stage-1 960×512. The 768×1280
    # bucket would have been 384×640 (~8 GiB optimistic on 20s).
    assert sorted(two_stage) == [512, 960]


def _catalog_item_with_fps(fps: object) -> IcLoraCatalogItem:
    return IcLoraCatalogItem(
        id="fps-probe",
        name="n",
        description="d",
        download=DownloadSpec(
            repo_id="r",
            variants=[DownloadVariant(id="default", label="Default", filename="f.safetensors", size_bytes=10, base_model="LTX-2.3")],
        ),
        requires_hf_login=False,
        input=InputSpec(kind="image"),
        preprocessing=[PreprocessingStep(utility="image_to_frames", params={"fps": fps})],
        instructions=[InstructionSection(kind="summary", title="What it does", body="b")],
        default_settings=IcLoraSettings(),
    )


def test_with_budgeted_duration_unsupported_clears_options() -> None:
    item = _catalog_item_with_fps(24).model_copy(
        update={
            "controls": [
                IcLoraControl(
                    id="duration",
                    label="Duration",
                    kind="int",
                    default=5,
                    options=[5, 8, 10, 20],
                )
            ]
        }
    )
    out = with_budgeted_duration_controls(
        item,
        job_filter=_job_filter(
            memory_gb=31, process_mode="unsupported"
        ),
    )
    duration = next(control for control in out.controls if control.id == "duration")
    assert duration.options == []


def _catalog_item_two_stage(*, fps: object = 24) -> IcLoraCatalogItem:
    base = _catalog_item_with_fps(fps).model_copy(
        update={
            "default_settings": IcLoraSettings(skip_stage_2=False),
            "controls": [
                IcLoraControl(
                    id="duration",
                    label="Duration",
                    kind="int",
                    default=5,
                    options=[5, 8, 10, 20],
                )
            ],
        }
    )
    return base



def test_list_time_layout_to_render_prices_the_look_still() -> None:
    """At 22.5 free a Layout to Render window asks 22.75 with its still (drops)
    and 22.15 without — the list must not offer what generate refuses."""
    item = _catalog_item_two_stage().model_copy(
        update={
            "default_settings": IcLoraSettings(skip_stage_2=False, stage_2_ic_lora=True),
            "allows_reference_image": True,
            "reference_image_frame": REFERENCE_STILL_FRAME_IDX,
        }
    )
    out = with_budgeted_duration_controls(
        item, job_filter=_windows_measured_filter(31, 64, free_gib=22.5)
    )
    duration = next(control for control in out.controls if control.id == "duration")
    assert duration.dropped_options is not None
    assert [(d.value, d.reason) for d in duration.dropped_options] == [
        (5, "stream_over"),
        (8, "stream_over"),
        (10, "stream_over"),
        (20, "stream_over"),
    ]


def test_with_budgeted_duration_two_stage_prices_past_the_ceiling_on_the_curve() -> None:
    """Two-stage is measured through the 14,880-token ceiling (D4): at the 1080p
    preview only the 20s cell (29,280 tokens) drops, with the curve's figures."""
    out = with_budgeted_duration_controls(
        _catalog_item_two_stage(),
        job_filter=_windows_measured_filter(31, 64, free_gib=31.0),
    )
    duration = next(control for control in out.controls if control.id == "duration")
    assert duration.options == [5, 8, 10]
    assert duration.dropped_options is not None
    assert [(d.value, d.reason) for d in duration.dropped_options] == [
        (20, "stream_over")
    ]


def test_with_budgeted_duration_single_stage_keeps_long_durations() -> None:
    """Control: the >5s fence must not leak onto single-stage items."""
    item = _catalog_item_two_stage().model_copy(
        update={"default_settings": IcLoraSettings(skip_stage_2=True)}
    )
    out = with_budgeted_duration_controls(
        item,
        job_filter=_windows_measured_filter(31, 64, free_gib=31.0),
    )
    duration = next(control for control in out.controls if control.id == "duration")
    assert duration.options == [5, 8, 10, 20]


def test_with_budgeted_duration_drops_carry_tier_and_over_gib() -> None:
    # The bug is a stream_over drop without figures, so the advisory tooltip
    # cannot say "over by X" while the 422 can.
    out = with_budgeted_duration_controls(
        _catalog_item_two_stage(), job_filter=_windows_measured_filter(16, 64, free_gib=16.0)
    )
    duration = next(c for c in out.controls if c.id == "duration")
    assert duration.dropped_options is not None
    over = next(d for d in duration.dropped_options if d.reason == "stream_over")
    assert over.tier == "over"
    assert over.over_gib is not None and over.over_gib > 0
    untested = [d for d in duration.dropped_options if d.reason == "unmeasured_cell"]
    for drop in untested:
        assert drop.tier == "unmeasured"
        assert drop.over_gib is None
        assert drop.unmeasured_basis is not None


def test_catalog_output_fps_coerces_junk_to_default() -> None:
    assert catalog_output_fps(_catalog_item_with_fps(24)) == 24
    assert catalog_output_fps(_catalog_item_with_fps("12")) == 12
    assert catalog_output_fps(_catalog_item_with_fps(None)) == 24
    assert catalog_output_fps(_catalog_item_with_fps({})) == 24
    assert catalog_output_fps(_catalog_item_with_fps("nope")) == 24
    assert catalog_output_fps(_catalog_item_with_fps(True)) == 24
    assert catalog_output_fps(_catalog_item_with_fps(0)) == 24


def test_tight_band_marks_headroom_under_max_floor_and_ratio() -> None:
    job_filter = _windows_measured_filter(14.5, 48)
    decision = VideoJobLoad(mode="streaming_models_loading", seq=1000.0, full_gib=20.0, stream_gib=14.14)
    assert load_headroom_gib(decision, job_filter) == approx(0.36)
    assert is_tight_fit(decision, job_filter)


def test_headroom_past_the_band_is_fit_not_tight() -> None:
    job_filter = _windows_measured_filter(16.0, 48)
    decision = VideoJobLoad(mode="streaming_models_loading", seq=1000.0, full_gib=20.0, stream_gib=14.14)
    assert not is_tight_fit(decision, job_filter)


def test_unknown_figures_stay_silent_not_tight() -> None:
    job_filter = _job_filter(memory_gb=31, process_mode="streaming_models_loading")
    decision = VideoJobLoad(mode="streaming_models_loading", seq=1000.0, full_gib=20.0, stream_gib=10.0)
    assert load_headroom_gib(decision, job_filter) is None


def test_free_bound_shortfall_reports_free_kind() -> None:
    job_filter = _windows_measured_filter(24, 48, free_gib=19.0)
    decision = VideoJobReject(seq=1000.0, full_gib=30.0, stream_gib=22.7, reason="stream_over")
    assert reject_shortfall_kind(decision, job_filter) == "free"
    assert reject_over_gib(decision, job_filter) == approx(3.7)


def test_total_only_shortfall_reports_total_kind() -> None:
    job_filter = _windows_measured_filter(16, 48, free_gib=4.0, total_only=True)
    decision = VideoJobReject(seq=1000.0, full_gib=25.0, stream_gib=17.2, reason="stream_over")
    assert reject_shortfall_kind(decision, job_filter) == "total"


def test_ram_shortfall_reports_ram_shortfall_only() -> None:
    job_filter = _windows_measured_filter(31, 32)
    decision = VideoJobReject(
        seq=1000.0, full_gib=20.0, stream_gib=14.0,
        reason="ram_shortfall", required_ram_bytes=40 * (1024**3),
    )
    assert reject_over_gib(decision, job_filter) is None
    assert reject_over_ram_gb(decision, job_filter) == 8
    assert reject_risk_tier(decision.reason) == "over"


def test_unsupported_has_no_risk_tier() -> None:
    job_filter = _windows_measured_filter(31, 48)
    decision = VideoJobReject(seq=0.0, full_gib=0.0, stream_gib=0.0, reason="unsupported")
    assert reject_risk_tier(decision.reason) is None
    assert risk_for_reject(decision, job_filter) is None


def test_mps_fence_has_no_risk_tier() -> None:
    # The bug is Brave tiering the MPS sequence fence (an uncatchable
    # SIGABRT, not a bet) as over-budget and allowing a fenced job.
    job_filter = _windows_measured_filter(31, 48)
    decision = VideoJobReject(
        seq=257_040.0, full_gib=80.7, stream_gib=89.6, reason="mps_sequence_over"
    )
    assert reject_risk_tier(decision.reason) is None
    assert risk_for_reject(decision, job_filter) is None


# MPS transformer sequence fence (Darwin-only): past 116,000 tokens in one
# forward the Metal kernel aborts the process (SIGABRT, uncatchable), so the
# gate refuses with a fixed-copy verdict instead of a pool comparison.


def test_mac_fast_over_mps_fence_refuses_with_fixed_copy() -> None:
    """A 257k-token job fits the 128 GB Mac pool but SIGABRTs the Metal kernel — it must refuse."""
    job_filter = _job_filter(
        process_mode="streaming_models_loading", **_mac_profile(pool_gib=107.52)
    )
    decision = _one_pass_priced(
        1920, 1088, compute_num_frames(20, 50), job_filter=job_filter
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "mps_sequence_over"
    message, code = reject_message_and_code(decision, job_filter)
    assert (message, code) == (MPS_SEQUENCE_OVER_MESSAGE, VIDEO_JOB_TOO_LARGE)


def test_mac_fast_at_mps_fence_boundary_loads() -> None:
    """Off-by-one guard: exactly 116,000 tokens loads, one frame past it refuses."""
    assert video_tokens(1280, 1600, 457) == 116_000
    load = _one_pass_priced(
        1600,
        1280,
        457,
        job_filter=_job_filter(
            process_mode="streaming_models_loading", **_mac_profile(pool_gib=107.52)
        ),
    )
    assert isinstance(load, VideoJobLoad)
    refuse = _one_pass_priced(
        1600,
        1280,
        458,
        job_filter=_job_filter(
            process_mode="streaming_models_loading", **_mac_profile(pool_gib=107.52)
        ),
    )
    assert isinstance(refuse, VideoJobReject)
    assert refuse.reason == "mps_sequence_over"


def test_mac_fence_binds_before_pool_shortfall() -> None:
    """A job past both the fence and the pool reports the fence — pool GB figures would mislead."""
    decision = _one_pass_job(
        1920, 1088, 20, 50,
        process_mode="streaming_models_loading",
        **_mac_profile(pool_gib=24.0),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "mps_sequence_over"


def test_mac_unsupported_wins_over_fence() -> None:
    """Order guard: an unsupported machine reports unsupported, not the fence."""
    decision = _video_job(
        1920, 1088, 20, 50,
        process_mode="unsupported",
        **_mac_profile(pool_gib=107.52),
    )
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "unsupported"


def test_mac_edit_window_trims_to_fence_on_huge_pool() -> None:
    """A 1080p edit window the pool allows (505f) still SIGABRTs past the fence — trim to 447f."""
    job_filter = _job_filter(
        process_mode="streaming_models_loading", **_mac_profile(pool_gib=107.52)
    )
    assert video_tokens(1088, 1920, 505) > 116_000
    assert max_edit_frames_that_load(1920, 1088, frame_cap=505, job_filter=job_filter) == 447


def test_iclora_dropped_message_for_mps_fence_is_fixed_copy() -> None:
    """The IC-LoRA list-time router must carry the shared fence verdict, not crash on it."""
    job_filter = _job_filter(
        process_mode="streaming_models_loading", **_mac_profile(pool_gib=107.52)
    )
    decision = VideoJobReject(
        seq=257_040.0, full_gib=80.7, stream_gib=89.6, reason="mps_sequence_over"
    )
    message, code = dropped_duration_message_and_code(decision, job_filter)
    assert (message, code) == (MPS_SEQUENCE_OVER_MESSAGE, VIDEO_JOB_TOO_LARGE)


def test_total_only_uses_startup_usable_vram_when_lower_than_total() -> None:
    """Picker/queue total_only must gate against startup usable VRAM so an OS/display reserve is not offered and then refused at execute."""
    budget = MemoryBudget(
        total_vram_bytes=24 * _GIB,
        usable_vram_bytes=int(22.4 * _GIB),
        free_vram_bytes=int(18.0 * _GIB),
        total_ram_bytes=64 * _GIB,
        memory_gb=24.0,
        mps_pool_bytes=None,
        total_only=True,
    )
    assert budget.total_vram_gib == pytest.approx(24.0)
    assert budget.usable_vram_gib == pytest.approx(22.4)
    assert budget.available_vram_gib == pytest.approx(22.4)
    profile = HardwareProfile(
        system="Windows",
        model_id="ltx-2.5-22b-distilled",
        process_mode="streaming_models_loading",
    )
    cautious_filter = LocalJobFilter(profile=profile, budget=budget, budget_mode="cautious")
    # 1080p / 10 s at 25 fps (249 frames) asks ~23.28 GiB in one pass (> 22.4 usable),
    # so auto splits it into 2 parts (~22.14 GiB) and admits it in Cautious mode.
    admitted = decide_video_job(1920, 1088, 249, job_filter=cautious_filter)
    assert isinstance(admitted, Admitted)
    assert admitted.priced.shape.windows == 2
    # In Brave mode, the 1-pass alternative is offered with an "over" bet.
    brave_filter = replace(cautious_filter, budget_mode="brave")
    alt = price_alternative_pass(1920, 1088, 249, job_filter=brave_filter, auto=admitted.priced)
    assert alt is not None and alt.shape.windows == 1
    assert alt.over_risk is not None and alt.over_risk.tier == "over"
    # Forcing one_pass in Cautious refuses with the card-ceiling message (not "free right now").
    one_pass_filter = replace(cautious_filter, pass_shape="one_pass")
    refused = decide_video_job(1920, 1088, 249, job_filter=one_pass_filter)
    assert isinstance(refused, Refused)
    assert isinstance(refused.reject, VideoJobReject)
    assert reject_shortfall_kind(refused.reject, one_pass_filter) == "total"
    readout = budget_readout_for_reject(refused.reject, one_pass_filter)
    assert readout is not None and readout.kind == "gpu"
    assert readout.have_gb == pytest.approx(22.4)
    msg, code = reject_message_and_code(refused.reject, one_pass_filter)
    assert code == VIDEO_JOB_TOO_LARGE
    assert "this card has 22.4 GB" in msg
    assert "free right now" not in msg
    # At execute time (total_only=False) on an idle card (free == usable == 22.4 GiB),
    # a 23.3 GiB reject is still classified as "total", not "free" / "gpu_busy".
    exec_filter = replace(
        one_pass_filter,
        budget=replace(budget, total_only=False, free_vram_bytes=int(22.4 * _GIB)),
    )
    assert reject_shortfall_kind(refused.reject, exec_filter) == "total"
    exec_readout = budget_readout_for_reject(refused.reject, exec_filter)
    assert exec_readout is not None and exec_readout.kind == "gpu"


