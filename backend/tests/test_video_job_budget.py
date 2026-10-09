"""Token/VRAM job budget: 5090-calibrated curve plus IC-LoRA wrapper."""

from __future__ import annotations

from api_types import (
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
from runtime_config.ic_lora_job_budget import (
    IcLoraJobLoad,
    IcLoraJobReject,
    advertised_duration_options,
    catalog_output_fps,
    catalog_preview_stage1,
    decide_ic_lora_job,
    ic_lora_sequence_tokens,
    with_budgeted_duration_controls,
)
from runtime_config.ic_lora_local_envelope import effective_denoise_stage1, ic_lora_output_canvas
from runtime_config.ic_lora_stage_mode import IcLoraStageMode
from runtime_config.video_job_budget import (
    OfferingCell,
    VideoJobLoad,
    VideoJobReject,
    advertised_fast_durations,
    enumerate_envelope,
    estimated_giB,
    fits,
    decide_edit_job,
    decide_video_job,
    max_edit_frames_that_load,
    memory_gb_for_job,
    video_tokens,
)
from services.retake_pipeline.window import (
    duration_to_extend_frames,
    window_for_extend,
    window_for_retake,
)


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
    decision = decide_ic_lora_job(
        w, h, compute_num_frames(5, 24), memory_gb=31, process_mode="full_models_loading",
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.mode == "full_models_loading"


def test_decide_31gb_8s_and_10s_540p_stream() -> None:
    w, h = _source_dim_stage1(1024, 576)
    for seconds in (8, 10):
        decision = decide_ic_lora_job(
            w, h, compute_num_frames(seconds, 24), memory_gb=31, process_mode="full_models_loading",
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
        )
        assert isinstance(decision, IcLoraJobLoad)
        assert decision.mode == "streaming_models_loading"


def test_decide_chunked_1080p_prices_one_window() -> None:
    w, h = _source_dim_stage1(1920, 1080)
    frames = compute_num_frames(8, 24)
    chunked = decide_ic_lora_job(
        w, h, frames, memory_gb=31, process_mode="full_models_loading", stage_mode=IcLoraStageMode.SINGLE_STAGE
    )
    assert isinstance(chunked, IcLoraJobLoad)
    assert chunked.mode == "streaming_models_loading"
    window = decide_ic_lora_job(
        w, h, 97, memory_gb=31, process_mode="full_models_loading", stage_mode=IcLoraStageMode.SINGLE_STAGE
    )
    assert isinstance(window, IcLoraJobLoad)
    assert chunked.seq == window.seq


def test_decide_stage_2_ic_lora_1080p_chunks_and_fits() -> None:
    decision = decide_ic_lora_job(
        960,
        544,
        compute_num_frames(9, 24),
        memory_gb=32,
        process_mode="full_models_loading",
        stage_mode=IcLoraStageMode.STAGE_2_LORA,
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_decide_1080p_single_stage_5s_on_22gb_chunks_and_loads_instead_of_rejecting() -> None:
    # Whole clip: 2 x 16 x 2025 = 64800 tokens, stream estimate 23.5 GiB, over 22.
    # One 97-frame window: 2 x 13 x 2025 = 52650 tokens, stream estimate 19.7 GiB.
    decision = decide_ic_lora_job(
        1920, 1080, compute_num_frames(5, 24), memory_gb=22, process_mode="full_models_loading",
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.mode == "streaming_models_loading"
    assert decision.chunk_pixel_frames == 97
    assert decision.seq == 52650.0


def test_decide_off_grid_123_frames_chunks_like_121_frames() -> None:
    # Both counts are 16 latent frames, so both need the same window on a 22 GB card.
    def window(frames: int) -> int | None:
        decision = decide_ic_lora_job(
            1920, 1088, frames, memory_gb=22, process_mode="full_models_loading",
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
                decide_ic_lora_job(
                    width, height, compute_num_frames(seconds, 24), memory_gb=memory_gb,
                    process_mode="full_models_loading", stage_mode=mode,
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
        priced = decide_ic_lora_job(
            width, height, frames, memory_gb=31, process_mode="full_models_loading", stage_mode=mode
        )
        window = decide_ic_lora_job(
            width, height, 97, memory_gb=31, process_mode="full_models_loading", stage_mode=mode
        )
        assert priced.seq > window.seq


def test_decide_31gb_20s_540p_streams() -> None:
    w, h = _source_dim_stage1(1024, 576)
    decision = decide_ic_lora_job(
        w, h, compute_num_frames(20, 24), memory_gb=31, process_mode="full_models_loading",
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_decide_31gb_5s_2160p_spatial_reject() -> None:
    w, h = _source_dim_stage1(3840, 2160)
    decision = decide_ic_lora_job(
        w, h, compute_num_frames(5, 24), memory_gb=31, process_mode="full_models_loading",
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
    )
    assert isinstance(decision, IcLoraJobReject)
    assert decision.reason == "spatial"


def test_decide_31gb_5s_2160p_two_stage_stage_1_lora_is_a_spatial_reject() -> None:
    # A 3840x2160 source gives a 3840x2048 canvas, so stage 1 is 1920x1024. That is inside
    # the 1920x1088 cell, but stage 2 runs at 3840x2048 and is over it.
    decision = decide_ic_lora_job(
        1920, 1024, compute_num_frames(5, 24), memory_gb=32, process_mode="full_models_loading",
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
    )
    assert isinstance(decision, IcLoraJobReject)
    assert decision.reason == "spatial"


def test_stage_1_lora_prices_the_stage_2_canvas_when_it_is_heavier_than_stage_1() -> None:
    # Stage 1 960x544, 121 frames: 16 latent frames of 17x30 cells, times 2 for the
    # reference = 16320 tokens. Stage 2 at 1920x1088 has no reference: 16 x 34 x 60 = 32640.
    decision = decide_ic_lora_job(
        960, 544, 121, memory_gb=32, process_mode="full_models_loading",
        stage_mode=IcLoraStageMode.STAGE_1_LORA,
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.seq == 32640.0


def test_a_reference_still_adds_one_latent_frame_of_tokens_to_the_stage_2_lora_price() -> None:
    # Stage 1 640x352 gives a 1280x704 canvas: 22 x 40 = 880 cells per latent frame.
    # 121 frames are 16 latent frames, doubled for the reference: 2 x 16 x 880 = 28160.
    # One still adds one latent frame of the canvas: 880.
    def price(stills: int) -> float:
        decision = decide_ic_lora_job(
            640, 352, 121, memory_gb=32, process_mode="full_models_loading",
            stage_mode=IcLoraStageMode.STAGE_2_LORA, stills=stills,
        )
        assert isinstance(decision, IcLoraJobLoad)
        return decision.seq

    assert price(0) == 28160.0
    assert price(1) == 29040.0


def test_decide_ic_lora_unsupported_beats_spatial() -> None:
    w, h = _source_dim_stage1(1280, 720)
    decision = decide_ic_lora_job(
        w, h, compute_num_frames(5, 24), memory_gb=31, process_mode="unsupported",
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
    )
    assert isinstance(decision, IcLoraJobReject)
    assert decision.reason == "unsupported"


def test_decide_ic_lora_unsupported_process_rejects() -> None:
    w, h = _source_dim_stage1(1024, 576)
    decision = decide_ic_lora_job(
        w, h, compute_num_frames(5, 24), memory_gb=31, process_mode="unsupported",
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
    )
    assert isinstance(decision, IcLoraJobReject)
    assert decision.reason == "unsupported"


def test_decide_24gb_stream_process_never_full() -> None:
    w, h = _source_dim_stage1(1024, 576)
    five = decide_ic_lora_job(
        w,
        h,
        compute_num_frames(5, 24),
        memory_gb=24,
        process_mode="streaming_models_loading",
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
    )
    twenty = decide_ic_lora_job(
        w,
        h,
        compute_num_frames(20, 24),
        memory_gb=24,
        process_mode="streaming_models_loading",
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
    )
    assert isinstance(five, IcLoraJobLoad)
    assert five.mode == "streaming_models_loading"
    assert isinstance(twenty, IcLoraJobReject)
    assert twenty.reason == "stream_over"


def test_decide_darwin_never_full() -> None:
    w, h = _source_dim_stage1(1024, 576)
    decision = decide_ic_lora_job(
        w,
        h,
        compute_num_frames(5, 24),
        memory_gb=48,
        process_mode="full_models_loading",
        darwin=True,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_decide_darwin_does_not_apply_cuda_stream_over() -> None:
    """Darwin skips the CUDA stream-over 422; free RAM is not 5090 reserved GiB."""
    w, h = _source_dim_stage1(1024, 576)
    decision = decide_ic_lora_job(
        w,
        h,
        compute_num_frames(20, 24),
        memory_gb=20,
        process_mode="full_models_loading",
        darwin=True,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
    )
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_memory_gb_for_job_uses_available_ram_on_darwin() -> None:
    assert memory_gb_for_job(vram_gb=31, available_ram_gb=48, darwin=True) == 48
    assert memory_gb_for_job(vram_gb=31, available_ram_gb=48, darwin=False) == 31


def test_enumerate_envelope_drops_measured_no_and_over_budget() -> None:
    grid = [
        OfferingCell("540p", 5, 24, 1024, 576),
        OfferingCell("1080p", 10, 24, 1920, 1088),
        OfferingCell("720p", 5, 24, 1280, 704),
    ]
    kept = enumerate_envelope(
        grid,
        "full_models_loading",
        31,
        measured_no=lambda c: c.resolution == "720p",
    )
    assert [(c.resolution, c.duration_seconds) for c in kept] == [("540p", 5)]


def test_ingredients_preview_stage1_is_576x960() -> None:
    w, h = catalog_preview_stage1(IcLoraSettings(skip_stage_2=True, resolution_factor=1.5))
    assert sorted((w, h)) == [576, 960]


def test_advertised_duration_drops_20s_on_10gb() -> None:
    w, h = catalog_preview_stage1(IcLoraSettings(skip_stage_2=True, resolution_factor=1.5))
    kept = advertised_duration_options(
        [5, 6, 8, 10, 20],
        stage1_width=w,
        stage1_height=h,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        fps=24,
        memory_gb=10,
        process_mode="full_models_loading",
        darwin=False,
    )
    assert 20 not in kept
    assert 5 in kept


def test_advertised_duration_keeps_20s_on_darwin() -> None:
    w, h = catalog_preview_stage1(IcLoraSettings(skip_stage_2=True, resolution_factor=1.5))
    kept = advertised_duration_options(
        [5, 6, 8, 10, 20],
        stage1_width=w,
        stage1_height=h,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        fps=24,
        memory_gb=20,
        process_mode="full_models_loading",
        darwin=True,
    )
    assert kept == [5, 6, 8, 10, 20]


def test_full_estimate_is_above_stream() -> None:
    seq = ic_lora_sequence_tokens(1024, 576, compute_num_frames(5, 24))
    assert estimated_giB("full_models_loading", seq) > estimated_giB("streaming_models_loading", seq)


def _video_job(width: int, height: int, seconds: int, **kwargs):
    return decide_video_job(
        width,
        height,
        compute_num_frames(seconds, 24),
        **kwargs,
    )


def test_decide_video_job_31gb_540p_5s_full() -> None:
    decision = _video_job(1024, 576, 5, memory_gb=31, process_mode="full_models_loading")
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "full_models_loading"


def test_decide_video_job_32gb_720p_20s_streams() -> None:
    decision = _video_job(1280, 704, 20, memory_gb=32, process_mode="full_models_loading")
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_decide_video_job_16gb_720p_20s_rejects() -> None:
    decision = _video_job(1280, 704, 20, memory_gb=16, process_mode="streaming_models_loading")
    assert isinstance(decision, VideoJobReject)


def test_decide_video_job_15gb_540p_5s_still_streams() -> None:
    decision = _video_job(1024, 576, 5, memory_gb=15, process_mode="streaming_models_loading")
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "streaming_models_loading"


def test_decide_edit_job_31gb_720p_extend_streams() -> None:
    """5090 extend 1280×704×337 estimated 27.2 GiB and reserved the whole card."""
    decision = decide_edit_job(
        1280,
        704,
        337,
        memory_gb=31,
        process_mode="full_models_loading",
    )
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "streaming_models_loading"
    reported_as_32 = decide_edit_job(
        1280, 704, 337, memory_gb=32, process_mode="full_models_loading"
    )
    assert isinstance(reported_as_32, VideoJobLoad)
    assert reported_as_32.mode == "streaming_models_loading"


def test_max_edit_frames_shrinks_1080p_on_31gb_and_keeps_720p() -> None:
    """1920×1088×505 streams past the 31 GB budget. The cap must load; 720p×505 already does."""
    kwargs = dict(memory_gb=31, process_mode="full_models_loading")
    cap = max_edit_frames_that_load(1920, 1088, frame_cap=505, **kwargs)
    assert cap < 505
    assert isinstance(decide_edit_job(1920, 1088, 505, **kwargs), VideoJobReject)
    assert isinstance(decide_edit_job(1920, 1088, cap, **kwargs), VideoJobLoad)
    assert isinstance(decide_edit_job(1920, 1088, cap + 1, **kwargs), VideoJobReject)
    assert (
        max_edit_frames_that_load(1280, 736, frame_cap=505, **kwargs) == 505
    )
    assert (
        max_edit_frames_that_load(
            1920, 1088, frame_cap=505, darwin=True, **kwargs
        )
        == 505
    )

    extend_frames = duration_to_extend_frames(4, 25)
    window = window_for_extend(
        source_frames=1000,
        extend_frames=extend_frames,
        mode="end",
        max_input_frames=cap,
    )
    assert isinstance(
        decide_edit_job(
            1920, 1088, window.context_frames + extend_frames, **kwargs
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
        decide_edit_job(1920, 1088, retake.encode_frames, **kwargs),
        VideoJobLoad,
    )


def test_decide_edit_job_31gb_short_720p_stays_full() -> None:
    decision = decide_edit_job(
        1280,
        704,
        compute_num_frames(5, 24),
        memory_gb=31,
        process_mode="full_models_loading",
    )
    assert isinstance(decision, VideoJobLoad)
    assert decision.mode == "full_models_loading"


def test_fast_budget_still_full_loads_the_720p_extend_clip() -> None:
    decision = decide_video_job(
        1280, 704, 337, memory_gb=31, process_mode="full_models_loading"
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
        darwin=True,
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
            darwin=True,
        )
        assert isinstance(decision, VideoJobLoad)
        assert decision.mode == "streaming_models_loading"


def test_decide_video_job_stream_over_rejects() -> None:
    decision = _video_job(1280, 704, 20, memory_gb=8, process_mode="full_models_loading")
    assert isinstance(decision, VideoJobReject)
    assert decision.reason == "stream_over"


def test_decide_video_job_16gb_1080p_10s_rejects() -> None:
    decision = _video_job(1920, 1088, 10, memory_gb=16, process_mode="streaming_models_loading")
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
        **kwargs,
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
    kwargs = dict(memory_gb=20, process_mode="streaming_models_loading", darwin=True)
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
        item, memory_gb=31, process_mode="unsupported", darwin=False
    )
    duration = next(control for control in out.controls if control.id == "duration")
    assert duration.options == []


def test_catalog_output_fps_coerces_junk_to_default() -> None:
    assert catalog_output_fps(_catalog_item_with_fps(24)) == 24
    assert catalog_output_fps(_catalog_item_with_fps("12")) == 12
    assert catalog_output_fps(_catalog_item_with_fps(None)) == 24
    assert catalog_output_fps(_catalog_item_with_fps({})) == 24
    assert catalog_output_fps(_catalog_item_with_fps("nope")) == 24
    assert catalog_output_fps(_catalog_item_with_fps(True)) == 24
    assert catalog_output_fps(_catalog_item_with_fps(0)) == 24
