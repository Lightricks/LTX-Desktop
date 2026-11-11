"""Tile plan and Windows budget of the 4K IC-LoRA cell (any catalog LoRA, tiled)."""

from __future__ import annotations

from runtime_config.budget_admission import Admitted, RiskAllowance
from runtime_config.budget_decision import IcLoraJobLoad, IcLoraJobReject, VideoJobReject
from runtime_config.ic_lora_high_res import high_res_tiling
from runtime_config.ic_lora_job_budget import decide_ic_lora_job
from runtime_config.ic_lora_stage_mode import IcLoraStageMode
from runtime_config.ic_lora_tiling import IcLoraTiling
from runtime_config.measured_cells import MEASURED_HIGH_RES_CELLS
from tests.test_video_job_budget import _windows_measured_filter

_4K = (3840, 2176)
_1440P = (2560, 1472)
_FRAMES_10S = 233


def _verdict(admission):
    return admission.load if isinstance(admission, Admitted) else admission.reject


def _decide_4k(vram_gib: float, tiling: IcLoraTiling):
    return _verdict(
        decide_ic_lora_job(
            *_4K,
            _FRAMES_10S,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(vram_gib, 64),
            tiling=tiling,
            allowance=RiskAllowance.none(),
        )
    )


def test_lora_without_a_tile_size_runs_the_local_cap_not_a_small_tile() -> None:
    # A 1024x576 tile needed 6x6 tiles at 4K and ran 210 s per step on the RTX 5090.
    tiling = high_res_tiling(None)
    assert (tiling.long_side, tiling.short_side) == (1920, 1088)


def test_lora_with_its_own_tile_size_keeps_it_in_the_high_cell() -> None:
    own = IcLoraTiling(long_side=960, short_side=544)
    tiling = high_res_tiling(own)
    assert (tiling.long_side, tiling.short_side) == (960, 544)


def test_4k_ten_second_job_loads_on_a_32gb_card() -> None:
    # The 25% margin on the canvas term priced this job at 32.1 GiB and refused it.
    decision = _decide_4k(31.84, high_res_tiling(None))
    assert isinstance(decision, IcLoraJobLoad)


def test_4k_ten_second_job_is_refused_on_a_card_below_its_ask() -> None:
    decision = _decide_4k(28, high_res_tiling(None))
    # The decode of the job's last window no longer fits, so the decode check refuses it before
    # the stream curve does.
    assert isinstance(decision, (VideoJobReject, IcLoraJobReject))


def _decide_4k_frames(frames: int):
    return _verdict(
        decide_ic_lora_job(
            *_4K,
            frames,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(64, 64),
            tiling=high_res_tiling(None),
            allowance=RiskAllowance.none(),
        )
    )


def test_iclora_4k_row_loads_at_10_windows_and_refuses_at_11() -> None:
    # 257 frames is 10 windows of 41; 281 is 11. Only the IC-LoRA key moves 9 to 10: the
    # Fast 4K row is 15 and the IC-LoRA one must not follow it.
    assert MEASURED_HIGH_RES_CELLS[("iclora", "windows_measured", "2160p")] == 10
    assert isinstance(_decide_4k_frames(257), IcLoraJobLoad)
    past_row = _decide_4k_frames(281)
    assert isinstance(past_row, VideoJobReject)
    assert past_row.reason == "unmeasured_cell"


def _decide_1440p_frames(frames: int):
    return _verdict(
        decide_ic_lora_job(
            *_1440P,
            frames,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(64, 64),
            tiling=high_res_tiling(None),
            allowance=RiskAllowance.none(),
        )
    )


def test_iclora_1440p_row_loads_at_5_windows_and_refuses_at_6() -> None:
    # 137 frames is 5 windows of 41 (measured); 161 is 6. The 1440p row must not inherit
    # the 4K row's 10 windows or the Fast 1440p row's 9.
    assert MEASURED_HIGH_RES_CELLS[("iclora", "windows_measured", "1440p")] == 5
    assert isinstance(_decide_1440p_frames(137), IcLoraJobLoad)
    past_row = _decide_1440p_frames(161)
    assert isinstance(past_row, VideoJobReject)
    assert past_row.reason == "unmeasured_cell"


_4K_REFINE = (3840, 2112)
_REFINE_TILING = high_res_tiling(IcLoraTiling(long_side=1024, short_side=576, window_frames=73))


def _decide_refine_4k(vram_gib: float, frames: int):
    return _verdict(
        decide_ic_lora_job(
            *_4K_REFINE,
            frames,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(vram_gib, 64),
            tiling=_REFINE_TILING,
            allowance=RiskAllowance.none(),
        )
    )


def test_refine_4k_137_frames_runs_49_frame_windows_on_a_32gb_card() -> None:
    # The decode budget read 27.8, 22.7 and 20.9 GiB on windows 1 to 3 of this job at 73 frames.
    # At 20.9 GiB a 73-frame decode does not fit, so a third full window would fail after denoise.
    decision = _decide_refine_4k(31.84, 137)
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.chunk_pixel_frames == 49


def test_refine_4k_one_window_clip_keeps_its_single_pass_on_a_32gb_card() -> None:
    # Window 1 reads 27.8 GiB and a 73-frame decode needs 22.3, so the clip runs as it did.
    decision = _decide_refine_4k(31.84, 73)
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.chunk_pixel_frames is None


def test_refine_4k_is_refused_before_denoise_when_no_window_decodes() -> None:
    decision = _decide_refine_4k(24, 137)
    assert isinstance(decision, IcLoraJobReject)
    assert decision.reason == "refine_decode_floor"


def test_refine_4k_keeps_its_catalog_window_on_a_card_with_room() -> None:
    decision = _decide_refine_4k(48, 137)
    assert isinstance(decision, IcLoraJobLoad)
    assert decision.chunk_pixel_frames == 73


def test_1440p_tiled_high_res_loads_on_16gb_cards_for_41f_and_stepped_down_refine() -> None:
    """C2 (9.79 GiB @ 16g cap) and C3/N2 (11.67 GiB @ 16g cap) proved 1440p 41f
    fits 16 GB cards (ask 11.50 GiB at 2w, 13.36 GiB at 3+w); flat 1.75*Mpx
    priced 1440p Water Simulation at 20.32 GiB and refused 16 GB cards."""
    water_2w = _verdict(
        decide_ic_lora_job(
            *_1440P,
            57,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(15.9, 64, free_gib=15.0),
            tiling=high_res_tiling(None),
            allowance=RiskAllowance.none(),
        )
    )
    assert isinstance(water_2w, IcLoraJobLoad)
    assert water_2w.chunk_pixel_frames == 41

    water_5w = _verdict(
        decide_ic_lora_job(
            *_1440P,
            137,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(15.9, 64, free_gib=15.0),
            tiling=high_res_tiling(None),
            allowance=RiskAllowance.none(),
        )
    )
    assert isinstance(water_5w, IcLoraJobLoad)
    assert water_5w.chunk_pixel_frames == 41

    # On a 16 GB card, 1440p Refine Details steps 73f down to 41f and fits at 13.36 GiB.
    refine_16g = _verdict(
        decide_ic_lora_job(
            *_1440P,
            113,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(15.9, 64, free_gib=15.0),
            ic_lora_id="refine-details",
            tiling=_REFINE_TILING,
            allowance=RiskAllowance.none(),
        )
    )
    assert isinstance(refine_16g, IcLoraJobLoad)
    assert refine_16g.chunk_pixel_frames == 41


def test_long_window_tiled_refine_1440p_and_4k_never_under_ask_measured_peaks() -> None:
    """T2 (1440p 241f 4w x 73f) peaked at 17.10 GiB and T3 (4K 137f 4w x 49f)
    peaked at 25.51 GiB: the old 9.90 + 1.75*Mpx curve under-asked T2 at 16.49 GiB
    and T3 at 23.50 GiB."""
    t2_short = _verdict(
        decide_ic_lora_job(
            *_1440P,
            241,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(24, 64, free_gib=17.10),
            ic_lora_id="refine-details",
            tiling=_REFINE_TILING,
            allowance=RiskAllowance.none(),
        )
    )
    assert isinstance(t2_short, VideoJobReject)
    assert t2_short.reason == "stream_over"

    t2_fits = _verdict(
        decide_ic_lora_job(
            *_1440P,
            241,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(24, 64, free_gib=18.80),
            ic_lora_id="refine-details",
            tiling=_REFINE_TILING,
            allowance=RiskAllowance.none(),
        )
    )
    assert isinstance(t2_fits, IcLoraJobLoad)
    assert t2_fits.chunk_pixel_frames == 73

    t3_short = _verdict(
        decide_ic_lora_job(
            *_4K,
            137,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(31.84, 64, free_gib=25.51),
            tiling=_REFINE_TILING,
            allowance=RiskAllowance.none(),
        )
    )
    assert isinstance(t3_short, VideoJobReject)
    assert t3_short.reason == "stream_over"

    t3_fits = _verdict(
        decide_ic_lora_job(
            *_4K,
            137,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(31.84, 64, free_gib=26.30),
            tiling=_REFINE_TILING,
            allowance=RiskAllowance.none(),
        )
    )
    assert isinstance(t3_fits, IcLoraJobLoad)
    assert t3_fits.chunk_pixel_frames == 49


def test_4k_square_tiled_prices_on_canvas_window_tokens_and_loads_on_16gb() -> None:
    """N1 (2176x2176 x 97f, 4 windows of 41f) peaked at 14.12 GiB: pricing on
    canvas-window tokens asks 15.45 GiB, whereas 13.72 + 1.75*Mpx asked 22.00 GiB."""
    n1_short = _verdict(
        decide_ic_lora_job(
            2176,
            2176,
            97,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(24.0, 64, free_gib=15.44),
            tiling=high_res_tiling(None),
            allowance=RiskAllowance.none(),
        )
    )
    assert isinstance(n1_short, VideoJobReject)
    assert n1_short.reason == "stream_over"

    n1_fits = _verdict(
        decide_ic_lora_job(
            2176,
            2176,
            97,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(24.0, 64, free_gib=15.46),
            tiling=high_res_tiling(None),
            allowance=RiskAllowance.none(),
        )
    )
    assert isinstance(n1_fits, IcLoraJobLoad)
    assert n1_fits.chunk_pixel_frames == 41


def test_short_second_window_restore_1440p_binds_on_window_1_decode() -> None:
    """T4 (Restore 1440p 137f: w1=97f, w2=57f with carry 17) peaked at 15.89 GiB
    with +0.00 GiB window-2 growth: max(dec(w1), dec(w2) + c_2w*t1) binds on
    w1 (17.84 GiB) rather than charging a full 97f second window (20.95 GiB)."""
    restore_high_res = high_res_tiling(IcLoraTiling(long_side=960, short_side=544, window_frames=97))
    t4_short = _verdict(
        decide_ic_lora_job(
            *_1440P,
            137,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(31.84, 64, free_gib=17.83),
            tiling=restore_high_res,
            allowance=RiskAllowance.none(),
        )
    )
    assert isinstance(t4_short, VideoJobReject)
    assert t4_short.reason == "stream_over"

    t4_fits = _verdict(
        decide_ic_lora_job(
            *_1440P,
            137,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(31.84, 64, free_gib=17.85),
            tiling=restore_high_res,
            allowance=RiskAllowance.none(),
        )
    )
    assert isinstance(t4_fits, IcLoraJobLoad)
    assert t4_fits.chunk_pixel_frames == 97


def test_refine_details_1440p_is_measured_through_4_windows_while_4k_stays_unmeasured() -> None:
    """T2 measured 1440p refine-details cleanly through 4 windows (241f at 73f
    windows), while T3 at 4K paged (0.60 GB shared) and stays unmeasured in Cautious mode."""
    refine_1440p_4w = _verdict(
        decide_ic_lora_job(
            *_1440P,
            241,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(31.84, 64),
            ic_lora_id="refine-details",
            tiling=_REFINE_TILING,
            allowance=RiskAllowance.none(),
        )
    )
    assert isinstance(refine_1440p_4w, IcLoraJobLoad)

    refine_1440p_5w = _verdict(
        decide_ic_lora_job(
            *_1440P,
            249,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(31.84, 64),
            ic_lora_id="refine-details",
            tiling=_REFINE_TILING,
            allowance=RiskAllowance.none(),
        )
    )
    assert isinstance(refine_1440p_5w, VideoJobReject)
    assert refine_1440p_5w.reason == "unmeasured_cell"

    refine_4k = _verdict(
        decide_ic_lora_job(
            *_4K_REFINE,
            137,
            stage_mode=IcLoraStageMode.SINGLE_STAGE,
            job_filter=_windows_measured_filter(31.84, 64),
            ic_lora_id="refine-details",
            tiling=_REFINE_TILING,
            allowance=RiskAllowance.none(),
        )
    )
    assert isinstance(refine_4k, VideoJobReject)
    assert refine_4k.reason == "unmeasured_cell"

