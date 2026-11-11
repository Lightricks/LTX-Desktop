"""ETA v1 table pins: exact cells, cold column, honest Nones."""

from __future__ import annotations

import pytest

from frame_math import compute_num_frames
from runtime_config.budget_evidence import JobShape
from runtime_config.device_calibration import envelope_for
from runtime_config.eta import (
    WallTimeBreakdown,
    WallTimeEstimate,
    estimate_envelope_wall_time,
    estimate_fast_wall_time_s,
    estimate_stage_breakdown_s,
    implied_envelope_factor,
    implied_machine_factor,
    is_measured_hardware_class,
    remaining_wall_time_s,
    spatial_tile_count,
)
from runtime_config.ic_lora_tiling import IcLoraTiling

# (width, height, seconds, full_cold, stream_cpu_cold)
_CELLS = [
    (1024, 576, 5, 121, 158),
    (1024, 576, 20, 206, 222),
    (1280, 704, 20, None, 280),
    (1920, 1088, 5, 215, 232),
    (1920, 1088, 10, None, 338),
]


@pytest.mark.parametrize(("width", "height", "seconds", "full", "cpu"), _CELLS)
def test_table_pins_cold_seconds(
    width: int, height: int, seconds: int, full: int | None, cpu: int
) -> None:
    frames = compute_num_frames(seconds, 24)
    stream = estimate_fast_wall_time_s(
        width=width,
        height=height,
        frames=frames,
        load_mode="streaming_models_loading",
        offload_mode="CPU",
        feature="t2v",
        machine_factor=1.0,
    )
    assert stream == WallTimeEstimate(seconds=cpu, basis="measured", t2v_reuse=False)
    full_est = estimate_fast_wall_time_s(
        width=width,
        height=height,
        frames=frames,
        load_mode="full_models_loading",
        offload_mode="NONE",
        feature="t2v",
        machine_factor=1.0,
    )
    if full is None:
        assert full_est is None
    else:
        assert full_est == WallTimeEstimate(
            seconds=full, basis="measured", t2v_reuse=False
        )


@pytest.mark.parametrize(
    ("width", "height", "seconds", "expected", "basis"),
    [
        # 1.6.0 W1 measured ratios: 1.06 (720p/20s), 1.10 (1080p/10s).
        (1280, 704, 20, round(280 * 1.06), "measured"),
        (1920, 1088, 10, round(338 * 1.10), "measured"),
        # Default 1.07 ratio elsewhere, labelled scaled.
        (1024, 576, 5, round(158 * 1.07), "scaled"),
        (1024, 576, 20, round(222 * 1.07), "scaled"),
        (1920, 1088, 5, round(232 * 1.07), "scaled"),
    ],
)
def test_disk_applies_ratio_with_basis(
    width: int, height: int, seconds: int, expected: int, basis: str
) -> None:
    est = estimate_fast_wall_time_s(
        width=width,
        height=height,
        frames=compute_num_frames(seconds, 24),
        load_mode="streaming_models_loading",
        offload_mode="DISK",
        feature="t2v",
        machine_factor=1.0,
    )
    assert est is not None
    assert est.seconds == expected
    assert est.basis == basis


def test_unlisted_cell_returns_scaled_three_part_estimate() -> None:
    """720p/10s is computed by the 3-part formula with load + denoise + decode == seconds."""
    est = estimate_fast_wall_time_s(
        width=1280,
        height=704,
        frames=compute_num_frames(10, 24),
        load_mode="streaming_models_loading",
        offload_mode="CPU",
        feature="t2v",
        machine_factor=1.0,
    )
    assert est is not None
    assert est.basis == "scaled"
    assert est.load_s + est.denoise_s + est.decode_s == est.seconds
    assert est.load_s > 0 and est.denoise_s > 0 and est.decode_s > 0


def test_full_mode_chunked_canvas_returns_none() -> None:
    assert (
        estimate_fast_wall_time_s(
            width=3840,
            height=2176,
            frames=compute_num_frames(5, 24),
            load_mode="full_models_loading",
            offload_mode="NONE",
            feature="t2v",
            machine_factor=1.0,
        )
        is None
    )


def test_mac_machine_factor_scales_gpu_terms_without_scaling_fixed_load() -> None:
    m5 = estimate_stage_breakdown_s(
        width=512,
        height=288,
        frames=compute_num_frames(4, 24),
        load_mode="streaming_models_loading",
        offload_mode="DISK",
        system="Darwin",
        machine_factor=1.0,
        stage_2=False,
    )
    m3 = estimate_stage_breakdown_s(
        width=512,
        height=288,
        frames=compute_num_frames(4, 24),
        load_mode="streaming_models_loading",
        offload_mode="DISK",
        system="Darwin",
        machine_factor=2.0,
        stage_2=False,
    )
    assert m5 is not None and m3 is not None
    assert m3.load_s == m5.load_s == 4
    assert m3.seconds < 2 * m5.seconds


def test_tiled_and_windowed_breakdown_scales_denoise_by_tiles_and_windows() -> None:
    untiled = estimate_stage_breakdown_s(
        width=1920,
        height=1088,
        frames=97,
        load_mode="streaming_models_loading",
        offload_mode="DISK",
        stage_2=False,
        ic_lora=True,
    )
    restore_tiled = estimate_stage_breakdown_s(
        width=1920,
        height=1088,
        frames=97,
        load_mode="streaming_models_loading",
        offload_mode="DISK",
        stage_2=False,
        ic_lora=True,
        tiling=IcLoraTiling(long_side=960, short_side=544),
    )
    windowed = estimate_stage_breakdown_s(
        width=1920,
        height=1088,
        frames=241,
        load_mode="streaming_models_loading",
        offload_mode="DISK",
        stage_2=False,
        ic_lora=True,
        chunk_pixel_frames=97,
        carry_frames=25,
        tiling=IcLoraTiling(long_side=960, short_side=544),
    )
    assert untiled is not None and restore_tiled is not None and windowed is not None
    assert restore_tiled.denoise_s > untiled.denoise_s * 3
    assert restore_tiled.decode_s == untiled.decode_s
    # Three windows. Each window's denoise time is rounded to whole seconds.
    assert abs(windowed.denoise_s - restore_tiled.denoise_s * 3) <= 3


def test_remaining_wall_time_counts_down_through_decode_phase() -> None:
    est = WallTimeEstimate(
        seconds=200, basis="scaled", t2v_reuse=False, load_s=20, denoise_s=120, decode_s=60
    )
    mid_denoise = remaining_wall_time_s(
        est, elapsed_s=80.0, current_step=4, total_steps=8
    )
    decode_start = remaining_wall_time_s(
        est, elapsed_s=140.0, current_step=8, total_steps=8, decode_elapsed_s=0.0
    )
    decode_half = remaining_wall_time_s(
        est, elapsed_s=170.0, current_step=8, total_steps=8, decode_elapsed_s=30.0
    )
    assert mid_denoise == 120
    assert decode_start == 60
    assert decode_half == 30


def test_remaining_wall_time_reaches_zero_when_overdue() -> None:
    est = WallTimeEstimate(
        seconds=100, basis="scaled", t2v_reuse=False, load_s=10, denoise_s=60, decode_s=30
    )
    assert remaining_wall_time_s(est, elapsed_s=120.0, current_step=0, total_steps=8) == 0
    assert (
        remaining_wall_time_s(
            est, elapsed_s=120.0, current_step=8, total_steps=8, decode_elapsed_s=45.0
        )
        == 0
    )


def test_incoherent_mode_combos_return_none() -> None:
    frames = compute_num_frames(5, 24)
    # Full-resident never offloads; streamed is never NONE.
    assert (
        estimate_fast_wall_time_s(
            width=1024,
            height=576,
            frames=frames,
            load_mode="full_models_loading",
            offload_mode="CPU",
            feature="t2v",
            machine_factor=1.0,
        )
        is None
    )
    assert (
        estimate_fast_wall_time_s(
            width=1024,
            height=576,
            frames=frames,
            load_mode="streaming_models_loading",
            offload_mode="NONE",
            feature="t2v",
            machine_factor=1.0,
        )
        is None
    )


def test_unknown_machine_returns_none() -> None:
    est = estimate_fast_wall_time_s(
        width=1024,
        height=576,
        frames=compute_num_frames(5, 24),
        load_mode="streaming_models_loading",
        offload_mode="CPU",
        feature="t2v",
        machine_factor=None,
    )
    assert est is None


def test_calibrated_factor_uses_stage_law_not_table_cell() -> None:
    """Table cells are reference walls at factor 1.0. A calibrated factor is relative to the
    stage law, so a 2x-slower machine gets the law's estimate, not 2x the table cell."""

    def cell(factor: float) -> WallTimeEstimate | None:
        return estimate_fast_wall_time_s(
            width=1024,
            height=576,
            frames=compute_num_frames(5, 24),
            load_mode="streaming_models_loading",
            offload_mode="CPU",
            feature="t2v",
            machine_factor=factor,
        )

    ref, slow = cell(1.0), cell(2.0)
    law = estimate_stage_breakdown_s(
        width=1024,
        height=576,
        frames=compute_num_frames(5, 24),
        load_mode="streaming_models_loading",
        offload_mode="CPU",
        machine_factor=2.0,
    )
    assert ref is not None and slow is not None and law is not None
    assert (ref.seconds, ref.basis) == (158, "measured")
    assert (slow.seconds, slow.basis) == (law.seconds, "scaled")
    assert (slow.load_s, slow.denoise_s, slow.decode_s) == (law.load_s, law.denoise_s, law.decode_s)


def _disk_breakdown(
    width: int, height: int, seconds: int, factor: float, *, stage_2: bool = True
) -> WallTimeBreakdown | None:
    return estimate_stage_breakdown_s(
        width=width,
        height=height,
        frames=compute_num_frames(seconds, 24),
        load_mode="streaming_models_loading",
        offload_mode="DISK",
        machine_factor=factor,
        stage_2=stage_2,
    )


def test_cuda_streaming_factor_scales_weight_streaming_only() -> None:
    """A RAM-starved DISK machine pays per-step streaming; compute, load, and decode stay."""
    ref_270p = _disk_breakdown(512, 288, 3, 1.0, stage_2=False)
    slow_270p = _disk_breakdown(512, 288, 3, 3.6, stage_2=False)
    slow_1440p = _disk_breakdown(2560, 1472, 5, 3.6)
    assert ref_270p is not None and slow_270p is not None and slow_1440p is not None
    assert (slow_270p.load_s, slow_270p.decode_s) == (ref_270p.load_s, ref_270p.decode_s)
    # Measured on an RTX 4090 / 32 GiB RAM / DISK offload: 270p 3s took 229-238 s in all, and
    # 1440p 5s denoised in 290 s, the same as 540p, because compute hides under streaming.
    assert 215 <= slow_270p.seconds <= 245
    assert 280 <= slow_1440p.denoise_s <= 305


def test_full_load_factor_scales_gpu_terms_only() -> None:
    def full(factor: float) -> WallTimeBreakdown | None:
        return estimate_stage_breakdown_s(
            width=1024,
            height=576,
            frames=compute_num_frames(5, 24),
            load_mode="full_models_loading",
            offload_mode="NONE",
            machine_factor=factor,
        )

    ref, slow = full(1.0), full(2.0)
    assert ref is not None and slow is not None
    assert slow.load_s == ref.load_s
    assert abs(slow.decode_s - 2 * ref.decode_s) <= 1
    assert ref.denoise_s < slow.denoise_s < 2 * ref.denoise_s


def test_implied_machine_factor_round_trips_cuda_estimate() -> None:
    est = estimate_stage_breakdown_s(
        width=1024,
        height=576,
        frames=compute_num_frames(5, 24),
        load_mode="streaming_models_loading",
        offload_mode="DISK",
        machine_factor=3.0,
    )
    assert est is not None
    factor = implied_machine_factor(
        wall_s=float(est.seconds), width=1024, height=576, frames=compute_num_frames(5, 24)
    )
    assert factor is not None and abs(factor - 3.0) < 0.05


@pytest.mark.parametrize("factor", [0.0, -1.0])
def test_non_positive_factor_returns_none(factor: float) -> None:
    est = estimate_fast_wall_time_s(
        width=1024,
        height=576,
        frames=compute_num_frames(5, 24),
        load_mode="streaming_models_loading",
        offload_mode="CPU",
        feature="t2v",
        machine_factor=factor,
    )
    assert est is None


@pytest.mark.parametrize(
    ("feature", "reuse"),
    [("t2v", False), ("i2v", True), ("a2v", True)],
)
def test_i2v_a2v_reuse_t2v_numbers_flagged(feature: str, reuse: bool) -> None:
    est = estimate_fast_wall_time_s(
        width=1024,
        height=576,
        frames=compute_num_frames(5, 24),
        load_mode="streaming_models_loading",
        offload_mode="CPU",
        feature=feature,  # type: ignore[arg-type]
        machine_factor=1.0,
    )
    assert est == WallTimeEstimate(seconds=158, basis="measured", t2v_reuse=reuse)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("NVIDIA GeForce RTX 5090", True),
        ("nvidia geforce rtx 5090", True),
        ("NVIDIA GeForce RTX 5090 Laptop GPU", False),
        ("NVIDIA GeForce RTX 4090", False),
        (None, False),
        ("", False),
    ],
)
def test_measured_hardware_class(name: str | None, expected: bool) -> None:
    assert is_measured_hardware_class(name) is expected


def test_conv_vae_decodes_at_measured_ratio_on_cuda() -> None:
    def run(conv_vae: bool) -> WallTimeBreakdown | None:
        return estimate_stage_breakdown_s(
            width=2560,
            height=1472,
            frames=compute_num_frames(5, 24),
            load_mode="streaming_models_loading",
            offload_mode="DISK",
            machine_factor=3.6,
            conv_vae=conv_vae,
        )

    diff_vae, conv = run(False), run(True)
    assert diff_vae is not None and conv is not None
    assert (conv.load_s, conv.denoise_s) == (diff_vae.load_s, diff_vae.denoise_s)
    assert abs(conv.decode_s / diff_vae.decode_s - 0.31) < 0.02
    # Measured on an RTX 4090 with ConvVAE: 1440p 5s decoded in 18.7-19.8 s.
    assert 15 <= conv.decode_s <= 35


def test_mac_ignores_conv_vae() -> None:
    def run(conv_vae: bool) -> WallTimeBreakdown | None:
        return estimate_stage_breakdown_s(
            width=1024,
            height=576,
            frames=compute_num_frames(5, 24),
            load_mode="streaming_models_loading",
            offload_mode="DISK",
            system="Darwin",
            conv_vae=conv_vae,
        )

    assert run(True) == run(False)


def test_conv_vae_swaps_table_cell_decode() -> None:
    def cell(conv_vae: bool) -> WallTimeEstimate | None:
        return estimate_fast_wall_time_s(
            width=1024,
            height=576,
            frames=compute_num_frames(5, 24),
            load_mode="streaming_models_loading",
            offload_mode="CPU",
            feature="t2v",
            machine_factor=1.0,
            conv_vae=conv_vae,
        )

    diff_vae, conv = cell(False), cell(True)
    assert diff_vae is not None and conv is not None
    assert (diff_vae.seconds, diff_vae.basis) == (158, "measured")
    assert conv.basis == "scaled"
    assert 0 < diff_vae.seconds - conv.seconds < diff_vae.decode_s


def test_implied_machine_factor_round_trips_conv_vae_estimate() -> None:
    est = estimate_stage_breakdown_s(
        width=1280,
        height=704,
        frames=compute_num_frames(5, 24),
        load_mode="streaming_models_loading",
        offload_mode="DISK",
        machine_factor=3.0,
        conv_vae=True,
    )
    assert est is not None
    factor = implied_machine_factor(
        wall_s=float(est.seconds),
        width=1280,
        height=704,
        frames=compute_num_frames(5, 24),
        conv_vae=True,
    )
    assert factor is not None and abs(factor - 3.0) < 0.05


def test_stage_1_share_weights_mac_and_cuda_streaming_countdown_differently() -> None:
    mac = estimate_fast_wall_time_s(
        width=1024,
        height=576,
        frames=compute_num_frames(5, 24),
        load_mode="streaming_models_loading",
        offload_mode="DISK",
        feature="t2v",
        machine_factor=1.0,
        system="Darwin",
    )
    cuda = estimate_fast_wall_time_s(
        width=1024,
        height=576,
        frames=compute_num_frames(5, 24),
        load_mode="streaming_models_loading",
        offload_mode="DISK",
        feature="t2v",
        machine_factor=3.6,
        system="Windows",
        conv_vae=True,
    )
    assert mac is not None and cuda is not None
    assert mac.stage_1_share is not None and cuda.stage_1_share is not None
    assert 0.35 < mac.stage_1_share < 0.55
    assert 0.68 < cuda.stage_1_share < 0.76
    rem_cuda = remaining_wall_time_s(
        cuda,
        elapsed_s=cuda.load_s + cuda.denoise_s * cuda.stage_1_share,
        current_step=8,
        total_steps=11,
    )
    true_rem = round(cuda.denoise_s * (1.0 - cuda.stage_1_share) + cuda.decode_s)
    assert rem_cuda is not None and abs(rem_cuda - true_rem) <= 2


def test_envelope_wall_time_uses_shape_tile_count() -> None:
    base = JobShape(
        feature="iclora",
        model_id="ltx-2.5-22b-distilled",
        adapter_id="refine-details",
        stage_mode="single_stage",
        tiled=True,
        chunked=False,
        seq=14_976.0,
        canvas_px=1920 * 1088,
        windows=1,
        tiles=1,
    )
    one_tile = estimate_envelope_wall_time(
        envelope_for(base, "streaming_models_loading", 14.0), machine_factor=1.0
    )
    nine_tiles = estimate_envelope_wall_time(
        envelope_for(
            JobShape(
                feature="iclora",
                model_id="ltx-2.5-22b-distilled",
                adapter_id="refine-details",
                stage_mode="single_stage",
                tiled=True,
                chunked=False,
                seq=14_976.0,
                canvas_px=3840 * 2176,
                windows=1,
                tiles=9,
            ),
            "streaming_models_loading",
            14.0,
        ),
        machine_factor=1.0,
    )
    assert one_tile is not None and nine_tiles is not None
    assert abs(nine_tiles.denoise_s - one_tile.denoise_s * 9) <= 2


def test_implausible_stored_shapes_are_rejected() -> None:
    """A tampered row must not size a huge window list."""
    shape = {
        "width": 1024,
        "height": 576,
        "load_mode": "streaming_models_loading",
        "offload_mode": "DISK",
    }
    assert estimate_stage_breakdown_s(**shape, frames=121) is not None  # type: ignore[arg-type]
    assert estimate_stage_breakdown_s(**shape, frames=100_001) is None  # type: ignore[arg-type]
    assert implied_machine_factor(wall_s=100.0, width=1024, height=576, frames=100_001) is None

    def envelope(windows: int):  # type: ignore[no-untyped-def]
        return envelope_for(
            JobShape(
                feature="fast",
                model_id="ltx-2.5-22b-distilled",
                adapter_id="",
                stage_mode="",
                tiled=False,
                chunked=False,
                seq=9216.0,
                canvas_px=1024 * 576,
                windows=windows,
            ),
            "streaming_models_loading",
            10.0,
        )

    assert estimate_envelope_wall_time(envelope(1_000), machine_factor=1.0) is not None
    assert estimate_envelope_wall_time(envelope(1_001), machine_factor=1.0) is None
    assert (
        implied_envelope_factor(
            "Windows", envelope(1_001).bucket, envelope(1_001).point, 100.0
        )
        is None
    )


def test_spatial_tile_count_follows_the_upstream_layout_and_the_run_overlap() -> None:
    """The count is upstream's ``split_by_size_pinned`` layout, and the run's overlap sets it.

    Documented layouts on 1920x1088: Refine's 1024x576 tiles lay out 2x2, Restore's 960x544 3x2.
    A lower overlap needs fewer tiles on a large canvas, down to the layout floor.
    """
    refine = IcLoraTiling(long_side=1024, short_side=576)
    assert spatial_tile_count(1920, 1088, refine) == 4
    assert spatial_tile_count(1920, 1088, IcLoraTiling(long_side=960, short_side=544)) == 6
    assert spatial_tile_count(1088, 1920, refine) == 4  # portrait follows the canvas
    assert spatial_tile_count(1920, 1088, None) == 1

    def count(overlap: float) -> int:
        tiling = IcLoraTiling(long_side=1024, short_side=576, overlap_fraction=overlap)
        return spatial_tile_count(3840, 2112, tiling)

    assert [count(o) for o in (0.0, 0.15, 0.25, 0.5)] == [16, 20, 25, 36]
