"""Auto carry: the job budget picks the carry with the window, and never a worse run than a fixed carry."""

from __future__ import annotations

import pytest

from runtime_config.budget_admission import Admitted, PricedJob, RiskAllowance, admit, safety_rank
from runtime_config.ic_lora_auto_carry import picked_carry_tiling
from runtime_config.ic_lora_high_res import high_res_tiling
from runtime_config.ic_lora_job_budget import price_ic_lora_job
from runtime_config.ic_lora_stage_mode import IcLoraStageMode
from runtime_config.ic_lora_tiling import (
    CARRY_PRESETS,
    IcLoraTiling,
    auto_carry_candidates,
    with_tiling_overrides,
)
from tests.test_video_job_budget import _windows_measured_filter

_BASE = IcLoraTiling(long_side=1024, short_side=576, adaptive_window=True)
_CELLS: dict[str, tuple[tuple[int, int], IcLoraTiling]] = {
    "720p": ((1280, 704), _BASE),
    "1080p": ((1920, 1088), _BASE),
    "1440p": ((2560, 1440), high_res_tiling(_BASE)),
    "2160p": ((3840, 2112), high_res_tiling(_BASE)),
}
_VRAM_GIB = (8, 10, 13.25, 16, 24, 48)
_FRAMES = (97, 241)


def _price(cell: str, vram_gib: float, frames: int, tiling: IcLoraTiling | None = None) -> PricedJob:
    size, cell_tiling = _CELLS[cell]
    return price_ic_lora_job(
        *size,
        frames,
        stage_mode=IcLoraStageMode.SINGLE_STAGE,
        job_filter=_windows_measured_filter(vram_gib, 64),
        ic_lora_id="refine-details",
        stills=0,
        tiling=cell_tiling if tiling is None else tiling,
    )


def _pinned(tiling: IcLoraTiling, carry: int) -> IcLoraTiling:
    return tiling.model_copy(update={"carry_frames": carry, "carry_pinned": True})


def test_the_candidates_are_the_presets_up_to_the_carry_of_the_cell() -> None:
    assert auto_carry_candidates(25, None) == [25, 17]
    assert auto_carry_candidates(17, None) == [17]
    assert auto_carry_candidates(33, None) == [33, 25, 17]


def test_a_chunk_length_pins_the_candidates_it_can_run_with() -> None:
    # A chunk length must be more than twice the carry.
    assert auto_carry_candidates(25, 41) == [17]
    assert auto_carry_candidates(25, 57) == [25, 17]


def test_a_chunk_length_the_default_carry_cannot_run_starts_auto_from_a_shorter_carry() -> None:
    tiling = with_tiling_overrides(_BASE, overlap_fraction=None, carry_frames=None, max_window_frames=41)
    assert tiling.auto_carry
    assert tiling.carry_frames == 17


def test_a_carry_the_run_chose_is_pinned_and_never_searched() -> None:
    tiling = with_tiling_overrides(_BASE, overlap_fraction=None, carry_frames=25, max_window_frames=None)
    assert tiling.carry_pinned
    assert not tiling.auto_carry
    assert _price("1080p", 16, 241, tiling).carry_frames is None


def test_auto_takes_the_short_carry_when_only_it_runs_clean() -> None:
    # At 1080p on 10 GiB the 57-frame shortest window of carry 25 is a bet. Carry 17 lets a 41-frame window run clean.
    auto = _price("1080p", 10, 241)
    assert auto.carry_frames == 17
    assert safety_rank(auto) == 3
    assert auto.decision.chunk_pixel_frames == 41
    assert safety_rank(_price("1080p", 10, 241, _pinned(_BASE, 25))) < 3


@pytest.mark.parametrize("cell", list(_CELLS))
@pytest.mark.parametrize("vram_gib", _VRAM_GIB)
@pytest.mark.parametrize("frames", _FRAMES)
def test_auto_is_never_less_safe_than_any_carry_it_may_pick(cell: str, vram_gib: float, frames: int) -> None:
    auto = _price(cell, vram_gib, frames)
    tiling = _CELLS[cell][1]
    fixed = [
        _price(cell, vram_gib, frames, _pinned(tiling, carry))
        for carry in auto_carry_candidates(tiling.carry_frames or 25, None)
    ]
    assert auto.carry_frames in CARRY_PRESETS
    assert safety_rank(auto) >= max(safety_rank(priced) for priced in fixed)


@pytest.mark.parametrize("cell", list(_CELLS))
@pytest.mark.parametrize("vram_gib", _VRAM_GIB)
def test_cautious_never_runs_a_bet_from_auto(cell: str, vram_gib: float) -> None:
    admission = admit(_price(cell, vram_gib, 241), RiskAllowance.for_mode("cautious"))
    assert not isinstance(admission, Admitted) or admission.risk is None


def test_brave_runs_the_unmeasured_cell_as_a_bet_and_cautious_refuses_it() -> None:
    priced = _price("2160p", 48, 137)
    brave = admit(priced, RiskAllowance.for_mode("brave"))
    assert isinstance(brave, Admitted)
    assert brave.risk is not None
    assert brave.risk.tier == "unmeasured"
    assert not isinstance(admit(priced, RiskAllowance.for_mode("cautious")), Admitted)


def test_the_run_tiling_takes_the_carry_auto_picked() -> None:
    priced = _price("1080p", 10, 241)
    run = picked_carry_tiling(_BASE, priced)
    assert run is not None
    assert run.carry_frames == 17
    assert run.carry_pinned
    assert picked_carry_tiling(_BASE, _price("1080p", 16, 241, _pinned(_BASE, 25))) == _BASE
