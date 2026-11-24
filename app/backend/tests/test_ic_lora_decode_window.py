"""Decode-safe window of a tiled high-res IC-LoRA job."""

from __future__ import annotations

import pytest

from runtime_config.ic_lora_decode_window import (
    DECODE_MAX_UNTILED_WINDOW,
    choose_decode_window,
    decode_budget_gib,
    decode_need_gib,
)

_RTX_5090_GIB = 31.84
_4K = (3840, 2112)
_1440P = (2560, 1472)
_1080P = (1920, 1088)

# Minimum decode budget the ltx-core planner probed offline (``recommended_decode_tiling_config``,
# GiB, 0.05 resolution): (width, height, frames, probe).
_PLANNER_PROBES = [
    (*_1080P, 25, 6.3),
    (*_1080P, 73, 8.15),
    (*_1440P, 41, 9.87),
    (*_1440P, 73, 12.08),
    (*_1440P, 97, 12.14),
    (*_4K, 25, 15.2),
    (*_4K, 41, 17.55),
    (*_4K, 49, 18.74),
    (*_4K, 57, 19.9),
    (*_4K, 65, 21.09),
    (*_4K, 73, 22.28),
    (*_4K, 97, 22.42),
]


@pytest.mark.parametrize(("width", "height", "frames", "probe"), _PLANNER_PROBES)
def test_need_sits_just_above_every_planner_probe(width: int, height: int, frames: int, probe: float) -> None:
    need = decode_need_gib(width, height, frames)
    assert probe <= need <= probe + 0.6


def test_budget_shrinks_as_windows_pile_up_and_saturates() -> None:
    budgets = [decode_budget_gib(_RTX_5090_GIB, windows) for windows in (1, 2, 3, 4, 9)]
    assert budgets == sorted(budgets, reverse=True)
    assert budgets[2] == budgets[3] == budgets[4]
    # Measured on the 5090: 27.8, 22.7 and 20.9 GiB on windows 1 to 3.
    assert budgets[0] == pytest.approx(27.84)
    assert budgets[1] <= 22.7
    assert budgets[2] <= 20.9


def _choose(
    frames: int, window: int | None, size: tuple[int, int], vram: float = _RTX_5090_GIB, carry: int = 17
):
    return choose_decode_window(
        frames=frames, window=window, width=size[0], height=size[1], total_vram_gib=vram, carry_frames=carry
    )


def test_4k_73_frame_windows_do_not_fit_a_third_window_on_a_5090() -> None:
    choice = _choose(137, 73, _4K)
    assert choice.fits
    assert choice.window == 49


def test_4k_cap_under_the_shortest_window_of_a_long_carry_refuses_instead_of_running_it() -> None:
    # A carry of 41 needs a window of 89 frames or more. The 4K cap of 73 breaks that, even
    # on a card with memory to spare.
    assert not _choose(300, 97, _4K, vram=80, carry=41).fits
    assert not _choose(300, None, _4K, vram=80, carry=41).fits
    assert _choose(300, 97, _4K, vram=80, carry=17).fits


def test_4k_longer_clips_keep_the_same_window() -> None:
    assert _choose(241, 73, _4K).window == 49


def test_4k_two_window_clip_keeps_a_longer_window() -> None:
    assert _choose(97, 73, _4K).window == 65


def test_4k_single_pass_clip_that_fits_stays_unchunked() -> None:
    choice = _choose(73, None, _4K)
    assert choice.fits
    assert choice.window is None


def test_4k_window_over_the_frame_tile_is_capped() -> None:
    # A 97-frame window splits the frame axis in the decode and stalled at 4K.
    choice = _choose(137, 97, _4K, vram=48)
    assert choice.window == DECODE_MAX_UNTILED_WINDOW


def test_4k_unchunked_clip_over_the_frame_tile_drops_to_windows() -> None:
    choice = _choose(97, None, _4K, vram=48)
    assert choice.fits
    assert choice.window is not None
    assert choice.window <= DECODE_MAX_UNTILED_WINDOW


def test_1440p_keeps_a_97_frame_window() -> None:
    # 1440p ran 97-frame windows; only canvases past the fence cap the window.
    choice = _choose(193, 97, _1440P)
    assert choice.fits
    assert choice.window == 97


def test_1440p_budget_leaves_room_for_41f_and_steps_73f_or_97f_down_to_41f_on_16gb() -> None:
    """C2/C3 measured 12.8–15.0 GiB decode budget under a 16.0 GiB cap at 1440p;
    using the 4K 11.0 GiB reserve would refuse 1440p 41f (need+margin 11.02 > 5.0)
    and fail to step 73f/97f down to 41f on 16 GB cards."""
    assert decode_budget_gib(16.0, 5, width=2560, height=1472) == pytest.approx(11.20)
    assert decode_budget_gib(15.9, 5, width=2560, height=1472) == pytest.approx(11.10)
    for vram in (16.0, 15.9):
        water = _choose(137, 41, _1440P, vram=vram)
        assert water.fits
        assert water.window == 41
        refine = _choose(241, 73, _1440P, vram=vram)
        assert refine.fits
        assert refine.window == 41
        restore = _choose(137, 97, _1440P, vram=vram)
        assert restore.fits
        assert restore.window == 41



def test_no_window_fits_a_card_with_too_little_memory() -> None:
    choice = _choose(137, 73, _4K, vram=24)
    assert not choice.fits


def test_a_card_with_room_keeps_the_lora_window() -> None:
    choice = _choose(137, 73, _4K, vram=48)
    assert choice.fits
    assert choice.window == 73


def test_a_window_under_the_shortest_working_one_is_only_tried_as_given() -> None:
    choice = _choose(137, 33, _1080P)
    assert choice.fits
    assert choice.window == 33
