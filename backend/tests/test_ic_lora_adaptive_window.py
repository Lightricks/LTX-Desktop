"""Adaptive chunk window of Refine Details: the resolution cap and the candidate list."""

from __future__ import annotations

import pytest

from runtime_config.ic_lora_adaptive_window import (
    adaptive_window_candidates,
    resolution_window_cap,
)
from runtime_config.ic_lora_tiling import ADAPTIVE_MAX_WINDOW_FRAMES, _CONTROLS  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("width", "height", "cap"),
    [
        (480, 256, None),
        (1280, 704, None),
        (1920, 1088, 97),
        (2560, 1440, 97),
        (3840, 2112, 73),
    ],
)
def test_resolution_cap(width: int, height: int, cap: int | None) -> None:
    assert resolution_window_cap(width, height) == cap


def test_the_window_presets_of_the_form_stay_within_the_longest_maximum() -> None:
    presets = _CONTROLS["windowFrames"]["presets"]
    assert max(presets) == ADAPTIVE_MAX_WINDOW_FRAMES


def test_a_short_clip_runs_in_one_pass_only() -> None:
    assert adaptive_window_candidates(49, cap=None, user_max=None, carry=25) == [None]


def test_an_uncapped_clip_tries_one_pass_then_every_window_below_it() -> None:
    candidates = adaptive_window_candidates(241, cap=None, user_max=None, carry=25)
    assert candidates[:3] == [None, 233, 225]
    assert candidates[-1] == 57  # shortest window for a carry of 25
    assert all(window is None or (window - 1) % 8 == 0 for window in candidates)


def test_a_cap_below_the_clip_removes_the_one_pass_candidate() -> None:
    candidates = adaptive_window_candidates(241, cap=97, user_max=None, carry=25)
    assert candidates[:3] == [97, 89, 81]
    assert None not in candidates


def test_a_clip_inside_the_cap_still_offers_shorter_windows_after_one_pass() -> None:
    assert adaptive_window_candidates(73, cap=97, user_max=None, carry=25) == [None, 65, 57]


def test_4k_walk_starts_at_the_cap() -> None:
    assert adaptive_window_candidates(137, cap=73, user_max=None, carry=17) == [73, 65, 57, 49, 41]


def test_a_user_maximum_below_the_cap_lowers_the_start() -> None:
    assert adaptive_window_candidates(241, cap=97, user_max=65, carry=17)[:2] == [65, 57]


def test_a_user_maximum_above_the_clip_keeps_one_pass() -> None:
    assert adaptive_window_candidates(49, cap=None, user_max=97, carry=17)[0] is None


def test_a_cap_under_the_shortest_window_leaves_no_candidate() -> None:
    assert adaptive_window_candidates(241, cap=73, user_max=None, carry=41) == []
