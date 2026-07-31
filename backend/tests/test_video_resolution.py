"""Unit tests for the local source-prep math (resolution + frame-count correction)."""

from __future__ import annotations

import pytest

from _routes._errors import HTTPError
from handlers.video_resolution import (
    correct_frame_count,
    correct_resolution,
    resolve_fast_video_dimensions,
)


def test_fast_540p_uses_aspect_aware_two_stage_grid_under_budget():
    width, height = resolve_fast_video_dimensions("540p", "16:9")
    assert (width, height) == (896, 512)
    assert width % 64 == 0 and height % 64 == 0
    assert width * height <= 960 * 544
    assert abs(width / height - 16 / 9) < abs(960 / 512 - 16 / 9)


def test_fast_540p_portrait_swaps_the_qualified_pair():
    assert resolve_fast_video_dimensions("540p", "9:16") == (512, 896)


def test_fast_resolution_rejects_unknown_catalog_or_aspect():
    with pytest.raises(ValueError):
        resolve_fast_video_dimensions("1440p", "16:9")
    with pytest.raises(ValueError):
        resolve_fast_video_dimensions("540p", "4:3")


def test_correct_resolution_snaps_height_down_to_div32():
    # 1080p: width already ÷32, height 1080 -> 1056 (33*32); never upscales.
    assert correct_resolution(1920, 1080, source_width=1920, source_height=1080) == (1920, 1056)


def test_correct_resolution_lower_tier_snapped():
    assert correct_resolution(1280, 720, source_width=1920, source_height=1080) == (1280, 704)


def test_correct_resolution_never_upscales():
    # Requesting above source is clamped to source, then snapped.
    assert correct_resolution(4000, 4000, source_width=1920, source_height=1080) == (1920, 1056)


def test_correct_resolution_portrait():
    assert correct_resolution(1080, 1920, source_width=1080, source_height=1920) == (1056, 1920)


def test_correct_frame_count_trims_down_to_8k_plus_1():
    assert correct_frame_count(97) == 97
    assert correct_frame_count(100) == 97
    assert correct_frame_count(9) == 9
    assert correct_frame_count(193) == 193


def test_correct_frame_count_rejects_too_short():
    with pytest.raises(HTTPError):
        correct_frame_count(8)
