"""Unit tests for the local source-prep math (resolution + frame-count correction)."""

from __future__ import annotations

import pytest

from _routes._errors import HTTPError
from handlers.video_resolution import correct_frame_count, correct_resolution


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
