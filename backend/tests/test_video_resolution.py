"""Unit tests for the local source-prep math (resolution + frame-count correction)."""

from __future__ import annotations

import logging

import pytest

from _routes._errors import HTTPError
from handlers.video_resolution import (
    cap_to_1080p,
    correct_frame_count,
    correct_resolution,
    generation_letterbox,
    resolve_home_edit_size,
    resolve_target_resolution,
)


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


def test_resolve_target_resolution_keeps_1080p_after_div32(caplog):
    with caplog.at_level(logging.INFO, logger="handlers.video_resolution"):
        assert resolve_target_resolution(None, 1920, 1080) == (1920, 1056)
    assert "Downscaling" not in caplog.text


def test_cap_to_1080p_fits_4k_into_the_home_box():
    assert cap_to_1080p(3840, 2160) == (1920, 1080)
    assert cap_to_1080p(2160, 3840) == (1080, 1920)
    # Already at or below the local 1080p cell: no upscale.
    assert cap_to_1080p(1920, 1088) == (1920, 1088)
    assert cap_to_1080p(1280, 704) == (1280, 704)


def test_home_edit_size_keeps_the_picture_and_leaves_padding_to_the_letterbox():
    # No request: 4K fits the 1080p cell. Height stays 1080; the letterbox pads it.
    assert resolve_home_edit_size(None, None, 3840, 2160) == (1920, 1080)
    # 720p on a 1080p source stays 720. It is not snapped down to 704.
    assert resolve_home_edit_size(1280, 720, 1920, 1080) == (1280, 720)
    # A 4K request on a 1080p source cannot upscale past the source.
    assert resolve_home_edit_size(3840, 2160, 1920, 1080) == (1920, 1080)
    # A request above a small source is clamped to that source, still unsnapped.
    assert resolve_home_edit_size(1280, 720, 640, 360) == (640, 360)


def test_queued_home_path_still_caps_then_snaps():
    # 1920×1080 snaps height to ÷32. Local 1080p (1920×1088) is already on that grid.
    capped = cap_to_1080p(3840, 2160)
    assert resolve_target_resolution(None, *capped) == (1920, 1056)
    local = cap_to_1080p(1920, 1088)
    assert resolve_target_resolution(None, *local) == (1920, 1088)
    portrait = cap_to_1080p(2160, 3840)
    assert resolve_target_resolution(None, *portrait) == (1056, 1920)


def test_resolve_target_resolution_leaves_smaller_sources(caplog):
    # Below 540p: no upscale; height 360 snaps down to ÷32.
    with caplog.at_level(logging.INFO, logger="handlers.video_resolution"):
        assert resolve_target_resolution(None, 640, 360) == (640, 352)
    assert "Downscaling" not in caplog.text


def test_generation_letterbox_pads_1080p_to_the_next_multiple_of_32():
    box = generation_letterbox(1920, 1080)
    assert (box.canvas_width, box.canvas_height) == (1920, 1088)
    assert (box.content_width, box.content_height) == (1920, 1080)
    assert (box.left, box.right, box.top, box.bottom) == (0, 0, 4, 4)


def test_generation_letterbox_pads_portrait_on_the_sides():
    box = generation_letterbox(1080, 1920)
    assert (box.canvas_width, box.canvas_height) == (1088, 1920)
    assert (box.left, box.right, box.top, box.bottom) == (4, 4, 0, 0)


def test_generation_letterbox_is_a_no_op_when_already_aligned():
    box = generation_letterbox(1920, 1088)
    assert (box.canvas_width, box.canvas_height) == (1920, 1088)
    assert (box.left, box.top) == (0, 0)


def test_correct_frame_count_trims_down_to_8k_plus_1():
    assert correct_frame_count(97) == 97
    assert correct_frame_count(100) == 97
    assert correct_frame_count(9) == 9
    assert correct_frame_count(193) == 193


def test_correct_frame_count_rejects_too_short():
    with pytest.raises(HTTPError):
        correct_frame_count(8)
