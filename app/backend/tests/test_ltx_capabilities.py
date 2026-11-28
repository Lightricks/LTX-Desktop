"""Unit tests for the Desktop LTX capabilities SSOT."""

from __future__ import annotations

import pytest

from api_types import LOCAL_MULTI_KEYFRAME_MAX_COUNT
from runtime_config.ltx_capabilities import (
    LocalOfferingCapabilities,
    a2v_pixels_for,
    api_caps,
    budget_pixels,
    budget_size,
    effective_local_caps,
    local_canvas,
    local_caps,
    pixels_for,
    supports,
)


def test_local_models_share_one_two_stage_pixel_map():
    """2.3 and 2.5 must advertise the same local sizes so a model switch cannot
    land 540p on 960×544, which the two-stage pipeline rejects (not ÷64)."""
    from runtime_config.model_download_specs import ALL_LTX_LOCAL_MODEL_IDS

    reference = local_caps("ltx-2.5-22b-distilled").resolution_pixels_16_9
    for model_id in ALL_LTX_LOCAL_MODEL_IDS:
        caps = local_caps(model_id)
        assert caps.resolution_pixels_16_9 == reference, model_id
        for resolution in reference:
            for aspect in ("21:9", "16:9", "3:2", "4:3", "1:1", "4:5", "9:16"):
                width, height = pixels_for(caps, resolution, aspect)
                assert width % 64 == 0 and height % 64 == 0, (
                    f"{model_id} {resolution} {aspect}: {width}x{height}"
                )


def test_local_2_3_540p_matches_2_5():
    caps = local_caps("ltx-2.3-22b-distilled-1.1")
    assert pixels_for(caps, "540p", "16:9") == (1024, 576)
    assert pixels_for(caps, "540p", "9:16") == (576, 1024)


def test_local_2_3_v10_shares_2_3_pixel_map():
    assert pixels_for(local_caps("ltx-2.3-22b-distilled"), "540p", "16:9") == (1024, 576)


def test_stage1_only_labels_use_the_parent_canvas():
    caps = local_caps("ltx-2.5-22b-distilled")
    video_270 = local_canvas("270p", mode="video")
    parent_540 = local_canvas("540p", mode="video")
    assert (video_270.width, video_270.height) == (parent_540.width, parent_540.height) == (1024, 576)
    assert video_270.skip_stage_2 is True
    assert parent_540.skip_stage_2 is False
    assert budget_size(video_270) == (512, 288)
    assert budget_size(local_canvas("360p", mode="video")) == (640, 352)
    audio_270 = local_canvas("270p", mode="a2v")
    assert (audio_270.width, audio_270.height) == (576, 320)
    assert audio_270.skip_stage_2 is False
    assert budget_size(audio_270) == (576, 320)
    assert pixels_for(caps, "270p", "16:9") == (1024, 576)
    assert pixels_for(caps, "360p", "16:9") == (1280, 704)
    assert pixels_for(caps, "270p", "9:16") == (576, 1024)


def test_a2v_small_labels_finish_on_their_own_canvas():
    caps = local_caps("ltx-2.5-22b-distilled")
    assert a2v_pixels_for(caps, "270p", "16:9") == (576, 320)
    assert a2v_pixels_for(caps, "270p", "9:16") == (320, 576)
    assert a2v_pixels_for(caps, "360p", "16:9") == (704, 384)
    assert a2v_pixels_for(caps, "360p", "9:16") == (384, 704)
    assert a2v_pixels_for(caps, "540p", "16:9") == (1024, 576)


def test_budget_pixels_uses_the_selected_ratio_and_stage1_half():
    caps = local_caps("ltx-2.5-22b-distilled")
    assert budget_pixels(caps, "540p", "16:9", mode="video") == (1024, 576)
    assert budget_pixels(caps, "540p", "21:9", mode="video") == (1344, 576)
    assert budget_pixels(caps, "270p", "16:9", mode="video") == (512, 288)
    assert budget_pixels(caps, "270p", "21:9", mode="video") == (672, 288)
    assert budget_pixels(caps, "270p", "21:9", mode="a2v") == (768, 320)


def test_short_side_formula_keeps_16_9_anchors_and_api_tiles():
    local = local_caps("ltx-2.5-22b-distilled")
    api = api_caps("fast-2.5")
    assert pixels_for(local, "1080p", "16:9") == (1920, 1088)
    assert pixels_for(local, "540p", "21:9") == (1344, 576)
    assert pixels_for(local, "1080p", "21:9") == (2560, 1088)
    assert pixels_for(local, "540p", "4:3") == (768, 576)
    assert pixels_for(local, "540p", "1:1") == (576, 576)
    # 720 is closer to 704 than to 768. Snapping up landed on portrait 4:3.
    assert pixels_for(local, "540p", "4:5") == (576, 704)
    # 1934 is closer to the 16:9 anchor's transpose (1920) than to 1984.
    assert pixels_for(local, "1080p", "9:16") == (1088, 1920)
    assert pixels_for(api, "1080p", "16:9") == (1920, 1080)
    assert pixels_for(api, "1080p", "3:2") == (1620, 1080)
    assert pixels_for(api, "1080p", "1:1") == (1080, 1080)
    assert pixels_for(api, "1080p", "9:16") == (1080, 1920)
    assert pixels_for(api, "720p", "3:2") == (1080, 720)


def test_local_2_5_540p_is_legal_16_9():
    caps = local_caps("ltx-2.5-22b-distilled")
    width, height = pixels_for(caps, "540p", "16:9")
    assert (width, height) == (1024, 576)
    assert width % 64 == 0 and height % 64 == 0


def test_local_2_5_allows_ic_lora_and_user_loras():
    caps = local_caps("ltx-2.5-22b-distilled")
    assert supports(caps, "ic_lora") is True
    assert supports(caps, "user_loras") is True
    assert supports(caps, "retake") is True
    assert supports(caps, "extend") is True
    assert supports(caps, "multi_keyframe") is True
    assert caps.multi_keyframe_max_count == LOCAL_MULTI_KEYFRAME_MAX_COUNT


def test_local_2_3_allows_ic_lora_user_loras_retake():
    caps = local_caps("ltx-2.3-22b-distilled-1.1")
    assert supports(caps, "ic_lora") is True
    assert supports(caps, "user_loras") is True
    assert supports(caps, "retake") is True
    assert supports(caps, "extend") is True
    assert supports(caps, "multi_keyframe") is True
    assert caps.multi_keyframe_max_count == LOCAL_MULTI_KEYFRAME_MAX_COUNT
    assert supports(caps, "auto_duration") is False


def test_local_2_5_keeps_t2v_i2v_a2v():
    caps = local_caps("ltx-2.5-22b-distilled")
    assert supports(caps, "t2v") is True
    assert supports(caps, "i2v") is True
    assert supports(caps, "a2v") is True
    assert supports(caps, "camera_motion") is True
    assert supports(caps, "auto_duration") is True


def test_local_2_5_auto_duration_requires_duration_head_ready():
    model_id = "ltx-2.5-22b-distilled"
    assert supports(effective_local_caps(model_id, duration_head_ready=True), "auto_duration") is True
    assert supports(effective_local_caps(model_id, duration_head_ready=False), "auto_duration") is False


def test_effective_local_caps_preserves_local_subclass():
    caps = effective_local_caps("ltx-2.5-22b-distilled", duration_head_ready=False)
    assert isinstance(caps, LocalOfferingCapabilities)


def test_local_2_3_auto_duration_stays_off_even_if_duration_head_ready():
    assert (
        supports(
            effective_local_caps("ltx-2.3-22b-distilled-1.1", duration_head_ready=True),
            "auto_duration",
        )
        is False
    )


def test_api_fast_2_3_has_no_a2v_or_auto_duration():
    caps = api_caps("fast")
    assert supports(caps, "a2v") is False
    assert supports(caps, "retake") is False
    assert supports(caps, "extend") is False
    assert supports(caps, "auto_duration") is False
    assert pixels_for(caps, "1080p", "16:9") == (1920, 1080)
    assert pixels_for(caps, "720p", "16:9") == (1280, 720)
    assert pixels_for(caps, "720p", "9:16") == (720, 1280)


def test_api_fast_2_5_has_a2v_and_auto_duration():
    caps = api_caps("fast-2.5")
    assert supports(caps, "a2v") is True
    assert supports(caps, "retake") is False
    assert supports(caps, "extend") is False
    assert supports(caps, "auto_duration") is True
    assert pixels_for(caps, "1080p", "16:9") == (1920, 1080)
    assert pixels_for(caps, "720p", "16:9") == (1280, 720)


def test_api_pro_2_3_has_a2v_and_retake():
    caps = api_caps("pro")
    assert supports(caps, "a2v") is True
    assert supports(caps, "retake") is True
    assert supports(caps, "extend") is True
    assert supports(caps, "auto_duration") is False


def test_api_pro_2_5_has_a2v_and_auto_duration_not_retake():
    caps = api_caps("pro-2.5")
    assert supports(caps, "a2v") is True
    assert supports(caps, "retake") is False
    assert supports(caps, "extend") is False
    assert supports(caps, "auto_duration") is True


def test_pixels_for_unknown_resolution_raises():
    with pytest.raises(KeyError):
        pixels_for(api_caps("fast"), "540p", "16:9")


def test_a2v_pixels_for_rejects_a_resolution_the_offering_does_not_list():
    with pytest.raises(KeyError):
        a2v_pixels_for(api_caps("fast"), "270p", "16:9")


def test_ic_lora_flag_is_on_for_every_local_model():
    from runtime_config.model_download_specs import ALL_LTX_LOCAL_MODEL_IDS

    for model_id in ALL_LTX_LOCAL_MODEL_IDS:
        assert supports(local_caps(model_id), "ic_lora") is True


@pytest.mark.parametrize("pipeline", ["fast", "fast-2.5", "pro", "pro-2.5"])
def test_multi_keyframe_is_off_for_every_api_pipeline(pipeline):
    caps = api_caps(pipeline)
    assert supports(caps, "multi_keyframe") is False
    assert caps.multi_keyframe_max_count == 0
