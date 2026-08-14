"""Unit tests for the Desktop LTX capabilities SSOT."""

from __future__ import annotations

import pytest

from runtime_config.ltx_capabilities import (
    api_caps,
    effective_local_caps,
    local_caps,
    pixels_for,
    supports,
)


def test_local_2_3_540p_is_historical_960x544():
    caps = local_caps("ltx-2.3-22b-distilled-1.1")
    assert pixels_for(caps, "540p", "16:9") == (960, 544)
    assert pixels_for(caps, "540p", "9:16") == (544, 960)


def test_local_2_3_v10_shares_2_3_pixel_map():
    assert pixels_for(local_caps("ltx-2.3-22b-distilled"), "540p", "16:9") == (960, 544)


def test_local_2_5_540p_is_legal_16_9():
    caps = local_caps("ltx-2.5-22b-distilled")
    width, height = pixels_for(caps, "540p", "16:9")
    assert (width, height) == (1024, 576)
    assert width % 64 == 0 and height % 64 == 0


def test_local_2_5_allows_ic_lora_and_user_loras():
    caps = local_caps("ltx-2.5-22b-distilled")
    assert supports(caps, "ic_lora") is True
    assert supports(caps, "user_loras") is True
    assert supports(caps, "retake") is False
    assert supports(caps, "extend") is False


def test_local_2_3_allows_ic_lora_user_loras_retake():
    caps = local_caps("ltx-2.3-22b-distilled-1.1")
    assert supports(caps, "ic_lora") is True
    assert supports(caps, "user_loras") is True
    assert supports(caps, "retake") is True
    assert supports(caps, "extend") is True
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


def test_api_fast_2_5_has_a2v_and_auto_duration():
    caps = api_caps("fast-2.5")
    assert supports(caps, "a2v") is True
    assert supports(caps, "retake") is False
    assert supports(caps, "extend") is False
    assert supports(caps, "auto_duration") is True
    assert pixels_for(caps, "1080p", "16:9") == (1920, 1080)


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


def test_ic_lora_flag_is_on_for_every_local_model():
    from runtime_config.model_download_specs import ALL_LTX_LOCAL_MODEL_IDS

    for model_id in ALL_LTX_LOCAL_MODEL_IDS:
        assert supports(local_caps(model_id), "ic_lora") is True
