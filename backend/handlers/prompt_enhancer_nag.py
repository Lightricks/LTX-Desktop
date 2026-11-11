"""Why NAG (negative guidance) is unavailable, per feature, for the prompt enhancer status."""

from __future__ import annotations

import os

from services.fast_video_pipeline.nag_config import decide_nag
from services.features.ic_lora_recipes import IC_LORA_RECIPES
from services.features.lora_recipes import queued_lora_recipes
from services.features.video import (
    AUDIO_TO_VIDEO_FEATURE,
    CUSTOM_LORA_FEATURE,
    EXTEND_FEATURE,
    FEATURE as TEXT_TO_VIDEO_FEATURE,
    IMAGE_TO_VIDEO_FEATURE,
    KEYFRAMES_FEATURE,
    RETAKE_FEATURE,
)


def nag_unavailable_reasons(
    *, nag_scale: float, local_encoding: bool, compile_on: bool
) -> tuple[str | None, dict[str, str]]:
    """The reason NAG is off for the base pipeline, and the reason for each feature."""
    base_reason = decide_nag(
        "x",
        os.environ,
        default_scale=nag_scale,
        local_encoding=local_encoding,
        compiled=False,
    ).user_message
    fast_reason = decide_nag(
        "x",
        os.environ,
        default_scale=nag_scale,
        local_encoding=local_encoding,
        compiled=compile_on,
    ).user_message
    fast_features = (
        TEXT_TO_VIDEO_FEATURE,
        IMAGE_TO_VIDEO_FEATURE,
        KEYFRAMES_FEATURE,
        CUSTOM_LORA_FEATURE,
        *(recipe.recipe_id for recipe in queued_lora_recipes()),
    )
    non_compiled_features = (
        AUDIO_TO_VIDEO_FEATURE,
        RETAKE_FEATURE,
        EXTEND_FEATURE,
        *(recipe.recipe_id for recipe in IC_LORA_RECIPES.values()),
    )
    reason_by_feature: dict[str, str] = {}
    if fast_reason is not None:
        for fid in fast_features:
            reason_by_feature[fid] = fast_reason
    if base_reason is not None:
        for fid in non_compiled_features:
            reason_by_feature[fid] = base_reason
    return base_reason, reason_by_feature
