"""Feature ids stored on Explore generations.

Desktop and remote Recently used both query this set, so the two apps show
the same features.
"""

from __future__ import annotations

from services.features.ic_lora_recipes import queued_ic_lora_recipes
from services.features.lora_recipes import queued_lora_recipes
from services.features.video import (
    AUDIO_TO_VIDEO_FEATURE,
    EXTEND_FEATURE,
    FEATURE as TEXT_TO_VIDEO_FEATURE,
    IMAGE_TO_VIDEO_FEATURE,
    RETAKE_FEATURE,
)

QUEUED_GENERATION_FEATURES: frozenset[str] = frozenset(
    {
        TEXT_TO_VIDEO_FEATURE,
        IMAGE_TO_VIDEO_FEATURE,
        AUDIO_TO_VIDEO_FEATURE,
        RETAKE_FEATURE,
        EXTEND_FEATURE,
    }
    | {recipe.recipe_id for recipe in queued_lora_recipes()}
    | {recipe.recipe_id for recipe in queued_ic_lora_recipes()}
)
