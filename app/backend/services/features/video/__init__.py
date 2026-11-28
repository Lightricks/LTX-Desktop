from typing import Any

from services.features.ic_lora_recipes import SynthesizedIcLoraKind
from services.features.video.ic_lora_recipe import IcLoraRecipeExecutor
from services.features.video.ic_lora_sheet_recipe import IcLoraSheetRecipeExecutor
from services.features.video.ic_lora_track_recipe import IcLoraTrackRecipeExecutor
from services.features.video.synthesized_ic_lora import SynthesizedIcLoraExecutor
from services.features.video.audio_to_video import (
    FEATURE as AUDIO_TO_VIDEO_FEATURE,
    AudioToVideoExecutor,
)
from services.features.video.image_to_video import (
    AssetResolver,
    FEATURE as IMAGE_TO_VIDEO_FEATURE,
    ImageToVideoExecutor,
    RebindableAssetResolver,
)
from services.features.video.keyframes import (
    FEATURE as KEYFRAMES_FEATURE,
    KeyframesExecutor,
)
from services.features.video.custom_lora import (
    CUSTOM_LORA_FEATURE,
    CustomLoraExecutor,
)
from services.features.video.extend import (
    FEATURE as EXTEND_FEATURE,
    ExtendExecutor,
)
from services.features.video.retake import (
    FEATURE as RETAKE_FEATURE,
    RetakeExecutor,
)
from services.features.video.text_to_video import (
    FEATURE,
    ReservedVideoGenerator,
    TextToVideoExecutor,
    VIDEO_OUTPUT_PLAN,
    require_video_output,
    require_supported_contract_version,
)

# A recipe kind with no entry here fails when the app starts, not when its first job runs.
SYNTHESIZED_IC_LORA_EXECUTORS: dict[
    SynthesizedIcLoraKind, type[SynthesizedIcLoraExecutor[Any]]
] = {"sheet": IcLoraSheetRecipeExecutor, "track": IcLoraTrackRecipeExecutor}

__all__ = [
    "AUDIO_TO_VIDEO_FEATURE",
    "CUSTOM_LORA_FEATURE",
    "CustomLoraExecutor",
    "IcLoraRecipeExecutor",
    "IcLoraSheetRecipeExecutor",
    "IcLoraTrackRecipeExecutor",
    "SYNTHESIZED_IC_LORA_EXECUTORS",
    "SynthesizedIcLoraExecutor",
    "AssetResolver",
    "AudioToVideoExecutor",
    "EXTEND_FEATURE",
    "ExtendExecutor",
    "FEATURE",
    "IMAGE_TO_VIDEO_FEATURE",
    "ImageToVideoExecutor",
    "KEYFRAMES_FEATURE",
    "KeyframesExecutor",
    "RETAKE_FEATURE",
    "RebindableAssetResolver",
    "ReservedVideoGenerator",
    "RetakeExecutor",
    "TextToVideoExecutor",
    "VIDEO_OUTPUT_PLAN",
    "require_video_output",
    "require_supported_contract_version",
]
