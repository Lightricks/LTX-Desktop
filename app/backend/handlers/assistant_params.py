"""Small tool inputs; the generation API still owns execution validation."""

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from api_types import (
    ExtendMode, IcLoraControlType, IcLoraRecipeResolution, InputAssetRef,
    LTXLocalA2VResolution, LTXQueuedAspectRatio, LTXVideoGenAspectRatio, LTXVideoGenFps,
    NonEmptyPrompt, OfferingId, PromptProvenance, RETAKE_MAX_DURATION_SECONDS, RetakeMode, TargetResolution, VideoCameraMotion,
)

# The model repeats its tool schema as capabilities, so offer only what local models run.
# list_video_models still decides which combinations a model supports.
# Keep in step with the local spec tables; test_assistant_video_schema_offers_exactly_the_local_spec_values
# fails when they differ.
LocalVideoResolution: TypeAlias = Literal["270p", "360p", "540p", "720p", "1080p", "1440p", "2160p"]
LocalVideoDuration: TypeAlias = Literal[2, 3, 4, 5, 6, 8, 10, 20]


class AssistantParams(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class VideoParams(AssistantParams):
    model: OfferingId
    # No default: small models omit it, and a default silently replaces the resolution the user asked for.
    resolution: LocalVideoResolution
    duration: LocalVideoDuration | None = 5
    fps: LTXVideoGenFps = 24
    cameraMotion: VideoCameraMotion = "none"
    negativePrompt: str = ""
    seed: int | None = None


class TextParams(VideoParams):
    prompt: NonEmptyPrompt
    aspectRatio: LTXVideoGenAspectRatio = "16:9"


class ImageParams(VideoParams):
    prompt: str
    aspectRatio: LTXQueuedAspectRatio = "auto"


class RecipeParams(VideoParams):
    prompt: NonEmptyPrompt
    # "auto" follows the start frame, so only image recipes take it.
    aspectRatio: LTXQueuedAspectRatio = "16:9"
    scale: float = Field(default=1.0, ge=0.0, le=4.0)


class RecipeInputs(AssistantParams):
    startFrame: InputAssetRef
    endFrame: InputAssetRef | None = None


class AudioParams(AssistantParams):
    model: OfferingId
    prompt: str
    aspectRatio: LTXQueuedAspectRatio = "16:9"
    resolution: LTXLocalA2VResolution = "540p"
    fps: Literal[24] = 24
    cameraMotion: VideoCameraMotion = "none"
    negativePrompt: str = ""
    seed: int | None = None


class ExtensionParams(AssistantParams):
    prompt: str = ""
    negativePrompt: str = ""
    model: OfferingId
    # Two names because "duration" means clip length in every other tool.
    added_seconds: float | None = Field(default=None, ge=2, le=20, allow_inf_nan=False, description='"Add 3 seconds"')
    total_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False, description='"Extend it to 10 seconds": the final length')
    mode: ExtendMode = "end"
    resolution: TargetResolution | None = Field(default=None, description="Omit to keep the source size; set only when the user asks")


class VideoRecipeParams(AssistantParams):
    model: OfferingId
    prompt: str = ""
    negativePrompt: str = ""
    resolution: IcLoraRecipeResolution = Field(default="720p", description="720p unless the user asks; lower is much faster")
    # Omitted keeps the source rate; a faster rate than the source is refused.
    fps: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    controlType: IcLoraControlType | None = Field(
        default=None,
        description="Only for a recipe with a control mode (Video Guide). Omit to use its default",
    )


# The cells and lengths the Home Ingredients form offers. The sheet has no source clip to inherit them from.
IngredientsResolution: TypeAlias = Literal["270p", "360p", "540p", "720p", "1080p"]
IngredientsDuration: TypeAlias = Literal[5, 6, 8, 10, 20]


class IngredientsParams(AssistantParams):
    model: OfferingId
    prompt: NonEmptyPrompt
    negativePrompt: str = ""
    resolution: IngredientsResolution = Field(default="720p", description="720p unless the user asks; lower is much faster")
    duration: IngredientsDuration = 5
    aspectRatio: LTXVideoGenAspectRatio = "16:9"


class ReplacementParams(AssistantParams):
    prompt: str = ""
    negativePrompt: str = ""
    model: OfferingId
    startTime: float = Field(ge=0, allow_inf_nan=False)
    duration: float = Field(ge=2, le=RETAKE_MAX_DURATION_SECONDS, allow_inf_nan=False)
    mode: RetakeMode = "replace_audio_and_video"
    # No resolution: a retake keeps the source's frame size, and small models fill it in from earlier clips.


def api_params[T: BaseModel](params: BaseModel, api_type: type[T], provenance: PromptProvenance, **extra: str) -> T:
    return api_type.model_validate({**params.model_dump(), **extra, "promptProvenance": provenance})
