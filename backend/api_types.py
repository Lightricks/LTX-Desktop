"""Pydantic request/response models and typed aliases for ltx2_server."""

from __future__ import annotations

from typing import Annotated
from typing import Literal, NamedTuple, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyPrompt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ModelFileType = Literal[
    "checkpoint",
    "upsampler",
    "distilled_lora",
    "ic_lora",
    "depth_processor",
    "person_detector",
    "pose_processor",
    "text_encoder",
    "zit",
]


class ImageConditioningInput(NamedTuple):
    """Image conditioning triplet used by all video pipelines."""

    path: str
    frame_idx: int
    strength: float


JsonObject: TypeAlias = dict[str, object]
VideoCameraMotion = Literal[
    "none",
    "dolly_in",
    "dolly_out",
    "dolly_left",
    "dolly_right",
    "jib_up",
    "jib_down",
    "static",
    "focus_shift",
]


# ============================================================
# Response Models
# ============================================================


class ModelStatusItem(BaseModel):
    id: str
    name: str
    loaded: bool
    downloaded: bool


class GpuTelemetry(BaseModel):
    name: str
    vram: int
    vramUsed: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    models_loaded: bool
    active_model: str | None
    gpu_info: GpuTelemetry
    sage_attention: bool
    models_status: list[ModelStatusItem]


class GpuInfoResponse(BaseModel):
    cuda_available: bool
    mps_available: bool = False
    gpu_available: bool = False
    gpu_name: str | None
    vram_gb: int | None
    gpu_info: GpuTelemetry


class RuntimePolicyResponse(BaseModel):
    force_api_generations: bool


class GenerationProgressResponse(BaseModel):
    status: Literal["idle", "running", "complete", "cancelled", "error"]
    phase: str
    progress: int
    currentStep: int | None
    totalSteps: int | None


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str


class ModelFileStatus(BaseModel):
    id: ModelFileType
    name: str
    description: str
    downloaded: bool
    size: int
    expected_size: int
    required: bool = True
    is_folder: bool = False
    optional_reason: str | None = None


class TextEncoderStatus(BaseModel):
    downloaded: bool
    size_bytes: int
    size_gb: float
    expected_size_gb: float


class ModelsStatusResponse(BaseModel):
    models: list[ModelFileStatus]
    all_downloaded: bool
    total_size: int
    downloaded_size: int
    total_size_gb: float
    downloaded_size_gb: float
    models_path: str
    has_api_key: bool
    text_encoder_status: TextEncoderStatus
    use_local_text_encoder: bool


class DownloadProgressRunningResponse(BaseModel):
    status: Literal["downloading"]
    current_downloading_file: ModelFileType | None
    current_file_progress: float
    total_progress: float
    total_downloaded_bytes: int
    expected_total_bytes: int
    completed_files: set[ModelFileType]
    all_files: set[ModelFileType]
    error: None = None
    speed_bytes_per_sec: float


class DownloadProgressCompleteResponse(BaseModel):
    status: Literal["complete"]


class DownloadProgressErrorResponse(BaseModel):
    status: Literal["error"]
    error: str


DownloadProgressResponse: TypeAlias = (
    DownloadProgressRunningResponse | DownloadProgressCompleteResponse | DownloadProgressErrorResponse
)


class SuggestGapPromptResponse(BaseModel):
    status: Literal["success"] = "success"
    suggested_prompt: str


class GenerateVideoCompleteResponse(BaseModel):
    status: Literal["complete"]
    video_path: str


class GenerateVideoCancelledResponse(BaseModel):
    status: Literal["cancelled"]


GenerateVideoResponse: TypeAlias = GenerateVideoCompleteResponse | GenerateVideoCancelledResponse


class GenerateImageCompleteResponse(BaseModel):
    status: Literal["complete"]
    image_paths: list[str]


class GenerateImageCancelledResponse(BaseModel):
    status: Literal["cancelled"]


GenerateImageResponse: TypeAlias = GenerateImageCompleteResponse | GenerateImageCancelledResponse


class CancelCancellingResponse(BaseModel):
    status: Literal["cancelling"]
    id: str


class CancelNoActiveGenerationResponse(BaseModel):
    status: Literal["no_active_generation"]


CancelResponse: TypeAlias = CancelCancellingResponse | CancelNoActiveGenerationResponse


class RetakeVideoResponse(BaseModel):
    status: Literal["complete"]
    video_path: str


class RetakePayloadResponse(BaseModel):
    status: Literal["complete"]
    result: JsonObject


class RetakeCancelledResponse(BaseModel):
    status: Literal["cancelled"]


RetakeResponse: TypeAlias = RetakeVideoResponse | RetakePayloadResponse | RetakeCancelledResponse


class IcLoraExtractResponse(BaseModel):
    conditioning: str
    original: str
    conditioning_type: ConditioningType
    frame_time: float


class IcLoraGenerateCompleteResponse(BaseModel):
    status: Literal["complete"]
    video_path: str


class IcLoraGenerateCancelledResponse(BaseModel):
    status: Literal["cancelled"]


IcLoraGenerateResponse: TypeAlias = IcLoraGenerateCompleteResponse | IcLoraGenerateCancelledResponse


class ModelDownloadStartResponse(BaseModel):
    status: Literal["started"]
    message: str
    sessionId: str


class TextEncoderDownloadStartedResponse(BaseModel):
    status: Literal["started"]
    message: str
    sessionId: str


class TextEncoderAlreadyDownloadedResponse(BaseModel):
    status: Literal["already_downloaded"]
    message: str


TextEncoderDownloadResponse: TypeAlias = TextEncoderDownloadStartedResponse | TextEncoderAlreadyDownloadedResponse


class StatusResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    error: str
    message: str | None = None


# ============================================================
# Request Models
# ============================================================


VideoResolution: TypeAlias = Literal["540p", "720p", "1080p", "1440p", "2160p"]
VideoModel: TypeAlias = Literal["fast", "pro"]


class GenerateVideoRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    prompt: NonEmptyPrompt
    resolution: VideoResolution = "1080p"
    model: VideoModel = "fast"
    cameraMotion: VideoCameraMotion = "none"
    negativePrompt: str = ""
    duration: int = Field(default=2, ge=1)
    fps: int = Field(default=24, ge=1)
    audio: bool = False
    imagePath: str | None = None
    audioPath: str | None = None
    aspectRatio: Literal["16:9", "9:16"] = "16:9"

    @model_validator(mode="after")
    def _validate_a2v_model(self) -> "GenerateVideoRequest":
        if self.audioPath is not None and self.model != "pro":
            raise ValueError("audioPath requires model='pro'")
        return self


class GenerateImageRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    prompt: NonEmptyPrompt
    width: int = Field(default=1024, ge=16)
    height: int = Field(default=1024, ge=16)
    numSteps: int = Field(default=4, ge=1)
    numImages: int = Field(default=1, ge=1)


def _default_model_types() -> set[ModelFileType]:
    return set()


class ModelDownloadRequest(BaseModel):
    modelTypes: set[ModelFileType] = Field(default_factory=_default_model_types)


class RequiredModelsResponse(BaseModel):
    modelTypes: list[ModelFileType]


GapPromptMode: TypeAlias = Literal["text-to-video", "image-to-video", "text-to-image"]


class SuggestGapPromptRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    beforePrompt: str = ""
    afterPrompt: str = ""
    beforeFrame: str | None = None
    afterFrame: str | None = None
    gapDuration: float = 5
    mode: GapPromptMode = "text-to-video"
    inputImage: str | None = None

    @model_validator(mode="after")
    def _validate_input_image_mode(self) -> "SuggestGapPromptRequest":
        if self.inputImage is not None and self.mode != "image-to-video":
            raise ValueError("inputImage is only valid for image-to-video mode")
        return self


RetakeMode: TypeAlias = Literal["replace_audio_and_video", "replace_video", "replace_audio"]


class RetakeRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    video_path: str
    start_time: float
    duration: float
    prompt: str = ""
    mode: RetakeMode = "replace_audio_and_video"


ConditioningType: TypeAlias = Literal["canny", "depth"]


class IcLoraExtractRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    video_path: str
    conditioning_type: ConditioningType = "canny"
    frame_time: float = 0


class IcLoraImageInput(BaseModel):
    model_config = ConfigDict(strict=True)

    path: str
    frame: int = 0
    strength: float = 1.0


def _default_ic_lora_images() -> list[IcLoraImageInput]:
    return []


class IcLoraGenerateRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    video_path: str
    conditioning_type: ConditioningType
    prompt: NonEmptyPrompt
    conditioning_strength: float = 1.0
    num_inference_steps: int = 30
    cfg_guidance_scale: float = 1.0
    negative_prompt: str = ""
    images: list[IcLoraImageInput] = Field(default_factory=_default_ic_lora_images)
