"""Pydantic request/response models and typed aliases for ltx2_server."""

from __future__ import annotations

from typing import Annotated
from datetime import datetime
from typing import Literal, NamedTuple, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyPrompt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ModelCheckpointID = Literal[
    "ltx-2.3-22b-distilled",
    "ltx-2.3-spatial-upscaler-x2-1.0",
    "ltx-2.3-22b-ic-lora-union-control-ref0.5",
    "dpt-hybrid-midas",
    "yolox-l-torchscript",
    "dw-ll-ucoco-384-bs5",
    "gemma-3-12b-it-qat-q4_0-unquantized",
    "z-image-turbo",
]
LTXLocalModelId = Literal["ltx-2.3-22b-distilled"]


class ImageConditioningInput(NamedTuple):
    """Image conditioning triplet used by all video pipelines."""

    path: str
    frame_idx: int
    strength: float


JsonObject: TypeAlias = dict[str, object]
MediaType: TypeAlias = Literal["image", "audio", "video"]
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


class ServerCapabilities(BaseModel):
    media_ids: Literal[True] = True
    artifact_downloads: Literal[True] = True
    legacy_path_inputs: bool
    models_dir_editable: bool


class ServerInfoResponse(BaseModel):
    api_version: Literal[2] = 2
    deployment_mode: Literal["managed_local", "standalone"]
    capabilities: ServerCapabilities


class MediaRef(BaseModel):
    media_id: str
    media_type: MediaType
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    expires_at: datetime


class ArtifactRef(BaseModel):
    artifact_id: str
    media_type: MediaType
    filename: str
    content_type: str
    size_bytes: int
    sha256: str
    expires_at: datetime


class GenerationProgressResponse(BaseModel):
    status: Literal["idle", "running", "complete", "cancelled", "error"]
    phase: str
    progress: int
    currentStep: int | None
    totalSteps: int | None
    generationId: str | None = None
    artifact: ArtifactRef | None = None
    artifacts: list[ArtifactRef] = Field(default_factory=lambda: list[ArtifactRef]())
    error: str | None = None


class DownloadProgressRunningResponse(BaseModel):
    status: Literal["downloading"]
    current_downloading_file: ModelCheckpointID | None
    current_file_progress: float
    total_progress: float
    total_downloaded_bytes: int
    expected_total_bytes: int
    completed_files: set[ModelCheckpointID]
    all_files: set[ModelCheckpointID]
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
    artifact: ArtifactRef


class GenerateVideoCancelledResponse(BaseModel):
    status: Literal["cancelled"]


GenerateVideoResponse: TypeAlias = GenerateVideoCompleteResponse | GenerateVideoCancelledResponse


class GenerateImageCompleteResponse(BaseModel):
    status: Literal["complete"]
    image_paths: list[str]
    artifacts: list[ArtifactRef]


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
    artifact: ArtifactRef


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
    artifact: ArtifactRef


class IcLoraGenerateCancelledResponse(BaseModel):
    status: Literal["cancelled"]


IcLoraGenerateResponse: TypeAlias = IcLoraGenerateCompleteResponse | IcLoraGenerateCancelledResponse


# ============================================================
# HuggingFace auth
# ============================================================


class HuggingFaceLoginResponse(BaseModel):
    client_id: str
    redirect_uri: str
    scope: str
    state: str
    code_challenge: str
    code_challenge_method: str


class HuggingFaceAuthStatusResponse(BaseModel):
    status: Literal["authenticated", "pending", "not_authenticated"]


class HuggingFaceLogoutResponse(BaseModel):
    status: Literal["logged_out"]


class ModelDownloadStartResponse(BaseModel):
    status: Literal["started"]
    message: str
    sessionId: str


class LtxDownloadRecommendationResponse(BaseModel):
    status: Literal["download"]
    cps_to_download: list[ModelCheckpointID]


class LtxUpgradeRecommendationResponse(BaseModel):
    status: Literal["upgrade"]
    ltx_model_id: LTXLocalModelId
    upgrade_message: str | None = None
    cps_to_download: list[ModelCheckpointID]
    cps_to_delete: list[ModelCheckpointID]


class LtxOkRecommendationResponse(BaseModel):
    status: Literal["ok"]


LtxRecommendationResponse: TypeAlias = (
    LtxDownloadRecommendationResponse | LtxUpgradeRecommendationResponse | LtxOkRecommendationResponse
)


class ImageGenRecommendationResponse(BaseModel):
    cp_to_download: ModelCheckpointID | None


class LtxIcLoraRecommendationResponse(BaseModel):
    cps_to_download: list[ModelCheckpointID]


class TextEncoderRecommendationResponse(BaseModel):
    cp_to_download: ModelCheckpointID | None
    expected_size_bytes: int
    expected_size_gb: float


class StatusResponse(BaseModel):
    status: str


class HTTPErrorResponse(BaseModel):
    code: str
    message: str


class LtxInsufficientFundsErrorResponse(BaseModel):
    code: Literal["LTX_INSUFFICIENT_FUNDS"]
    message: str


# ============================================================
# Request Models
# ============================================================


LTXVideoGenResolution: TypeAlias = Literal["540p", "720p", "1080p", "1440p", "2160p"]
LTXVideoGenDuration: TypeAlias = Literal[5, 6, 8, 10, 12, 14, 16, 18, 20]
LTXVideoGenFps: TypeAlias = Literal[24, 25, 48, 50]
LTXVideoGenPipeline: TypeAlias = Literal["fast", "pro"]


class LTXVideoGenerationResolutionSpec(BaseModel):
    fps_to_durations: dict[LTXVideoGenFps, list[LTXVideoGenDuration]]


class LTXVideoGenerationSpec(BaseModel):
    display_name: str
    supported_resolutions_durations: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec]
    a2v_supported_resolutions_durations: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] | None = None


class LTXVideoGenerationModelSpecItem(BaseModel):
    pipeline: LTXVideoGenPipeline
    spec: LTXVideoGenerationSpec


class GenerateVideoModelsSpecsResponse(BaseModel):
    local_models: list[LTXVideoGenerationModelSpecItem]
    api_models: list[LTXVideoGenerationModelSpecItem]


class GenerateVideoRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    prompt: NonEmptyPrompt
    generationId: str | None = Field(default=None, min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    resolution: LTXVideoGenResolution = "1080p"
    model: LTXVideoGenPipeline = "fast"
    cameraMotion: VideoCameraMotion = "none"
    negativePrompt: str = ""
    duration: LTXVideoGenDuration = 5
    fps: LTXVideoGenFps = 24
    audio: bool = False
    imagePath: str | None = None
    audioPath: str | None = None
    imageMediaId: str | None = None
    audioMediaId: str | None = None
    aspectRatio: Literal["16:9", "9:16"] = "16:9"

    @model_validator(mode="after")
    def _validate_media_refs(self) -> "GenerateVideoRequest":
        if self.imagePath and self.imageMediaId:
            raise ValueError("imagePath and imageMediaId are mutually exclusive")
        if self.audioPath and self.audioMediaId:
            raise ValueError("audioPath and audioMediaId are mutually exclusive")
        return self


class GenerateImageRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    prompt: NonEmptyPrompt
    generationId: str | None = Field(default=None, min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    width: int = Field(default=1024, ge=16)
    height: int = Field(default=1024, ge=16)
    numSteps: int = Field(default=4, ge=1)
    numImages: int = Field(default=1, ge=1)


def _default_model_types() -> set[ModelCheckpointID]:
    return set()


class ModelDownloadRequest(BaseModel):
    type: Literal["download", "upgrade"] = "download"
    cp_ids: set[ModelCheckpointID] = Field(default_factory=_default_model_types)


ModelAccessStatus: TypeAlias = Literal["authorized", "not_authorized"]


class CheckModelAccessRequest(BaseModel):
    cp_ids: set[ModelCheckpointID] = Field(default_factory=_default_model_types)


class CheckModelAccessResponse(BaseModel):
    access: dict[str, ModelAccessStatus]


class ModelDeleteRequest(BaseModel):
    cp_ids: set[ModelCheckpointID] = Field(default_factory=_default_model_types)


GapPromptMode: TypeAlias = Literal["text-to-video", "image-to-video", "text-to-image"]


class SuggestGapPromptRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    beforePrompt: str = ""
    afterPrompt: str = ""
    beforeFrame: str | None = None
    afterFrame: str | None = None
    beforeFrameMediaId: str | None = None
    afterFrameMediaId: str | None = None
    gapDuration: float = 5
    mode: GapPromptMode = "text-to-video"
    inputImage: str | None = None
    inputImageMediaId: str | None = None

    @model_validator(mode="after")
    def _validate_input_image_mode(self) -> "SuggestGapPromptRequest":
        if (self.inputImage is not None or self.inputImageMediaId is not None) and self.mode != "image-to-video":
            raise ValueError("inputImage is only valid for image-to-video mode")
        for path_value, id_value, name in (
            (self.beforeFrame, self.beforeFrameMediaId, "beforeFrame"),
            (self.afterFrame, self.afterFrameMediaId, "afterFrame"),
            (self.inputImage, self.inputImageMediaId, "inputImage"),
        ):
            if path_value and id_value:
                raise ValueError(f"{name} path and media ID are mutually exclusive")
        return self


RetakeMode: TypeAlias = Literal["replace_audio_and_video", "replace_video", "replace_audio"]


class RetakeRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    video_path: str | None = None
    video_media_id: str | None = None
    start_time: float
    duration: float
    prompt: str = ""
    mode: RetakeMode = "replace_audio_and_video"

    @model_validator(mode="after")
    def _validate_video_ref(self) -> "RetakeRequest":
        if bool(self.video_path) == bool(self.video_media_id):
            raise ValueError("exactly one of video_path or video_media_id is required")
        return self


ConditioningType: TypeAlias = Literal["canny", "depth"]


class IcLoraExtractRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    video_path: str | None = None
    video_media_id: str | None = None
    conditioning_type: ConditioningType = "canny"
    frame_time: float = 0

    @model_validator(mode="after")
    def _validate_video_ref(self) -> "IcLoraExtractRequest":
        if bool(self.video_path) == bool(self.video_media_id):
            raise ValueError("exactly one of video_path or video_media_id is required")
        return self


class IcLoraImageInput(BaseModel):
    model_config = ConfigDict(strict=True)

    path: str | None = None
    media_id: str | None = None
    frame: int = 0
    strength: float = 1.0

    @model_validator(mode="after")
    def _validate_image_ref(self) -> "IcLoraImageInput":
        if bool(self.path) == bool(self.media_id):
            raise ValueError("exactly one of path or media_id is required")
        return self


def _default_ic_lora_images() -> list[IcLoraImageInput]:
    return []


class IcLoraGenerateRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    video_path: str | None = None
    video_media_id: str | None = None
    conditioning_type: ConditioningType
    prompt: NonEmptyPrompt
    conditioning_strength: float = 1.0
    num_inference_steps: int = 30
    cfg_guidance_scale: float = 1.0
    negative_prompt: str = ""
    images: list[IcLoraImageInput] = Field(default_factory=_default_ic_lora_images)

    @model_validator(mode="after")
    def _validate_video_ref(self) -> "IcLoraGenerateRequest":
        if bool(self.video_path) == bool(self.video_media_id):
            raise ValueError("exactly one of video_path or video_media_id is required")
        return self
