"""Pydantic request/response models and TypedDicts for ltx2_server."""

from __future__ import annotations

from typing import Literal, NamedTuple, TypeAlias, TypedDict
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

NonEmptyPrompt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ImageConditioningInput(NamedTuple):
    """Image conditioning triplet used by all video pipelines."""

    path: str
    frame_idx: int
    strength: float


# ============================================================
# TypedDicts for module-level state globals
# ============================================================


class GenerationState(TypedDict):
    id: str | None
    cancelled: bool
    result: str | list[str] | None
    error: str | None
    status: str  # "idle" | "running" | "complete" | "cancelled" | "error"
    phase: str
    progress: int
    current_step: int
    total_steps: int


class ModelDownloadState(TypedDict):
    status: str  # "idle" | "downloading" | "complete" | "error"
    current_file: str
    current_file_progress: int
    total_progress: int
    downloaded_bytes: int
    total_bytes: int
    files_completed: int
    total_files: int
    error: str | None
    speed_mbps: int


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
    status: str
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
    status: str
    phase: str
    progress: int
    currentStep: int | None
    totalSteps: int | None


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str


class ModelFileStatus(BaseModel):
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


class DownloadProgressResponse(BaseModel):
    status: str
    currentFile: str
    currentFileProgress: int
    totalProgress: int
    downloadedBytes: int
    totalBytes: int
    filesCompleted: int
    totalFiles: int
    error: str | None
    speedMbps: int


class IcLoraModel(BaseModel):
    name: str
    path: str
    conditioning_type: str
    reference_downscale_factor: int


class IcLoraListResponse(BaseModel):
    models: list[IcLoraModel]
    directory: str


class SuggestGapPromptResponse(BaseModel):
    status: str = "success"
    suggested_prompt: str


class GenerateVideoResponse(BaseModel):
    status: str
    video_path: str | None = None


class GenerateImageResponse(BaseModel):
    status: str
    image_paths: list[str] | None = None


class CancelResponse(BaseModel):
    status: str
    id: str | None = None


class RetakeResponse(BaseModel):
    status: str
    video_path: str | None = None
    result: JsonObject | None = None


class IcLoraExtractResponse(BaseModel):
    conditioning: str
    original: str
    conditioning_type: str
    frame_time: float


class IcLoraDownloadResponse(BaseModel):
    status: str
    path: str | None = None
    already_existed: bool | None = None
    already_exists: bool | None = None


class IcLoraGenerateResponse(BaseModel):
    status: str
    video_path: str | None = None


class ModelDownloadStartResponse(BaseModel):
    status: str
    message: str | None = None
    skippingTextEncoder: bool | None = None


class TextEncoderDownloadResponse(BaseModel):
    status: str
    message: str | None = None


class StatusResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    error: str
    message: str | None = None


class QueueJobResponse(BaseModel):
    id: str
    type: str
    model: str
    params: dict[str, object] = {}
    status: str
    slot: str
    progress: int
    phase: str
    result_paths: list[str] = []
    error: str | None = None
    created_at: str = ""
    batch_id: str | None = None
    batch_index: int = 0
    tags: list[str] = []


class QueueStatusResponse(BaseModel):
    jobs: list[QueueJobResponse]


class QueueSubmitResponse(BaseModel):
    id: str
    status: str


# ============================================================
# Gallery Models
# ============================================================

GalleryAssetType = Literal["image", "video"]


class GalleryAsset(BaseModel):
    id: str
    filename: str
    path: str
    url: str
    type: GalleryAssetType
    size_bytes: int
    created_at: str
    model_name: str | None = None


class GalleryListResponse(BaseModel):
    items: list[GalleryAsset]
    total: int
    page: int
    per_page: int
    total_pages: int


# ============================================================
# Request Models
# ============================================================


class GenerateVideoRequest(BaseModel):
    prompt: NonEmptyPrompt
    resolution: str = "512p"
    model: str = "fast"
    cameraMotion: VideoCameraMotion = "none"
    negativePrompt: str = ""
    duration: str = "2"
    fps: str = "24"
    audio: str = "false"
    imagePath: str | None = None
    audioPath: str | None = None
    lastFramePath: str | None = None
    aspectRatio: Literal["16:9", "9:16"] = "16:9"
    loraPath: str | None = None
    loraWeight: float = 1.0


class GenerateImageRequest(BaseModel):
    prompt: NonEmptyPrompt
    width: int = 1024
    height: int = 1024
    numSteps: int = 4
    numImages: int = 1
    loraPath: str | None = None
    loraWeight: float = 1.0
    sourceImagePath: str | None = None
    strength: float = 0.65


class QueueSubmitRequest(BaseModel):
    type: Literal["video", "image"]
    model: str
    params: dict[str, object] = {}


# --- Batch Generation Types ---


class BatchJobItem(BaseModel):
    type: Literal["video", "image"]
    model: str
    params: dict[str, object] = {}


class SweepAxis(BaseModel):
    param: str
    values: list[object]
    mode: Literal["replace", "search_replace"] = "replace"
    search: str | None = None


class SweepDefinition(BaseModel):
    base_type: Literal["video", "image"]
    base_model: str
    base_params: dict[str, object] = {}
    axes: list[SweepAxis]


class PipelineStep(BaseModel):
    type: Literal["video", "image"]
    model: str
    params: dict[str, object] = {}
    auto_prompt: bool = False


class PipelineDefinition(BaseModel):
    steps: list[PipelineStep]


class BatchSubmitRequest(BaseModel):
    mode: Literal["list", "sweep", "pipeline"]
    target: Literal["local", "cloud"]
    jobs: list[BatchJobItem] | None = None
    sweep: SweepDefinition | None = None
    pipeline: PipelineDefinition | None = None


class BatchSubmitResponse(BaseModel):
    batch_id: str
    job_ids: list[str]
    total_jobs: int


class BatchReport(BaseModel):
    batch_id: str
    total: int
    succeeded: int
    failed: int
    cancelled: int
    duration_seconds: float
    avg_job_seconds: float
    result_paths: list[str]
    failed_indices: list[int]
    sweep_axes: list[str] | None = None


class BatchStatusResponse(BaseModel):
    batch_id: str
    total: int
    completed: int
    failed: int
    running: int
    queued: int
    cancelled: int = 0
    jobs: list[QueueJobResponse]
    report: BatchReport | None = None


class ModelDownloadRequest(BaseModel):
    skipTextEncoder: bool = False


class SuggestGapPromptRequest(BaseModel):
    beforePrompt: str = ""
    afterPrompt: str = ""
    beforeFrame: str | None = None
    afterFrame: str | None = None
    gapDuration: float = 5
    mode: str = "t2v"
    inputImage: str | None = None


class RetakeRequest(BaseModel):
    video_path: str
    start_time: float
    duration: float
    prompt: str = ""
    mode: str = "replace_audio_and_video"


class IcLoraDownloadRequest(BaseModel):
    model: str


class IcLoraExtractRequest(BaseModel):
    video_path: str
    conditioning_type: str = "canny"
    frame_time: float = 0


class IcLoraImageInput(BaseModel):
    path: str
    frame: int = 0
    strength: float = 1.0


def _default_ic_lora_images() -> list[IcLoraImageInput]:
    return []


class IcLoraGenerateRequest(BaseModel):
    video_path: str
    lora_path: str
    conditioning_type: str = "canny"
    prompt: NonEmptyPrompt
    conditioning_strength: float = 1.0
    seed: int = 42
    height: int = 512
    width: int = 768
    num_frames: int = 121
    frame_rate: float = 24
    num_inference_steps: int = 30
    cfg_guidance_scale: float = 1.0
    negative_prompt: str = ""
    images: list[IcLoraImageInput] = Field(default_factory=_default_ic_lora_images)


# ============================================================
# Receive Job (from Palette)
# ============================================================


class ReceiveJobSettings(BaseModel):
    resolution: str = "512p"
    duration: str = "2"
    fps: str = "24"
    aspect_ratio: Literal["16:9", "9:16"] = "16:9"


class ReceiveJobRequest(BaseModel):
    prompt: NonEmptyPrompt
    model: str = "ltx-fast"
    settings: ReceiveJobSettings = Field(default_factory=ReceiveJobSettings)
    character_id: str | None = None
    first_frame_url: str | None = None
    last_frame_url: str | None = None
    priority: int = 0


class ReceiveJobResponse(BaseModel):
    id: str
    status: str


# ============================================================
# Contact Sheet
# ============================================================


class GenerateContactSheetRequest(BaseModel):
    reference_image_path: str
    subject_description: NonEmptyPrompt
    style: str = ""


class GenerateContactSheetResponse(BaseModel):
    job_ids: list[str]


# ============================================================
# Style Guide Grid
# ============================================================


class GenerateStyleGuideRequest(BaseModel):
    style_name: NonEmptyPrompt
    style_description: str = ""
    reference_image_path: str | None = None


class GenerateStyleGuideResponse(BaseModel):
    job_ids: list[str]


# ============================================================
# Library Models (Characters, Styles, References)
# ============================================================

LibraryReferenceCategory = Literal["people", "places", "props", "other"]


class CharacterCreate(BaseModel):
    name: str
    role: str = ""
    description: str = ""
    reference_image_paths: list[str] = Field(default_factory=list)


class CharacterUpdate(BaseModel):
    name: str | None = None
    role: str | None = None
    description: str | None = None
    reference_image_paths: list[str] | None = None


class CharacterResponse(BaseModel):
    id: str
    name: str
    role: str
    description: str
    reference_image_paths: list[str]
    created_at: str


class CharacterListResponse(BaseModel):
    characters: list[CharacterResponse]


class StyleCreate(BaseModel):
    name: str
    description: str = ""
    reference_image_path: str = ""


class StyleResponse(BaseModel):
    id: str
    name: str
    description: str
    reference_image_path: str
    created_at: str


class StyleListResponse(BaseModel):
    styles: list[StyleResponse]


class ReferenceCreate(BaseModel):
    name: str
    category: LibraryReferenceCategory
    image_path: str = ""


class ReferenceResponse(BaseModel):
    id: str
    name: str
    category: LibraryReferenceCategory
    image_path: str
    created_at: str


class ReferenceListResponse(BaseModel):
    references: list[ReferenceResponse]


# ============================================================
# Prompt Library Models
# ============================================================


class SavedPromptResponse(BaseModel):
    id: str
    text: str
    tags: list[str]
    category: str
    used_count: int
    created_at: str
    last_used_at: str | None


class PromptListResponse(BaseModel):
    prompts: list[SavedPromptResponse]


class SavePromptRequest(BaseModel):
    text: NonEmptyPrompt
    tags: list[str] = Field(default_factory=list)
    category: str = ""


class IncrementUsageResponse(BaseModel):
    status: str
    used_count: int


class WildcardResponse(BaseModel):
    id: str
    name: str
    values: list[str]
    created_at: str


class WildcardListResponse(BaseModel):
    wildcards: list[WildcardResponse]


class CreateWildcardRequest(BaseModel):
    name: str = Field(min_length=1)
    values: list[str] = Field(min_length=1)


class UpdateWildcardRequest(BaseModel):
    values: list[str] = Field(min_length=1)


class ExpandWildcardsRequest(BaseModel):
    prompt: NonEmptyPrompt
    mode: Literal["all", "random"] = "random"
    count: int = Field(default=1, ge=1, le=1000)


class ExpandWildcardsResponse(BaseModel):
    expanded: list[str]
