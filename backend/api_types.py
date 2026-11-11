"""Pydantic request/response models and typed aliases for ltx2_server."""

from __future__ import annotations

import re
from typing import Annotated, Literal, NamedTuple, Self, TypeAlias, cast, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
    Tag,
)
from pydantic.json_schema import SkipJsonSchema

from api_motion_path import MOTION_PATH_MAX_PATHS, MotionPath
from api_types_lora import (
    CUSTOM_LORA_DESCRIPTION_MAX as CUSTOM_LORA_DESCRIPTION_MAX,
    CUSTOM_LORA_EXAMPLE_MAX as CUSTOM_LORA_EXAMPLE_MAX,
    CUSTOM_LORA_EXAMPLES_MAX as CUSTOM_LORA_EXAMPLES_MAX,
    CUSTOM_LORA_TRIGGER_MAX as CUSTOM_LORA_TRIGGER_MAX,
    ActiveCustomLoraImportResponse as ActiveCustomLoraImportResponse,
    ActiveIcLoraDownloadResponse as ActiveIcLoraDownloadResponse,
    ActiveLoraDownloadResponse as ActiveLoraDownloadResponse,
    CatalogDownloadStartResponse as CatalogDownloadStartResponse,
    CatalogDownloadStatus as CatalogDownloadStatus,
    CustomLoraDescription as CustomLoraDescription,
    CustomLoraExample as CustomLoraExample,
    CustomLoraId as CustomLoraId,
    CustomLoraImportProgressResponse as CustomLoraImportProgressResponse,
    CustomLoraImportRequest as CustomLoraImportRequest,
    CustomLoraImportStartResponse as CustomLoraImportStartResponse,
    CustomLoraItem as CustomLoraItem,
    CustomLoraLinkInfoRequest as CustomLoraLinkInfoRequest,
    CustomLoraLinkInfoResponse as CustomLoraLinkInfoResponse,
    CustomLoraListResponse as CustomLoraListResponse,
    CustomLoraRestoreRequest as CustomLoraRestoreRequest,
    CustomLoraSource as CustomLoraSource,
    CustomLoraTrigger as CustomLoraTrigger,
    IcLoraDeleteRequest as IcLoraDeleteRequest,
    IcLoraDownloadProgressResponse as IcLoraDownloadProgressResponse,
    IcLoraDownloadRequest as IcLoraDownloadRequest,
    LoraDeleteRequest as LoraDeleteRequest,
    LoraDownloadProgressResponse as LoraDownloadProgressResponse,
    LoraDownloadRequest as LoraDownloadRequest,
    LoraEntry as LoraEntry,
)
from assistant_limits import ASSISTANT_LIMITS
from frame_math import compute_num_frames
from runtime_config.eta import EtaBasis, EtaFeature
from runtime_config.reject_reasons import (
    BudgetMode,
    IcLoraDroppedReason,
    JobRiskTier,
    PassShape,
    ShortfallKind,
    UnmeasuredBasis,
    VideoJobRejectReason,
    WarnedCellTier,
)
from runtime_config.ic_lora_stage_mode import IcLoraStageMode, ic_lora_stage_mode
from runtime_config.ic_lora_tiling import (
    MAX_TILE_OVERLAP,
    MIN_CARRY_FRAMES,
    MIN_TILE_OVERLAP,
    MIN_WINDOW_FRAMES,
    IcLoraTileSize,
    IcLoraTiling,
)
from runtime_config.nag_limits import (
    MAX_ALPHA as NAG_MAX_ALPHA,
    MAX_SCALE as NAG_MAX_SCALE,
    MAX_TAU as NAG_MAX_TAU,
    NEGATIVE_PROMPT_MAX_LENGTH,
)

NonEmptyPrompt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
LocalAssistantModel = Literal["qwen3.5:9b"]


class NagParams(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    negativePrompt: str = Field(default="", max_length=NEGATIVE_PROMPT_MAX_LENGTH)
    nagScale: float | None = Field(default=None, ge=0.0, le=NAG_MAX_SCALE, allow_inf_nan=False)
    nagAlpha: float | None = Field(default=None, ge=0.0, le=NAG_MAX_ALPHA, allow_inf_nan=False)
    nagTau: float | None = Field(default=None, ge=0.0, le=NAG_MAX_TAU, allow_inf_nan=False)



class AssistantModelDownloadRequest(BaseModel):
    model: LocalAssistantModel


class AssistantModelItem(BaseModel):
    model: LocalAssistantModel
    name: str
    size_bytes: int
    installed: bool


class AssistantModelDownloadProgress(BaseModel):
    model: LocalAssistantModel
    status: Literal["downloading", "complete", "error"]
    downloaded_bytes: int = 0
    expected_bytes: int = 0
    progress: float = 0
    error: str | None = None


class AssistantModelsResponse(BaseModel):
    available: bool
    models: list[AssistantModelItem]
    download: AssistantModelDownloadProgress | None = None
    error: str | None = None


ModelCheckpointID = Literal[
    "ltx-2.3-22b-distilled",
    "ltx-2.3-22b-distilled-1.1",
    "ltx-2.3-spatial-upscaler-x2-1.0",
    "ltx-2.3-spatial-upscaler-x2-1.1",
    "ltx-2.3-22b-ic-lora-union-control-ref0.5",
    "ltx-2.5-22b-distilled",
    "ltx-2.5-spatial-upscaler-x2-1.0",
    "ltx-2.5-video-vae",
    "ltx-2.5-video-vae-conv",
    "ltx-2.5-audio-vae",
    "ltx-2.5-duration-head",
    "dpt-hybrid-midas",
    "yolox-l-torchscript",
    "dw-ll-ucoco-384-bs5",
    "gemma-3-12b-it-qat-q4_0-unquantized",
    "gemma4-12b-with-proj-ltx-2.5",
    "gemma-4-e2b-it",
    "z-image-turbo",
]
LTXLocalModelId = Literal[
    "ltx-2.5-22b-distilled",
    "ltx-2.3-22b-distilled-1.1",
    "ltx-2.3-22b-distilled",
]
# T2V/I2V/A2V/LoRA create-body `params.model`. Not a GenSpace pipeline (`fast`)
# and not a checkpoint id. Future API rows (`ltx-2.5-pro`, …) join this union.
OfferingId = Literal["ltx-2.3-fast", "ltx-2.5-fast"]


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


# Wire mirror of DriveType in services.storage_info.storage_info (import would cycle
# back through services/__init__ into this partially-initialized module).
CheckpointDriveType = Literal["nvme", "ssd", "hdd"]


class GpuInfoResponse(BaseModel):
    cuda_available: bool
    mps_available: bool = False
    gpu_available: bool = False
    gpu_name: str | None
    vram_gb: int | None
    gpu_info: GpuTelemetry
    # Raw capacities the renderer cannot observe itself; read once per generation
    # for telemetry. All None-able: probes fail soft off-CUDA and in containers.
    total_vram_bytes: int | None = None
    free_vram_bytes: int | None = None
    total_ram_bytes: int | None = None
    system: str | None = None
    checkpoint_drive_type: CheckpointDriveType | None = None
    # MPS allocator pool (recommended_max) in bytes. Gates the Darwin Fast
    # profile; None off-MPS or when the probe fails.
    mps_pool_bytes: int | None = None


class MpsMemoryResponse(BaseModel):
    """Read-only Apple Silicon MPS memory snapshot, MiB. Fields are None off MPS."""

    available: bool
    allocated_mib: int | None = None
    driver_mib: int | None = None
    recommended_max_mib: int | None = None


class RuntimePolicyResponse(BaseModel):
    force_api_generations: bool
    local_viable: bool


class FeatureFlags(BaseModel):
    """Dev feature flags, persisted in feature_flags.json. Adding a flag = one field here."""

    customIcLora: bool = False
    advancedIcLoraControls: bool = False
    # Offer 1440p and 4K in the video-to-video flows: Retake, Extend and the IC-LoRA
    # recipes. Text, image and audio to video offer both without a flag. The job budget
    # still gates every job, and a cell with no measured run (measured_cells.py) is
    # refused in Standard mode. The flag only adds the cell. Off by default.
    highResVideoToVideo: bool = False
    # Show "Create with AI" next to the start frame, end frame and reference image fields.
    # It opens the image generator and puts the image in the field. Off by default.
    createWithAi: bool = False
    # Show a "Negative prompt:" line under each result's meta bar, for every feature
    # that stores a negative prompt. For comparing runs with and without one. Off by default.
    showGenerationNegativePrompt: bool = False
    # Offer the Keyframes feature on Home. The backend refuses the create route while it is
    # off, and Home hides the feature. GenSpace multi-keyframe has no flag. Off by default.
    keyframes: bool = False


class UiPreferences(BaseModel):
    """The user settings that change what a form shows. Read-only, so Remote may read them."""

    advancedControls: bool = False


class FeatureFlagsPatch(BaseModel):
    """Partial update: only the fields present in the body are applied."""

    model_config = ConfigDict(strict=True, extra="forbid")

    customIcLora: bool | None = None
    advancedIcLoraControls: bool | None = None
    highResVideoToVideo: bool | None = None
    createWithAi: bool | None = None
    showGenerationNegativePrompt: bool | None = None
    keyframes: bool | None = None


class GenerationProgressResponse(BaseModel):
    status: Literal["idle", "running", "complete", "cancelled", "error"]
    phase: str
    progress: int
    currentStep: int | None
    totalSteps: int | None
    result: str | list[str] | None = None
    # Identifies which generation this snapshot belongs to (None only for "idle" — nothing has
    # run yet). Lets a polling client confirm it's still looking at its own generation rather
    # than a different, unrelated one that reused the single global progress slot in the
    # meantime.
    id: str | None = None
    # Same poll that disables Generate (`status == "running"`). Stop is allowed only for a
    # local GPU slot — reservation, denoise, or cancelled-in-flight unwind. LTX/FAL API jobs
    # occupy the slot (Generate stays locked) but have no public cancel.
    cancellable: bool
    # Whole-run wall-time estimate fixed at start (ETA v1: local Fast video on
    # measured hardware only). None means unknown, not zero. Shared with the
    # pre-generate estimate endpoint so both surfaces speak one contract.
    estimate: WallTimeEstimateResponse | None = None
    remainingSeconds: int | None = None


class WallTimeEstimateResponse(BaseModel):
    seconds: int
    basis: EtaBasis
    t2vReuse: bool


class EstimateVideoRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    resolution: LTXVideoGenResolution = "1080p"
    # Full local ratio list, like generate: the estimate budgets the picked
    # shape (anchor + ultrawide bump), so a narrower Literal would 422 jobs
    # the picker offers.
    aspectRatio: LTXVideoGenAspectRatio = "16:9"
    # None = automatic duration: length unknown until generation, so the
    # estimate is always null (valid selection, unknown time — not an error).
    duration: LTXVideoGenDuration | None = 5
    fps: LTXVideoGenFps = 24
    feature: EtaFeature = "t2v"


class EstimateVideoResponse(BaseModel):
    estimate: WallTimeEstimateResponse | None = None
    risk: GenerationRisk | None = None


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
    DownloadProgressRunningResponse
    | DownloadProgressCompleteResponse
    | DownloadProgressErrorResponse
)


class SuggestGapPromptResponse(BaseModel):
    status: Literal["success"] = "success"
    suggested_prompt: str


# Wire mirror of LoadMode in runtime_config.video_job_budget (import would cycle
# back through services/__init__ into this partially-initialized module).
VideoLoadMode = Literal["full_models_loading", "streaming_models_loading"]
# Wire mirror of ltx_pipelines OffloadMode member names (import would drag heavy
# deps into the API boundary). NONE = resident, CPU = pinned-host streaming,
# DISK = mapped-from-checkpoint streaming.
PipelineOffloadMode = Literal["NONE", "CPU", "DISK"]


class GenerateVideoCompleteResponse(BaseModel):
    status: Literal["complete"]
    video_path: str
    # Local route only: the decided load mode and the offload the routing
    # selects for it. None on the API route, which runs nothing locally.
    load_mode: VideoLoadMode | None = None
    offload_mode: PipelineOffloadMode | None = None


class GenerateVideoCancelledResponse(BaseModel):
    status: Literal["cancelled"]
    load_mode: VideoLoadMode | None = None
    offload_mode: PipelineOffloadMode | None = None


GenerateVideoResponse: TypeAlias = (
    GenerateVideoCompleteResponse | GenerateVideoCancelledResponse
)


class GenerateImageCompleteResponse(BaseModel):
    status: Literal["complete"]
    image_paths: list[str]


class GenerateImageCancelledResponse(BaseModel):
    status: Literal["cancelled"]


GenerateImageResponse: TypeAlias = (
    GenerateImageCompleteResponse | GenerateImageCancelledResponse
)


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


RetakeResponse: TypeAlias = (
    RetakeVideoResponse | RetakePayloadResponse | RetakeCancelledResponse
)


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


IcLoraGenerateResponse: TypeAlias = (
    IcLoraGenerateCompleteResponse | IcLoraGenerateCancelledResponse
)


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


class ActiveDownloadResponse(BaseModel):
    # The currently-running download session (at most one); session_id is null and cp_ids
    # empty when idle. Lets a UI that remounts mid-download (e.g. a reopened settings modal)
    # reattach to it.
    session_id: str | None
    cp_ids: list[ModelCheckpointID]


class LtxDownloadRecommendationResponse(BaseModel):
    status: Literal["download"]
    cps_to_download: list[ModelCheckpointID]
    # Checkpoints left out of cps_to_download only because an LTX API key covers what they do.
    # Offered as an opt-in so a user who wants to generate offline can take the download now
    # instead of discovering later that the key is the only thing making generation work.
    optional_cp_ids: list[ModelCheckpointID] = []
    # Quality extras recommended for the bundle being installed (currently 2.5's preferred
    # Gemma 4 E2B enhancer). Distinct from required and from API-excused optional_cp_ids.
    # Missing or skipped items must not block app entry or generation.
    recommended_quality_cp_ids: list[ModelCheckpointID] = []


class LtxUpgradeRecommendationResponse(BaseModel):
    status: Literal["upgrade"]
    ltx_model_id: LTXLocalModelId
    upgrade_message: str | None = None
    cps_to_download: list[ModelCheckpointID]
    cps_to_delete: list[ModelCheckpointID]
    # True when deleting the old bundle would also remove built-in Union Control IC-LoRA
    # that the target model does not share. The upgrade UI defaults "delete old" off so
    # users don't silently lose the easy path back to depth/canny/pose control.
    loses_built_in_control: bool = False
    recommended_quality_cp_ids: list[ModelCheckpointID] = []


class LtxOkRecommendationResponse(BaseModel):
    status: Literal["ok"]


LtxRecommendationResponse: TypeAlias = (
    LtxDownloadRecommendationResponse
    | LtxUpgradeRecommendationResponse
    | LtxOkRecommendationResponse
)


class ImageGenRecommendationResponse(BaseModel):
    cp_to_download: ModelCheckpointID | None


class LtxIcLoraRecommendationResponse(BaseModel):
    cps_to_download: list[ModelCheckpointID]
    # False when the active model has no built-in Union Control IC-LoRA: distinct from
    # "supported but already downloaded" (both have empty cps_to_download).
    supported: bool = True


class TextEncoderRecommendationResponse(BaseModel):
    cp_to_download: ModelCheckpointID | None
    expected_size_bytes: int
    expected_size_gb: float
    # False when the active model can't be encoded by the LTX API, making the local encoder
    # mandatory instead of one of two interchangeable options.
    api_encoding_supported: bool
    ltx_version_label: str
    # False when no generative checkpoint local Enhance can run is downloaded, so Enhance
    # needs Gemini. True if the preferred enhancer *or* a fallback (Gemma 3 on 2.5) is present.
    local_enhancement_supported: bool
    # The separate generative checkpoint local Enhance prefers, when the encoder can only encode
    # (2.5). None when the encoder enhances too (2.3), so there's no extra download to offer.
    local_enhancer_cp: ModelCheckpointID | None
    local_enhancer_expected_size_gb: float | None
    # The checkpoint Enhance will actually load, after E2B-then-Gemma-3 fallback. None if local
    # Enhance cannot run. Equal to local_enhancer_cp when the preferred extra model is present.
    active_local_enhancer_cp: ModelCheckpointID | None
    # Installed local text encoder for the active LTX model (for delete in Settings). None if missing.
    active_local_text_encoder_cp: ModelCheckpointID | None = None
    # Whether that encoder may actually be deleted right now. False while local encoding is the
    # only way to encode for this model (no API encoding support, no API key, or the user chose
    # the local encoder), in which case /api/models/delete protects it and Settings disables the
    # action instead of offering a request that is guaranteed to 409.
    local_text_encoder_removable: bool = False


class LtxModelVersionItem(BaseModel):
    model_id: LTXLocalModelId
    label: str
    model_cp: ModelCheckpointID
    size_bytes: int
    installed: bool
    active: bool
    is_newest: bool
    cps_to_download: list[ModelCheckpointID]


class LtxModelVersionsResponse(BaseModel):
    versions: list[LtxModelVersionItem]


class SetActiveLtxModelRequest(BaseModel):
    model_id: LTXLocalModelId


CheckpointRole = Literal["base", "upscaler", "text_encoder", "vae", "image", "support", "prompt_enhancer"]


class CheckpointDescriptor(BaseModel):
    cp_id: ModelCheckpointID
    name: str
    # User-facing explanatory copy lives in the frontend keyed off `role` (so wording/i18n
    # iterates without a backend deploy); the backend only ships the stable role enum.
    role: CheckpointRole
    size_bytes: int
    downloaded: bool


class DescribeCheckpointsResponse(BaseModel):
    checkpoints: list[CheckpointDescriptor]


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


LTXVideoGenResolution: TypeAlias = Literal[
    "270p", "360p", "540p", "720p", "1080p", "1440p", "2160p"
]
LTXLocalA2VResolution: TypeAlias = Literal[
    "270p", "360p", "540p", "720p", "1080p", "1440p", "2160p"
]
LTXVideoGenDuration: TypeAlias = Literal[2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20]
LTXVideoGenFps: TypeAlias = Literal[24, 25, 48, 50]
LTXVideoGenPipeline: TypeAlias = Literal["fast", "pro", "fast-2.5", "pro-2.5"]
# Widest to tallest. A cell advertises a subset of this; 2:3 is intentionally absent.
LTXVideoGenAspectRatio: TypeAlias = Literal["21:9", "16:9", "3:2", "4:3", "1:1", "4:5", "9:16"]
LTXQueuedAspectRatio: TypeAlias = Literal["auto", "21:9", "16:9", "3:2", "4:3", "1:1", "4:5", "9:16"]


class LTXVideoGenerationResolutionSpec(BaseModel):
    fps_to_durations: dict[LTXVideoGenFps, list[LTXVideoGenDuration]]
    aspect_ratios: list[LTXVideoGenAspectRatio]


class LTXOfferingCapabilitiesSpec(BaseModel):
    """Feature flags for one local model or API pipeline. Pixel maps stay backend-only."""

    t2v: bool
    i2v: bool
    a2v: bool
    ic_lora: bool
    retake: bool
    extend: bool
    multi_keyframe: bool
    multi_keyframe_max_count: int
    user_loras: bool
    camera_motion: bool
    auto_duration: bool


# Specs surface budget-layer reasons; one vocabulary, no parallel Literal.
LocalUnavailableReason = VideoJobRejectReason

BudgetReadoutKind = Literal["gpu", "gpu_busy", "mac", "ram", "tight", "untested"]


class BudgetReadout(BaseModel):
    """Need and have for a picker badge. The sentence stays on ``message``."""

    model_config = ConfigDict(strict=True, extra="forbid")

    kind: BudgetReadoutKind
    need_gb: float | None = None
    have_gb: float | None = None


class WarnedFastCell(BaseModel):
    """One kept grid cell carrying a warning tier: a tight fit, or a Brave bet
    (over budget or untested) the echoed mode admits.

    Kept + warned render the full grid for warned-with-tooltip pickers:
    ``fps_to_durations`` stays the kept set, this list marks the risky subset.
    ``headroom_gib`` is set on tight cells only.
    ``message`` is backend-composed, same builder as the create annotation, so
    the picker tooltip and the queue badge read identically.
    """

    resolution: LTXVideoGenResolution
    fps: LTXVideoGenFps
    duration: LTXVideoGenDuration
    tier: WarnedCellTier
    headroom_gib: float | None = None
    message: str
    readout: BudgetReadout | None = None
    audio: bool


class WarnedFastAspectCell(BaseModel):
    """One kept (cell, wider shape) pair carrying a warning tier. Same
    contract as WarnedFastCell; aspect pickers read this, the
    resolution/fps/duration pickers never do."""

    resolution: LTXVideoGenResolution
    fps: LTXVideoGenFps
    duration: LTXVideoGenDuration
    aspectRatio: LTXVideoGenAspectRatio
    tier: WarnedCellTier
    headroom_gib: float | None = None
    message: str
    readout: BudgetReadout | None = None
    audio: bool


class ChunkedFastCell(BaseModel):
    """One kept cell the local budget runs in overlapping windows, not one pass.

    The picker tells the user it takes longer. ``windows`` is the job's window count and
    ``window_frames`` the pixel frames of one window. A cell that loads in a single pass never
    appears. Same ``audio`` split as the warned cells (A2V rides its own grid).
    """

    resolution: LTXVideoGenResolution
    fps: LTXVideoGenFps
    duration: LTXVideoGenDuration
    windows: int
    window_frames: int
    audio: bool


class DroppedFastCell(BaseModel):
    """One static grid cell the budget withheld, with its refusal to show.

    Kept + dropped reconstruct the full grid for disabled-with-tooltip
    pickers. ``message`` is backend-composed (same builder as the 422s), so
    list-time and request-time refusals read identically. ``audio`` selects
    the A2V matrix, whose durations differ from the plain one.
    """

    resolution: LTXVideoGenResolution
    fps: LTXVideoGenFps
    duration: LTXVideoGenDuration
    reason: LocalUnavailableReason
    message: str
    audio: bool
    over_gib: float | None = None
    over_ram_gb: int | None = None
    shortfall_kind: ShortfallKind | None = None
    readout: BudgetReadout | None = None
    unmeasured_basis: UnmeasuredBasis | None = None
    # Set when the run this cell would get (a Brave run, or an unmeasured bet) is in
    # overlapping windows: the window count and the pixel frames of one window. A picker that
    # offers the cell anyway tells the user, as it does for a kept ``ChunkedFastCell``.
    windows: int | None = None
    window_frames: int | None = None


class DroppedFastAspectCell(BaseModel):
    """One anchor-kept cell a wider shape cannot run, with its refusal.

    The grid (``fps_to_durations`` + ``dropped_cells``) is priced at the 16:9
    anchor: a kept duration runs at 16:9. Only a wider frame (21:9 today) can
    exceed the anchor, so these rows name exactly those (cell, shape) pairs —
    same refusal Generate would return — for disabled-with-tooltip aspect
    pickers. Never emitted for the anchor itself or for an anchor-dropped cell.
    """

    resolution: LTXVideoGenResolution
    fps: LTXVideoGenFps
    duration: LTXVideoGenDuration
    aspectRatio: LTXVideoGenAspectRatio
    reason: LocalUnavailableReason
    message: str
    audio: bool
    over_gib: float | None = None
    over_ram_gb: int | None = None
    shortfall_kind: ShortfallKind | None = None
    readout: BudgetReadout | None = None
    unmeasured_basis: UnmeasuredBasis | None = None


MemoryConfidence = Literal["solid", "estimate", "rough"]


class GenerationRisk(BaseModel):
    """Create-time snapshot of the bet a Brave job took.

    Stored privately as spec["_risk"] (no store migration), promoted to this
    typed field on every read. Drives the queue badge and the failure backstop
    copy. An `over` tier also waives a later memory miss at execute. An
    `unmeasured` tier does not. `over_gib`/`over_ram_gb` mirror the dropped-
    cell figure semantics; `message` is the snapshotted tier line.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    tier: JobRiskTier
    shortfall_kind: ShortfallKind | None = None
    over_gib: float | None = None
    over_ram_gb: int | None = None
    readout: BudgetReadout | None = None
    message: str
    unmeasured_basis: UnmeasuredBasis | None = None


class JobMemoryAsk(BaseModel):
    """What one job asks of the memory the gate compares.

    ``ask_gib`` is the figure admission compares against ``MemoryCapacity.capacity_gib``, rounded
    up to 0.001 GiB, so a meter drawn from it never shows an over job as fitting. ``windows`` is
    1 for a single pass. ``tight`` is the gate's own tight-fit flag. The band is sent on
    ``estimate`` asks only: one ask per Fast grid point and aspect makes this the bulk of the
    specs payload. ``risk`` is sent on ``FastCellAsk.alternative`` when the alternative pass
    shape is an ``over`` or ``unmeasured`` bet.
    """

    ask_gib: float
    confidence: MemoryConfidence
    windows: int = 1
    tight: bool = False
    band_low_gib: float | None = Field(default=None, exclude_if=lambda value: value is None)
    band_high_gib: float | None = Field(default=None, exclude_if=lambda value: value is None)
    risk: GenerationRisk | None = Field(default=None, exclude_if=lambda value: value is None)
    estimated_seconds: int | None = Field(default=None, exclude_if=lambda value: value is None)


class FastCellAsk(JobMemoryAsk):
    """One Fast grid cell's ask, kept or withheld.

    The ask is the shape the budget picks (``windows`` 1 is one pass). ``alternative`` is the
    other shape's ask, sent only where a Brave form offers the choice, so a shape the form
    shows is always one create runs.
    """

    resolution: LTXVideoGenResolution
    fps: LTXVideoGenFps
    duration: LTXVideoGenDuration
    audio: bool
    alternative: JobMemoryAsk | None = Field(default=None, exclude_if=lambda value: value is None)


class MemoryCapacity(BaseModel):
    """The figure every ``FastCellAsk`` is compared against.

    ``gpu`` is total VRAM (list time gates on total). ``mac`` is the MPS pool macOS recommends
    for this Mac, which is the whole allowance video may use. None when the backend compared
    nothing (unknown VRAM, unsupported process).
    """

    device: Literal["gpu", "mac"]
    capacity_gib: float | None


class LTXVideoGenerationAspectSpec(BaseModel):
    """Per-aspect kept grids and cell annotations priced at that ratio's real canvas."""

    supported_resolutions_durations: dict[
        LTXVideoGenResolution, LTXVideoGenerationResolutionSpec
    ]
    a2v_supported_resolutions_durations: (
        dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] | None
    ) = None
    dropped_cells: list[DroppedFastCell] = Field(default_factory=list[DroppedFastCell])
    warned_cells: list[WarnedFastCell] = Field(default_factory=list[WarnedFastCell])
    chunked_cells: list[ChunkedFastCell] = Field(default_factory=list[ChunkedFastCell])
    cell_asks: list[FastCellAsk] = Field(default_factory=list[FastCellAsk])


class LTXVideoGenerationSpec(BaseModel):
    display_name: str
    supported_resolutions_durations: dict[
        LTXVideoGenResolution, LTXVideoGenerationResolutionSpec
    ]
    a2v_supported_resolutions_durations: (
        dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] | None
    ) = None
    capabilities: LTXOfferingCapabilitiesSpec | None = None
    # Withheld cells for disabled-with-tooltip pickers. None where no gating
    # ran (API specs, unfiltered validation ceiling); possibly empty locally.
    dropped_cells: list[DroppedFastCell] | None = None
    # Anchor-kept cells a wider shape cannot run (21:9 only, today). Same
    # None/empty contract as dropped_cells; aspect pickers read this, the
    # resolution/fps/duration pickers never do.
    dropped_aspect_cells: list[DroppedFastAspectCell] | None = None
    warned_cells: list[WarnedFastCell] | None = None
    warned_aspect_cells: list[WarnedFastAspectCell] | None = None
    # Kept cells that run in more than one window (the anchor shape; a wider shape can need
    # more windows). Same None/empty contract as warned_cells.
    chunked_cells: list[ChunkedFastCell] | None = None
    # Every priced cell's ask, kept and withheld, at the anchor shape. Same None/empty contract.
    cell_asks: list[FastCellAsk] | None = None
    # Per-aspect kept grids and cell annotations priced at each ratio's real canvas.
    # None where no gating ran (API specs, unfiltered validation ceiling).
    aspect_specs: dict[LTXVideoGenAspectRatio, LTXVideoGenerationAspectSpec] | None = None


class LTXVideoGenerationModelSpecItem(BaseModel):
    pipeline: LTXVideoGenPipeline
    spec: LTXVideoGenerationSpec


class DownloadedLocalVideoGenerationModelSpecItem(BaseModel):
    """One downloaded offering. `local_models` stays active-pipeline-shaped."""

    model: OfferingId
    pipeline: LTXVideoGenPipeline
    spec: LTXVideoGenerationSpec


class GenerateVideoModelsSpecsResponse(BaseModel):
    local_models: list[LTXVideoGenerationModelSpecItem]
    api_models: list[LTXVideoGenerationModelSpecItem]
    downloaded_local_models: list[DownloadedLocalVideoGenerationModelSpecItem] = Field(
        default_factory=list[DownloadedLocalVideoGenerationModelSpecItem]
    )
    # Settings-active checkpoint mapped to an offering id. Home prefers this over
    # joining local_models[0].display_name to downloaded rows (both share pipeline fast).
    active_offering: OfferingId | None = None
    # Why the local grid is empty, when exactly one constraint explains it.
    # None whenever any local cell survives — or when emptiness has another
    # cause (unsupported process, unknown VRAM, non-gated profile).
    local_unavailable_reason: LocalUnavailableReason | None = None
    # Sibling numbers, populated only alongside a ram_shortfall reason: the RAM
    # floor that gated this decision and the detected total RAM. None otherwise —
    # a stream_over/unsupported grid names no RAM numbers. Kept as siblings (not
    # nested) so the reason stays a plain alias of the budget-layer vocabulary.
    local_unavailable_required_ram_gb: int | None = None
    local_unavailable_detected_ram_gb: int | None = None
    # The backend machine is slow for heavy local runs. The backend decides, so the
    # desktop app and the Remote app show the same warnings.
    low_performance_machine: bool = False
    budgetMode: BudgetMode = "brave"
    # "Never split into parts" the grid was priced with, echoed like the mode.
    neverSplitIntoParts: bool = False
    memory: MemoryCapacity | None = None


class InstalledModelResponse(BaseModel):
    path: str
    name: str
    kind: str  # "safetensors"
    size_bytes: int
    is_lora: bool
    is_ic_lora: bool


class InstalledModelsResponse(BaseModel):
    models: list[InstalledModelResponse]


# Local Distilled offering cap. API pipelines stay at 0 via multi_keyframe=False.
LOCAL_MULTI_KEYFRAME_MAX_COUNT = 10


class KeyframeInput(BaseModel):
    """CamelCase of LTXV keyframe-edit `{image_uri, frame_index, strength}`."""

    model_config = ConfigDict(strict=True)

    imagePath: str
    frameIndex: int = Field(ge=0)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)


class KeyframeRef(BaseModel):
    """One still on the timeline. An asset id, never a path."""

    model_config = ConfigDict(strict=True, extra="forbid")

    assetId: str = Field(min_length=1)
    frameIndex: int = Field(ge=0)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)


# Where the submitted prompt came from. "typed" (the default for any client that omits it) is
# eligible for automatic enhancement; "enhanced" is the exact text a successful manual Enhance
# produced and the user chose to keep, so generation must submit it untouched rather than
# rewrite a rewrite. Editing the prompt returns it to "typed".
PromptProvenance: TypeAlias = Literal["typed", "enhanced"]


class GenerateVideoRequest(NagParams):
    model_config = ConfigDict(strict=True)

    prompt: Annotated[str, StringConstraints(strip_whitespace=True)]
    promptProvenance: PromptProvenance = "typed"
    resolution: LTXVideoGenResolution = "1080p"
    model: LTXVideoGenPipeline = "fast"
    cameraMotion: VideoCameraMotion = "none"
    # None = automatic duration (API 2.5 t2v/i2v only): the worker picks length from the prompt.
    duration: LTXVideoGenDuration | None = 5
    fps: LTXVideoGenFps = 24
    audio: bool = False
    imagePath: str | None = None
    lastImagePath: str | None = None
    keyframes: list[KeyframeInput] = Field(default_factory=list[KeyframeInput])
    audioPath: str | None = None
    aspectRatio: LTXVideoGenAspectRatio = "16:9"
    seed: int | None = None
    loras: list[LoraEntry] = Field(default_factory=list[LoraEntry])

    @model_validator(mode="after")
    def _require_prompt_or_start_image(self) -> "GenerateVideoRequest":
        if not self.prompt.strip() and not (self.imagePath or "").strip():
            raise ValueError("Prompt is required unless an image is provided")
        return self


# Z-Image Turbo is distilled for four steps. Every image path uses it.
ZIT_DEFAULT_STEPS = 4


class GenerateImageRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    prompt: NonEmptyPrompt
    width: int = Field(default=1024, ge=16)
    height: int = Field(default=1024, ge=16)
    numSteps: int = Field(default=ZIT_DEFAULT_STEPS, ge=1)
    numImages: int = Field(default=1, ge=1)
    imagePath: str | None = None
    strength: float = Field(default=0.6, ge=0.0, le=1.0)


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


class DescribeCheckpointsRequest(BaseModel):
    cp_ids: list[ModelCheckpointID]


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


RetakeMode: TypeAlias = Literal[
    "replace_audio_and_video", "replace_video", "replace_audio"
]

# ltxv-api /v2/retake and /v2/extend accept ltx-2-pro / ltx-2-3-pro. Desktop maps
# those to pipeline "pro" — narrower than LTXVideoGenPipeline on purpose.
RetakeExtendModel: TypeAlias = Literal["pro"]


class TargetResolution(BaseModel):
    """Desired output resolution for a local generation. The backend corrects it to the
    nearest valid size (snapped down to a multiple of 32, never above the source)."""

    model_config = ConfigDict(strict=True)

    width: int = Field(gt=0)
    height: int = Field(gt=0)


class RetakeRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    video_path: str
    start_time: float
    duration: float
    prompt: str = ""
    prompt_provenance: PromptProvenance = "typed"
    mode: RetakeMode = "replace_audio_and_video"
    resolution: TargetResolution | None = None
    # API-mode only; ignored for local generation (there is no local model choice).
    model: RetakeExtendModel = "pro"


ExtendMode: TypeAlias = Literal["start", "end"]


class ExtendRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    video_path: str
    # gt=0 + allow_inf_nan=False so NaN/negative fail validation (clean 422) instead of
    # slipping past the handler's min/max guards and blowing up in frame-count math.
    duration: float = Field(gt=0, allow_inf_nan=False)
    prompt: str = ""
    prompt_provenance: PromptProvenance = "typed"
    mode: ExtendMode = "end"
    resolution: TargetResolution | None = None
    # API-mode only; ignored for local generation (there is no local model choice).
    model: RetakeExtendModel = "pro"


# Extend returns the same shapes as retake (video file, remote payload, or cancelled).
ExtendResponse: TypeAlias = RetakeResponse


ConditioningType: TypeAlias = Literal["canny", "depth"]

# Generation can additionally run a user-supplied IC-LoRA against a pre-rendered
# control video ("custom"), bypassing the built-in canny/depth preprocessing.
IcLoraGenerateConditioning: TypeAlias = Literal["canny", "depth", "custom"]
IcLoraAudioMode: TypeAlias = Literal["source", "generated", "off"]
# How a guide clip is read per frame (Video Guide): body pose, depth map, or Canny edges.
IcLoraControlType: TypeAlias = Literal["pose", "depth", "canny"]
# The catalog control that picks it. It is also the key of `IcLoraGenerateRequest.control_values`.
IC_LORA_CONTROL_TYPE_ID = "control_type"


class IcLoraExtractRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    video_path: str
    conditioning_type: ConditioningType = "canny"
    frame_time: float = 0


# The frame index of a reference still (the look still of Layout To Render). Upstream
# rejects a negative index, so the pipeline keeps the still apart from the clip frames.
# Only a catalog entry can pin it (``reference_image_frame``). A request cannot send it.
REFERENCE_STILL_FRAME_IDX = -1


class IcLoraImageInput(BaseModel):
    model_config = ConfigDict(strict=True)

    path: str
    # A clip frame. A negative index is a reference still, which only the catalog pins.
    frame: int = Field(default=0, ge=0)
    strength: float = 1.0


def _default_ic_lora_images() -> list[IcLoraImageInput]:
    return []


class OutpaintPads(BaseModel):
    """Per-edge pixels to add around the source video (extend-only, so all ≥ 0).

    The +100% cap (each ≤ the source's axis size) needs the source dimensions, so it's enforced
    in `_outpaint_canvas`, not here.
    """

    model_config = ConfigDict(strict=True)
    left: int = Field(ge=0)
    right: int = Field(ge=0)
    top: int = Field(ge=0)
    bottom: int = Field(ge=0)


class IcLoraGenerateRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    # Optional: not required in IC-LoRA mode (input comes via input_path).
    video_path: str = ""
    conditioning_type: IcLoraGenerateConditioning
    # May be empty for catalog IC-LoRAs whose entry sets allows_empty_prompt (e.g. outpainting).
    # Non-emptiness is enforced in the handler per path, not at the DTO level.
    prompt: str
    prompt_provenance: PromptProvenance = "typed"
    # These five are overlay settings: None means "use the default" — the IC-LoRA's
    # default_settings in IC-LoRA mode, or the built-in default otherwise. The UI sends
    # an explicit value only when the user edits it (advanced controls).
    conditioning_strength: float | None = None
    # Skip Stage 2 (upscale + refine). Default False = current two-stage behavior.
    # Transformation IC-LoRAs (cross-eye, etc.) need this True or Stage 2 repaints
    # from the prompt and erases the effect.
    skip_stage_2: bool | None = None
    # Deprecated (LTXP-514): accepted for API compatibility, always ignored.
    # Stage 2 never keeps the IC-LoRA; catalog never set this.
    use_lora_in_stage_2: bool | None = None
    # Target output resolution. Two-stage snaps down to a valid size, never above
    # source. skip_stage_2 with a resolution renders that size (canvas 2×, factor 1).
    # skip_stage_2 without a resolution keeps the 768-wide catalog bucket.
    resolution: TargetResolution | None = None
    # Stage-1-only canvas multiplier (only affects skip_stage_2). Stage 1 runs at
    # half the passed canvas, so 2.0 = native target resolution, 1.0 = half (faster).
    # Intermediate values trade quality for speed/VRAM. Canvas is rounded to a
    # multiple of 64. Sentinel 0 = "source dimensions": ignore the multiplier and
    # size the output to the input video's resolution (see ic_lora_handler).
    resolution_factor: float | None = Field(default=None, ge=0.0, le=2.0)
    # Audio: "generated" = model audio from the prompt (current default); "source" =
    # mux the input clip's soundtrack (transformation LoRAs); "off" = no audio.
    audio_mode: IcLoraAudioMode | None = None
    # LoRA adapter merge weight (baked at load time). Bounded to [0, 2] to match the
    # desktop IC-LoRA strength control (same practical ceiling as the plain-LoRA UI slider).
    lora_strength: float | None = Field(default=None, ge=0.0, le=2.0)
    # Advanced controls of a tiled catalog entry (Refine Details). None keeps the catalog
    # value, or the high cell default for the carry. A request for an entry without a tile
    # plan, or a pair the pipeline cannot run, is refused (422).
    tile_overlap: float | None = Field(default=None, ge=MIN_TILE_OVERLAP, le=MAX_TILE_OVERLAP)
    carry_frames: int | None = Field(default=None, ge=MIN_CARRY_FRAMES)
    tile_size: IcLoraTileSize | None = None
    max_window_frames: int | None = Field(default=None, ge=MIN_WINDOW_FRAMES)
    # Override the control-video fps by temporally resampling it. Lower fps = fewer frames =
    # less compute/VRAM (fits longer clips), at the cost of choppier motion. None = source fps.
    # Only decimates (ignored if >= source fps).
    fps_override: float | None = Field(default=None, gt=0.0)
    # Rate of the looped control video of an image entry (``image_to_frames``). It wins over
    # the catalog step's own fps. Unlike ``fps_override`` it applies to an image, not a clip.
    # Ignored unless the entry has an ``image_to_frames`` step.
    loop_fps: int | None = Field(default=None, gt=0)
    # Length of that looped control video. It replaces the catalog ``duration`` control for an
    # image-list entry (``input.max_count``) and skips the options check on that control. The two
    # lists differ on purpose: the catalog list (5 to 20 s) is the GenSpace picker, and Home
    # offers 2 to 20 s. Ignored for every other entry.
    loop_seconds: LTXVideoGenDuration | None = None
    num_inference_steps: int = 30
    cfg_guidance_scale: float = 1.0
    negative_prompt: str = Field(default="", max_length=NEGATIVE_PROMPT_MAX_LENGTH)
    nag_scale: float | None = Field(default=None, ge=0.0, le=NAG_MAX_SCALE, allow_inf_nan=False)
    nag_alpha: float | None = Field(default=None, ge=0.0, le=NAG_MAX_ALPHA, allow_inf_nan=False)
    nag_tau: float | None = Field(default=None, ge=0.0, le=NAG_MAX_TAU, allow_inf_nan=False)
    images: list[IcLoraImageInput] = Field(default_factory=_default_ic_lora_images)
    # "custom" conditioning: the user's own IC-LoRA + a pre-rendered control video.
    # Ignored for canny/depth; required (both) when conditioning_type == "custom".
    custom_lora_ref: str | None = None
    control_video_path: str | None = None
    # IC-LoRA mode: when set, the backend resolves catalog weights, builds the control
    # video via the IC-LoRA's preprocessing pipeline, and runs the custom inference path.
    ic_lora_id: str | None = None
    # Optional catalog download.variants[].id. When set, that checkpoint must be on disk.
    # When omitted, any installed candidate is used (preferred filename first).
    variant_id: str | None = None
    # Values for the IC-LoRA's declared `controls`, keyed by control id (e.g. {"duration": 5}).
    # Missing keys fall back to each control's default; values are validated against the control's
    # options. The handler maps known ids to behaviour (duration → frame count).
    control_values: dict[str, int | str] = Field(default_factory=dict)
    # Outpainting only: per-edge pixels to add around the source (extend-only). The canvas UI
    # emits these directly; aspect presets are computed front-end into pads. None for other entries.
    outpaint_pads: OutpaintPads | None = None
    # The user's input media for IC-LoRA mode (image or video per ic_lora.input.kind).
    input_path: str | None = None


# --- LoRA / IC-LoRA catalog ---
# Shared base `LoraCatalogItem` (used as-is for plain LoRAs); `IcLoraCatalogItem` extends it
# with the IC-specific preprocessing / controls / default settings.
InputKind: TypeAlias = Literal["image", "video"]
InstructionTitle: TypeAlias = Literal[
    "What it does", "Input", "Prompt", "Tips", "Notes"
]
# Machine-readable counterpart to `title`, so consumers (e.g. a prompt enhancer) can select
# instruction blocks by purpose without string-matching display titles. "tips" also covers
# the "Notes" title; "summary" covers "What it does".
InstructionKind: TypeAlias = Literal["summary", "prompting", "tips", "input"]
# Families a catalog adapter is known to run on. Distinct from `base_model` (what it was
# trained on). 2.3 adapters often run on 2.5; that is recorded here after validation, not
# inferred from the training tag.
LtxCatalogModelFamily: TypeAlias = Literal["LTX-2.3", "LTX-2.5"]
# What a checkpoint was trained on. Wider than the runnable families: some camera
# LoRAs are LTX-2 weights that still run on 2.3 and 2.5.
LtxVariantBaseModel: TypeAlias = Literal["LTX-2", "LTX-2.3", "LTX-2.5"]
# Variant id stored on runs created before ids carried the base model and LoRA name.
LEGACY_DEFAULT_VARIANT_ID = "default"
# Where a trigger word/phrase must appear in the prompt. Not set (and not applicable) when
# `prompt_template` is present, since the template's placeholder position already encodes it.
TriggerPlacement: TypeAlias = Literal["first_token", "anywhere"]

_TEMPLATE_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


class InstructionSection(BaseModel):
    model_config = ConfigDict(strict=True)
    kind: InstructionKind
    title: InstructionTitle
    body: str | list[str]


class PromptTemplatePlaceholder(BaseModel):
    model_config = ConfigDict(strict=True)
    # None = free text (a description Gemma fills in). A list = the value must be exactly one
    # of these choices (e.g. crossview's azimuth/elevation/distance vocabulary).
    choices: list[str] | None = None


class PromptTemplateSpec(BaseModel):
    """A fixed prompt structure a LoRA/IC-LoRA requires verbatim (e.g. a two-part
    "Reference shows X. Edited shows Y." restore template, or crossview's enum-constrained
    camera vocabulary). `template` contains `{name}` placeholders; each must have a matching
    entry in `placeholders`. A degenerate template with no placeholders (e.g. "upscale") is a
    fixed, non-generated prompt.
    """

    model_config = ConfigDict(strict=True)
    template: str
    placeholders: dict[str, PromptTemplatePlaceholder] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_placeholders_match_template(self) -> "PromptTemplateSpec":
        referenced = set(_TEMPLATE_PLACEHOLDER_RE.findall(self.template))
        declared = set(self.placeholders)
        if referenced != declared:
            raise ValueError(
                f"prompt_template placeholders {sorted(declared)} must exactly match "
                f"template references {sorted(referenced)}"
            )
        return self


class DownloadVariant(BaseModel):
    """One downloadable weights file under a catalog item (e.g. strong vs light)."""

    model_config = ConfigDict(strict=True)
    id: str
    label: str
    filename: str
    size_bytes: int
    # Training family for the mismatch notice. Required so a catalog entry that
    # forgets it is not treated as LTX-2.3.
    base_model: LtxVariantBaseModel
    # Set when this file lives in a different Hugging Face repo than DownloadSpec.repo_id.
    repo_id: str | None = None

    def resolved_repo_id(self, parent_repo_id: str) -> str:
        return self.repo_id if self.repo_id is not None else parent_repo_id


class DownloadSpec(BaseModel):
    """HF download location for a catalog item.

    ``variants`` is always non-empty. The first entry is the default checkpoint
    (preferred for generation / omit-``variant_id`` downloads). See the comment on
    ``LoraCatalogHandler`` listing helpers.
    """

    model_config = ConfigDict(strict=True)
    repo_id: str
    variants: list[DownloadVariant]

    @model_validator(mode="after")
    def _check_variants(self) -> "DownloadSpec":
        if not self.variants:
            raise ValueError(
                "download.variants must contain at least one entry (first = default)"
            )
        filenames = [v.filename for v in self.variants]
        if len(filenames) != len(set(filenames)):
            raise ValueError("download.variants filenames must be unique")
        ids = [v.id for v in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError("download.variants ids must be unique")
        return self

    def default_variant(self) -> DownloadVariant:
        return self.variants[0]

    def legacy_variant(self) -> DownloadVariant:
        """The checkpoint stored runs meant before variant ids were unique.

        New checkpoints are added at index 0, so the original one is last.
        """
        return self.variants[-1]

    def resolve_variant(self, variant_id: str | None) -> DownloadVariant | None:
        """Resolve by id; ``None`` → first variant (the default).

        A stored ``"default"`` id (written before ids were renamed) resolves to
        the original checkpoint when no variant carries that id.
        """
        if variant_id is None:
            return self.variants[0]
        match = next((v for v in self.variants if v.id == variant_id), None)
        if match is None and variant_id == LEGACY_DEFAULT_VARIANT_ID:
            return self.legacy_variant()
        return match

    def variant_for_run(
        self, variant_id: str | None, installed_ids: list[str]
    ) -> DownloadVariant | None:
        """Checkpoint a generate should use.

        An explicit id must exist (``None`` when it does not). An omitted id is
        the only checkpoint when the entry has one, otherwise index 0 only when
        nothing is installed; once one of several files is on disk the caller
        sends its id.
        """
        if variant_id is not None:
            return self.resolve_variant(variant_id)
        if len(self.variants) == 1:
            return self.variants[0]
        installed = set(installed_ids)
        if any(variant.id in installed for variant in self.variants):
            return None
        return self.variants[0]

    def candidate_filenames(self) -> list[str]:
        """Filenames in preference order (default first)."""
        return [v.filename for v in self.variants]


# The most images one image-list recipe (Ingredients) takes. The catalog `input.max_count`, the
# create body and the Enhance body all stay within it.
IC_LORA_SHEET_MAX_IMAGES = 10


class InputSpec(BaseModel):
    model_config = ConfigDict(strict=True)
    kind: InputKind
    # False (default) = required, matching every existing entry (IC-LoRAs always require their
    # driving input; a plain LoRA with `input` set has, until now, always meant "requires one").
    # True = accepted/preferred but not mandatory (e.g. product-ad-style: works with or without
    # a product image, image preferred).
    optional: bool = False
    # Set on an image input that takes a list (Ingredients): the most images one run takes.
    # The recipe path then joins them into one sheet and loops it for a `duration` control.
    max_count: int | None = Field(default=None, ge=2, le=IC_LORA_SHEET_MAX_IMAGES)


class LicenseSpec(BaseModel):
    model_config = ConfigDict(strict=True)
    name: str
    url: str | None = None


AuthorAffiliation: TypeAlias = Literal["ltx", "community"]


class AuthorSpec(BaseModel):
    # Credit back to the community that builds these LoRAs. Extensible (more fields later).
    model_config = ConfigDict(strict=True)
    name: str
    url: str | None = None
    # Product/legal bucket for disclaimer gating. Defaults to community so a missing
    # value fails closed to the safer copy; set "ltx" for official LTX catalog items.
    affiliation: AuthorAffiliation = "community"


class MediaSpec(BaseModel):
    model_config = ConfigDict(strict=True)
    thumbnail: str | None = None
    demo_video: str | None = None


def _empty_instructions() -> list[InstructionSection]:
    return []


def _empty_tags() -> list[str]:
    return []


def _default_supported_models() -> list[LtxCatalogModelFamily]:
    return ["LTX-2.3", "LTX-2.5"]


def _empty_checkpoints() -> list[ModelCheckpointID]:
    return []


class LoraCatalogItem(BaseModel):
    """Base catalog entry — used as-is for a plain LoRA."""

    model_config = ConfigDict(strict=True)
    id: str
    name: str
    description: str
    download: DownloadSpec
    requires_hf_login: bool
    # Optional: IC-LoRAs always take a driving image/video; many plain (t2v) LoRAs take none.
    input: InputSpec | None = None
    instructions: list[InstructionSection] = Field(default_factory=_empty_instructions)
    license: LicenseSpec | None = None
    author: AuthorSpec | None = None
    media: MediaSpec | None = None
    # ISO-8601 date the model was created at the source (e.g. HuggingFace).
    created_at: str | None = None
    # What the adapter was trained on. Not a compatibility list — see `supported_models`.
    base_model: str | None = None
    # Families this adapter is known to run on. Currently both, so 2.5 testing can
    # use the full catalog; trim after validation.
    supported_models: list[LtxCatalogModelFamily] = Field(
        default_factory=_default_supported_models
    )
    tags: list[str] = Field(default_factory=_empty_tags)
    # Model checkpoints the entry reads besides its own file (the depth, pose and person
    # detector models of Video Guide). The catalog download fetches them after the entry's
    # file, and the entry counts as installed only when all of them are on disk.
    requires_checkpoints: list[ModelCheckpointID] = Field(default_factory=_empty_checkpoints)
    # Trigger phrase to include in the prompt if the LoRA needs one (e.g. "ADD WATER").
    trigger: str | None = None
    # Required together with `trigger` (and only then), unless `prompt_template` is set —
    # the template's placeholder position already encodes where the trigger goes.
    trigger_placement: TriggerPlacement | None = None
    # Set when the LoRA/IC-LoRA requires a fixed prompt structure rather than a free rewrite
    # (e.g. colorization's two-part restore template, or crossview's enum-constrained camera
    # vocabulary). Some LoRAs/IC-LoRAs with a template have no separate `trigger` at all.
    prompt_template: PromptTemplateSpec | None = None
    # Extra few-shot example prompts for a prompt enhancer to draw on — never rendered in the
    # LoRA info UI (unlike the single illustrative example inside a "prompting" instruction).
    # Multi-shot grounding helps an LLM rewrite generalize past one example's specific subject.
    enhancement_examples: list[str] = Field(default_factory=list)
    # Suggested LoRA scale when a plain LoRA is selected (IC-LoRAs use default_settings instead).
    recommended_strength: float | None = None
    # Generic negative prompt from the Hugging Face model card, when the card names one.
    default_negative_prompt: str | None = None
    # When true, generation may run with an empty prompt (e.g. outpainting fills from the scene).
    # Enforced for the catalog IC-LoRA path; plain-LoRA t2v enforcement is not wired yet.
    allows_empty_prompt: bool = False

    def supports_family(self, family: LtxCatalogModelFamily) -> bool:
        return family in self.supported_models

    @field_validator("supported_models")
    @classmethod
    def _check_supported_models(
        cls, value: list[LtxCatalogModelFamily]
    ) -> list[LtxCatalogModelFamily]:
        if not value:
            raise ValueError("supported_models must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("supported_models must be unique")
        return value

    @model_validator(mode="after")
    def _check_trigger_placement(self) -> "LoraCatalogItem":
        if self.prompt_template is not None:
            if self.trigger_placement is not None:
                raise ValueError(
                    f"'{self.id}': trigger_placement is redundant when prompt_template is set"
                )
        elif (self.trigger is None) != (self.trigger_placement is None):
            raise ValueError(
                f"'{self.id}': trigger and trigger_placement must be set together"
            )
        return self


class IcLoraSettings(BaseModel):
    model_config = ConfigDict(strict=True)
    skip_stage_2: bool = False
    # Deprecated (LTXP-514): accepted for API compatibility, always ignored.
    use_lora_in_stage_2: bool = False
    # Keep the IC-LoRA and its reference (video and stills) on stage 2 (Layout To Render).
    # Ignored when skip_stage_2 is true. Unlike the deprecated flag above, this one is honoured.
    stage_2_ic_lora: bool = False
    # Tile size of a LoRA trained on one tile (Restore, Refine Details). Catalog only: a
    # request cannot set it. Needs skip_stage_2 (see IcLoraCatalogItem).
    tiling: IcLoraTiling | None = None
    # 2.0 = native, 1.0 = half; sentinel 0 = source dimensions (see ic_lora_handler).
    resolution_factor: float = Field(default=2.0, ge=0.0, le=2.0)
    audio_mode: IcLoraAudioMode = "generated"
    lora_strength: float = Field(default=1.0, ge=0.0, le=2.0)
    conditioning_strength: float = 1.0

    @property
    def stage_mode(self) -> IcLoraStageMode:
        """The stage layout of these settings. The job budget and the chunk rule read it."""
        return ic_lora_stage_mode(skip_stage_2=self.skip_stage_2, stage_2_ic_lora=self.stage_2_ic_lora)


IcLoraControlKind: TypeAlias = Literal["int", "select", "position_canvas"]


class DroppedIcLoraDuration(BaseModel):
    """One duration option the list-time budget withheld, with its refusal.

    Advisory: list-time filtering previews a mid-size canvas without the user
    image, so a dropped duration may still pass on a small source (generate
    re-decides on real pixels). ``options`` stays filtered; kept + dropped
    reconstruct the catalog grid for tooltip pickers.
    """

    model_config = ConfigDict(strict=True)
    value: int
    reason: IcLoraDroppedReason
    message: str
    tier: JobRiskTier | None = None
    over_gib: float | None = None
    over_ram_gb: int | None = None
    shortfall_kind: ShortfallKind | None = None
    readout: BudgetReadout | None = None
    unmeasured_basis: UnmeasuredBasis | None = None


class IcLoraControl(BaseModel):
    # A user-facing knob the IC-LoRA exposes in the gen UI. `id` is the field it drives
    # (e.g. "duration", "outpaint_pads"); the frontend renders each control by id + kind.
    # `kind` picks the value type: "int" → integer options (e.g. duration seconds);
    # "select" → string options (e.g. a region); "position_canvas" → the outpainting editor
    # for positioning/sizing the source within the output frame, whose value is structured
    # (per-edge pads) and travels in the typed `outpaint_pads` request field, so it declares
    # no options/default. `kind` defaults to "int" so pre-existing int-only entries parse unchanged.
    model_config = ConfigDict(strict=True)
    id: str
    label: str
    kind: IcLoraControlKind = "int"
    # int/select carry their value via `options`; "position_canvas" leaves both None (value is typed elsewhere).
    default: int | str | None = None
    options: list[int] | list[str] | None = None
    # Display-only metadata so the frontend renders controls generically (data-driven):
    # `unit` is a suffix appended to the value (e.g. "s", "%"); `value_labels` maps an option
    # (as a string) to a friendlier label (e.g. "all" → "All sides"). Both optional.
    unit: str | None = None
    value_labels: dict[str, str] | None = None
    # Durations the list-time budget withheld. None when nothing was dropped
    # (or no gating ran); the catalog file never sets this, generate ignores
    # it, and only duration controls ever carry it.
    dropped_options: list[DroppedIcLoraDuration] | None = None

    @model_validator(mode="after")
    def _check_value_type(self) -> "IcLoraControl":
        if self.kind == "position_canvas":
            if self.default is not None or self.options is not None:
                raise ValueError(
                    "position_canvas control takes no default/options (value is structured)"
                )
            return self
        if self.default is None or self.options is None:
            raise ValueError(f"{self.kind} control requires a default and options")
        if self.kind == "int":
            ok = isinstance(self.default, int) and all(
                isinstance(o, int) for o in self.options
            )
            if not ok:
                raise ValueError("int control requires an int default and int options")
        else:
            ok = isinstance(self.default, str) and all(
                isinstance(o, str) for o in self.options
            )
            if not ok:
                raise ValueError(
                    "select control requires a str default and str options"
                )
        return self


class PreprocessingStep(BaseModel):
    model_config = ConfigDict(strict=True)
    utility: str
    params: dict[str, JsonValue] = Field(default_factory=dict)


def _empty_preprocessing() -> list[PreprocessingStep]:
    return []


def _empty_controls() -> list[IcLoraControl]:
    return []


# Every field is_home_recipe has classified. A field added to IcLoraCatalogItem
# is absent from this set, so is_home_recipe fails closed until it is classified
# here and in the method.
_HOME_RECIPE_FIELDS = frozenset({
    "allows_empty_prompt",
    "allows_reference_image",
    "author",
    "base_model",
    "controls",
    "created_at",
    "default_negative_prompt",
    "default_settings",
    "description",
    "download",
    "enhancement_examples",
    "id",
    "input",
    "instructions",
    "license",
    "media",
    "name",
    "preprocessing",
    "prompt_template",
    "recommended_strength",
    "reference_image_frame",
    "reference_image_required",
    "requires_checkpoints",
    "requires_hf_login",
    "supported_models",
    "tags",
    "trigger",
    "trigger_placement",
})


class IcLoraCatalogItem(LoraCatalogItem):
    """IC-LoRA — base plus IC-specific preprocessing / controls / default settings.

    IC-LoRAs always take a driving image/video; `input` is required in practice (the base
    field is optional for plain LoRAs). Enforced by the validator below, so handlers can rely
    on it being present.
    """

    preprocessing: list[PreprocessingStep] = Field(default_factory=_empty_preprocessing)
    controls: list[IcLoraControl] = Field(default_factory=_empty_controls)
    default_settings: IcLoraSettings = Field(default_factory=IcLoraSettings)
    # When true, the UI offers an optional reference image that seeds the first frame
    # (image conditioning at frame 0, strength 1.0) — e.g. a photoreal seed for 3D-render.
    allows_reference_image: bool = False
    # Pins the reference image to this pixel frame and ignores the request frame.
    # REFERENCE_STILL_FRAME_IDX is the look still of Layout To Render. None keeps the
    # request frame (0).
    reference_image_frame: int | None = None
    # The Home form refuses a create without the reference image.
    reference_image_required: bool = False

    def is_home_recipe(self) -> bool:
        """True when the Home recipe path can run this entry.

        A Home recipe has a video input, no controls and no preprocessing, or it
        reads the clip per frame (``is_video_control_recipe``), or it
        declares an image list (``is_sheet_recipe``). A video recipe takes an
        optional or required reference image, pinned to the catalog
        ``reference_image_frame``. Stage 2 is skipped, or it keeps the IC-LoRA
        (``stage_2_ic_lora``). Home sizes an explicit resolution and ignores
        ``resolution_factor``, which is only correct for those two cases.
        The model default for ``skip_stage_2`` is false, so an entry that omits
        ``default_settings`` is not a Home recipe.
        """
        if frozenset(type(self).model_fields) != _HOME_RECIPE_FIELDS:
            return False
        return (
            self.input is not None
            and (
                self.is_sheet_recipe()
                or self.is_video_control_recipe()
                or (self.input.kind == "video" and not self.controls and not self.preprocessing)
            )
            # A reference image is pinned to a catalog frame, never a request frame.
            and (not self.allows_reference_image or self.reference_image_frame is not None)
            and (self.default_settings.skip_stage_2 or self.default_settings.stage_2_ic_lora)
        )

    def is_sheet_recipe(self) -> bool:
        """True for an entry whose input is an image list (``input.max_count``)."""
        return self.input is not None and self.input.max_count is not None

    def is_video_control_recipe(self) -> bool:
        """True for an entry that turns the clip into a pose, depth or Canny guide clip."""
        return any(step.utility == "video_control" for step in self.preprocessing)

    @model_validator(mode="after")
    def _require_input(self) -> "IcLoraCatalogItem":
        # An IC-LoRA always drives off an image/video input — reject a malformed catalog entry
        # at parse time rather than letting it trip the handler's assert at generate time.
        if self.input is None:
            raise ValueError(f"IC-LoRA '{self.id}' must declare an input spec")
        if self.reference_image_required and not self.allows_reference_image:
            raise ValueError(f"IC-LoRA '{self.id}' requires a reference image it does not allow")
        if self.input.max_count is not None and (
            self.input.kind != "image"
            or self.allows_reference_image
            or [step.utility for step in self.preprocessing] != ["image_to_frames"]
            or [control.id for control in self.controls] != ["duration"]
        ):
            raise ValueError(
                f"IC-LoRA '{self.id}' declares an image list: it needs an image input, "
                "no reference image, one image_to_frames step and one duration control"
            )
        if self.is_video_control_recipe() and (
            self.input.kind != "video"
            or [step.utility for step in self.preprocessing] != ["video_control"]
            or [control.id for control in self.controls] != [IC_LORA_CONTROL_TYPE_ID]
            or self.controls[0].options != list(get_args(IcLoraControlType))
            or self.controls[0].default != get_args(IcLoraControlType)[0]
        ):
            raise ValueError(
                f"IC-LoRA '{self.id}' video_control needs a video input, one video_control step "
                f"and one {IC_LORA_CONTROL_TYPE_ID} control with the options "
                f"{get_args(IcLoraControlType)} and the first as its default"
            )
        if self.reference_image_frame not in (None, 0, REFERENCE_STILL_FRAME_IDX):
            # Another frame would reach the pipeline as a negative still or an unknown index.
            raise ValueError(
                f"IC-LoRA '{self.id}' reference_image_frame must be 0 or {REFERENCE_STILL_FRAME_IDX}, "
                f"got {self.reference_image_frame}"
            )
        if self.default_settings.tiling is not None:
            if not self.default_settings.skip_stage_2:
                raise ValueError(f"IC-LoRA '{self.id}' tiling needs skip_stage_2")
            if self.reference_image_frame is not None and self.reference_image_frame < 0:
                # ltx-pipelines keeps a negative-time token whole in every tile. It does not
                # crop the still to each tile, as the LoRA cards do.
                raise ValueError(f"IC-LoRA '{self.id}' tiling cannot take a reference still")
            if self.default_settings.tiling.output_cap() is not None:
                # Only the high cells raise the spatial cap, behind their flag and budget.
                # A catalog row that did so would price as a high cell with no flag.
                raise ValueError(f"IC-LoRA '{self.id}' tiling cannot set max_output sides")
        return self


def _empty_ic_loras() -> list[IcLoraCatalogItem]:
    return []


def _empty_loras() -> list[LoraCatalogItem]:
    return []


CATALOG_SCHEMA_VERSION = 1


class LoraCatalogFile(BaseModel):
    model_config = ConfigDict(strict=True)
    schema_version: int
    ic_loras: list[IcLoraCatalogItem] = Field(default_factory=_empty_ic_loras)
    loras: list[LoraCatalogItem] = Field(default_factory=_empty_loras)


def parse_lora_catalog(raw: str) -> LoraCatalogFile:
    catalog = LoraCatalogFile.model_validate_json(raw)
    if catalog.schema_version != CATALOG_SCHEMA_VERSION:
        # A future v2 doc would otherwise parse against v1 fields with no clear error.
        raise ValueError(
            f"Unsupported catalog schema_version {catalog.schema_version} (expected {CATALOG_SCHEMA_VERSION})"
        )
    return catalog


class ReplacedWindow(BaseModel):
    """Seconds ``start`` to ``end`` of a source video that the job replaces (a Retake selection)."""

    model_config = ConfigDict(strict=True)
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def _check_end_after_start(self) -> "ReplacedWindow":
        if self.end <= self.start:
            raise ValueError("replacedWindow end must be after start")
        return self


class EnhancePromptRequest(BaseModel):
    """Regular-LoRA, IC-LoRA, and built-in conditioning-type selection are mutually exclusive UI
    surfaces — a request never carries more than one of loraCatalogIds/customLoraIds/icLoraId/conditioningType.
    """

    model_config = ConfigDict(strict=True)
    # Empty is allowed when imagePath or keyframes are set — the rewrite is then captioned
    # from the attached frame(s), plus any extra text the user did type.
    prompt: str
    negativePrompt: str | None = Field(default=None, max_length=NEGATIVE_PROMPT_MAX_LENGTH)
    loraCatalogIds: list[str] = Field(default_factory=list)
    # The user's saved custom LoRA. The enhancer reads what the user told it about the LoRA.
    customLoraIds: list[str] = Field(default_factory=list, max_length=1)
    icLoraId: str | None = None
    # Set only for the built-in (non-catalog) "bring your own IC-LoRA" canny/depth conditioning
    # modes — those have no catalog entry to draw a system prompt from, but still need the same
    # "describe the reference scene faithfully, don't invent a different one" discipline a
    # catalog IC-LoRA's template-fill path gets, instead of Gemma's default free-rewrite prompt.
    conditioningType: Literal["canny", "depth"] | None = None
    imagePath: str | None = None
    lastImagePath: str | None = None
    # Explore sends asset ids (Remote has no filesystem paths). Mutually exclusive with
    # imagePath / lastImagePath.
    imageAssetId: str | None = None
    lastImageAssetId: str | None = None
    # Same shape as GenerateVideoRequest.keyframes. Mutually exclusive with imagePath/
    # lastImagePath — middle markers have nowhere to go on the two-slot i2v path.
    keyframes: list[KeyframeInput] = Field(default_factory=list[KeyframeInput])
    # The same stills as asset ids. Explore sends these and the route resolves them to
    # `keyframes`. Mutually exclusive with `keyframes`, and bound by the same rules.
    keyframeAssets: list[KeyframeRef] = Field(default_factory=list[KeyframeRef])
    # The panel images of an image-list IC-LoRA (Ingredients), in order. Explore sends asset
    # ids and the route resolves them to paths. Needs an icLoraId whose entry takes an image
    # list. Mutually exclusive with every other image input and with a video.
    imagePaths: list[str] = Field(default_factory=list[str], max_length=IC_LORA_SHEET_MAX_IMAGES)
    imageAssetIds: list[str] = Field(
        default_factory=list[str], max_length=IC_LORA_SHEET_MAX_IMAGES
    )
    # The source video of an IC-LoRA, Extend, or Retake. The rewrite sees a few stills of it.
    # With icLoraId the stills are the IC-LoRA's input; without, they are scene context only and
    # the typed prompt leads. Mutually exclusive with every image input. Explore sends the asset
    # id; the route resolves it to videoPath.
    videoPath: str | None = None
    videoAssetId: str | None = None
    # The part of the video a Retake replaces. The stills then come from around it, never from
    # inside it. Needs a video, and does not combine with icLoraId.
    replacedWindow: ReplacedWindow | None = None
    # Optional clip timing for multi-keyframe enhance: the rewriter needs duration/fps
    # and each still's clock time on the user turn. Ignored for t2v/first-last/image.
    duration: int | None = Field(default=None, gt=0)
    fps: int | None = Field(default=None, gt=0)
    # Optional sheet canvas shape for an image-list IC-LoRA (Ingredients) so manual Enhance
    # builds the reference sheet at the exact size Generate will use.
    aspectRatio: LTXVideoGenAspectRatio | None = None
    resolution: LTXVideoGenResolution | None = None
    # "local" runs the on-device Gemma text encoder (default, matches every existing caller);
    # "api" calls Gemini's hosted API instead — no local checkpoint required, gated on
    # AppSettings.gemini_api_key being set.
    provider: Literal["local", "api"] = "local"
    # "video" (default, matches every existing caller) routes through the video-catalog LoRA/
    # IC-LoRA/conditioning-type selection below. "image" is Z-Image-Turbo generation/editing —
    # no catalog LoRA concept, so none of loraCatalogIds/icLoraId/conditioningType apply;
    # imagePath's presence distinguishes editing (img2img) from generation, same convention as
    # video's t2v/i2v split.
    mediaType: Literal["video", "image"] = "video"

    @model_validator(mode="after")
    def _check_selection_is_mutually_exclusive(self) -> "EnhancePromptRequest":
        selected = [
            bool(self.loraCatalogIds),
            bool(self.customLoraIds),
            self.icLoraId is not None,
            self.conditioningType is not None,
        ]
        if sum(selected) > 1:
            raise ValueError(
                "loraCatalogIds, customLoraIds, icLoraId, and conditioningType are mutually exclusive"
            )
        if self.mediaType == "image" and sum(selected) > 0:
            raise ValueError(
                "loraCatalogIds, customLoraIds, icLoraId, and conditioningType only apply to mediaType='video'"
            )
        has_first_path = bool((self.imagePath or "").strip())
        has_last_path = bool((self.lastImagePath or "").strip())
        has_first_asset = bool((self.imageAssetId or "").strip())
        has_last_asset = bool((self.lastImageAssetId or "").strip())
        if has_first_path and has_first_asset:
            raise ValueError("Use imagePath or imageAssetId, not both")
        if has_last_path and has_last_asset:
            raise ValueError("Use lastImagePath or lastImageAssetId, not both")
        has_first = has_first_path or has_first_asset
        has_last = has_last_path or has_last_asset
        if self.keyframes and self.keyframeAssets:
            raise ValueError("Use keyframes or keyframeAssets, not both")
        keyframe_indices = [
            keyframe.frameIndex for keyframe in (self.keyframes or self.keyframeAssets)
        ]
        has_keyframes = bool(keyframe_indices)
        if self.mediaType == "image" and has_keyframes:
            raise ValueError("keyframes only apply to mediaType='video'")
        # Explore text-to-image has no input image. A path is the GenSpace edit rewrite.
        if self.mediaType == "image" and (has_first_asset or has_last_asset or self.imageAssetIds):
            raise ValueError("mediaType='image' takes no image asset ids")
        if has_keyframes and (has_first or has_last):
            raise ValueError("Keyframes cannot be combined with a first or last frame")
        if has_last and not has_first:
            raise ValueError("Last frame requires a first-frame image")
        if has_keyframes:
            if len(keyframe_indices) > LOCAL_MULTI_KEYFRAME_MAX_COUNT:
                raise ValueError(
                    f"You can place up to {LOCAL_MULTI_KEYFRAME_MAX_COUNT} keyframes"
                )
            indices = keyframe_indices
            if len(set(indices)) != len(indices):
                raise ValueError("Keyframe frame indices must be unique")
            if self.duration is not None and self.fps is not None and self.fps > 0:
                last_frame = compute_num_frames(self.duration, self.fps) - 1
                for frame_idx in indices:
                    if frame_idx > last_frame:
                        raise ValueError(
                            f"Keyframe frame index {frame_idx} is outside 0..{last_frame}"
                        )
        has_video_path = bool((self.videoPath or "").strip())
        has_video_asset = bool((self.videoAssetId or "").strip())
        if has_video_path and has_video_asset:
            raise ValueError("Use videoPath or videoAssetId, not both")
        has_video = has_video_path or has_video_asset
        if has_video and self.mediaType == "image":
            raise ValueError("A video only applies to mediaType='video'")
        # An IC-LoRA with a look image takes one image beside its video. Nothing else mixes.
        if has_video and (has_last or has_keyframes):
            raise ValueError("A video cannot be combined with last-frame or keyframe input")
        if has_video and has_first and self.icLoraId is None:
            raise ValueError("A video combines with an image only for an IC-LoRA look image")
        if has_video and (
            self.loraCatalogIds or self.customLoraIds or self.conditioningType is not None
        ):
            raise ValueError("A video only combines with icLoraId or no selection")
        if self.replacedWindow is not None and (not has_video or self.icLoraId is not None):
            raise ValueError("replacedWindow needs a video and does not combine with icLoraId")
        if self.imagePaths and self.imageAssetIds:
            raise ValueError("Use imagePaths or imageAssetIds, not both")
        has_panels = bool(self.imagePaths or self.imageAssetIds)
        if has_panels and self.icLoraId is None:
            raise ValueError("A list of images needs an icLoraId")
        if has_panels and (has_first or has_last or has_keyframes or has_video):
            raise ValueError("A list of images cannot be combined with another image or a video")
        if (
            not self.prompt.strip()
            and not has_first
            and not has_keyframes
            and not has_video
            and not has_panels
        ):
            raise ValueError("Prompt is required unless an image or video is provided")
        return self


class EnhancePromptResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    enhancedPrompt: str


class PromptEnhancerStatusResponse(BaseModel):
    """Availability for the Explore Enhance control. Does not expose API keys or paths."""

    model_config = ConfigDict(strict=True)
    exploreAutoEnhancePrompts: bool
    hasGeminiApiKey: bool
    localEnhancementSupported: bool
    defaultProvider: Literal["local", "api"]
    canToggleProvider: bool
    showManualEnhance: bool
    nagUnavailableReason: str | None = None
    nagUnavailableReasonByFeature: dict[str, str] = Field(default_factory=dict)
    # The Desktop setting that shows the per-run NAG strength row; Remote follows it.
    showNagStrengthControls: bool = False


class IcLoraListItem(BaseModel):
    model_config = ConfigDict(strict=True)
    ic_lora: IcLoraCatalogItem
    downloaded: bool
    # Subset of download.variants[].id present on disk (empty when none installed).
    downloaded_variant_ids: list[str] = Field(default_factory=list)


class IcLoraListResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    ic_loras: list[IcLoraListItem]
    budgetMode: BudgetMode = "brave"


# --- Plain LoRA catalog (list + download). Mirrors the IC-LoRA flow with a separate
#     download session so it never interferes with an in-flight IC-LoRA download. ---
class LoraListItem(BaseModel):
    model_config = ConfigDict(strict=True)
    lora: LoraCatalogItem
    downloaded: bool
    # Subset of download.variants[].id present on disk (empty when none installed).
    downloaded_variant_ids: list[str] = Field(default_factory=list)


class LoraListResponse(BaseModel):
    model_config = ConfigDict(strict=True)
    loras: list[LoraListItem]


class GeminiModelOptionPayload(BaseModel):
    model_config = ConfigDict(strict=True)
    id: str
    displayName: str
    description: str = ""


class GeminiModelsResponsePayload(BaseModel):
    model_config = ConfigDict(strict=True)
    models: list[GeminiModelOptionPayload]
    resolvedModel: str


MediaKind = Literal["image", "video", "audio"]
Origin = Literal["uploaded", "generated"]
GenerationStatus = Literal[
    "queued", "running", "cancelling", "succeeded", "failed", "cancelled"
]
GenerationErrorCode = Literal[
    "INTERRUPTED",
    "INPUT_MISSING",
    "OUTPUT_MISSING",
    "OUTPUT_UNREADABLE",
    "EXECUTOR_FAILED",
    # Mid-run CUDA out-of-memory, classified by message (is_cuda_oom_error).
    # A policy-miss signal: the job was admitted but did not fit. Distinct
    # from EXECUTOR_FAILED so capacity analysis can separate the two.
    "CUDA_OOM",
    # Execute-time budget miss: the job passed create-time checks but did not
    # fit when its turn came (free VRAM drifted while queued). The runner maps
    # HTTPError/CapabilityFailedError budget codes here instead of swallowing
    # them as EXECUTOR_FAILED/CAPABILITY_FAILED. Distinct from CUDA_OOM
    # (mid-run death) — this one never started.
    "JOB_TOO_LARGE",
    "CAPABILITY_FAILED",
    # A queued job's LTX key was rejected, or prompt embedding failed.
    # Other capability failures stay CAPABILITY_FAILED.
    "LTX_INVALID_API_KEY",
    "LTX_API_PROMPT_EMBEDDING_FAILED",
    # The image provider's safety checker returned no image for the prompt.
    "IMAGE_BLOCKED_BY_SAFETY_CHECKER",
]


class ImageMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow", strict=True)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class VideoMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow", strict=True)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    durationMs: int = Field(ge=1)
    sizeBytes: int = Field(ge=0)
    audioStreamCount: int = Field(ge=0)
    # Written at ingest. Older rows omit it; create probes the file then.
    fps: float | None = Field(default=None, gt=0)


class AudioMeta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow", strict=True)
    durationMs: int = Field(ge=1)
    bitrate: int | None = Field(default=None, ge=1)


class ImageAssetMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    mediaType: Literal["image"]
    metadata: ImageMeta


class VideoAssetMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    mediaType: Literal["video"]
    metadata: VideoMeta


class AudioAssetMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    mediaType: Literal["audio"]
    metadata: AudioMeta


AssetMetadata = Annotated[
    ImageAssetMetadata | VideoAssetMetadata | AudioAssetMetadata,
    Field(discriminator="mediaType"),
]


def require_media_kind_matches_metadata(
    media_kind: MediaKind, metadata: AssetMetadata
) -> None:
    if metadata.mediaType != media_kind:
        raise ValueError(
            f"media_kind {media_kind!r} does not match metadata.mediaType {metadata.mediaType!r}"
        )


def require_failed_has_error_code(
    status: GenerationStatus, error_code: GenerationErrorCode | None
) -> None:
    if status == "failed" and error_code is None:
        raise ValueError("failed generation requires error_code")
    if status != "failed" and error_code is not None:
        raise ValueError("error_code is only valid when status is failed")


class AssetCore(BaseModel):
    """Identity, media kind, and metadata shared by Desktop and Remote assets.

    Filesystem paths stay on Desktop `Asset` only. Remote `RemoteAsset` must not
    inherit those fields.
    """

    model_config = ConfigDict(strict=True)
    id: str = Field(min_length=1)
    media_kind: MediaKind
    origin: Origin
    mime_type: str = Field(min_length=1)
    name: str = Field(min_length=1)
    metadata: AssetMetadata
    created_at: int = Field(ge=0)

    @model_validator(mode="after")
    def _media_kind_matches_metadata(self) -> Self:
        require_media_kind_matches_metadata(self.media_kind, self.metadata)
        return self


class Asset(AssetCore):
    model_config = ConfigDict(strict=True)
    path: str = Field(min_length=1)
    thumbnail_path: str | None = None


class AssetListItem(Asset):
    in_use: bool


class AssetListQuery(BaseModel):
    model_config = ConfigDict(strict=True)
    media_kind: MediaKind | None = None
    sort: str = "created_at-desc"
    q: str | None = None
    cursor: str | None = None
    limit: int = Field(default=100, ge=1, le=200)


class AssetListResponse(BaseModel):
    items: list[AssetListItem]
    next_cursor: str | None = None


class GenerationCore(BaseModel):
    """Lifecycle fields shared by Desktop and Remote generations.

    `spec` and `outputs` stay on the concrete models: Desktop keeps generic JSON
    plus `Asset` outputs; Remote keeps the allowlisted spec plus `RemoteAsset`
    outputs.
    """

    model_config = ConfigDict(strict=True)
    id: str = Field(min_length=1)
    feature: str = Field(min_length=1)
    contract_version: int = Field(ge=1)
    status: GenerationStatus
    error_code: GenerationErrorCode | None = None
    created_at: int = Field(ge=0)
    queued_at: int = Field(ge=0)
    attempt_count: int = Field(ge=0)
    started_at: int | None = None
    finished_at: int | None = None
    risk: GenerationRisk | None = None
    # The execution lane the job runs on ("local" or a provider). Set when the job is queued.
    lane: str = Field(default="local", min_length=1)
    # The id of the job at its provider. Null for a local job and until the provider accepts it.
    provider_job_id: str | None = None

    @model_validator(mode="after")
    def _failed_has_error_code(self) -> Self:
        require_failed_has_error_code(self.status, self.error_code)
        return self


class Generation(GenerationCore):
    model_config = ConfigDict(strict=True)
    spec: dict[str, JsonValue]
    outputs: list[Asset]


# What a provider reports for a running external job. Memory only, never stored.
ApiActivityStatus = Literal["pending", "processing"]

# Who rewrote the prompt of a run: local Gemma, Gemini, or the provider's own expansion.
PromptRewriteSource = Literal["local", "gemini", "fal_expansion"]


class QueueProgress(BaseModel):
    """Live progress for the persisted active queue entry."""

    model_config = ConfigDict(strict=True)
    phase: str
    progress: int
    currentStep: int | None
    totalSteps: int | None
    remainingSeconds: int | None = Field(default=None, exclude_if=lambda value: value is None)


class QueueEntry(BaseModel):
    model_config = ConfigDict(strict=True)
    generation: Generation
    input_assets: list[Asset]
    progress: QueueProgress | None = None
    # Live queue facts, never stored. A queued or finished row is always cancellable.
    cancellable: bool = True
    api_status: ApiActivityStatus | None = None


class QueueSnapshot(BaseModel):
    model_config = ConfigDict(strict=True)
    # One entry per running job. Jobs on different lanes run at the same time.
    active: list[QueueEntry] = Field(default_factory=list[QueueEntry])
    queued: list[QueueEntry]
    done: list[QueueEntry] = Field(default_factory=list[QueueEntry])
    failed: list[QueueEntry] = Field(default_factory=list[QueueEntry])
    unseen_ids: list[str] = Field(default_factory=list[str])


class IngestAssetRequest(BaseModel):
    model_config = ConfigDict(strict=True)
    path: str = Field(min_length=1)


class ReorderGenerationQueueRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    generation_id: str = Field(min_length=1)
    before_generation_id: str | None = Field(default=None, min_length=1)


class TrimMediaRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    startSec: float = Field(ge=0, allow_inf_nan=False)
    endSec: float = Field(gt=0, allow_inf_nan=False)


class _QueuedVideoParams(NagParams):
    model_config = ConfigDict(strict=True, extra="forbid")

    resolution: LTXVideoGenResolution = "1080p"
    model: OfferingId
    cameraMotion: VideoCameraMotion = "none"
    duration: LTXVideoGenDuration | None = 5
    fps: LTXVideoGenFps = 24
    seed: int | None = None
    promptProvenance: PromptProvenance = "typed"
    passShape: PassShape = "auto"


class TextToVideoParams(_QueuedVideoParams):
    """Non-media text-to-video params stored on a generation row."""

    prompt: NonEmptyPrompt
    aspectRatio: LTXVideoGenAspectRatio = "16:9"
    loras: list[LoraEntry] = Field(default_factory=list[LoraEntry])


class ImageToVideoParams(_QueuedVideoParams):
    """Non-media image-to-video params stored on a generation row."""

    prompt: str
    aspectRatio: LTXQueuedAspectRatio = "auto"


class ImageToVideoRecipeParams(ImageToVideoParams):
    """Stored params for an i2v LoRA recipe.

    Public ``CreateImageToVideoRequest`` stays loras-free. Recipe rows carry the
    catalog id + scale (empty ``ref``) the same way t2v recipes do; execute
    hydrates ``ref`` via ``lora_resolver``.
    """

    loras: list[LoraEntry] = Field(default_factory=list[LoraEntry])


class LoraRecipeParams(_QueuedVideoParams):
    """Create-body params for a LoRA recipe.

    The client sends the catalog id + scale of the recipe's adapter, never a
    filesystem ``ref``. Create stores ``loras: [{ ref: "", scale, catalogId }]``;
    execute hydrates the installed path (see ``make_recipe_lora_resolver``).
    ``prompt`` is the raw user scene; the recipe's prompt scaffold is applied at
    execution time. ``aspectRatio`` includes ``auto`` for i2v recipes (resolved
    from the start frame at execute); t2v recipes reject ``auto``.
    """

    prompt: NonEmptyPrompt
    aspectRatio: LTXQueuedAspectRatio = "16:9"
    catalogId: str = Field(pattern=r"^[a-z0-9-]+$", max_length=64)
    scale: float = Field(default=1.0, ge=0.0, le=4.0)
    # Omitted resolves to the only checkpoint, or to index 0 when nothing is installed.
    variantId: str | None = None


class InputAssetRef(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    assetId: str = Field(min_length=1)


class ImageToVideoInputs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    startFrame: InputAssetRef
    endFrame: InputAssetRef | None = None


class KeyframesParams(_QueuedVideoParams):
    """Non-media params of a Home Keyframes job. Several stills guide one clip.

    The keyframes need a known length, so ``duration`` cannot be ``None``. A blank prompt
    is refused: the generate request takes a blank one only with a start image.
    """

    prompt: NonEmptyPrompt
    aspectRatio: LTXQueuedAspectRatio = "auto"

    @field_validator("duration")
    @classmethod
    def _duration_is_set(cls, value: LTXVideoGenDuration | None) -> LTXVideoGenDuration:
        if value is None:
            raise ValueError("Keyframes cannot be combined with automatic duration")
        return value


class KeyframesInputs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    keyframes: list[KeyframeRef] = Field(min_length=1, max_length=LOCAL_MULTI_KEYFRAME_MAX_COUNT)


class LoraRecipeInputs(BaseModel):
    """Start frame for an i2v LoRA recipe. t2v recipes omit this object.

    ``endFrame`` is only valid for recipes that require it (Transition).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    startFrame: InputAssetRef
    endFrame: InputAssetRef | None = None


class CreateLoraRecipeRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    params: LoraRecipeParams
    inputs: LoraRecipeInputs | None = None
    contract_version: int = Field(default=1, ge=1)


class CustomLoraParams(_QueuedVideoParams):
    """Create-body params for a custom LoRA run.

    The client sends the custom LoRA id + scale, never a filesystem path. Create stores
    ``loras: [{ ref: "", scale, displayName, customLoraId }]`` and execute hydrates the
    weights path (see ``make_custom_lora_resolver``). ``aspectRatio`` includes ``auto`` only
    when the run has a start frame (resolved from the start frame at execute); a run without
    one rejects ``auto``.
    """

    prompt: NonEmptyPrompt
    aspectRatio: LTXQueuedAspectRatio = "16:9"
    customLoraId: CustomLoraId
    scale: float = Field(default=1.0, ge=0.0, le=4.0)


class CreateCustomLoraRequest(BaseModel):
    """``inputs`` carries the optional Start Frame. With it the run is image to video. Without it, text to video."""

    model_config = ConfigDict(strict=True, extra="forbid")

    params: CustomLoraParams
    inputs: LoraRecipeInputs | None = None
    contract_version: int = Field(default=1, ge=1)


class CreateTextToVideoRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    params: TextToVideoParams
    contract_version: int = Field(default=1, ge=1)


ImageProvider: TypeAlias = Literal["local", "fal"]

ImageProviderBlocker: TypeAlias = Literal[
    "FAL_API_KEY_NOT_CONFIGURED",
    "LOCAL_GENERATION_UNSUPPORTED",
    "IMAGE_MODEL_NOT_DOWNLOADED",
]

IMAGE_MIN_SIDE = 256
IMAGE_MAX_SIDE = 2048
IMAGE_SIDE_GRID = 16


class _TextToImageFields(BaseModel):
    """Shared text-to-image params. No image input exists: Z-Image edit is not used."""

    model_config = ConfigDict(strict=True, extra="forbid")

    prompt: NonEmptyPrompt = Field(max_length=4000)
    promptProvenance: PromptProvenance = "typed"
    width: int = Field(default=1024, ge=IMAGE_MIN_SIDE, le=IMAGE_MAX_SIDE)
    height: int = Field(default=1024, ge=IMAGE_MIN_SIDE, le=IMAGE_MAX_SIDE)
    seed: int | None = None


class TextToImageParams(_TextToImageFields):
    """Create-body params. An omitted provider resolves from the settings at create time."""

    provider: ImageProvider | None = None


class TextToImageStoredParams(_TextToImageFields):
    """Params stored on the row: the provider is always resolved, so a retry keeps it."""

    provider: ImageProvider


class ImageGenerationStatusResponse(BaseModel):
    """Which image providers can run now. A blocker is the code create would refuse with."""

    model_config = ConfigDict(strict=True, extra="forbid")
    defaultProvider: ImageProvider
    localBlocker: ImageProviderBlocker | None
    falBlocker: ImageProviderBlocker | None


class CreateTextToImageRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    params: TextToImageParams
    contract_version: int = Field(default=1, ge=1)


_UUID_PATTERN = r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"


class AssistantTurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(pattern=_UUID_PATTERN)
    conversation_id: str = Field(pattern=_UUID_PATTERN)
    text: NonEmptyPrompt = Field(max_length=12000)
    asset_ids: list[str] = Field(default_factory=list, max_length=ASSISTANT_LIMITS.attachments)
    provider: Literal["gemini", "local"] | None = None
    model: str | None = None
    # Set by the route, never the client: which app the user sent from, for generation analytics.
    origin: SkipJsonSchema[Literal["desktop", "remote"]] = "desktop"


class AssistantTurnResponse(BaseModel):
    text: str
    generation_ids: list[str] = Field(default_factory=list)
    asset_ids: list[str] = Field(default_factory=list)
    run_id: str
    status: Literal["waiting", "paused", "complete", "stopped"] = "complete"


class AssistantConversationTurn(BaseModel):
    request: AssistantTurnRequest
    response: AssistantTurnResponse | None
    provider: Literal["gemini", "local"] | None = None
    model: str = ""
    # Without a response, False means the turn failed and can be resent unchanged.
    running: bool = False


class AssistantQueueRequest(BaseModel):
    """A message for the chat's queue. `id` becomes the turn's request id when it starts."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=_UUID_PATTERN)
    text: NonEmptyPrompt = Field(max_length=12000)
    asset_ids: list[str] = Field(default_factory=list, max_length=ASSISTANT_LIMITS.attachments)
    provider: Literal["gemini", "local"] | None = None
    model: str | None = None


class AssistantQueuedMessage(BaseModel):
    id: str
    text: str
    asset_ids: list[str]
    provider: Literal["gemini", "local"] | None = None
    model: str | None = None
    # Why it could not start on its last attempt; the chat is on hold until it is released or removed.
    error: str | None = None


class AssistantConversation(BaseModel):
    conversation_id: str
    turns: list[AssistantConversationTurn]
    queue: list[AssistantQueuedMessage] = Field(default_factory=list[AssistantQueuedMessage])
    # Stop and a Desktop restart hold the queue until someone releases it.
    queue_on_hold: bool = False


class AssistantConversationSummary(BaseModel):
    conversation_id: str
    title: str
    updated_at: int


class AssistantSelection(BaseModel):
    """The Desktop's assistant choice, readable by Remote, which cannot open Settings."""

    provider: Literal["gemini", "local"]
    local_model: LocalAssistantModel
    gemini_model: str
    has_gemini_api_key: bool


class AssistantSelectionRequest(BaseModel):
    """The one assistant choice Desktop Settings and both composers share."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["gemini", "local"]
    local_model: LocalAssistantModel | None = None


class AssistantProgress(BaseModel):
    active: bool = False
    activity: str = ""
    text: str = ""
    # Last event of the stream: the run needs the user before it can progress again.
    done: bool = False


class CreateImageToVideoRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    params: ImageToVideoParams
    inputs: ImageToVideoInputs
    contract_version: int = Field(default=1, ge=1)


class CreateKeyframesRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    params: KeyframesParams
    inputs: KeyframesInputs
    contract_version: int = Field(default=1, ge=1)


class CreateAudioToVideoParams(NagParams):
    """Public non-media A2V inputs; the server derives inference frames from audio."""

    model_config = ConfigDict(strict=True, extra="forbid")

    prompt: str
    aspectRatio: LTXQueuedAspectRatio = "16:9"
    resolution: LTXLocalA2VResolution = "540p"
    model: OfferingId
    fps: LTXVideoGenFps = 24
    cameraMotion: VideoCameraMotion = "none"
    seed: int | None = None
    promptProvenance: PromptProvenance = "typed"
    passShape: PassShape = "auto"


class AudioToVideoParams(CreateAudioToVideoParams):
    """Authoritative A2V parameters persisted on the queued generation."""

    numFrames: int = Field(ge=9)

    @model_validator(mode="after")
    def _require_a2v_frame_grid(self) -> "AudioToVideoParams":
        if (self.numFrames - 1) % 8 != 0:
            raise ValueError("numFrames must be on the 8k+1 frame grid")
        return self


class AudioToVideoInputs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    audio: InputAssetRef
    startFrame: InputAssetRef | None = None


class CreateAudioToVideoRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    params: CreateAudioToVideoParams
    inputs: AudioToVideoInputs
    contract_version: int = Field(default=1, ge=1)


# Longest selection one Home/Remote retake regenerates, at every resolution.
# Product limit on new requests (`RetakeRequestParams`); the legacy editor
# endpoint (`RetakeRequest`) is not capped.
RETAKE_MAX_DURATION_SECONDS = 10.0


class RetakeParams(NagParams):
    """Queued Home/Remote retake.

    ``resolution`` omitted keeps the source size. Either way the picture is
    capped at the local 1080p cell. The model encodes a black-bar letterbox
    up to the next multiple of 32, and the stitch crops back to this picture.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    prompt: str = ""
    model: OfferingId
    startTime: float = Field(ge=0, allow_inf_nan=False)
    duration: float = Field(ge=2, allow_inf_nan=False)
    mode: RetakeMode = "replace_audio_and_video"
    resolution: TargetResolution | None = None
    seed: int | None = None
    # Accepted so the shared Home form can submit provenance like every other
    # queued video feature. It decides whether auto-enhance may rewrite the prompt
    # ("enhanced" means the user already did). The server-side enhance flag stays off.
    promptProvenance: PromptProvenance = "typed"


class RetakeInputs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    video: InputAssetRef


class RetakeRequestParams(RetakeParams):
    """New retakes only: specs queued before the cap still parse as ``RetakeParams``."""

    duration: float = Field(
        ge=2, le=RETAKE_MAX_DURATION_SECONDS, allow_inf_nan=False
    )


class CreateRetakeRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    params: RetakeRequestParams
    inputs: RetakeInputs
    contract_version: int = Field(default=1, ge=1)


class ExtendParams(NagParams):
    """Queued Home/Remote extend.

    ``resolution`` omitted keeps the source size. Either way the picture is
    capped at the local 1080p cell. The model encodes a black-bar letterbox
    up to the next multiple of 32, and the stitch crops back to this picture.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    prompt: str = ""
    model: OfferingId
    duration: float = Field(ge=2, le=20, allow_inf_nan=False)
    mode: ExtendMode = "end"
    resolution: TargetResolution | None = None
    seed: int | None = None
    # Accepted so the shared Home form can submit provenance like every other
    # queued video feature. It decides whether auto-enhance may rewrite the prompt
    # ("enhanced" means the user already did). The server-side enhance flag stays off.
    promptProvenance: PromptProvenance = "typed"


class ExtendInputs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    video: InputAssetRef


class CreateExtendRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    params: ExtendParams
    inputs: ExtendInputs
    contract_version: int = Field(default=1, ge=1)


EditPreflightFeature: TypeAlias = Literal["retake", "extend"]


class EditPreflightCandidate(BaseModel):
    """One resolution tier the UI displays, echoed back by key.

    ``width``/``height`` None selects the source size ("original"); otherwise
    both must be set. The backend resolves each exactly like a create's
    ``params.resolution`` (1080p cap, then the ÷32 snap), so verdicts match
    what Generate would do.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    key: str = Field(min_length=1)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_size_is_both_or_neither(self) -> Self:
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height must be set together")
        return self


class EditPreflightRequest(BaseModel):
    """Advisory per-cell verdicts for Home retake/extend pickers.

    ``durations`` always includes the smallest selectable duration (extend
    presets, retake minimum) so every candidate has an exact resolution-level
    verdict; callers add the durations they display (extend presets, retake
    current selection) for exact per-cell messages. ``startTime`` only affects
    retake windowing, ``mode`` only extend stitching; both default so each
    feature sends what it needs.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    feature: EditPreflightFeature
    assetId: str = Field(min_length=1)
    startTime: float = Field(default=0, ge=0, allow_inf_nan=False)
    mode: ExtendMode = "end"
    candidates: list[EditPreflightCandidate] = Field(min_length=1, max_length=8)
    durations: list[float] = Field(min_length=1, max_length=32)

    @field_validator("durations")
    @classmethod
    def _check_durations_positive(cls, value: list[float]) -> list[float]:
        for duration in value:
            # NaN fails the first comparison, +inf the second; both 422 here
            # instead of blowing up in frame-count math downstream.
            if not (duration > 0 and duration < float("inf")):
                raise ValueError("durations must be positive")
        return value


class EditPreflightSource(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    frames: int = Field(gt=0)
    durationSeconds: float = Field(gt=0)


class EditPreflightResolution(BaseModel):
    """Per-candidate summary. ``width``/``height`` is the resolved encode size.

    ``maxDurationSeconds`` is the longest duration that passes at this size,
    floored to 0.5s for slider clamps; 0 when even the smallest requested
    duration fails. Authoritative per-cell verdicts live in ``cells`` — this
    row never overrides them.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    key: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    maxDurationSeconds: float = Field(ge=0)


class EditPreflightCell(BaseModel):
    """Verdict for one (candidate, duration) cell. ``ok`` means Generate would
    enqueue it in the echoed mode. An ok cell admitted on a bet carries the bet
    line as ``message``; a failing cell carries the exact refusal Generate
    would return."""

    model_config = ConfigDict(strict=True, extra="forbid")

    key: str = Field(min_length=1)
    duration: float
    ok: bool
    message: str | None = None
    code: str | None = None
    # The bet: admitted on an ok cell, refused on a failing one. Hard refusals
    # (window math, unsupported) leave these empty.
    tier: JobRiskTier | None = None
    over_gib: float | None = None
    over_ram_gb: int | None = None
    shortfall_kind: ShortfallKind | None = None
    readout: BudgetReadout | None = None
    estimated_seconds: int | None = None


class EditPreflightResponse(BaseModel):
    """Path-free (asset dims + verdicts only), so the Remote router reuses it."""

    model_config = ConfigDict(strict=True, extra="forbid")

    source: EditPreflightSource
    resolutions: list[EditPreflightResolution]
    cells: list[EditPreflightCell]
    budgetMode: BudgetMode = "brave"


class IcLoraPreflightRequest(BaseModel):
    """Advisory verdicts for a catalog IC-LoRA input before Generate.

    Mirrors the generate probe check (source pixels → output canvas → budget)
    without preprocessing or loading weights. ``durations`` lists the seconds
    to verdict for image-input entries; video-input entries transform the
    whole clip, so requested durations are ignored and one cell at the clip
    length is returned. Sizing overrides mirror the generate request (None
    keeps the entry default); ``fps_override`` is intentionally absent —
    generate ignores it on the catalog probe path, and preflight must agree.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    ic_lora_id: str = Field(min_length=1)
    input_path: str = Field(min_length=1)
    durations: list[float] = Field(min_length=1, max_length=32)
    skip_stage_2: bool | None = None
    resolution_factor: float | None = Field(default=None, ge=0.0, le=2.0)
    resolution: TargetResolution | None = None

    @field_validator("durations")
    @classmethod
    def _check_durations_positive(cls, value: list[float]) -> list[float]:
        for duration in value:
            # NaN fails the first comparison, +inf the second; both 422 here
            # instead of blowing up in frame-count math downstream.
            if not (duration > 0 and duration < float("inf")):
                raise ValueError("durations must be positive")
        return value


class IcLoraPreflightSource(BaseModel):
    """Probed source dims. ``frames`` is the snapped clip length for video
    inputs; images have no source frames, so it rides None there."""

    model_config = ConfigDict(strict=True, extra="forbid")

    kind: InputKind
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frames: int | None = Field(default=None, gt=0)
    fps: float = Field(gt=0)


class IcLoraPreflightVerdict(BaseModel):
    """Verdict for one candidate. ``ok`` means Generate would enqueue it in the echoed mode.

    A failing cell carries the exact refusal Generate would return. An ok cell admitted on a
    bet carries the bet (``tier`` and ``message``).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    ok: bool
    message: str | None = None
    code: str | None = None
    # The bet: admitted on an ok cell, refused on a failing one. Hard refusals
    # leave these empty.
    tier: JobRiskTier | None = None
    over_gib: float | None = None
    over_ram_gb: int | None = None
    shortfall_kind: ShortfallKind | None = None
    readout: BudgetReadout | None = None


class IcLoraPreflightCell(IcLoraPreflightVerdict):
    """Verdict for one duration. ``message``/``code`` are None when ``ok``;
    otherwise the exact refusal Generate would return for this input."""

    duration: float


class IcLoraPreflightResponse(BaseModel):
    """Source dims + verdicts only; the input path is never echoed."""

    model_config = ConfigDict(strict=True, extra="forbid")

    source: IcLoraPreflightSource
    cells: list[IcLoraPreflightCell]
    budgetMode: BudgetMode = "brave"


# 1440p and 2160p run on AlphaGen only, behind the highResVideoToVideo feature flag.
IcLoraRecipeResolution: TypeAlias = Literal[
    "270p", "360p", "540p", "720p", "1080p", "1440p", "2160p"
]


class IcLoraRecipeMemorySource(BaseModel):
    """The picked clip: its file size and its frames at the output rate."""

    model_config = ConfigDict(strict=True, extra="forbid")

    width: int = Field(gt=0, le=16384)
    height: int = Field(gt=0, le=16384)
    frames: int = Field(gt=0, le=10000)


class IcLoraRecipeMemoryRequest(BaseModel):
    """Memory advice for a Home video recipe, before and after a clip is picked.

    ``cells`` are the output cells the form offers. Without ``source`` each is priced for a 16:9
    clip of that size. ``resolution`` is the selected cell, priced for ``source`` when given.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    model: OfferingId
    resolution: IcLoraRecipeResolution
    cells: list[IcLoraRecipeResolution] = Field(max_length=7)
    audioMode: IcLoraAudioMode | None = None
    scale: float | None = Field(default=None, ge=0.0, le=2.0)
    source: IcLoraRecipeMemorySource | None = None
    # The Advanced choices of the form, as in the create body. None is Auto. The advice then
    # prices the same tiling the run uses.
    tilingOverlap: float | None = Field(default=None, ge=MIN_TILE_OVERLAP, le=MAX_TILE_OVERLAP)
    carryFrames: int | None = Field(default=None, ge=MIN_CARRY_FRAMES)
    tilingSize: IcLoraTileSize | None = None
    maxWindowFrames: int | None = Field(default=None, ge=MIN_WINDOW_FRAMES)


class IcLoraCellFit(BaseModel):
    """How long a clip one output cell fits on this machine.

    ``max_frames`` is the longest clip, in frames at the output rate, whose job loads. 0 when none
    does. ``full`` is a clip at the input cap.
    """

    resolution: IcLoraRecipeResolution
    max_frames: int
    full: bool


class IcLoraRecipeMemoryResponse(BaseModel):
    memory: MemoryCapacity
    # The longest clip a recipe takes, in frames. Every ``IcLoraCellFit`` is measured against it.
    max_input_frames: int
    cells: list[IcLoraCellFit]
    # The selected cell for the picked clip. None without a source, or when the cell cannot run.
    selected: JobMemoryAsk | None = None


class _IcLoraRecipeCore(NagParams):
    model_config = ConfigDict(strict=True, extra="forbid")

    prompt: str = ""
    model: OfferingId
    resolution: IcLoraRecipeResolution = "720p"
    seed: int | None = None
    promptProvenance: PromptProvenance = "typed"


class IcLoraRecipeParams(_IcLoraRecipeCore):
    """Create-body params. Strength is ``scale``; the stored spec uses ``loras``.

    A ``loras`` array on the create body is rejected. The path id is the recipe,
    so the body does not carry a catalog id. The server writes the checkpoint
    it actually ran. Omitted audio and strength are filled from the catalog.
    """

    audioMode: IcLoraAudioMode | None = None
    scale: float | None = Field(default=None, ge=0.0, le=2.0)
    # Omitted fps is the supported rate closest to the source, never above it.
    fps: float | None = Field(default=None, gt=0)
    # Omitted resolves to the only checkpoint, or to index 0 when nothing is installed.
    variantId: str | None = None


class IcLoraVideoRecipeParams(IcLoraRecipeParams):
    """Create-body params of a video recipe."""

    # Only a recipe whose catalog entry has a control_type control takes it. Create fills the
    # catalog default when it is omitted, and refuses it for any other recipe.
    controlType: IcLoraControlType | None = None
    # Advanced controls of a recipe whose catalog entry has a tile plan (Refine Details). A
    # recipe without one refuses them. Omitted keeps the catalog tile overlap and the carry
    # of the cell.
    tilingOverlap: float | None = Field(default=None, ge=MIN_TILE_OVERLAP, le=MAX_TILE_OVERLAP)
    carryFrames: int | None = Field(default=None, ge=MIN_CARRY_FRAMES)
    tilingSize: IcLoraTileSize | None = None
    maxWindowFrames: int | None = Field(default=None, ge=MIN_WINDOW_FRAMES)


IcLoraTilePreset: TypeAlias = Literal["standard", "large", "extra-large"]


class IcLoraTilePreflightRequest(BaseModel):
    """Advisory verdict per tile size and per window for a recipe create.

    The body is the create body. Each requested tile size replaces its ``tilingSize``, each
    requested maximum chunk length replaces its ``maxWindowFrames``, and each requested carry replaces its
    ``carryFrames``. The other choice of the body stays. ``standard`` is the tile of the
    catalog entry. An empty prompt is accepted here. Create still refuses one. The response
    also gives the carry, window and tile the run uses when the body omits them.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    params: IcLoraVideoRecipeParams
    inputs: IcLoraRecipeInputs
    tileSizes: list[IcLoraTilePreset] = Field(default_factory=list[IcLoraTilePreset], max_length=3)
    maxWindows: list[Annotated[int, Field(ge=MIN_WINDOW_FRAMES)]] = Field(default_factory=list[int], max_length=4)
    # Carries to judge against the window the body holds. The form sends the shared preset list.
    carries: list[Annotated[int, Field(ge=MIN_CARRY_FRAMES)]] = Field(default_factory=list[int], max_length=8)


class IcLoraTilePreflightCell(IcLoraPreflightVerdict):
    tileSize: IcLoraTilePreset


class IcLoraMaxWindowPreflightCell(IcLoraPreflightVerdict):
    maxWindowFrames: int


class IcLoraCarryPreflightCell(IcLoraPreflightVerdict):
    carryFrames: int


class IcLoraAutoSinglePass(BaseModel):
    """Auto runs the whole clip in one pass."""

    model_config = ConfigDict(strict=True, extra="forbid")

    kind: Literal["single-pass"] = "single-pass"


class IcLoraAutoWindowed(BaseModel):
    """Auto runs the clip in chunks of ``frames`` frames."""

    model_config = ConfigDict(strict=True, extra="forbid")

    kind: Literal["windowed"] = "windowed"
    frames: int = Field(gt=0)


IcLoraAutoChunking: TypeAlias = Annotated[
    IcLoraAutoSinglePass | IcLoraAutoWindowed, Field(discriminator="kind")
]


class IcLoraTilingDefaults(BaseModel):
    """What Auto runs when the request omits a tiling control.

    ``carryFrames`` is the Auto carry. ``overlapFraction`` is the Auto tile overlap of the LoRA.
    ``auto`` is the Auto chunk length, picked from the clip and the resolution. It is None when
    Auto is refused: the cells carry the refusal.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    carryFrames: int
    overlapFraction: float | None = None
    auto: IcLoraAutoChunking | None = None


class IcLoraTilePreflightResponse(BaseModel):
    """Verdicts only. The input is never echoed."""

    model_config = ConfigDict(strict=True, extra="forbid")

    cells: list[IcLoraTilePreflightCell]
    maxWindowCells: list[IcLoraMaxWindowPreflightCell] = Field(default_factory=list[IcLoraMaxWindowPreflightCell])
    carryCells: list[IcLoraCarryPreflightCell] = Field(default_factory=list[IcLoraCarryPreflightCell])
    defaults: IcLoraTilingDefaults
    budgetMode: BudgetMode = "brave"


class IcLoraRecipeStoredParams(_IcLoraRecipeCore):
    """Stored IC-LoRA recipe spec.

    The executor picks the closest local aspect from the source video. Resolution
    stays inside the local 1080p IC-LoRA envelope. ``fps`` is the output rate:
    the crop filter decimates when it is below the source.
    ``loras`` is the one checkpoint this run used. ``audioMode`` is required
    because create resolves it from the catalog when the request omits it.
    """

    audioMode: IcLoraAudioMode
    fps: float = Field(gt=0)
    loras: list[LoraEntry] = Field(default_factory=list[LoraEntry])
    # Advanced tiling controls of a recipe with a catalog tile plan. None keeps the plan.
    tilingOverlap: float | None = Field(default=None, ge=MIN_TILE_OVERLAP, le=MAX_TILE_OVERLAP)
    carryFrames: int | None = Field(default=None, ge=MIN_CARRY_FRAMES)
    tilingSize: IcLoraTileSize | None = None
    maxWindowFrames: int | None = Field(default=None, ge=MIN_WINDOW_FRAMES)

    @model_validator(mode="after")
    def _require_one_catalog_lora(self) -> "IcLoraRecipeStoredParams":
        if len(self.loras) != 1 or not self.loras[0].catalogId:
            raise ValueError("an IC-LoRA recipe spec requires one catalog LoRA")
        return self


class IcLoraVideoRecipeStoredParams(IcLoraRecipeStoredParams):
    """Stored params of a video recipe."""

    controlType: IcLoraControlType | None = None


class IcLoraRecipeInputs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    video: InputAssetRef
    # Reference image for a catalog entry with allows_reference_image.
    image: InputAssetRef | None = None


class CreateIcLoraRecipeRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    params: IcLoraVideoRecipeParams
    inputs: IcLoraRecipeInputs
    contract_version: int = Field(default=1, ge=1)


class _IcLoraSheetShape(BaseModel):
    """What a sheet run takes from the request: there is no source clip to give them."""

    duration: LTXVideoGenDuration
    aspectRatio: LTXVideoGenAspectRatio


class IcLoraSheetParams(IcLoraRecipeParams, _IcLoraSheetShape):
    """Params of an image-list recipe (Ingredients). An omitted ``fps`` is the catalog rate."""


class IcLoraSynthesizedStoredParams(IcLoraRecipeStoredParams, _IcLoraSheetShape):
    """Stored params of a run that makes its own control clip (a sheet or a track)."""


class IcLoraSheetInputs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    # The panels. The job stitches them into one sheet when it runs.
    images: list[InputAssetRef] = Field(min_length=1, max_length=IC_LORA_SHEET_MAX_IMAGES)


class CreateIcLoraSheetRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    params: IcLoraSheetParams
    inputs: IcLoraSheetInputs
    contract_version: int = Field(default=1, ge=1)


class IcLoraTrackParams(IcLoraRecipeParams):
    """Params of the Motion Track recipe. An omitted ``fps`` is the catalog rate.

    There is no ``aspectRatio``: create takes the closest ratio from the image.
    """

    duration: LTXVideoGenDuration


class IcLoraTrackInputs(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    # The first frame. The model also reads it as the picture the paths move over.
    image: InputAssetRef
    # One dot for each path, all on the same image. Each path has its own pacing.
    paths: list[MotionPath] = Field(min_length=1, max_length=MOTION_PATH_MAX_PATHS)


class CreateIcLoraTrackRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    params: IcLoraTrackParams
    inputs: IcLoraTrackInputs
    contract_version: int = Field(default=1, ge=1)


def _ic_lora_create_kind(value: object) -> str:
    if isinstance(value, CreateIcLoraSheetRequest):
        return "sheet"
    if isinstance(value, CreateIcLoraTrackRequest):
        return "track"
    if isinstance(value, dict):
        inputs = cast(dict[str, object], value).get("inputs")
        if isinstance(inputs, dict):
            if "images" in cast(dict[str, object], inputs):
                return "sheet"
            if "paths" in cast(dict[str, object], inputs):
                return "track"
    return "video"


# One create route takes every body. A body with `inputs.images` is a sheet body and one
# with `inputs.paths` is a track body. Every other body is a video body, so a sheet or track
# request that sends neither key fails as a video body that lacks `inputs.video`.
IcLoraRecipeCreate: TypeAlias = Annotated[
    Annotated[CreateIcLoraRecipeRequest, Tag("video")]
    | Annotated[CreateIcLoraSheetRequest, Tag("sheet")]
    | Annotated[CreateIcLoraTrackRequest, Tag("track")],
    Discriminator(_ic_lora_create_kind),
]
