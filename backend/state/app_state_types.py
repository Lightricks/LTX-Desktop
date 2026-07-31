"""Canonical state model for backend runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NewType, Protocol

from api_types import ModelCheckpointID
from state.conditioning_cache import ConditioningCache

if TYPE_CHECKING:
    from state.app_settings import AppSettings
    from services.interfaces import (
        A2VPipeline,
        DepthProcessorPipeline,
        FastVideoPipeline,
        ImageGenerationPipeline,
        IcLoraPipeline,
        PoseProcessorPipeline,
        RetakePipeline,
        TextEncoder,
    )
    import torch


# Download session
# ============================================================


DownloadSessionId = NewType("DownloadSessionId", str)


@dataclass(frozen=True)
class DownloadSessionComplete:
    status: str = "complete"


@dataclass(frozen=True)
class DownloadSessionError:
    error_message: str
    status: str = "error"


DownloadSessionResult = DownloadSessionComplete | DownloadSessionError


def _default_completed_download_sessions() -> dict[DownloadSessionId, DownloadSessionResult]:
    return {}


@dataclass
class FileDownloadRunning:
    file_type: ModelCheckpointID
    target_path: str
    downloaded_bytes: int
    speed_bytes_per_sec: float


@dataclass
class DownloadingSession:
    id: DownloadSessionId
    current_running_file: FileDownloadRunning | None
    files_to_download: set[ModelCheckpointID]
    completed_files: set[ModelCheckpointID]
    completed_bytes: int


# One shape for both catalog download kinds (IC-LoRA + plain LoRA). The kinds keep separate
# state slots below so an IC-LoRA download never collides with a plain-LoRA one; `item_id` is
# the catalog id of whichever kind owns the slot.
@dataclass
class CatalogDownloadSession:
    id: DownloadSessionId
    item_id: str
    downloaded_bytes: int
    expected_bytes: int
    speed_bytes_per_sec: float = 0.0


def _default_completed_ic_lora_download_sessions() -> dict[DownloadSessionId, DownloadSessionResult]:
    return {}


def _default_completed_lora_download_sessions() -> dict[DownloadSessionId, DownloadSessionResult]:
    return {}


# ============================================================
# Text encoding
# ============================================================


@dataclass
class TextEncodingResult:
    video_context: torch.Tensor
    audio_context: torch.Tensor | None


class CachedTextEncoder(Protocol):
    def to(self, device: torch.device) -> "CachedTextEncoder":
        ...


def _new_prompt_cache() -> dict[tuple[str, bool], TextEncodingResult]:
    return {}


@dataclass
class TextEncoderState:
    service: TextEncoder
    prompt_cache: dict[tuple[str, bool], TextEncodingResult] = field(default_factory=_new_prompt_cache)
    api_embeddings: TextEncodingResult | None = None
    cached_encoder: CachedTextEncoder | None = None


# ============================================================
# Pipeline state
# ============================================================


@dataclass
class VideoPipelineState:
    pipeline: FastVideoPipeline
    is_compiled: bool
    runtime_engine: Literal["torch", "mlx"] = "torch"
    loras: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    # gemma_root the pipeline's text encoder was built with. Part of the cache key: switching
    # text-encoding mode (API<->local) changes it, and a cached pipeline built for the other
    # mode must be rebuilt (an API-mode pipeline has a stub encoder that can't encode locally).
    gemma_root: str | None = None


@dataclass
class PoseResources:
    pipeline: PoseProcessorPipeline
    person_detector_model_path: str
    pose_model_path: str


@dataclass
class ICLoraState:
    pipeline: IcLoraPipeline
    lora_path: str
    depth_pipeline: DepthProcessorPipeline | None
    depth_model_path: str | None
    lora_strength: float = 1.0
    pose_resources: PoseResources | None = None
    conditioning_cache: ConditioningCache = field(default_factory=ConditioningCache)
    gemma_root: str | None = None  # cache key — see VideoPipelineState.gemma_root


@dataclass
class A2VPipelineState:
    pipeline: A2VPipeline
    loras: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    gemma_root: str | None = None  # cache key — see VideoPipelineState.gemma_root


@dataclass
class RetakePipelineState:
    pipeline: RetakePipeline
    distilled: bool
    quantized: bool
    gemma_root: str | None = None  # cache key — see VideoPipelineState.gemma_root


# ============================================================
# Generation state
# ============================================================


@dataclass
class GenerationProgress:
    phase: str
    progress: int
    current_step: int | None
    total_steps: int | None


@dataclass
class GenerationRunning:
    id: str
    progress: GenerationProgress


@dataclass
class GenerationComplete:
    id: str
    # None = completed successfully but produced no locally-recoverable artifact
    # (e.g. an API retake that returned a remote payload instead of video bytes).
    result: str | list[str] | None


@dataclass
class GenerationError:
    id: str
    error: str


@dataclass
class GenerationCancelled:
    id: str


GenerationState = GenerationRunning | GenerationComplete | GenerationError | GenerationCancelled


@dataclass
class GpuGeneration:
    state: GenerationState


@dataclass
class ApiGeneration:
    state: GenerationState


ActiveGeneration = GpuGeneration | ApiGeneration


# ============================================================
# Device slots
# ============================================================


@dataclass
class GpuSlot:
    active_pipeline: VideoPipelineState | ICLoraState | A2VPipelineState | RetakePipelineState | ImageGenerationPipeline


@dataclass
class CpuSlot:
    active_pipeline: ImageGenerationPipeline


# HuggingFace auth
# ============================================================


@dataclass(frozen=True)
class HfNotAuthenticated:
    pass


@dataclass(frozen=True)
class HfOAuthPending:
    state: str
    code_verifier: str
    created_at: float


@dataclass(frozen=True)
class HfAuthenticated:
    access_token: str
    expires_at: float


HfAuthState = HfNotAuthenticated | HfOAuthPending | HfAuthenticated


# ============================================================
# Top-level state
# ============================================================


@dataclass
class AppState:
    downloading_session: DownloadingSession | None
    gpu_slot: GpuSlot | None
    active_generation: ActiveGeneration | None
    cpu_slot: CpuSlot | None
    text_encoder: TextEncoderState | None
    app_settings: AppSettings
    completed_download_sessions: dict[DownloadSessionId, DownloadSessionResult] = field(
        default_factory=_default_completed_download_sessions
    )
    hf_auth_state: HfAuthState = field(default_factory=HfNotAuthenticated)
    ic_lora_download_session: CatalogDownloadSession | None = None
    completed_ic_lora_download_sessions: dict[DownloadSessionId, DownloadSessionResult] = field(
        default_factory=_default_completed_ic_lora_download_sessions
    )
    lora_download_session: CatalogDownloadSession | None = None
    completed_lora_download_sessions: dict[DownloadSessionId, DownloadSessionResult] = field(
        default_factory=_default_completed_lora_download_sessions
    )
    # Timestamp (time.monotonic()) of when a generation request passed its "not already running"
    # check, before it does any slow pre-work (pipeline load — can take tens of seconds, worse
    # for image models loading checkpoint shards) and long before active_generation reflects it
    # as GenerationRunning. Without this, a second concurrent request's own "not already running"
    # check would also pass during that window, letting two generations race to load pipelines
    # and start concurrently — whichever loses the start_generation() race gets an error that
    # (via fail_generation) can overwrite the WINNER's still-legitimately-running state.
    # A timestamp (not a bool) so try_reserve_generation_start() can self-expire it — a request
    # that raises on some validation path before ever reaching start_generation()/
    # fail_generation() (both of which clear it) must not block every future generation forever.
    generation_starting_since: float | None = None
    generation_starting_id: str | None = None
    generation_starting_phase: str = "starting"
    generation_start_cancelled: bool = False
