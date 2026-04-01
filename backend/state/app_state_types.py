"""Canonical state model for backend runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from api_types import ModelFileType
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


# ============================================================
# Model file availability (disk truth)
# ============================================================

# Availability and download are orthogonal concerns.
AvailableFiles = dict[ModelFileType, Path | None]


# ============================================================
# Download session
# ============================================================


@dataclass
class FileDownloadRunning:
    file_type: ModelFileType
    target_path: str
    downloaded_bytes: int
    speed_bytes_per_sec: float


@dataclass
class DownloadingSession:
    id: str
    current_running_file: FileDownloadRunning | None
    files_to_download: set[ModelFileType]
    completed_files: set[ModelFileType]
    completed_bytes: int


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


class VideoPipelineWarmth(Enum):
    COLD = "cold"
    WARMING = "warming"
    WARM = "warm"


@dataclass
class VideoPipelineState:
    pipeline: FastVideoPipeline
    warmth: VideoPipelineWarmth
    is_compiled: bool


@dataclass
class ICLoraState:
    pipeline: IcLoraPipeline
    lora_path: str
    depth_pipeline: DepthProcessorPipeline
    depth_model_path: str
    pose_pipeline: PoseProcessorPipeline | None = None
    person_detector_model_path: str | None = None
    pose_model_path: str | None = None
    conditioning_cache: ConditioningCache = field(default_factory=ConditioningCache)


@dataclass
class A2VPipelineState:
    pipeline: A2VPipeline


@dataclass
class RetakePipelineState:
    pipeline: RetakePipeline
    distilled: bool
    quantized: bool


# ============================================================
# Generation state
# ============================================================


@dataclass
class GenerationProgress:
    phase: str
    progress: float
    current_step: int | None
    total_steps: int | None


@dataclass
class GenerationRunning:
    id: str
    progress: GenerationProgress


@dataclass
class GenerationComplete:
    id: str
    result: str | list[str]


@dataclass
class GenerationError:
    id: str
    error: str


@dataclass
class GenerationCancelled:
    id: str


GenerationState = GenerationRunning | GenerationComplete | GenerationError | GenerationCancelled


# ============================================================
# Device slots
# ============================================================


@dataclass
class GpuSlot:
    active_pipeline: VideoPipelineState | ICLoraState | A2VPipelineState | RetakePipelineState | ImageGenerationPipeline
    generation: GenerationState | None


@dataclass
class CpuSlot:
    active_pipeline: ImageGenerationPipeline


# ============================================================
# Startup lifecycle
# ============================================================

# Internal warmup lifecycle markers consumed by AppHandler.default_warmup().

@dataclass
class StartupPending:
    message: str


@dataclass
class StartupLoading:
    current_step: str
    progress: float


@dataclass
class StartupReady:
    pass


@dataclass
class StartupError:
    error: str


StartupState = StartupPending | StartupLoading | StartupReady | StartupError


# ============================================================
# Top-level state
# ============================================================


@dataclass
class AppState:
    available_files: AvailableFiles
    downloading_session: DownloadingSession | None
    gpu_slot: GpuSlot | None
    api_generation: GenerationState | None
    cpu_slot: CpuSlot | None
    text_encoder: TextEncoderState | None
    startup: StartupState
    app_settings: AppSettings
    completed_download_sessions: dict[str, str] = field(default_factory=lambda: {})
