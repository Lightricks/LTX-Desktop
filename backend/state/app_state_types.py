"""Canonical state model for backend runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from state.app_settings import AppSettings
    from services.interfaces import (
        A2VPipeline,
        FastVideoPipeline,
        ImageGenerationPipeline,
        IcLoraPipeline,
        RetakePipeline,
        TextEncoder,
    )
    import torch


# ============================================================
# Model file availability (disk truth)
# ============================================================

ModelFileType = Literal["checkpoint", "upsampler", "text_encoder", "text_encoder_abliterated", "zit"]

# Availability and download are orthogonal concerns.
AvailableFiles = dict[ModelFileType, Path | None]


# ============================================================
# Download session
# ============================================================


@dataclass
class FileDownloadRunning:
    target_path: str
    progress: float
    downloaded_bytes: int
    total_bytes: int
    speed_mbps: float


@dataclass
class FileDownloadCompleted:
    pass


FileDownloadState = FileDownloadRunning | FileDownloadCompleted


@dataclass
class DownloadError:
    error: str


DownloadingSession = None | dict[ModelFileType, FileDownloadState] | DownloadError


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
    lora_path: str | None = None


@dataclass
class ICLoraState:
    pipeline: IcLoraPipeline
    lora_path: str


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
    downloading_session: DownloadingSession
    gpu_slot: GpuSlot | None
    api_generation: GenerationState | None
    cpu_slot: CpuSlot | None
    text_encoder: TextEncoderState | None
    startup: StartupState
    app_settings: AppSettings

    @property
    def is_downloading(self) -> bool:
        match self.downloading_session:
            case dict() as files:
                return any(isinstance(download_state, FileDownloadRunning) for download_state in files.values())
            case _:
                return False
