"""Shared base types for state handlers."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar

from _routes._errors import HTTPError
from runtime_config.model_download_specs import resolve_active_ltx_model_id
from runtime_config.runtime_policy import (
    CUDA_CPU_OFFLOAD_RAM_FLOOR_GB,
    streaming_prefetch_count_for_mode,
)
from runtime_config.video_job_budget import (
    LOCAL_GENERATION_UNSUPPORTED,
    LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
    build_local_job_filter,
    resolve_pass_shape,
)
from services.ltx_pipeline_common import offload_mode_for_prefetch_count
from state.app_settings import MAX_GENERATION_SEED, resolved_use_conv_vae
from state.app_state_types import AppState

if TYPE_CHECKING:
    from ltx_pipelines.utils.types import OffloadMode

    from api_types import LTXLocalModelId
    from runtime_config.runtime_config import RuntimeConfig
    from runtime_config.reject_reasons import PassShape
    from runtime_config.runtime_policy import LocalGenerationMode
    from runtime_config.video_job_budget import LocalJobFilter
    from services.gpu_info.gpu_info import GpuInfo

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")
_S = TypeVar("_S", bound="StateHandlerBase")

def resolve_models_dir(state: AppState | None, config: RuntimeConfig) -> Path:
    """Effective models dir: custom from settings, or the startup default.

    Shared by ``StateHandlerBase.models_dir`` and the queued-generation handler so
    the custom-vs-default rule lives in one place. ``state`` may be ``None`` for
    callers that only hold a config.
    """
    custom = state.app_settings.models_dir if state is not None else None
    return Path(custom) if custom else config.default_models_dir


class StateHandlerBase:
    """Base handler with shared state and lock references."""

    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
        *,
        gpu_info: GpuInfo | None = None,
    ) -> None:
        self._state = state
        self._lock = lock
        self._config = config
        self._gpu_info = gpu_info

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def lock(self) -> RLock:
        return self._lock

    @property
    def config(self) -> RuntimeConfig:
        return self._config

    @property
    def models_dir(self) -> Path:
        """Effective models dir: custom from settings, or startup default."""
        return resolve_models_dir(self._state, self._config)

    @property
    def _gpu(self) -> GpuInfo:
        """Injected GPU probes. Budget gates need them; non-budget handlers never call."""
        if self._gpu_info is None:
            raise RuntimeError(
                f"{type(self).__name__} needs gpu_info for budget gates; pass it to __init__"
            )
        return self._gpu_info

    def _local_job_filter(
        self,
        *,
        model_id: LTXLocalModelId | None = None,
        total_only: bool = False,
        pass_shape: PassShape | None = None,
    ) -> LocalJobFilter:
        """Profile + budget for this request. Built once per public method so
        free VRAM and the MPS pool are sampled a single time per request,
        never per check. Single source for every budget gate.

        ``pass_shape=None`` follows the live "Never split" setting; a queued job
        passes the shape it resolved at create."""
        with self._lock:
            settings = self.state.app_settings
            calibration = self.state.device_calibration
        if pass_shape is None:
            pass_shape = resolve_pass_shape(
                "auto", never_split=settings.never_split_into_parts
            )
        return build_local_job_filter(
            self.config,
            model_id=(
                model_id
                if model_id is not None
                else resolve_active_ltx_model_id(
                    self.models_dir, settings.active_ltx_model_id
                )
            ),
            free_vram_bytes=self._gpu.get_free_cuda_vram_bytes(),
            mps_pool_bytes=self._gpu.get_mps_pool_bytes(),
            total_only=total_only,
            budget_mode=settings.budget_mode,
            calibration=calibration,
            pass_shape=pass_shape,
            conv_vae=resolved_use_conv_vae(settings, system=self.config.system),
        )

    def _offload_mode(
        self,
        load_mode: LocalGenerationMode,
        *,
        cpu_ram_floor_gb: int | None = CUDA_CPU_OFFLOAD_RAM_FLOOR_GB,
    ) -> OffloadMode:
        """Weight-streaming kind for a pipeline load. Computed once per load from
        frozen config facts; pipelines take the verdict, never the raw triple."""
        return offload_mode_for_prefetch_count(
            streaming_prefetch_count_for_mode(load_mode),
            self.config.device,
            drive_type=self.config.checkpoint_drive_type,
            system=self.config.system,
            ram_bytes=self.config.total_ram_bytes,
            cpu_ram_floor_gb=cpu_ram_floor_gb,
        )

    def _require_local_generation_possible(self, *, api_key: str | None = None) -> None:
        """422 when this machine cannot generate locally and no API route is left.

        Pass the relevant key for entry points that can fall back to the API; omit it
        for local-only entry points (the Explore queue, IC-LoRA), which can never
        succeed on unsupported hardware no matter what keys are configured.
        """
        if self.config.local_generations_mode != "unsupported":
            return
        if api_key is not None and api_key.strip():
            return
        raise HTTPError(
            422,
            LOCAL_GENERATION_UNSUPPORTED_MESSAGE,
            code=LOCAL_GENERATION_UNSUPPORTED,
        )

    def _resolve_seed(self) -> int:
        """Resolve the generation seed from the lock or a fresh random draw."""
        settings = self.state.app_settings
        if settings.seed_locked:
            logger.info("Using locked seed: %s", settings.locked_seed)
            return settings.locked_seed
        return random.randint(0, MAX_GENERATION_SEED)


def with_state_lock(
    method: Callable[Concatenate[_S, _P], _R],
) -> Callable[Concatenate[_S, _P], _R]:
    @wraps(method)
    def wrapped(self: _S, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        with self.lock:
            return method(self, *args, **kwargs)

    return wrapped
