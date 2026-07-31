"""Runtime configuration model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from runtime_config.runtime_policy import (
    FastVideoEngineDecision,
    FastVideoEnginePreference,
    LocalGenerationMode,
    MLX_BF16_MODEL_SOURCE,
    decide_fast_video_engine,
)


@dataclass
class RuntimeConfig:
    device: torch.device
    app_data_dir: Path
    default_models_dir: Path
    outputs_dir: Path
    settings_file: Path
    ltx_api_base_url: str
    local_generations_mode: LocalGenerationMode
    use_sage_attention: bool
    camera_motion_prompts: dict[str, str]
    default_negative_prompt: str
    dev_mode: bool
    backend_port: int
    fast_video_engine_preference: FastVideoEnginePreference = "auto"
    mlx_runtime_eligible: bool = False
    mlx_model_cached: bool = False
    mlx_model_source: str = MLX_BF16_MODEL_SOURCE
    mlx_model_variant: Literal["bf16", "q8"] = "bf16"
    mlx_runtime_version: str = "not installed"
    mlx_runtime_revision: str | None = None
    mlx_runtime_source: str = "not found"
    mlx_runtime_dirty: bool | None = None
    mlx_core_version: str | None = None
    mlx_framework_version: str | None = None
    available_ram_gb: int | None = None
    hf_oauth_client_id: str = ""
    lora_catalog_source: str = ""
    # Bundled catalog used as a fallback when lora_catalog_source is a URL that fails to fetch.
    lora_catalog_fallback_path: str = ""

    @property
    def force_api_generations(self) -> bool:
        """Derived: local generation is unavailable for this runtime."""
        return self.local_generations_mode == "unsupported"

    def decide_fast_video_engine(self, *, use_local_text_encoding: bool) -> FastVideoEngineDecision:
        return decide_fast_video_engine(
            preference=self.fast_video_engine_preference,
            mlx_runtime_eligible=self.mlx_runtime_eligible,
            mlx_model_cached=self.mlx_model_cached,
            use_local_text_encoding=use_local_text_encoding,
            mlx_quality_qualified=self.mlx_model_variant == "bf16",
        )
