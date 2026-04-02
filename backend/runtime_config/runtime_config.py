"""Runtime configuration model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from runtime_config.model_download_specs import ModelFileDownloadSpec
from state.app_state_types import ModelFileType


@dataclass
class RuntimeConfig:
    device: torch.device
    default_models_dir: Path
    model_download_specs: Mapping[ModelFileType, ModelFileDownloadSpec]
    required_model_types: frozenset[ModelFileType]
    outputs_dir: Path
    settings_file: Path
    ltx_api_base_url: str
    force_api_generations: bool
    use_sage_attention: bool
    camera_motion_prompts: dict[str, str]
    default_negative_prompt: str
    dev_mode: bool

    def spec_for(self, model_type: ModelFileType) -> ModelFileDownloadSpec:
        return self.model_download_specs[model_type]
