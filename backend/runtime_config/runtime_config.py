"""Runtime configuration model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from runtime_config.runtime_policy import LocalGenerationMode
from runtime_config.server_config import DeploymentMode


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
    hf_oauth_client_id: str = ""
    hf_gating_enabled: bool = False
    deployment_mode: DeploymentMode = "managed_local"
    public_base_url: str = ""
    allow_legacy_path_inputs: bool = True
    models_dir_editable: bool = True
    models_dir_override: Path | None = None

    @property
    def force_api_generations(self) -> bool:
        """Derived: local generation is unavailable for this runtime."""
        return self.local_generations_mode == "unsupported"

    @property
    def effective_public_base_url(self) -> str:
        return self.public_base_url.rstrip("/") or f"http://127.0.0.1:{self.backend_port}"
