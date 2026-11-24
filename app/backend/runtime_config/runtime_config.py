"""Runtime configuration model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from runtime_config.assets_layout import in_flight_folder
from runtime_config.runtime_policy import LocalGenerationMode

if TYPE_CHECKING:
    # Runtime import would cycle: api_types initializes through this module
    # (runtime_config/__init__) while services/__init__ needs api_types.
    from services.storage_info.storage_info import DriveType


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
    lora_catalog_source: str = ""
    # Bundled catalog used as a fallback when lora_catalog_source is a URL that fails to fetch.
    lora_catalog_fallback_path: str = ""
    # Discrete GPU VRAM (CUDA). On Darwin this is total unified memory (SKU).
    vram_gb: int | None = None
    # Darwin: available (free) RAM at process start, used for per-job sizing.
    available_ram_gb: int | None = None
    # MPS unified-memory machines never load full; tests leave this False to simulate CUDA.
    darwin_unified_memory: bool = False
    # Actual OS family from platform.system() ("Windows", "Linux", "Darwin", ...).
    system: str = ""
    # Exact CUDA device total VRAM in bytes. None if CUDA is unavailable or the query failed.
    vram_bytes: int | None = None
    # Free CUDA VRAM sampled once at backend startup before any model loads, used as the
    # usable ceiling for total_only (picker, model specs, queue create).
    usable_vram_bytes: int | None = None
    # Exact total system RAM in bytes. None if the query failed.
    total_ram_bytes: int | None = None
    # Drive class of the checkpoint volume, probed once at startup. None when unknown.
    checkpoint_drive_type: DriveType | None = None
    # CUDA device name (torch.cuda.get_device_name), probed once at startup.
    # None off-CUDA or when the query failed. Powers hardware-class gates.
    gpu_device_name: str | None = None
    # Native fp8 (compute capability >= 8.9), probed once at startup. Gates the
    # measured budget route: pre-Ada streaming runs bf16, which the fp8-fit
    # laws never saw. Default True preserves current routing where unprobed.
    fp8_capable: bool = True
    # Kill switch for parallel queue lanes. False registers only the local lane: every job
    # then runs one at a time on the local slot, as before lanes existed.
    parallel_queue_lanes: bool = True
    # User-chosen assets folder (LTX_ASSETS_DIR). None means <app_data_dir>/assets.
    assets_dir: Path | None = None

    @property
    def in_flight_dir(self) -> Path:
        return in_flight_folder(self.app_data_dir, self.assets_dir)

    @property
    def force_api_generations(self) -> bool:
        """Derived: local generation is unavailable for this runtime."""
        return self.local_generations_mode == "unsupported"
