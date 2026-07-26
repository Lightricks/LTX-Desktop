"""Runtime policy decisions for local generation mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LocalGenerationMode = Literal[
    "full_models_loading",
    "streaming_models_loading",
    "unsupported",
]
FastVideoEngine = Literal["torch", "mlx"]
FastVideoEnginePreference = Literal["auto", "torch", "mlx"]
FastVideoExecutionMode = Literal["eager", "low_ram", "unsupported"]

MLX_RUNTIME_VERSION = "0.14.20.dev1"
MLX_RUNTIME_REVISION = "3171bac4ba901c0237faea2678c34034b37abc2a"
MLX_BF16_MODEL_SOURCE = "dgrauet/ltx-2.3-mlx"
MLX_Q8_MODEL_SOURCE = "dgrauet/ltx-2.3-mlx-q8"
TORCH_RUNTIME_REVISION = "9377758131b1ffde4b7f766804590a6617bf2ab9"

# BF16 block streaming was validated by the MLX runtime for 32 GB Macs. Eager
# materialization needs substantially more headroom, so keep the automatic
# threshold conservative and use block streaming below it.
MLX_BF16_EAGER_FLOOR_GB = 64


@dataclass(frozen=True, slots=True)
class FastVideoEngineDecision:
    engine: FastVideoEngine
    reason: str

# Below this, local generation isn't viable at all on Darwin. Roughly matches the
# measured ~13 GB RSS of the distilled pipeline streaming on an M4 Pro, plus margin
# — a single data point, not a hardware sweep.
DARWIN_STREAMING_FLOOR_GB = 15

# Full-resident on Darwin means the *full* bf16 transformer (no fp8 path on MPS,
# unlike CUDA) sitting in unified memory alongside the Gemma text encoder, VAE,
# upsampler, activations, and the OS/Electron/app itself. A live test at ram_gb=19
# (48 GB machine, other apps open) picked this tier before this floor existed and
# stalled — torch/driver allocation climbed past torch.mps.recommended_max_memory()
# and the process sat there paging instead of progressing or erroring (no crash, no
# OOM, just stuck) — so the failure mode this floor guards against doesn't announce
# itself; there's no clean OOM to notice in testing by accident.
#
# 85 is arithmetic from real checkpoint bytes (model_download_specs.py), not a
# hardware measurement, converted to GiB since get_available_ram_gb() divides by
# 1024**3:
#   - transformer (ltx-2.3-22b-distilled-1.1.safetensors): 46_149_345_334 B = 42.98 GiB
#   - Gemma text encoder (gemma-3-12b-it-qat-q4_0-unquantized, bf16): 25_000_000_000 B = 23.28 GiB
#   - VAE + active latent workspace: ~7.3 GiB estimate (unverified) for a 5s/1080p
#     clip (this app's max local resolution/duration); decode is tiled
#     (default_tiling_config() in ltx_pipeline_common.py), so this term likely
#     doesn't scale up much with resolution the way a naive pixel-count estimate
#     would suggest, but that's not confirmed either
#   -> subtotal 73.5 GiB, +~15% margin for MPS allocator fragmentation/driver
#      overhead against the silent-stall failure mode above -> 85
#
# Still UNVERIFIED on real hardware. Do not lower further without confirming on a
# machine that actually has this much RAM free.
DARWIN_FULL_RESIDENT_FLOOR_GB = 85


def decide_local_generation_mode(
    system: str,
    cuda_available: bool,
    vram_gb: int | None,
    mps_available: bool = False,
    ram_gb: int | None = None,
) -> LocalGenerationMode:
    """Pick the local-generation mode for this runtime.

    - "unsupported": local generation is not viable; caller must route to the API.
    - "streaming_models_loading": enough memory to run, but model weights must be
      streamed from pinned host RAM (15-30 GB range on CUDA).
    - "full_models_loading": enough memory to hold the whole model resident, so
      streaming is skipped to avoid unnecessary host-RAM pressure.

    On CUDA (Windows/Linux) the memory figure is discrete (total) VRAM, and
    "full_models_loading" (>=31 GB) holds the fp8-halved (~23 GB) transformer
    resident. On Apple Silicon (Darwin) there is no discrete VRAM — the GPU shares
    system RAM — so ``ram_gb`` is the *available* (free) RAM, not total: total RAM
    overstates real headroom once the OS/Electron/app are already running. Gated on
    MPS being available (i.e. Apple Silicon, not an Intel Mac, which has no MPS
    backend and stays unsupported).

    Darwin's "full_models_loading" (>=``DARWIN_FULL_RESIDENT_FLOOR_GB``) is NOT the
    same regime as CUDA's: MPS has no fp8 path, so it holds the *full* ~46 GB bf16
    transformer resident — roughly 2x CUDA's resident footprint at its own
    full-loading floor. Below that, Darwin streams (mmap via OffloadMode.DISK, not
    CUDA's pinned-host-RAM-then-copy OffloadMode.CPU — see
    ``services.ltx_pipeline_common.offload_mode_for_prefetch_count``). Streaming was
    the only Darwin path validated end-to-end (M4 Pro, 48 GB); the full-resident
    tier above is unverified on real hardware, see ``DARWIN_FULL_RESIDENT_FLOOR_GB``.
    """
    if system == "Darwin":
        if not mps_available:
            return "unsupported"
        if ram_gb is None:
            return "unsupported"
        if ram_gb < DARWIN_STREAMING_FLOOR_GB:
            return "unsupported"
        if ram_gb >= DARWIN_FULL_RESIDENT_FLOOR_GB:
            return "full_models_loading"
        return "streaming_models_loading"

    if system in ("Windows", "Linux"):
        if not cuda_available:
            return "unsupported"
        if vram_gb is None:
            return "unsupported"
        if vram_gb < 15:
            return "unsupported"
        if vram_gb < 31:
            return "streaming_models_loading"
        return "full_models_loading"

    # Fail closed for non-target platforms unless explicitly relaxed.
    return "unsupported"


def streaming_prefetch_count_for_mode(mode: LocalGenerationMode) -> int | None:
    """Return the streaming_prefetch_count to pass to a local pipeline.

    Must not be called when local generation is unsupported — callers should
    route through the API instead.
    """
    if mode == "unsupported":
        raise AssertionError(
            "streaming_prefetch_count_for_mode called with 'unsupported' mode; "
            "callers must route to the API instead of constructing a local pipeline."
        )
    if mode == "full_models_loading":
        return None
    if mode == "streaming_models_loading":
        return 2
    raise AssertionError(f"Unexpected LocalGenerationMode: {mode!r}")


def decide_fast_video_execution_mode(
    engine: FastVideoEngine,
    local_mode: LocalGenerationMode,
    available_ram_gb: int | None,
) -> FastVideoExecutionMode:
    """Choose eager vs low-RAM loading for the selected Fast pipeline.

    Torch keeps its existing, separately qualified policy. MLX BF16 can stream
    transformer blocks directly from mmap and therefore uses a lower, explicit
    eager threshold without changing the policy used by Retake/Extend/A2V/
    IC-LoRA Torch fallbacks.
    """
    if local_mode == "unsupported":
        return "unsupported"
    if engine == "torch":
        return "eager" if local_mode == "full_models_loading" else "low_ram"
    if available_ram_gb is not None and available_ram_gb >= MLX_BF16_EAGER_FLOOR_GB:
        return "eager"
    return "low_ram"


def decide_fast_video_engine(
    *,
    preference: FastVideoEnginePreference,
    mlx_runtime_eligible: bool,
    mlx_model_cached: bool,
    use_local_text_encoding: bool,
    mlx_quality_qualified: bool = True,
) -> FastVideoEngineDecision:
    """Resolve Fast T2V/I2V without silently dropping prepared embeddings.

    ``auto`` is capability-aware: MLX is selected only for requests whose text
    is encoded locally by the MLX pipeline, and only when both runtime and BF16
    weights are already available. An explicit MLX preference may download the
    configured model on first use, but still fails closed to Torch when the
    platform/runtime cannot execute MLX.
    """
    if preference == "torch":
        return FastVideoEngineDecision("torch", "Torch was explicitly selected.")
    if preference == "mlx":
        if mlx_runtime_eligible:
            suffix = (
                " BF16 weights are cached."
                if mlx_model_cached
                else " BF16 weights are not cached and may download on first use."
            )
            return FastVideoEngineDecision("mlx", "MLX was explicitly selected." + suffix)
        return FastVideoEngineDecision(
            "torch",
            "MLX was explicitly requested but is unavailable on this runtime; using Torch.",
        )
    if not use_local_text_encoding:
        return FastVideoEngineDecision(
            "torch",
            "Prepared/API text embeddings require the feature-complete Torch pipeline.",
        )
    if not mlx_quality_qualified:
        return FastVideoEngineDecision(
            "torch",
            "MLX q8 is expert-only and is never auto-selected after quality qualification.",
        )
    if not mlx_runtime_eligible:
        return FastVideoEngineDecision(
            "torch",
            "MLX auto-selection requires Apple Silicon, MPS, and the pinned MLX runtime.",
        )
    if not mlx_model_cached:
        return FastVideoEngineDecision(
            "torch",
            "MLX BF16 weights are not cached; auto mode avoids an unannounced model download.",
        )
    return FastVideoEngineDecision(
        "mlx",
        "MLX auto-selected for Fast T2V/I2V with local text encoding and cached BF16 weights.",
    )
