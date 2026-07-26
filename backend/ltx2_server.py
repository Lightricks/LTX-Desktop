"""FastAPI composition root for the LTX backend server."""
import faulthandler
import os
import sys

faulthandler.enable(file=sys.stderr, all_threads=True)
from typing import Any, cast

if os.environ.get("BACKEND_DEBUG") == "1":
    try:
        import debugpy  # type: ignore[reportMissingImports]

        if not bool(debugpy.is_client_connected()):  # type: ignore[reportUnknownMemberType]
            try:
                # Connect to an already-listening IDE debugger (compound launch)
                debugpy.connect(("127.0.0.1", 5678))  # type: ignore[reportUnknownMemberType]
            except (ConnectionRefusedError, ConnectionError, OSError):
                # IDE not listening — start a debug server for manual attach
                debugpy.listen(("127.0.0.1", 5678))  # type: ignore[reportUnknownMemberType]
    except (ImportError, RuntimeError) as exc:
        print(f"Debugpy setup failed: {exc}", file=sys.stderr)

import logging
from pathlib import Path

# expandable_segments reduces CUDA allocator fragmentation near full VRAM (helps long/large
# IC-LoRA runs). Not supported on all platforms (older Windows / non-CUDA) — PyTorch warns and
# ignores it there. Must be set before importing torch; setdefault lets an explicit env override.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# macOS: stage mps-sdpa's prebuilt zero-copy attention extension cache before mps-sdpa
# is imported, so it loads without a runtime compiler. This MUST run before the patch
# imports below: several of them transitively import mps_sdpa (via ltx_core/ltx_pipelines
# attention), and mps-sdpa decides mpsgraph_zc's availability at its first import — if that
# happens before this patch, the zero-copy backend fails to JIT-build on a clean Mac and
# silently falls back to the leaky pyobjc backend. No-op off Darwin. See
# backend/mps_prebuilt_ext.py and docs/mps-attention-memory-leak.md.
import mps_prebuilt_ext as _mps_prebuilt_ext  # pyright: ignore[reportUnusedImport]
_mps_prebuilt_ext.setup_prebuilt_mps_extension()

import services.patches.record_stream_fix as _record_stream_fix  # pyright: ignore[reportUnusedImport]  # Remove once ltx-core includes the fix
del _record_stream_fix
import services.patches.safetensors_loader_fix as _safetensors_loader_fix  # pyright: ignore[reportUnusedImport]  # Remove once safetensors/PyTorch fix the mmap issue
del _safetensors_loader_fix
import services.patches.safetensors_metadata_fix as _safetensors_metadata_fix  # pyright: ignore[reportUnusedImport]  # Remove once safetensors supports read-only mmap
del _safetensors_metadata_fix
import services.patches.pinned_pool_fix as _pinned_pool_fix  # pyright: ignore[reportUnusedImport]  # Remove once ltx-core restores bounded pinned pool
del _pinned_pool_fix
import services.patches.ic_lora_stage2_lora as _ic_lora_stage2_lora  # pyright: ignore[reportUnusedImport]  # EXPERIMENTAL: remove once upstream ships PR #494 (use_lora_in_stage_2)
del _ic_lora_stage2_lora
import services.patches.diffusion_stage_cache as _diffusion_stage_cache  # pyright: ignore[reportUnusedImport]  # EXPERIMENTAL: remove once DiffusionStage caches/reuses identical builds upstream
del _diffusion_stage_cache

from state.app_settings import AppSettings

# ============================================================
# Logging Configuration
# ============================================================

import io
import platform

# Windows consoles default to a legacy code page (e.g. cp1252) that can't encode
# non-ASCII log content — the arrows in the LTX API logs, but also accented file paths or
# provider error strings — which crashes the logging handler's emit(). Force UTF-8 on the
# log streams so logging can never fail on an unencodable character.
for _stream in (sys.stdout, sys.stderr):
    if isinstance(_stream, io.TextIOWrapper):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")

# Backend logs to console only — Electron captures stdout/stderr and writes
# them to the session log file. This ensures *all* output (including early
# import errors and unhandled tracebacks) reaches the log, not just messages
# that go through Python's logging module.
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO, handlers=[console_handler])
logger = logging.getLogger(__name__)

# Now that logging is configured, report which mps-sdpa attention backend is live
# (the setup call at import time logs too early to be captured). No-op off Darwin.
_mps_prebuilt_ext.log_mps_backend_status()

# ============================================================
# SageAttention Integration
# ============================================================
use_sage_attention = os.environ.get("USE_SAGE_ATTENTION", "1") == "1"
_sageattention_runtime_fallback_logged = False

if use_sage_attention:
    try:
        from sageattention import sageattn  # type: ignore[reportMissingImports]
        import torch.nn.functional as F

        _original_sdpa = F.scaled_dot_product_attention

        _SAGE_SUPPORTED_HEADDIMS = {64, 96, 128}

        @torch.compiler.disable(recursive=True)  # type: ignore[reportUntypedFunctionDecorator]
        def _sageattn_call(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, is_causal: bool) -> torch.Tensor:
            # torch.compile must not trace into sageattn's Python internals (quant.py,
            # core.py): it bakes the current call's shapes into the compiled graph's
            # guards, which then hard-fails (ConstraintViolationError) on any later call
            # with different video/audio dimensions. Disabling keeps this call eager —
            # a graph break here, not a recompile — while the surrounding transformer
            # forward stays compiled.
            return cast(torch.Tensor, sageattn(query, key, value, is_causal=is_causal, tensor_layout="HND"))  # type: ignore[reportUnnecessaryCast]

        def patched_sdpa(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale: float | None = None,
            **kwargs: Any,
        ) -> torch.Tensor:
            global _sageattention_runtime_fallback_logged
            try:
                if (
                    query.dim() == 4
                    and attn_mask is None
                    and dropout_p == 0.0
                    and query.shape[-1] in _SAGE_SUPPORTED_HEADDIMS
                ):
                    return _sageattn_call(query, key, value, is_causal)
                else:
                    return _original_sdpa(query, key, value, attn_mask=attn_mask,
                                         dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
            except Exception:
                if not _sageattention_runtime_fallback_logged:
                    logger.warning("SageAttention failed during runtime; falling back to default attention", exc_info=True)
                    _sageattention_runtime_fallback_logged = True
                return _original_sdpa(query, key, value, attn_mask=attn_mask,
                                     dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)

        F.scaled_dot_product_attention = patched_sdpa
        logger.info("SageAttention enabled - attention operations will be faster")
    except ImportError:
        logger.warning("SageAttention not installed - using default attention")
        use_sage_attention = False
    except Exception:
        logger.warning("Failed to enable SageAttention", exc_info=True)
        use_sage_attention = False

# ============================================================
# Constants & Paths
# ============================================================

from runtime_config.port_constant import PORT


def _get_device() -> torch.device:
    # Delegates to ltx_core's device selection (CUDA -> MPS -> CPU) so the app and the
    # inference library agree on the preferred device, including MPS on Apple Silicon.
    from ltx_core.devices import get_preferred_device

    return get_preferred_device()


DEVICE = _get_device()
DTYPE = torch.bfloat16

def _resolve_app_data_dir() -> Path:
    env_path = os.environ.get("LTX_APP_DATA_DIR")
    if not env_path:
        raise RuntimeError(
            "LTX_APP_DATA_DIR environment variable must be set. "
            "When running standalone, set it to the desired data directory."
        )
    candidate = Path(env_path)
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


APP_DATA_DIR = _resolve_app_data_dir()

DEFAULT_MODELS_DIR = APP_DATA_DIR / "models"
DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = APP_DATA_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Models directory: {DEFAULT_MODELS_DIR}")

# ============================================================
# Settings
# ============================================================

SETTINGS_DIR = APP_DATA_DIR
SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = SETTINGS_DIR / "settings.json"

DEFAULT_APP_SETTINGS = AppSettings()

from app_factory import DEFAULT_ALLOWED_ORIGINS, create_app
from state import RuntimeConfig, build_initial_state
from runtime_config.mlx_runtime import discover_mlx_runtime
from runtime_config.runtime_policy import (
    FastVideoEnginePreference,
    LocalGenerationMode,
    MLX_BF16_MODEL_SOURCE,
    MLX_Q8_MODEL_SOURCE,
    decide_local_generation_mode,
)
from server_utils.model_layout_migration import migrate_legacy_models_layout
from services.gpu_info.gpu_info_impl import GpuInfoImpl

migrate_legacy_models_layout(APP_DATA_DIR)

LTX_API_BASE_URL = "https://api.ltx.video"


def _resolve_local_generations_mode() -> LocalGenerationMode:
    gpu_info = GpuInfoImpl()
    system = platform.system()
    cuda_available = gpu_info.get_cuda_available()
    mps_available = gpu_info.get_mps_available()
    vram_gb = gpu_info.get_vram_total_gb()
    # On Darwin there's no discrete VRAM (unified memory), so gate on *available* RAM,
    # not total — total overstates real headroom once the OS/Electron/app are running.
    # See GpuInfoImpl.get_available_ram_gb.
    available_ram_gb = gpu_info.get_available_ram_gb() if system == "Darwin" else None

    # Server-owned source of truth for mode selection.
    mode = decide_local_generation_mode(
        system=system,
        cuda_available=cuda_available,
        vram_gb=vram_gb,
        mps_available=mps_available,
        ram_gb=available_ram_gb,
    )
    logger.info(
        "Runtime policy local_generations_mode=%s (system=%s cuda_available=%s mps_available=%s "
        "vram_gb=%s available_ram_gb=%s)",
        mode,
        system,
        cuda_available,
        mps_available,
        vram_gb,
        available_ram_gb,
    )
    return mode


LOCAL_GENERATIONS_MODE = _resolve_local_generations_mode()


def _resolve_fast_video_engine_preference() -> FastVideoEnginePreference:
    raw = os.environ.get("LTX_FAST_VIDEO_ENGINE", "auto").strip().lower()
    if raw in {"auto", "torch", "mlx"}:
        return cast(FastVideoEnginePreference, raw)
    logger.warning("Ignoring invalid LTX_FAST_VIDEO_ENGINE=%r; using auto", raw)
    return "auto"


def _is_mlx_model_cached(model_source: str) -> bool:
    local_path = Path(model_source).expanduser()
    if local_path.exists():
        return True
    try:
        from huggingface_hub import scan_cache_dir

        return any(repo.repo_id == model_source for repo in scan_cache_dir().repos)
    except Exception:
        logger.warning("Failed to inspect the Hugging Face cache for %s", model_source, exc_info=True)
        return False


FAST_VIDEO_ENGINE_PREFERENCE = _resolve_fast_video_engine_preference()
_mlx_model_variant_raw = os.environ.get("LTX_MLX_MODEL_VARIANT", "bf16").strip().lower()
if _mlx_model_variant_raw not in {"bf16", "q8"}:
    logger.warning("Ignoring invalid LTX_MLX_MODEL_VARIANT=%r; using bf16", _mlx_model_variant_raw)
    _mlx_model_variant_raw = "bf16"
MLX_MODEL_VARIANT = _mlx_model_variant_raw
MLX_MODEL_SOURCE = os.environ.get(
    "LTX_MLX_MODEL_ID",
    MLX_Q8_MODEL_SOURCE if MLX_MODEL_VARIANT == "q8" else MLX_BF16_MODEL_SOURCE,
)
_gpu_info_for_runtime = GpuInfoImpl()
AVAILABLE_RAM_GB = (
    _gpu_info_for_runtime.get_available_ram_gb()
    if platform.system() == "Darwin"
    else None
)
MLX_RUNTIME = discover_mlx_runtime()
MLX_RUNTIME_ELIGIBLE = bool(
    platform.system() == "Darwin"
    and platform.machine().lower() in {"arm64", "aarch64"}
    and _gpu_info_for_runtime.get_mps_available()
    and MLX_RUNTIME.compatible
)
MLX_MODEL_CACHED = _is_mlx_model_cached(MLX_MODEL_SOURCE)
logger.info(
    "Fast runtime preference=%s mlx_eligible=%s runtime_version=%s runtime_revision=%s "
    "model=%s cached=%s variant=%s",
    FAST_VIDEO_ENGINE_PREFERENCE,
    MLX_RUNTIME_ELIGIBLE,
    MLX_RUNTIME.version,
    MLX_RUNTIME.revision,
    MLX_MODEL_SOURCE,
    MLX_MODEL_CACHED,
    MLX_MODEL_VARIANT,
)

CAMERA_MOTION_PROMPTS = {
    "none": "",
    "static": ", static camera, locked off shot, no camera movement",
    "focus_shift": ", focus shift, rack focus, changing focal point",
    "dolly_in": ", dolly in, camera pushing forward, smooth forward movement",
    "dolly_out": ", dolly out, camera pulling back, smooth backward movement",
    "dolly_left": ", dolly left, camera tracking left, lateral movement",
    "dolly_right": ", dolly right, camera tracking right, lateral movement",
    "jib_up": ", jib up, camera rising up, upward crane movement",
    "jib_down": ", jib down, camera lowering down, downward crane movement",
}

DEFAULT_NEGATIVE_PROMPT = """blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field"""

HF_OAUTH_CLIENT_ID = "a8189e14-9246-4f19-bd6a-a307bdcb9276"

runtime_config = RuntimeConfig(
    device=DEVICE,
    app_data_dir=APP_DATA_DIR,
    default_models_dir=DEFAULT_MODELS_DIR,
    outputs_dir=OUTPUTS_DIR,
    settings_file=SETTINGS_FILE,
    ltx_api_base_url=LTX_API_BASE_URL,
    local_generations_mode=LOCAL_GENERATIONS_MODE,
    use_sage_attention=use_sage_attention,
    camera_motion_prompts=CAMERA_MOTION_PROMPTS,
    default_negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    dev_mode=os.environ.get("LTX_DEV_MODE") == "1",
    hf_oauth_client_id=HF_OAUTH_CLIENT_ID,
    backend_port=int(os.environ.get("LTX_PORT", "") or PORT),
    fast_video_engine_preference=FAST_VIDEO_ENGINE_PREFERENCE,
    mlx_runtime_eligible=MLX_RUNTIME_ELIGIBLE,
    mlx_model_cached=MLX_MODEL_CACHED,
    mlx_model_source=MLX_MODEL_SOURCE,
    mlx_model_variant=cast(Any, MLX_MODEL_VARIANT),
    mlx_runtime_version=MLX_RUNTIME.version,
    mlx_runtime_revision=MLX_RUNTIME.revision,
    mlx_runtime_source=MLX_RUNTIME.source,
    mlx_runtime_dirty=MLX_RUNTIME.dirty,
    mlx_core_version=MLX_RUNTIME.core_version,
    mlx_framework_version=MLX_RUNTIME.mlx_version,
    available_ram_gb=AVAILABLE_RAM_GB,
    lora_catalog_source=str(Path(__file__).parent / "runtime_config" / "lora_catalog.json"),
    lora_catalog_fallback_path=str(Path(__file__).parent / "runtime_config" / "lora_catalog.json"),
)

handler = build_initial_state(runtime_config, DEFAULT_APP_SETTINGS)

auth_token = os.environ.get("LTX_AUTH_TOKEN", "")
admin_token = os.environ.get("LTX_ADMIN_TOKEN", "")

app = create_app(handler=handler, allowed_origins=DEFAULT_ALLOWED_ORIGINS, auth_token=auth_token, admin_token=admin_token)


def precache_model_files(model_dir: Path) -> int:
    if not model_dir.exists():
        return 0
    total_bytes = 0
    for f in model_dir.rglob("*"):
        if f.is_file() and f.suffix in (".safetensors", ".bin", ".pt", ".pth", ".onnx", ".model"):
            try:
                size = f.stat().st_size
                with open(f, "rb") as fh:
                    while fh.read(8 * 1024 * 1024):
                        pass
                total_bytes += size
            except Exception:
                logger.warning("Failed to precache model file: %s", f, exc_info=True)
    return total_bytes

def log_hardware_info() -> None:
    """Log runtime hardware and environment details."""
    gpu = GpuInfoImpl()
    gpu_info = gpu.get_gpu_info()
    vram_gb = gpu_info["vram"] // 1024 if gpu_info["vram"] else 0

    logger.info(f"Platform: {platform.system()} ({platform.machine()})")
    logger.info(f"Device: {DEVICE}  |  Dtype: {DTYPE}")
    gpu_line = f"GPU: {gpu_info['name']}  |  VRAM: {vram_gb} GB"
    # On Apple Silicon there's no discrete VRAM — the figure above is total unified
    # memory. Local generation is gated on *available* RAM at startup, so surface it.
    if gpu.get_mps_available():
        avail = gpu.get_available_ram_gb()
        gpu_line += f"  |  Available RAM: {avail if avail is not None else '?'} GB"
    logger.info(gpu_line)
    logger.info(f"SageAttention: {'enabled' if use_sage_attention else 'disabled'}")
    logger.info(f"Python: {sys.version.split()[0]}  |  Torch: {torch.__version__}")


if __name__ == "__main__":
    import asyncio
    import uvicorn

    port = runtime_config.backend_port
    logger.info("=" * 60)
    logger.info("LTX-2 Video Generation Server (FastAPI + Uvicorn)")
    log_hardware_info()
    logger.info("=" * 60)

    # Use our root logging config so uvicorn logs go to stdout (not its
    # default stderr), letting Electron tag them correctly as INFO.
    log_config: dict[str, object] = {
        "version": 1,
        "disable_existing_loggers": False,
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO"},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
        },
    }

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info", access_log=False, log_config=log_config)
    server = uvicorn.Server(config)

    _orig_startup = server.startup

    async def _startup_with_ready_msg(sockets: object = None) -> None:
        await _orig_startup(sockets=sockets)  # type: ignore[arg-type]
        if server.started:
            # Machine-parseable ready message — Electron matches this line
            print(f"Server running on http://127.0.0.1:{port}", flush=True)

    server.startup = _startup_with_ready_msg  # type: ignore[assignment]

    asyncio.run(server.serve())
