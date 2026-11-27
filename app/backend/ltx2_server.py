"""FastAPI composition root for the LTX backend server."""
import faulthandler
import os
import sys

from server_utils.win_dll_search import remove_cwd_from_dll_search_path

# Before torch / native extensions: do not search CWD for DLLs (Windows hijack).
remove_cwd_from_dll_search_path()

faulthandler.enable(file=sys.stderr, all_threads=True)
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
if os.environ.get("LTX_REQUIRE_ROCM") != "1":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Before any mps-sdpa import. Calibration benches the slow pyobjc backend at
# L<=2048 and can cache fused_min_bytes=None ("always stock"). That then also
# disables mpsgraph_zc, so video-length attention hits stock MPS SDPA, which
# materializes S×S (~47 GiB at 720p/5s) and OOMs / wedged-U-state
# (https://github.com/Lightricks/LTX-Desktop/issues/161). setdefault so a
# developer override still wins. Remove once mps-sdpa treats None as "use
# defaults" for large sequences, or calibrates against mpsgraph_zc / video L.
if sys.platform == "darwin":
    os.environ.setdefault("MPS_SDPA_SKIP_CALIBRATION", "1")

import torch
from runtime_config.accelerator import accelerator_backend, accelerator_supports_fp8, verify_rocm_execution

# Windows source launches can use an existing HIP venv without an installer receipt.
if sys.platform == "win32" and "LTX_GPU_RUNTIME" not in os.environ and getattr(torch.version, "hip", None):
    os.environ.setdefault("LTX_REQUIRE_ROCM", "1")

if os.environ.get("LTX_GPU_RUNTIME") == "cuda":
    if accelerator_backend() != "cuda":
        raise RuntimeError("The installed CUDA runtime cannot access its NVIDIA GPU. Check the graphics driver and rerun the Windows installer if the GPU changed.")
    print(f"NVIDIA GPU selected: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}", flush=True)
if os.environ.get("LTX_REQUIRE_ROCM") == "1":
    _rocm_name = verify_rocm_execution(require_gfx1151=os.environ.get("LTX_GPU_RUNTIME") == "rocm")
    print(f"ROCm GPU verified: {_rocm_name} | HIP {torch.version.hip} | BF16 matmul and attention passed", flush=True)

# macOS: stage mps-sdpa's prebuilt zero-copy attention extension cache before mps-sdpa
# is imported, so it loads without a runtime compiler. This MUST run before the patch
# imports below: several of them transitively import mps_sdpa (via ltx_core/ltx_pipelines
# attention), and mps-sdpa decides mpsgraph_zc's availability at its first import — if that
# happens before this patch, the zero-copy backend fails to JIT-build on a clean Mac and
# silently falls back to the leaky pyobjc backend. No-op off Darwin. See
# backend/mps_prebuilt_ext.py and docs/mps-attention-memory-leak.md.
import mps_prebuilt_ext as _mps_prebuilt_ext  # pyright: ignore[reportUnusedImport]
_mps_prebuilt_ext.setup_prebuilt_mps_extension()

# Darwin transfer fence. Must run before apply_patches(), which imports
# ltx-pipelines attention. CUDA patches are applied after logging is configured.
import services.patches.mps_transfer_sync as _mps_transfer_sync  # pyright: ignore[reportUnusedImport]  # Remove once ltx-core waits for queued MPS compute before a transfer and a 1080p stage 2 passes on a 48 GB Mac without it
del _mps_transfer_sync
import services.patches.mps_tiling_weights_sync as _mps_tiling_weights_sync  # pyright: ignore[reportUnusedImport]  # Remove once ltx-core _device_weights uploads with non_blocking=allow_async_transfer(...) and Refine Details gives a real picture on a Mac without it (LTX-Desktop#199)
del _mps_tiling_weights_sync

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

# CUDA patches + SageAttention wrap (gated on USE_SAGE_ATTENTION, default on).
# LLM: bumping ltx-core / ltx-pipelines MUST re-verify every import in
# inference/apply_patches.py (and mps_sdpa_torch below). Delete-first.
# Import the function from its module. Binding it on the inference package
# would shadow the submodule of the same name.
from inference.apply_patches import apply_patches
from runtime_config.smart_app_control import (
    detect_smart_app_control,
    smart_app_control_blocks_unsigned_code,
)

_smart_app_control_state = detect_smart_app_control()
smart_app_control_on = smart_app_control_blocks_unsigned_code(_smart_app_control_state)
logger.info("Windows Smart App Control state: %s", _smart_app_control_state)
use_sage_attention = apply_patches(smart_app_control_on=smart_app_control_on, allow_sage_attention=accelerator_backend() == "cuda")

# state/__init__.py imports the queue runner, which imports CUDA patches.
# Importing AppSettings earlier installs those patches before logging exists.
from state.app_settings import AppSettings

# Desktop-only LTXTextEncoder hook (split from safetensors_metadata_fix patch 4).
import services.patches.ltx_text_encoder_checkpoint_id as _ltx_text_encoder_checkpoint_id  # pyright: ignore[reportUnusedImport]  # Remove once safetensors supports read-only mmap
del _ltx_text_encoder_checkpoint_id

# Darwin: Diffusers/Z-Image call F.sdpa (stock MPS SDPA materializes S×S and can
# freeze the machine). Video already uses mps-sdpa via ltx-core; this covers Z-Image.
# Remove once Diffusers native attention on MPS uses a fused / mps-sdpa backend.
import services.patches.mps_sdpa_torch as _mps_sdpa_torch
_mps_sdpa_torch.install()

# Darwin: MPS conv3d runs at ~1/5 of conv2d throughput, which makes the conv VAE
# decode ~4x slower than it needs to be. The hook is installed off; the experimental
# Settings toggle turns it on per generation (GenerationHandler.start_generation).
# Remove once torch MPS conv3d reaches conv2d throughput.
import services.patches.mps_conv3d_as_conv2d as _mps_conv3d_as_conv2d
_mps_conv3d_as_conv2d.install(enabled=False)
del _mps_conv3d_as_conv2d

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

# Optional. Electron sets it when the user picked an assets folder. The store does not
# create it: the folder must exist and hold the .ltx-assets marker.
_assets_dir_env = os.environ.get("LTX_ASSETS_DIR")
ASSETS_DIR = Path(_assets_dir_env) if _assets_dir_env else None

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
from runtime_config.local_runtime import LocalRuntime, resolve_local_runtime
from runtime_config.macos_version import detect_macos_major
from runtime_config.runtime_policy import LocalGenerationMode
from server_utils.model_layout_migration import migrate_legacy_models_layout
from services.gpu_info.gpu_info_impl import GpuInfoImpl, platform_label
from services.storage_info.storage_info_impl import StorageInfoImpl

migrate_legacy_models_layout(APP_DATA_DIR)

LTX_API_BASE_URL = "https://api.ltx.video"


def _resolve_startup_local_runtime() -> LocalRuntime:
    gpu_info = GpuInfoImpl()
    system = platform.system()
    fp8_capable = accelerator_supports_fp8()
    macos_major = detect_macos_major()
    if system == "Darwin" and macos_major is None:
        logger.warning("Could not read the macOS version. The macOS minimum for local generation is not enforced.")
    runtime = resolve_local_runtime(
        gpu_info,
        system=system,
        fp8_capable=fp8_capable,
        storage_info=StorageInfoImpl(APP_DATA_DIR),
        macos_major=macos_major,
    )
    logger.info(
        "Runtime policy local_generations_mode=%s (system=%s macos_major=%s cuda_available=%s mps_available=%s "
        "vram_gb=%s available_ram_gb=%s fp8_capable=%s checkpoint_drive=%s)",
        runtime.mode,
        system,
        macos_major,
        gpu_info.get_cuda_available(),
        gpu_info.get_mps_available(),
        runtime.vram_gb,
        runtime.available_ram_gb,
        fp8_capable,
        runtime.checkpoint_drive_type,
    )
    return runtime


_LOCAL_RUNTIME = _resolve_startup_local_runtime()
LOCAL_GENERATIONS_MODE = _LOCAL_RUNTIME.mode


def _effective_local_generations_mode(mode: LocalGenerationMode) -> LocalGenerationMode:
    """Dev-only: unlock Explore model specs when RAM check failed at launch."""
    if (
        os.environ.get("LTX_DEV_MODE") == "1"
        and os.environ.get("LTX_DEV_FORCE_LOCAL_VIABLE") == "1"
        and mode == "unsupported"
    ):
        logger.warning(
            "Dev override LTX_DEV_FORCE_LOCAL_VIABLE: using streaming_models_loading "
            "(detected mode was unsupported)"
        )
        return "streaming_models_loading"
    return mode


_EFFECTIVE_LOCAL_GENERATIONS_MODE = _effective_local_generations_mode(
    LOCAL_GENERATIONS_MODE
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

HF_OAUTH_CLIENT_ID = "508843e5-65c3-4565-aeed-fb17e94cf394"

runtime_config = RuntimeConfig(
    device=DEVICE,
    app_data_dir=APP_DATA_DIR,
    assets_dir=ASSETS_DIR,
    default_models_dir=DEFAULT_MODELS_DIR,
    outputs_dir=OUTPUTS_DIR,
    settings_file=SETTINGS_FILE,
    ltx_api_base_url=LTX_API_BASE_URL,
    use_sage_attention=use_sage_attention,
    smart_app_control_on=smart_app_control_on,
    camera_motion_prompts=CAMERA_MOTION_PROMPTS,
    default_negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    dev_mode=os.environ.get("LTX_DEV_MODE") == "1",
    parallel_queue_lanes=os.environ.get("LTX_PARALLEL_QUEUE_LANES", "1") != "0",
    hf_oauth_client_id=HF_OAUTH_CLIENT_ID,
    backend_port=int(os.environ.get("LTX_PORT", "") or PORT),
    lora_catalog_source=str(Path(__file__).parent / "runtime_config" / "lora_catalog.json"),
    lora_catalog_fallback_path=str(Path(__file__).parent / "runtime_config" / "lora_catalog.json"),
    **_LOCAL_RUNTIME.runtime_config_kwargs(
        local_generations_mode=_EFFECTIVE_LOCAL_GENERATIONS_MODE
    ),
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

    logger.info(f"Platform: {platform_label()}")
    logger.info(f"Device: {DEVICE}  |  Dtype: {DTYPE}")
    gpu_line = f"GPU: {gpu_info['name']}  |  VRAM: {vram_gb} GB"
    # On Apple Silicon there's no discrete VRAM — the figure above is total unified
    # memory, which is also the local-generation eligibility check.
    if gpu.get_mps_available():
        total = gpu.get_vram_total_gb()
        avail = gpu.get_available_ram_gb()
        gpu_line += f"  |  Total RAM: {total if total is not None else '?'} GB"
        gpu_line += f"  |  Available RAM: {avail if avail is not None else '?'} GB"
        logger.info(
            "LTX 2.5 decode uses eager SDPA on Mac (no Triton; slower than Linux/Windows)."
        )
    logger.info(gpu_line)
    logger.info(f"Accelerator: {accelerator_backend()}  |  HIP: {getattr(torch.version, 'hip', None)}")
    logger.info(f"SageAttention: {'enabled' if use_sage_attention else 'disabled'}")
    logger.info(f"Python: {sys.version.split()[0]}  |  Torch: {torch.__version__}")


if __name__ == "__main__":
    import asyncio
    import uvicorn

    port = runtime_config.backend_port
    logger.info("=" * 60)
    logger.info("LTX Desktop - AMD 395 MAX video generation server (FastAPI + Uvicorn)")
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
