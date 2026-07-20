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

# Note: expandable_segments is not supported on all platforms

import torch

import services.patches.record_stream_fix as _record_stream_fix  # pyright: ignore[reportUnusedImport]  # Remove once ltx-core includes the fix
del _record_stream_fix
import services.patches.safetensors_loader_fix as _safetensors_loader_fix  # pyright: ignore[reportUnusedImport]  # Remove once safetensors/PyTorch fix the mmap issue
del _safetensors_loader_fix
import services.patches.safetensors_metadata_fix as _safetensors_metadata_fix  # pyright: ignore[reportUnusedImport]  # Remove once safetensors supports read-only mmap
del _safetensors_metadata_fix
import services.patches.pinned_pool_fix as _pinned_pool_fix  # pyright: ignore[reportUnusedImport]  # Remove once ltx-core restores bounded pinned pool
del _pinned_pool_fix

from state.app_settings import AppSettings

# ============================================================
# Logging Configuration
# ============================================================

import platform

# Backend logs to console only — Electron captures stdout/stderr and writes
# them to the session log file. This ensures *all* output (including early
# import errors and unhandled tracebacks) reaches the log, not just messages
# that go through Python's logging module.
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO, handlers=[console_handler])
logger = logging.getLogger(__name__)

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
                    return cast(torch.Tensor, sageattn(query, key, value, is_causal=is_causal, tensor_layout="HND"))  # type: ignore[reportUnnecessaryCast]
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

from runtime_config.server_config import load_server_config

server_config = load_server_config()


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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

models_dir_env = os.environ.get("LTX_MODELS_DIR", "").strip()
MODELS_DIR_OVERRIDE = Path(models_dir_env).expanduser().resolve() if models_dir_env else None
if MODELS_DIR_OVERRIDE is not None:
    MODELS_DIR_OVERRIDE.mkdir(parents=True, exist_ok=True)

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

from app_factory import create_app
from state import RuntimeConfig, build_initial_state
from runtime_config.runtime_policy import LocalGenerationMode, decide_local_generation_mode
from server_utils.model_layout_migration import migrate_legacy_models_layout
from services.gpu_info.gpu_info_impl import GpuInfoImpl

migrate_legacy_models_layout(APP_DATA_DIR)

LTX_API_BASE_URL = "https://api.ltx.video"


def _resolve_local_generations_mode() -> LocalGenerationMode:
    gpu_info = GpuInfoImpl()
    system = platform.system()
    cuda_available = gpu_info.get_cuda_available()
    vram_gb = gpu_info.get_vram_total_gb()

    # Server-owned source of truth for mode selection.
    mode = decide_local_generation_mode(
        system=system,
        cuda_available=cuda_available,
        vram_gb=vram_gb,
    )
    logger.info(
        "Runtime policy local_generations_mode=%s (system=%s cuda_available=%s vram_gb=%s)",
        mode,
        system,
        cuda_available,
        vram_gb,
    )
    return mode


LOCAL_GENERATIONS_MODE = _resolve_local_generations_mode()

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

HF_OAUTH_CLIENT_ID = os.environ.get("LTX_HF_OAUTH_CLIENT_ID", "a8189e14-9246-4f19-bd6a-a307bdcb9276")

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
    backend_port=server_config.port,
    hf_gating_enabled=os.environ.get("LTX_HF_GATING_ENABLED") == "1",
    deployment_mode=server_config.deployment_mode,
    public_base_url=server_config.public_base_url,
    allow_legacy_path_inputs=server_config.allow_legacy_path_inputs,
    models_dir_editable=server_config.models_dir_editable,
    models_dir_override=MODELS_DIR_OVERRIDE,
)

handler = build_initial_state(runtime_config, DEFAULT_APP_SETTINGS)

app = create_app(
    handler=handler,
    allowed_origins=list(server_config.allowed_origins),
    auth_token=server_config.auth_token,
    admin_token=server_config.admin_token,
)


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
    logger.info(f"GPU: {gpu_info['name']}  |  VRAM: {vram_gb} GB")
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

    config = uvicorn.Config(
        app,
        host=server_config.bind_host,
        port=port,
        log_level="info",
        access_log=False,
        log_config=log_config,
    )
    server = uvicorn.Server(config)

    _orig_startup = server.startup

    async def _startup_with_ready_msg(sockets: object = None) -> None:
        await _orig_startup(sockets=sockets)  # type: ignore[arg-type]
        if server.started:
            # Machine-parseable ready message — Electron matches this line
            ready_host = "127.0.0.1" if server_config.bind_host in {"0.0.0.0", "::"} else server_config.bind_host
            print(f"Server running on http://{ready_host}:{port}", flush=True)

    server.startup = _startup_with_ready_msg  # type: ignore[assignment]

    asyncio.run(server.serve())
