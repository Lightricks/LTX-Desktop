"""Central configuration + HTTP helpers for the perf suite.

Everything backend-specific lives here; the measurement scripts (soak_test,
output_ab, decompose, sanity) import from this module and stay generic. To point
the suite at a different backend, edit the four sections below.

The backend is a local FastAPI server. We drive real generations through it over
HTTP so the patch's per-generation hooks fire exactly as in production, and read
GPU memory via `nvidia-smi` so the measurement path never couples to backend
internals.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# 1. Backend location + GPU
# --------------------------------------------------------------------------- #
# The backend binds 127.0.0.1 on PORT (runtime_config/port_constant.py = 41954),
# overridable by the LTX_PORT env var (ltx2_server.py reads os.environ["LTX_PORT"]).
# PERF_BASE_URL / PERF_GPU_INDEX let you point the suite elsewhere without edits.
_PORT = os.environ.get("LTX_PORT") or "41954"
BASE_URL = os.environ.get("PERF_BASE_URL") or f"http://127.0.0.1:{_PORT}"
HEALTH_PATH = "/health"                              # backend/_routes/health.py
GPU_INDEX = int(os.environ.get("PERF_GPU_INDEX", "0"))  # nvidia-smi index of the 5090
HTTP_TIMEOUT_S = 1800                                 # a 1080p/20s gen can take minutes

# Auth: the backend requires a Bearer token on every request when it's launched
# with an auth token set. The desktop app generates a random token per launch;
# copy it from the app's Logs footer (the "•••••" control) into the dashboard's
# token field, or export PERF_AUTH_TOKEN for the CLI. `pnpm perf:dev` injects a
# token automatically. A backend started without a token leaves this empty.
AUTH_TOKEN = os.environ.get("PERF_AUTH_TOKEN") or os.environ.get("LTX_AUTH_TOKEN") or ""

# Windows: usually on PATH (C:\Windows\System32\nvidia-smi.exe). If not, set the
# full path here. Test with `nvidia-smi` in the session terminal first — if the
# sampler can't find it, the soak floor silently reads 0.
NVIDIA_SMI = os.environ.get("NVIDIA_SMI", "nvidia-smi")

# Single output folder for ALL run artifacts — logs, soak CSVs, sanity results —
# so the dashboard and the CLI scripts share one location. Override with PERF_RUNS_DIR.
RUNS_DIR = Path(os.environ.get("PERF_RUNS_DIR") or Path(__file__).resolve().parent / "dashboard_runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _app_data_dir() -> Path:
    """App data dir, mirroring electron/app-paths.ts, so we can find the backend's
    session logs. LTX_APP_DATA_DIR (set for the headless backend) wins."""
    env = os.environ.get("LTX_APP_DATA_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "LTXDesktop"


# Where the backend writes its session logs (with tracebacks). The dashboard tails
# the newest one so a failed run's backend traceback is visible in-browser.
BACKEND_LOGS_DIR = _app_data_dir() / "logs"

# --------------------------------------------------------------------------- #
# 2. Trigger ONE generation and BLOCK until it finishes.
#    /api/generate is SYNCHRONOUS: route_generate calls
#    handler.video_generation.generate(req) and returns the terminal
#    GenerateVideoResponse directly — {"status":"complete","video_path":...}
#    or {"status":"cancelled"}. So a blocking POST is the completion signal;
#    the progress poll below is only a fallback / liveness aid.
#
#    Payload = GenerateVideoRequest (backend/api_types.py), model_config
#    strict=True -> field names and types must match EXACTLY (no extras).
# --------------------------------------------------------------------------- #
GENERATE_PATH = "/api/generate"                       # POST, blocks until terminal
STATUS_PATH = "/api/generation/progress"              # GET, poll fallback

# Default: 540p / 5s "fast" distilled two-stage. The leak accumulates per
# build/evict cycle, not per second of video, so the shortest duration gives the
# same reclaim-path sensitivity with a faster per-gen wall time (more cycles/hour).
# Keep duration CONSTANT across a soak — varying it triggers torch.compile shape
# recompiles and shifts the floor baseline.
# duration/fps/resolution are Literals in api_types.py; keep to allowed values.
GEN_PAYLOAD_TEMPLATE: dict[str, Any] = {
    "prompt": "a calm ocean at sunset, slow camera push in",
    "resolution": "540p",          # Literal: 540p|720p|1080p|1440p|2160p
    "model": "fast",               # Literal: fast|pro  (fast = distilled -> cache hits)
    "cameraMotion": "none",
    "negativePrompt": "",
    "duration": 5,                 # Literal: 5|6|8|10|12|14|16|18|20
    "fps": 24,                     # Literal: 24|25|48|50
    "audio": False,
    "imagePath": None,             # set for i2v/ia2v runs
    "audioPath": None,             # set for a2v/ia2v runs
    "aspectRatio": "16:9",         # Literal: 16:9|9:16
    "seed": 42,                    # fixed seed -> deterministic A/B (output_ab.py)
    "loras": [],                   # list[{"ref": str, "scale": float}]
}


def _extract_output(resp: Any) -> str | None:
    """Pull an output file path out of a synchronous generation response.

    Handles the shapes across routes: video_path (generate/extend/retake/ic-lora),
    image_paths[] (generate-image), and result (str | list) (retake payload).
    Raises on a non-complete terminal status.
    """
    if not isinstance(resp, dict):
        return None
    status = resp.get("status")
    if status in ("cancelled", "error"):
        raise RuntimeError(f"generation not complete: {resp}")
    if resp.get("video_path"):
        return resp["video_path"]
    imgs = resp.get("image_paths")
    if isinstance(imgs, list) and imgs:
        return imgs[0]
    r = resp.get("result")
    if isinstance(r, str):
        return r
    if isinstance(r, list) and r:
        return r[0]
    return None


def trigger(route: str, payload: dict[str, Any]) -> str | None:
    """POST a full payload to an arbitrary synchronous generation route and return
    its output path. Used by scenarios that hit routes other than /api/generate
    (generate-image, extend, retake, ic-lora/generate)."""
    return _extract_output(_post(route, payload))


def trigger_generation(overrides: dict[str, Any] | None = None) -> str | None:
    """Trigger one /api/generate video generation and block until complete.

    /api/generate is synchronous, so the POST return value is already terminal.
    Falls back to polling /api/generation/progress only if the route ever
    returns before completion. Raises on cancellation / error / non-2xx.
    """
    payload = {**GEN_PAYLOAD_TEMPLATE, **(overrides or {})}
    out = _extract_output(_post(GENERATE_PATH, payload))
    if out is not None:
        return out
    # Fallback: poll the progress endpoint to a terminal state.
    return _poll_until_done()


def set_torch_compile_enabled(enabled: bool) -> None:
    """Flip AppSettings.use_torch_compile.

    The soak must run with Torch Compile ON — the whole point is the cumulative
    compile-path leak. Prefer starting the backend already compiled; this helper
    is here so a run block can assert/force it.
    """
    _post(SETTINGS_PATH, {"useTorchCompile": enabled})
    time.sleep(0.5)


# --------------------------------------------------------------------------- #
# 3. Set the cache patch on/off for a run block.
#    Live setting: AppSettings.diffusion_stage_cache_enabled, pushed into
#    diffusion_stage_cache.set_enabled() by GenerationHandler on every
#    generation — so flipping it here takes effect on the NEXT generation.
#    POST /api/settings takes a partial patch; settings are camelCase-aliased
#    (populate_by_name=True also accepts snake_case), extra="forbid".
# --------------------------------------------------------------------------- #
SETTINGS_PATH = "/api/settings"                       # backend/_routes/settings.py
CACHE_SETTING_KEY = "diffusionStageCacheEnabled"      # camelCase alias of diffusion_stage_cache_enabled


def set_cache_enabled(enabled: bool) -> None:
    """Set the patch toggle. Takes effect on the next generation."""
    _post(SETTINGS_PATH, {CACHE_SETTING_KEY: enabled})
    time.sleep(0.5)


def get_settings() -> dict:
    """Read the live backend settings (GET /api/settings). Keys are camelCase
    aliases (diffusionStageCacheEnabled, useTorchCompile, ...)."""
    return _get(SETTINGS_PATH) or {}


_MODELS_DIR: Path | None = None


def models_dir() -> Path | None:
    """The backend's models directory (user-configurable), read once from settings.
    Used to check whether a scenario's LoRA / IC-LoRA weights are actually on disk —
    catalog weights live under models_dir/loras/<id>/ and models_dir/ic-loras/<id>/.
    None if the backend can't be reached or doesn't report it."""
    global _MODELS_DIR
    if _MODELS_DIR is not None:
        return _MODELS_DIR
    try:
        s = get_settings()
    except Exception:  # noqa: BLE001
        return None
    raw = s.get("modelsDir") or s.get("models_dir")
    if raw:
        _MODELS_DIR = Path(raw)
    return _MODELS_DIR


# --------------------------------------------------------------------------- #
# Generic helpers — no need to edit below.
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    label: str
    gen_index: int
    wall_s: float
    peak_mib: int
    floor_mib: int
    output_path: str | None = None
    ram_floor_mib: float | None = None    # host-RAM floor in the settle window (spill signal)
    temp_c: float | None = None           # GPU temperature during the gen (thermal/throttle)
    clock_mhz: float | None = None        # mean SM clock during the gen (drops = throttling)
    power_w: float | None = None          # mean power draw during the gen (-> energy per gen)


def gpu_used_mib(gpu_index: int = GPU_INDEX) -> int:
    """Current used VRAM in MiB via nvidia-smi (robust, backend-agnostic)."""
    out = subprocess.check_output(
        [
            NVIDIA_SMI,
            f"--id={gpu_index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(out.strip().splitlines()[0])


_GPU_FIELDS = ("util", "used_mib", "total_mib", "temp_c", "clock_mhz", "power_w")


def gpu_stats(gpu_index: int = GPU_INDEX) -> dict[str, float | None]:
    """One nvidia-smi snapshot: utilization, VRAM, temp, SM clock, power draw, plus
    host RAM. Surfaces contention/thermal at a glance (low util + low clock while
    'busy' = the GPU is starved by another process, not the workload), and — since a
    desktop app shares the machine — how much system RAM the backend is holding."""
    out = subprocess.check_output(
        [NVIDIA_SMI, f"--id={gpu_index}",
         "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,clocks.sm,power.draw",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    vals = [v.strip() for v in out.strip().splitlines()[0].split(",")]
    stats: dict[str, float | None] = {}
    for key, raw in zip(_GPU_FIELDS, vals):
        try:
            stats[key] = float(raw) if "." in raw else int(raw)
        except ValueError:
            stats[key] = None
    stats.update(host_ram())
    return stats


def host_ram() -> dict[str, float | None]:
    """System RAM used/total in MiB. Uses psutil if present, else /proc/meminfo on
    Linux; None values if neither works (e.g. Windows without psutil). Desktop shares
    the machine, so a climbing host-RAM floor across gens = a spill/leak the user feels."""
    try:
        import psutil  # optional; installed with the backend on most setups
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        return {"ram_used_mib": round(vm.used / 1048576), "ram_total_mib": round(vm.total / 1048576),
                "ram_avail_mib": round(vm.available / 1048576), "swap_used_mib": round(sm.used / 1048576)}
    except Exception:  # noqa: BLE001
        pass
    try:
        info: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, rest = line.partition(":")
            info[k] = int(rest.strip().split()[0])  # kB
        total, avail = info.get("MemTotal"), info.get("MemAvailable")
        swt, swf = info.get("SwapTotal"), info.get("SwapFree")
        swap = round((swt - swf) / 1024) if (swt is not None and swf is not None) else None
        if total and avail is not None:
            return {"ram_used_mib": round((total - avail) / 1024), "ram_total_mib": round(total / 1024),
                    "ram_avail_mib": round(avail / 1024), "swap_used_mib": swap}
    except Exception:  # noqa: BLE001
        pass
    return {"ram_used_mib": None, "ram_total_mib": None, "ram_avail_mib": None, "swap_used_mib": None}


def disk_free() -> dict[str, int | None]:
    """Free/total disk (GiB) on the app-data volume — models + outputs live there, and a
    full disk fails generations. Falls back to $HOME if the app-data dir doesn't exist yet."""
    for base in (_app_data_dir(), Path.home()):
        try:
            u = shutil.disk_usage(str(base))
            return {"disk_free_gb": round(u.free / 1073741824), "disk_total_gb": round(u.total / 1073741824)}
        except OSError:
            continue
    return {"disk_free_gb": None, "disk_total_gb": None}


# Common desktop GPU VRAM tiers (GiB). A desktop shares the GPU with the OS/other
# apps, so a config only "fits" a tier if its peak VRAM stays under it with headroom.
CARD_TIERS_GIB = (8, 12, 16, 24)
_FIT_HEADROOM = 0.90


def vram_fit(peak_mib: int) -> str:
    """Smallest common card tier this peak VRAM fits under (with headroom), or a
    warning if it exceeds the largest tier. Answers 'what card can run this config'."""
    if not peak_mib:
        return ""
    for gib in CARD_TIERS_GIB:
        if peak_mib < gib * 1024 * _FIT_HEADROOM:
            return f"fits ≥{gib}GB"
    return f">{CARD_TIERS_GIB[-1]}GB (high-end only)"


_GPU_TOTAL_MIB: int | None = None


def gpu_total_mib(gpu_index: int = GPU_INDEX) -> int | None:
    """Total VRAM (MiB) for this GPU, read once. Used with peak VRAM to report how close
    a run came to OOM (headroom %). None where nvidia-smi isn't available (e.g. MPS)."""
    global _GPU_TOTAL_MIB
    if _GPU_TOTAL_MIB is None:
        try:
            t = gpu_stats(gpu_index).get("total_mib")
            _GPU_TOTAL_MIB = int(t) if t else None
        except Exception:  # noqa: BLE001
            return None
    return _GPU_TOTAL_MIB


def ffprobe_info(path: str | None) -> dict | None:
    """Probe a produced media file with ffprobe -> width/height/duration/fps/
    nb_frames/has_audio. None if the path is missing, ffprobe is unavailable, or the
    file won't parse. Used to verify the output matches what was requested."""
    if not path or not os.path.exists(path):
        return None
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", path],
            text=True, stderr=subprocess.DEVNULL,
        )
        data = json.loads(out)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return None
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})
    dur = None
    for src in ((v or {}).get("duration"), fmt.get("duration")):
        try:
            dur = float(src)
            break
        except (TypeError, ValueError):
            continue
    fps = None
    rate = (v or {}).get("avg_frame_rate", "")
    if rate and rate != "0/0" and "/" in rate:
        num, den = rate.split("/")
        try:
            fps = round(float(num) / float(den), 2) if float(den) else None
        except (ValueError, ZeroDivisionError):
            fps = None
    nb = int(v["nb_frames"]) if v and str(v.get("nb_frames", "")).isdigit() else None

    def _int(x: object) -> int | None:
        try:
            return int(x)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    size = _int(fmt.get("size"))
    if size is None:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = None
    return {
        "width": (v or {}).get("width"), "height": (v or {}).get("height"),
        "duration": dur, "fps": fps, "nb_frames": nb, "has_audio": a is not None,
        "size": size, "bit_rate": _int(fmt.get("bit_rate")),
    }


def gpu_busy(threshold_util: float = 20.0, samples: int = 5, interval: float = 0.2) -> tuple[bool, float]:
    """Sample GPU utilization briefly; return (busy?, median util). Use before a soak
    to catch another process already using the GPU (which would contaminate wall time)."""
    utils = []
    for _ in range(samples):
        try:
            u = gpu_stats().get("util")
            if u is not None:
                utils.append(u)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(interval)
    med = statistics.median(utils) if utils else 0.0
    return med > threshold_util, med


def wait_for_backend(timeout_s: int = 120) -> None:
    """Block until /health responds ok."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            h = _get(HEALTH_PATH)
            if h:
                return
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(1.0)
    raise RuntimeError(f"backend not healthy after {timeout_s}s (last: {last})")


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else {}


def _get(path: str) -> Any:
    req = urllib.request.Request(BASE_URL + path, method="GET", headers=_auth_headers())
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as r:
        body = r.read().decode()
    return json.loads(body) if body else None


def _post(path: str, payload: dict[str, Any]) -> Any:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE_URL + path, data=data, method="POST",
        headers={"Content-Type": "application/json", **_auth_headers()},
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as r:
        body = r.read().decode()
    return json.loads(body) if body else None


def _poll_until_done(interval_s: float = 1.0) -> str | None:
    """Poll GET /api/generation/progress to a terminal status.

    GenerationProgressResponse.status is one of
    idle|running|complete|cancelled|error; the output (if any) is in `result`
    (str | list[str] | None).
    """
    while True:
        st = _get(STATUS_PATH) or {}
        status = st.get("status", "")
        if status == "complete":
            result = st.get("result")
            if isinstance(result, list):
                return result[0] if result else None
            return result
        if status in {"error", "cancelled"}:
            raise RuntimeError(f"generation failed: {st}")
        time.sleep(interval_s)
