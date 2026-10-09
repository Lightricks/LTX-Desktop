"""Project 181's process mode + decide_video_job into the published compat matrix.

Not a second Yes table. CUDA 16 GB that cannot stream 720p/20s is No for that
cell in the Fast picker; the T2V row stays Yes because 540p still loads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from api_types import (
    LTXLocalModelId,
    LTXVideoGenResolution,
    LTXVideoGenerationResolutionSpec,
)
from frame_math import compute_num_frames
from runtime_config.ic_lora_job_budget import (
    HOME_IC_LORA_PREVIEW_SETTINGS,
    IcLoraJobLoad,
    catalog_preview_stage1,
    decide_ic_lora_job,
)
from runtime_config.ltx_capabilities import LtxCapabilityFeature, local_caps, pixels_for, supports
from runtime_config.model_download_specs import get_latest_ltx_model_id, get_ltx_model_spec
from runtime_config.runtime_policy import (
    CUDA_FULL_VRAM_GB,
    CUDA_VRAM_FLOOR_GB,
    DARWIN_ADVERTISED_FLOOR_GB,
    DARWIN_STREAMING_FLOOR_GB,
    LocalGenerationMode,
    decide_local_generation_mode,
)
from runtime_config.video_job_budget import advertised_fast_durations, memory_gb_for_job

_REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = _REPO_ROOT / "docs" / "local-compat-matrix.md"

# Typical 16 GB CUDA SKU used as the stream-column example (floor is 15).
_CUDA_STREAM_EXAMPLE_GB = 16
_DARWIN_STREAM_EXAMPLE_GB = 48

# Retake/Extend are hardware-viable but only on 2.3, which is not the Explore default.
_LEGACY_LOCAL_MODEL_ID: LTXLocalModelId = "ltx-2.3-22b-distilled-1.1"
_LEGACY_ONLY_CELL = "2.3 only"


@dataclass(frozen=True, slots=True)
class _Band:
    label: str
    system: str
    cuda_available: bool
    vram_gb: int | None
    mps_available: bool
    ram_gb: int | None


_BANDS = (
    _Band("CUDA unsupported", "Linux", True, CUDA_VRAM_FLOOR_GB - 1, False, 64),
    _Band("CUDA stream", "Linux", True, _CUDA_STREAM_EXAMPLE_GB, False, 64),
    _Band("CUDA full", "Linux", True, CUDA_FULL_VRAM_GB, False, 64),
    _Band("Darwin unsupported", "Darwin", False, None, True, DARWIN_STREAMING_FLOOR_GB - 1),
    _Band("Darwin stream", "Darwin", False, None, True, _DARWIN_STREAM_EXAMPLE_GB),
    _Band("Intel Mac / no GPU", "Darwin", False, None, False, 64),
)

_ROWS = (
    "T2V",
    "I2V",
    "A2V",
    "LoRAs",
    "IC-LoRA",
    "Retake",
    "Extend",
    "local text encoder",
    "2.5 download",
)

_CAPABILITY_BY_ROW: dict[str, LtxCapabilityFeature] = {
    "T2V": "t2v",
    "I2V": "i2v",
    "A2V": "a2v",
}


def _durations_at_24(
    specs: dict[LTXVideoGenResolution, LTXVideoGenerationResolutionSpec] | None,
) -> dict[LTXVideoGenResolution, list[int]]:
    if not specs:
        return {}
    return {
        res: [int(duration) for duration in rspec.fps_to_durations.get(24, [])]
        for res, rspec in specs.items()
    }


def _fast_duration_maps() -> tuple[
    dict[LTXVideoGenResolution, list[int]],
    dict[LTXVideoGenResolution, list[int]],
]:
    spec = get_ltx_model_spec(get_latest_ltx_model_id()).supported_pipelines[0][1]
    return (
        _durations_at_24(spec.supported_resolutions_durations),
        _durations_at_24(spec.a2v_supported_resolutions_durations),
    )


def _process_mode(band: _Band) -> LocalGenerationMode:
    return decide_local_generation_mode(
        band.system,
        band.cuda_available,
        band.vram_gb,
        mps_available=band.mps_available,
        ram_gb=band.ram_gb,
    )


def _job_memory(band: _Band) -> float | None:
    darwin = band.system == "Darwin"
    return memory_gb_for_job(
        vram_gb=band.vram_gb,
        available_ram_gb=band.ram_gb,
        darwin=darwin,
    )


def _any_fast_cell(
    band: _Band,
    process_mode: LocalGenerationMode,
    duration_map: dict[LTXVideoGenResolution, list[int]],
) -> bool:
    caps = local_caps(get_latest_ltx_model_id())
    darwin = band.system == "Darwin"
    memory_gb = _job_memory(band)
    for resolution, durations in duration_map.items():
        width, height = pixels_for(caps, resolution, "16:9")
        kept = advertised_fast_durations(
            width,
            height,
            24,
            durations,
            memory_gb=memory_gb,
            process_mode=process_mode,
            darwin=darwin,
        )
        if kept:
            return True
    return False


def _ic_lora_viable(band: _Band, process_mode: LocalGenerationMode) -> bool:
    """True when the lightest Home recipe (single-stage, 5 s) fits this band.

    A stage-2 recipe such as Layout To Render at 1080p is heavier. It can be rejected
    on 12-20 GB cards where this returns True.
    """
    stage1_w, stage1_h = catalog_preview_stage1(HOME_IC_LORA_PREVIEW_SETTINGS)
    decision = decide_ic_lora_job(
        stage1_w,
        stage1_h,
        compute_num_frames(5, 24),
        memory_gb=_job_memory(band),
        process_mode=process_mode,
        stage_mode=HOME_IC_LORA_PREVIEW_SETTINGS.stage_mode,
        darwin=band.system == "Darwin",
    )
    return isinstance(decision, IcLoraJobLoad)


def _retake_extend_cell(feature: LtxCapabilityFeature, *, viable: bool) -> str:
    """Never a bare Yes while the Explore default cannot run the feature.

    Explore defaults to local 2.5, which has no Retake/Extend; 2.3 still offers both.
    Turns into a plain Yes on its own once 2.5 gains them.
    """
    if not viable:
        return "No"
    if supports(local_caps(get_latest_ltx_model_id()), feature):
        return "Yes"
    if supports(local_caps(_LEGACY_LOCAL_MODEL_ID), feature):
        return _LEGACY_ONLY_CELL
    return "No"


def _cell(
    row: str,
    band: _Band,
    t2v_i2v_map: dict[LTXVideoGenResolution, list[int]],
    a2v_map: dict[LTXVideoGenResolution, list[int]],
) -> str:
    process_mode = _process_mode(band)
    viable = process_mode != "unsupported"
    latest = get_latest_ltx_model_id()
    if row in _CAPABILITY_BY_ROW:
        if not supports(local_caps(latest), _CAPABILITY_BY_ROW[row]):
            return "No"
        duration_map = a2v_map if row == "A2V" else t2v_i2v_map
        return "Yes" if _any_fast_cell(band, process_mode, duration_map) else "No"
    if row == "IC-LoRA":
        return "Yes" if _ic_lora_viable(band, process_mode) else "No"
    if row in ("Retake", "Extend"):
        return _retake_extend_cell("retake" if row == "Retake" else "extend", viable=viable)
    if row == "LoRAs":
        return "Yes" if viable and supports(local_caps(latest), "user_loras") else "No"
    if row in ("local text encoder", "2.5 download"):
        return "Yes" if viable else "No"
    raise AssertionError(row)


def render_compat_matrix() -> str:
    t2v_i2v_map, a2v_map = _fast_duration_maps()
    headers = ["Tool", *(band.label for band in _BANDS)]
    lines = [
        "# Local Explore compatibility matrix",
        "",
        "Generated from `decide_local_generation_mode` and `decide_video_job`.",
        "A machine that cannot run a Fast cell is not offered that cell.",
        "Regenerate: `cd backend && uv run python -m runtime_config.local_compat_matrix`.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in _ROWS:
        cells = [_cell(row, band, t2v_i2v_map, a2v_map) for band in _BANDS]
        lines.append("| " + " | ".join([row, *cells]) + " |")
    lines.extend(
        [
            "",
            f"Floors: CUDA integer **{CUDA_VRAM_FLOOR_GB}** GiB VRAM (`total_memory // 1024**3`);",
            f"Darwin **{DARWIN_ADVERTISED_FLOOR_GB}** GiB total RAM; ~72 GB disk for the 2.5 Fast core pack",
            "(kitchen-sink installs with 2.3, image models, and processors are larger).",
            "",
            f"CUDA stream ({_CUDA_STREAM_EXAMPLE_GB} GB) still runs T2V/I2V/A2V; the Fast picker hides 720p/20s",
            f"and 1080p/10s because `decide_video_job` would reject them. CUDA full ({CUDA_FULL_VRAM_GB} GB)",
            "keeps those cells (they stream). Darwin stream keeps the static Fast ceiling.",
            "",
        ]
    )
    if any(_LEGACY_ONLY_CELL in line for line in lines):
        lines.extend(
            [
                f"`{_LEGACY_ONLY_CELL}`: hardware can run it, but not on Explore's default local 2.5",
                f"(no local Retake/Extend there) — only after switching to {_LEGACY_LOCAL_MODEL_ID}.",
                "",
            ]
        )
    lines.extend(
        [
            "Footnote: Darwin IC-LoRA duration is unbounded past the 540p spatial cap",
            "(no token 422). Do not invent a CUDA-curve ceiling for Darwin.",
            "",
        ]
    )
    return "\n".join(lines)


def write_compat_matrix(path: Path = MATRIX_PATH) -> Path:
    path.write_text(render_compat_matrix(), encoding="utf-8")
    return path


if __name__ == "__main__":
    written = write_compat_matrix()
    print(written)
