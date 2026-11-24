"""Project 181's process mode + decide_video_job into the published compat matrix.

Not a second Yes table. CUDA 16 GB that cannot stream 720p/20s is No for that
cell in the Fast picker; the T2V row stays Yes because 540p still loads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from api_types import (
    LTXLocalModelId,
    LTXVideoGenResolution,
    LTXVideoGenerationResolutionSpec,
)
from frame_math import compute_num_frames
from runtime_config.budget_admission import Admitted
from runtime_config.ic_lora_job_budget import (
    HOME_IC_LORA_PREVIEW_SETTINGS,
    MAC_IC_LORA_FLOOR_GIB,
    catalog_preview_stage1,
    decide_ic_lora_job,
)
from runtime_config.ltx_capabilities import LtxCapabilityFeature, local_caps, pixels_for, supports
from runtime_config.model_download_specs import get_latest_ltx_model_id, get_ltx_model_spec
from runtime_config.runtime_policy import (
    CUDA_FULL_VRAM_GB,
    CUDA_VRAM_FLOOR_GB,
    DARWIN_ADVERTISED_FLOOR_GB,
    DARWIN_MIN_MACOS_MAJOR,
    DARWIN_STREAMING_FLOOR_GB,
    LINUX_VRAM_FLOOR_GB,
    LocalGenerationMode,
    decide_local_generation_mode,
)
from runtime_config.video_job_budget import (
    MAC_FAST_FLOOR_GIB,
    NOMINAL_MPS_POOL_RATIO,
    WINDOWS_EDIT_CPU_STREAMING_RAM_GB,
    WINDOWS_EDIT_DISK_STREAMING_RAM_GB,
    WINDOWS_FAST_DISK_STREAMING_RAM_GB,
    WINDOWS_FAST_STREAMING_RAM_BYTES,
    WINDOWS_ICLORA_CPU_STREAMING_RAM_GB,
    WINDOWS_ICLORA_DISK_STREAMING_RAM_GB,
    HardwareProfile,
    LocalJobFilter,
    MemoryBudget,
    advertised_fast_durations,
    mac_disk_stream_host_gib,
    memory_gb_for_job,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = _REPO_ROOT / "docs" / "local-compat-matrix.md"
_GIB = 1024**3

# Typical 16 GB CUDA SKU used as the stream-column example (floor is 9).
_CUDA_STREAM_EXAMPLE_GB = 16
_DARWIN_STREAM_EXAMPLE_GB = 48

# Retake/Extend are hardware-viable but only on 2.3, which is not the Explore default.
_LEGACY_LOCAL_MODEL_ID: LTXLocalModelId = "ltx-2.3-22b-distilled-1.1"
_LEGACY_ONLY_CELL = "2.3 only"


@dataclass(frozen=True, slots=True)
class _Band:
    label: str
    system: Literal["Windows", "Linux", "Darwin"]
    cuda_available: bool
    vram_gb: int | None
    mps_available: bool
    ram_gb: int | None
    macos_major: int | None = None


_WINDOWS_STREAM_VRAM_GB = 16
_WINDOWS_STREAM_RAM_GB = 40
_WINDOWS_GATED_RAM_GB = 32

_BANDS = (
    _Band("CUDA unsupported", "Linux", True, LINUX_VRAM_FLOOR_GB - 1, False, 64),
    _Band("CUDA stream", "Linux", True, _CUDA_STREAM_EXAMPLE_GB, False, 64),
    _Band("CUDA full", "Linux", True, CUDA_FULL_VRAM_GB, False, 64),
    _Band("Windows stream", "Windows", True, _WINDOWS_STREAM_VRAM_GB, False, _WINDOWS_STREAM_RAM_GB),
    _Band("Windows RAM-gated", "Windows", True, _WINDOWS_STREAM_VRAM_GB, False, _WINDOWS_GATED_RAM_GB),
    _Band(
        "Darwin unsupported", "Darwin", False, None, True, DARWIN_STREAMING_FLOOR_GB - 1, DARWIN_MIN_MACOS_MAJOR
    ),
    _Band(
        f"macOS below {DARWIN_MIN_MACOS_MAJOR}",
        "Darwin",
        False,
        None,
        True,
        _DARWIN_STREAM_EXAMPLE_GB,
        DARWIN_MIN_MACOS_MAJOR - 1,
    ),
    _Band("Darwin stream", "Darwin", False, None, True, _DARWIN_STREAM_EXAMPLE_GB, DARWIN_MIN_MACOS_MAJOR),
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


# Anchor rationale (fps/ratio audit): the Fast rows answer "can this band run
# ANY Fast cell", and that existence is decided at the 24 fps / 16:9 anchor.
# fps: the static ceiling lists identical durations at every rate while frames
# (hence tokens) grow with fps, so a band with no kept 24 fps cell has none
# anywhere — higher rates can only subtract. aspect: lists price every cell at
# the 16:9 anchor and request time only ever upgrades wider (21:9), never
# discounts narrower — so a band with no kept anchor cell offers nothing, and
# checking narrower shapes would claim cells the picker hides and the budget
# refuses. Extension past the anchor is vacuous (fps, 21:9) or a lie
# (narrower); the anchor stands.
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
        macos_major=band.macos_major,
    )


def _job_memory(band: _Band) -> float | None:
    darwin = band.system == "Darwin"
    return memory_gb_for_job(
        vram_gb=band.vram_gb,
        available_ram_gb=band.ram_gb,
        darwin=darwin,
    )


def _band_bytes(gb: int | None) -> int | None:
    """Nominal band GiB as exact bytes. Bands are hardware classes, not probes."""
    return None if gb is None else gb * _GIB


def _band_pool_bytes(band: _Band) -> int | None:
    """Nominal MPS pool for a band, or None where no MPS pool exists."""
    if band.system != "Darwin" or not band.mps_available or band.ram_gb is None:
        return None
    return int(band.ram_gb * _GIB * NOMINAL_MPS_POOL_RATIO)


def _band_filter(band: _Band) -> LocalJobFilter:
    """Hardware class as the same filter the app serves, with nominal bytes.

    Strict mode: the published matrix lists only what the lab has measured.
    """
    return LocalJobFilter(
        profile=HardwareProfile(
            system=band.system,
            model_id=get_latest_ltx_model_id(),
            process_mode=_process_mode(band),
        ),
        budget=MemoryBudget(
            total_vram_bytes=_band_bytes(band.vram_gb),
            free_vram_bytes=None,
            total_ram_bytes=_band_bytes(band.ram_gb),
            memory_gb=_job_memory(band),
            mps_pool_bytes=_band_pool_bytes(band),
        ),
        budget_mode="cautious",
    )


def _any_fast_cell(
    band: _Band,
    duration_map: dict[LTXVideoGenResolution, list[int]],
) -> bool:
    caps = local_caps(get_latest_ltx_model_id())
    job_filter = _band_filter(band)
    for resolution, durations in duration_map.items():
        width, height = pixels_for(caps, resolution, "16:9")
        kept = advertised_fast_durations(
            width,
            height,
            24,
            durations,
            job_filter=job_filter,
        )
        if kept:
            return True
    return False


def _ic_lora_viable(band: _Band) -> bool:
    """True when the lightest Home recipe (single-stage, 5 s) fits this band.

    A stage-2 recipe such as Layout To Render at 1080p is heavier. It can be rejected
    on 12-20 GB cards where this returns True.
    """
    stage1_w, stage1_h = catalog_preview_stage1(HOME_IC_LORA_PREVIEW_SETTINGS)
    admission = decide_ic_lora_job(
        stage1_w,
        stage1_h,
        compute_num_frames(5, 24),
        stage_mode=HOME_IC_LORA_PREVIEW_SETTINGS.stage_mode,
        job_filter=_band_filter(band),
    )
    return isinstance(admission, Admitted)


def _retake_extend_cell(feature: LtxCapabilityFeature, band: _Band) -> str:
    """Viability + RAM floor; VRAM is input-dependent (asterisk).

    No representative edit cell exists without an arbitrary canonical input
    (cost depends on the source window). But the RAM floor is input-free —
    it comes from the profile, not the footage (see
    windows_edit_required_ram_bytes) — so bands answer from floors alone.
    Bands carry no drive class, so Windows bands take the CPU path
    (WINDOWS_EDIT_CPU_STREAMING_RAM_GB floor; NVMe machines map from disk at
    WINDOWS_EDIT_DISK_STREAMING_RAM_GB instead). "Yes*" means the floor
    passes and the default model advertises the feature; VRAM still
    depends on the source length (the gate decides each window live).
    """
    if _process_mode(band) == "unsupported":
        return "No"
    if band.system == "Windows":
        floor_bytes = _band_filter(band).profile.windows_edit_required_ram_bytes
        if band.ram_gb is None or band.ram_gb * _GIB < floor_bytes:
            return "No"
    if supports(local_caps(get_latest_ltx_model_id()), feature):
        return "Yes*"
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
        return "Yes" if _any_fast_cell(band, duration_map) else "No"
    if row == "IC-LoRA":
        return "Yes" if _ic_lora_viable(band) else "No"
    if row in ("Retake", "Extend"):
        return _retake_extend_cell("retake" if row == "Retake" else "extend", band)
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
        "Generated from `decide_local_generation_mode`, `decide_video_job`, and the edit RAM floors.",
        "A machine that cannot run a Fast cell is not offered that cell.",
        "Fast rows are decided at the 24 fps / 16:9 anchor: higher rates offer a",
        "subset of these durations, and wider shapes can refuse a kept top cell",
        "(the picker shows it; the budget enforces it).",
        "Retake/Extend rows are viability + RAM floor (`Yes*` — VRAM depends on",
        "your source length): no canonical input exists for a representative",
        "cell, but the floor is input-free. Bands carry no drive class, so",
        "Windows bands take the CPU path (64 GB floor; NVMe streams from disk at",
        f"{WINDOWS_EDIT_DISK_STREAMING_RAM_GB} GB instead).",
        "Regenerate: `cd backend && uv run python -m runtime_config.local_compat_matrix`.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in _ROWS:
        cells = [_cell(row, band, t2v_i2v_map, a2v_map) for band in _BANDS]
        lines.append("| " + " | ".join([row, *cells]) + " |")
    windows_ram_floor = WINDOWS_FAST_STREAMING_RAM_BYTES // _GIB
    lines.extend(
        [
            "",
            f"Floors: Windows **{CUDA_VRAM_FLOOR_GB}** (fp8 cards; **{LINUX_VRAM_FLOOR_GB}** without fp8) / Linux **{LINUX_VRAM_FLOOR_GB}** GiB total VRAM (exact bytes, integer fallback);",
            f"Windows streamed Fast needs **{windows_ram_floor}** GiB total RAM;",
            f"Darwin **{DARWIN_ADVERTISED_FLOOR_GB}** GiB total RAM and macOS **{DARWIN_MIN_MACOS_MAJOR}** or later",
            "(macOS 14 and 15 run the app but not local generation); Fast cells gate on the MPS pool",
            f"({MAC_FAST_FLOOR_GIB} GiB floor, plus {mac_disk_stream_host_gib(get_latest_ltx_model_id()):.2f} GiB",
            "of loader host buffers); ~72 GB disk for the 2.5 Fast core pack",
            "(kitchen-sink installs with 2.3, image models, and processors are larger).",
            "",
            f"CUDA stream ({_CUDA_STREAM_EXAMPLE_GB} GB) still runs T2V/I2V/A2V;",
            "the Fast picker disables 720p/20s and 1080p at 8s and up (greyed out with the reason)",
            f"because `decide_video_job` would reject them. CUDA full ({CUDA_FULL_VRAM_GB} GB)",
            "keeps those cells (they stream) except 1080p/20s, whose stream ask exceeds `CUDA_FULL_VRAM_GB`.",
            "Darwin stream gates each Fast cell against the sampled MPS",
            f"pool (M2/F1 model, {MAC_FAST_FLOOR_GIB} GiB floor); the 48 GB example keeps the full grid (1080p/20s in windows)",
            f"(nominal {NOMINAL_MPS_POOL_RATIO} pool ratio — min of 2 F0 probes, SKU-dependent).",
            "",
            f"Windows stream ({_WINDOWS_STREAM_VRAM_GB} GB VRAM, {_WINDOWS_STREAM_RAM_GB} GB RAM) runs every Fast cell",
            "through 1080p/20s, in time windows where one pass does not fit (1080p/10s runs in 3); what each card size runs is in",
            f"`docs/hardware-compatibility.md`. Below {windows_ram_floor} GB RAM, streamed Fast cells are not",
            f"offered (`ram_shortfall`). Streamed edit cells gate on E3-measured floors ({WINDOWS_EDIT_CPU_STREAMING_RAM_GB} GB CPU,",
            f"{WINDOWS_EDIT_DISK_STREAMING_RAM_GB} GB NVMe DISK); streamed IC-LoRA on L5-measured floors ({WINDOWS_ICLORA_CPU_STREAMING_RAM_GB} GB CPU,",
            f"{WINDOWS_ICLORA_DISK_STREAMING_RAM_GB} GB DISK, L5-32 holds both temps). The live picker lists every total-fitting cell;",
            "GenSpace refuses cells over free VRAM at click, and queued jobs re-check at execute.",
            f"On NVMe drives below {windows_ram_floor} GB, streamed Fast reads weights from disk and the RAM floor is **{WINDOWS_FAST_DISK_STREAMING_RAM_GB}** GiB instead.",
            "",
            "1440p and 4K Fast cells run long clips in overlapping time windows, and the job budget prices one window.",
            "1440p and 4K are offered for Fast, A2V and the Home recipes; Retake, Extend and IC-LoRA need `highResVideoToVideo` (Refine Details does not). Both are provisional;",
            "Windows and Mac each fit their own window extra. The cells above are the 24 fps / 16:9 anchor and do not list these two rows.",
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
            "Footnote: Darwin IC-LoRA gates each duration against the sampled MPS",
            f"pool (I1/I2 model, {MAC_IC_LORA_FLOOR_GIB} GiB floor, growth line",
            "P20-refit on 1.4.1: 5.75 + 6.105e-4 per token, +0.50 for",
            "ingredients); the 48 GB example keeps 20s portrait.",
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
