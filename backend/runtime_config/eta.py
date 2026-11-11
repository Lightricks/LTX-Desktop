"""Whole-run and 3-part wall-time estimates for local video generation (ETA).

Reference table and 3-part stage law (load + denoise + decode) measured on:
- Windows 11 / RTX 5090 / 63.45 GiB RAM (Fast, Edit, IC-LoRA, 540p to 4K)
- macOS / Apple M5 Max / 128 GB unified memory (Fast, Edit, IC-LoRA, 540p to 4K)
- macOS / Apple M3 Max / 36 GB unified memory (270p to 720p Fast, Custom LoRA, IC-LoRA)

Exact measured cells return their recorded seconds with ``basis="measured"``.
Other shapes use the 3-part stage law (`estimate_stage_breakdown_s`):
1. ``load_s``: cold pipeline + text encoding (fixed per platform/mode) plus
   any input-video VAE encoding (`source_frames`).
2. ``denoise_s``: per-step weight streaming (fixed per step) plus GPU token
   compute across stage-1 and stage-2 steps, summed over temporal windows and
   multiplied by spatial tiles.
3. ``decode_s``: VAE decoding and media encoding proportional to output latent
   tokens across windows.

``machine_factor`` is this computer's speed relative to the platform reference.
On CUDA streaming it scales the per-step weight streaming. Transfer overlaps
compute and bounds each step, so GPU compute, load, and decode stay at the
reference. A 32 GiB-RAM machine on DISK offload cannot page-cache the
transformer and pays ~26 s/step where the reference pays ~6.6 s/step, whatever
the token count. Full-load CUDA and Mac scale the GPU terms.
Evidence: Windows 11 / RTX 4090 / 32 GiB RAM, DISK offload, Fast 270p-1440p
and 24-48 fps: every 1-stage run 229-238 s and every 2-stage run 309-334 s
(26.0-26.9 s/step); scaling only GPU compute clamped the factor at 8.0 and
left ETAs 34-64% short.

Unknown anything -> None. An ETA that says unknown beats one that lies.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from runtime_config.fast_chunking import (
    CARRY_FRAMES,
    is_chunked_canvas,
    last_window_frames,
    window_candidates,
    window_count,
)
from runtime_config.ic_lora_tiling import IcLoraTiling

if TYPE_CHECKING:
    from runtime_config.budget_evidence import JobShape
    from runtime_config.device_calibration import CalibrationBucket, EnvelopePoint, JobEnvelope

EtaBasis = Literal["measured", "scaled"]
EtaFeature = Literal["t2v", "i2v", "a2v"]
EtaLoadMode = Literal["full_models_loading", "streaming_models_loading"]
EtaOffloadMode = Literal["NONE", "CPU", "DISK"]
EtaSystem = Literal["Windows", "Linux", "Darwin"]

_STAGE_1_STEPS = 8
_STAGE_2_STEPS = 3
_HALF_RES_REF_IC_LORAS = frozenset({"video-guide", "motion-track"})
# Bounds on stored shapes. A tampered row must not size a huge window list (CWE-770).
_MAX_FRAMES = 100_000
_MAX_WINDOWS = 1_000
_MIN_MACHINE_FACTOR = 0.25
_MAX_MACHINE_FACTOR = 8.0
# CUDA decode term is a DiffVAE rate. ConvVAE (Settings -> Fast decode) decodes the same latents
# at about 0.31x of it.
# Evidence: Windows 11 / RTX 4090, 2026-11-08 ETA matrix: ConvVAE decode / reference term
# median 0.31 (0.22-0.39) over 18 Fast runs 360p-1440p; DiffVAE 540p/720p 5s at 1.12x/1.22x.
_CONV_VAE_DECODE_RATIO = 0.31


def _latent_tokens(height: int, width: int, frames: int) -> float:
    return ((max(1, frames) - 1) / 8.0 + 1.0) * (height / 32.0) * (width / 32.0)


@dataclass(frozen=True, slots=True)
class WallTimeBreakdown:
    load_s: int
    denoise_s: int
    decode_s: int
    stage_1_share: float | None = field(default=None, compare=False)

    @property
    def seconds(self) -> int:
        return self.load_s + self.denoise_s + self.decode_s


@dataclass(frozen=True, slots=True)
class WallTimeEstimate:
    seconds: int
    basis: EtaBasis
    t2v_reuse: bool
    load_s: int = field(default=0, compare=False)
    denoise_s: int = field(default=0, compare=False)
    decode_s: int = field(default=0, compare=False)
    stage_1_share: float | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class _Cell:
    full_cold_s: int | None
    stream_cpu_cold_s: int
    disk_ratio: float
    disk_ratio_measured: bool


# Mean of the measured 1.6.0 same-session DISK/CPU ratios across F1 and F2
# cold+warm. Applied only where no direct 1.6.0 DISK measurement exists, and
# labelled "scaled" there.
_DEFAULT_DISK_RATIO = 1.07

# (width, height, frames) at 24 fps. 720p/20s and 1080p/10s never full-load,
# so their full column is None rather than a guess. Keys are fps-agnostic: a
# same-frames job at another rate (5s@25fps is 121f like 5s@24fps) matches the
# same key wherever its (w, h) is listed — no new cells needed there.
_TABLE: dict[tuple[int, int, int], _Cell] = {
    (1024, 576, 121): _Cell(121, 158, _DEFAULT_DISK_RATIO, False),
    (1024, 576, 481): _Cell(206, 222, _DEFAULT_DISK_RATIO, False),
    (1280, 704, 481): _Cell(None, 280, 1.06, True),
    (1920, 1088, 121): _Cell(215, 232, _DEFAULT_DISK_RATIO, False),
    (1920, 1088, 241): _Cell(None, 338, 1.10, True),
    # 4K runs in windows and never full-loads. Both cells come from one run each.
    (3840, 2176, 73): _Cell(None, 416, _DEFAULT_DISK_RATIO, False),
    (3840, 2176, 241): _Cell(None, 1537, _DEFAULT_DISK_RATIO, False),
}

# Mac M5 Max measured Fast cells (streaming DISK).
_MAC_TABLE: dict[tuple[int, int, int], int] = {
    (1024, 576, 121): 177,
    (1280, 704, 241): 512,
    (1920, 1088, 121): 733,
    (1024, 576, 481): 866,
    (1280, 704, 481): 1450,
    (1920, 1088, 241): 1739,
    (2560, 1472, 241): 3528,
    (2560, 1472, 481): 7222,
    (3840, 2176, 73): 2508,
    (3840, 2176, 241): 10154,
}


def is_measured_hardware_class(device_name: str | None) -> bool:
    """Desktop RTX 5090 class. A substring, deliberately: driver string
    variations must not silently disable the feature, and any 5090-variant
    SKU is close enough to the measured box for a cold whole-run estimate.
    Laptops match the name family but not the clocks, so they are out."""
    if not device_name:
        return False
    lowered = device_name.lower()
    return "5090" in lowered and "laptop" not in lowered


def default_machine_factor(device_name: str | None) -> float | None:
    """Baseline compute factor relative to the platform reference (5090 on CUDA, M5 Max on Mac).

    Stage 2 device calibration overrides this with the machine's measured factor.
    """
    if not device_name:
        return None
    if is_measured_hardware_class(device_name):
        return 1.0
    lowered = device_name.lower()
    if "apple m5 max" in lowered:
        return 1.0
    if "apple m4 max" in lowered:
        return 1.35
    if "apple m3 max" in lowered:
        return 2.0
    if "apple m4 pro" in lowered:
        return 3.5
    if "apple m2 max" in lowered or "apple m3 pro" in lowered:
        return 2.4
    if "apple m1 max" in lowered or "apple m2 pro" in lowered:
        return 2.8
    return None


def _window_frames_list(
    width: int,
    height: int,
    frames: int,
    chunk_pixel_frames: int | None,
    carry_frames: int = CARRY_FRAMES,
) -> list[int]:
    win = chunk_pixel_frames
    if win is None and is_chunked_canvas(width, height):
        candidates = window_candidates(width, height)
        if candidates and frames > candidates[0]:
            win = candidates[0]
    if win is None or frames <= win:
        return [frames]
    count = window_count(frames, win, carry_frames)
    last = last_window_frames(frames, win, carry_frames)
    return [win] * (count - 1) + [last]


_LATENT_PX = 32


def _axis_tile_count(size: int, tile_size: int, overlap_fraction: float) -> int:
    """Tiles on one axis, as upstream ``split_by_size_pinned`` lays them out.

    Upstream works in 32 px latent cells. The overlap is a minimum: the count grows until it is
    met, capped so that no cell is covered by three tiles. The cap binds first at high overlap,
    so the count stops growing once the overlap is high enough.
    """
    if size <= tile_size:
        return 1
    dim = size // _LATENT_PX
    tile = tile_size // _LATENT_PX
    min_overlap = int(tile_size * overlap_fraction) // _LATENT_PX
    limit = max(2, 1 + (dim - tile) // ((tile + 1) // 2))
    count = 2
    while count < limit and (count * tile - dim) / (count - 1) < min_overlap:
        count += 1
    return count


def spatial_tile_count(stage_width: int, stage_height: int, tiling: IcLoraTiling | None) -> int:
    """Number of sequential spatial tiles per step on a stage canvas."""
    if tiling is None:
        return 1
    tw, th = tiling.tile_size(stage_width, stage_height)
    return _axis_tile_count(stage_width, tw, tiling.overlap_fraction) * _axis_tile_count(
        stage_height, th, tiling.overlap_fraction
    )


def _ic_lora_ref_multiplier(ic_lora_id: str | None) -> float:
    return 1.25 if ic_lora_id in _HALF_RES_REF_IC_LORAS else 2.0


def _stage_time_parts(
    *,
    system: EtaSystem,
    load_mode: EtaLoadMode,
    offload_mode: EtaOffloadMode,
    window_tokens: Sequence[tuple[float, float, float]],
    src_tokens: float,
    stage_2: bool,
    tiles: int,
    conv_vae: bool = False,
) -> tuple[float, float, float, float, float]:
    """Return ``(fixed_load_s, gpu_src_s, fixed_denoise_s, gpu_denoise_s, gpu_decode_s)`` at factor 1.0.

    Each entry in ``window_tokens`` is ``(seq1, seq2, out_tok)`` for one temporal window.
    """
    steps = _STAGE_1_STEPS + (_STAGE_2_STEPS if stage_2 else 0)
    windows = len(window_tokens)
    if system == "Darwin":
        # Streaming starts during denoise, so this load is text encode only.
        # A weight-load constant here makes every short clip late and pins
        # machine_factor to the floor. 1-stage decode is the half-res ConvVAE
        # pass; 2-stage decode walks the full canvas. One rate for both
        # overshoots 270p and misses 540p and above.
        fixed_load = 4.0
        gpu_src = 0.0040 * src_tokens
        fixed_denoise = windows * tiles * (0.6 * steps)
        gpu_denoise = 0.0
        gpu_decode = 0.0
        decode_per_token = 0.0095 if stage_2 else 0.0065
        for seq1, seq2, out_tok in window_tokens:
            win_den = _STAGE_1_STEPS * (0.0018 * seq1 + 3.6e-8 * (seq1**2))
            if stage_2:
                win_den += _STAGE_2_STEPS * (0.0018 * seq2 + 3.6e-8 * (seq2**2))
            gpu_denoise += tiles * win_den
            gpu_decode += decode_per_token * out_tok + 2.5e-8 * (out_tok**2)
        return fixed_load, gpu_src, fixed_denoise, gpu_denoise, gpu_decode

    fixed_load = 30.0 if offload_mode == "DISK" else 36.0
    gpu_src = 0.0004 * src_tokens
    if load_mode == "full_models_loading":
        step_stream_s = 5.0
        c_step_tok = 0.00025
    elif offload_mode == "CPU":
        step_stream_s = 8.0
        c_step_tok = 0.00025
    else:
        step_stream_s = 6.2 * _DEFAULT_DISK_RATIO
        c_step_tok = 0.00010 * _DEFAULT_DISK_RATIO
    fixed_denoise = windows * tiles * (step_stream_s * steps)
    gpu_denoise = 0.0
    gpu_decode = 0.0
    decode_per_token = 0.00145 * (_CONV_VAE_DECODE_RATIO if conv_vae else 1.0)
    for seq1, seq2, out_tok in window_tokens:
        win_den = _STAGE_1_STEPS * (c_step_tok * seq1)
        if stage_2:
            win_den += _STAGE_2_STEPS * (c_step_tok * seq2)
        gpu_denoise += tiles * win_den
        gpu_decode += decode_per_token * out_tok
    return fixed_load, gpu_src, fixed_denoise, gpu_denoise, gpu_decode


_StageParts = tuple[float, float, float, float, float]


def _scales_weight_streaming(system: EtaSystem, load_mode: EtaLoadMode) -> bool:
    """CUDA streaming: per-step weight transfer bounds the step and hides GPU compute under it."""
    return system != "Darwin" and load_mode == "streaming_models_loading"


def _stage_1_denoise_share(
    *,
    system: EtaSystem,
    load_mode: EtaLoadMode,
    parts: _StageParts,
    window_tokens: Sequence[tuple[float, float, float]],
    stage_2: bool,
    tiles: int,
    machine_factor: float,
) -> float | None:
    """Fraction of a window's denoise cost spent in the 8 stage-1 steps (or None if 1-stage)."""
    if not stage_2:
        return None
    _fixed_load, _gpu_src, fixed_denoise, gpu_denoise, _gpu_decode = parts
    steps = _STAGE_1_STEPS + _STAGE_2_STEPS
    fixed_s1_ratio = _STAGE_1_STEPS / steps
    if system == "Darwin":
        gpu_s1 = sum(
            tiles * _STAGE_1_STEPS * (0.0018 * seq1 + 3.6e-8 * (seq1**2))
            for seq1, _seq2, _out in window_tokens
        )
    else:
        gpu_total_tok = sum(_STAGE_1_STEPS * seq1 + _STAGE_2_STEPS * seq2 for seq1, seq2, _ in window_tokens)
        gpu_s1 = (
            gpu_denoise * sum(_STAGE_1_STEPS * seq1 for seq1, _, _ in window_tokens) / gpu_total_tok
            if gpu_total_tok > 0
            else gpu_denoise * fixed_s1_ratio
        )
    if _scales_weight_streaming(system, load_mode):
        s1 = machine_factor * fixed_denoise * fixed_s1_ratio + gpu_s1
        total = machine_factor * fixed_denoise + gpu_denoise
    else:
        s1 = fixed_denoise * fixed_s1_ratio + machine_factor * gpu_s1
        total = fixed_denoise + machine_factor * gpu_denoise
    return round(s1 / total, 4) if total > 0 else fixed_s1_ratio


def _compose_stages(
    system: EtaSystem, load_mode: EtaLoadMode, parts: _StageParts, machine_factor: float
) -> tuple[int, int, int]:
    """Apply ``machine_factor`` to the reference parts and return ``(load_s, denoise_s, decode_s)``.

    CUDA streaming scales the per-step weight streaming and keeps GPU compute and decode at
    the reference. Everything else scales the GPU terms.
    """
    fixed_load, gpu_src, fixed_denoise, gpu_denoise, gpu_decode = parts
    load = fixed_load + machine_factor * gpu_src
    if _scales_weight_streaming(system, load_mode):
        denoise = machine_factor * fixed_denoise + gpu_denoise
        decode = gpu_decode
    else:
        denoise = fixed_denoise + machine_factor * gpu_denoise
        decode = machine_factor * gpu_decode
    return max(1, round(load)), max(1, round(denoise)), max(1, round(decode))


def _implied_factor(
    system: EtaSystem,
    load_mode: EtaLoadMode,
    parts: _StageParts,
    wall_s: float,
    *,
    reject_clamped: bool = False,
) -> float | None:
    """Invert `_compose_stages`: the factor that makes the parts sum to ``wall_s``."""
    fixed_load, gpu_src, fixed_denoise, gpu_denoise, gpu_decode = parts
    if _scales_weight_streaming(system, load_mode):
        fixed_s = fixed_load + gpu_denoise + gpu_decode
        scaled_s = gpu_src + fixed_denoise
    else:
        fixed_s = fixed_load + fixed_denoise
        scaled_s = gpu_src + gpu_denoise + gpu_decode
    if scaled_s <= 1.0:
        return None
    raw = max(1.0, wall_s - fixed_s) / scaled_s
    if reject_clamped and not (_MIN_MACHINE_FACTOR < raw < _MAX_MACHINE_FACTOR):
        return None
    return max(_MIN_MACHINE_FACTOR, min(_MAX_MACHINE_FACTOR, raw))


def estimate_stage_breakdown_s(
    *,
    width: int,
    height: int,
    frames: int,
    load_mode: EtaLoadMode,
    offload_mode: EtaOffloadMode,
    system: EtaSystem = "Windows",
    machine_factor: float | None = 1.0,
    stage_2: bool = True,
    ic_lora: bool = False,
    ic_lora_id: str | None = None,
    stage_2_ic_lora: bool = False,
    source_frames: int = 0,
    chunk_pixel_frames: int | None = None,
    carry_frames: int = CARRY_FRAMES,
    tiling: IcLoraTiling | None = None,
    conv_vae: bool = False,
) -> WallTimeBreakdown | None:
    """Three-part wall-time breakdown (load_s, denoise_s, decode_s) for any local video job."""
    if machine_factor is None or machine_factor <= 0 or width <= 0 or height <= 0:
        return None
    if not 0 < frames <= _MAX_FRAMES:
        return None
    if load_mode == "full_models_loading" and offload_mode != "NONE":
        return None
    if load_mode == "streaming_models_loading" and offload_mode == "NONE":
        return None
    if system == "Darwin" and (load_mode != "streaming_models_loading" or offload_mode != "DISK"):
        return None

    win_list = _window_frames_list(width, height, frames, chunk_pixel_frames, carry_frames)
    s1_w = width // 2 if stage_2 else width
    s1_h = height // 2 if stage_2 else height
    tiles = spatial_tile_count(s1_w, s1_h, tiling)
    pass_s1_w, pass_s1_h = tiling.tile_size(s1_w, s1_h) if tiling is not None else (s1_w, s1_h)
    src_tokens = _latent_tokens(height, width, source_frames) if source_frames > 0 else 0.0
    ref_mult = _ic_lora_ref_multiplier(ic_lora_id)

    window_tokens = [
        (
            ref_mult * _latent_tokens(pass_s1_h, pass_s1_w, wf)
            if ic_lora
            else _latent_tokens(pass_s1_h, pass_s1_w, wf),
            (
                ref_mult * _latent_tokens(height, width, wf)
                if stage_2_ic_lora
                else _latent_tokens(height, width, wf)
            )
            if stage_2
            else 0.0,
            _latent_tokens(height, width, wf),
        )
        for wf in win_list
    ]
    parts = _stage_time_parts(
        system=system,
        load_mode=load_mode,
        offload_mode=offload_mode,
        window_tokens=window_tokens,
        src_tokens=src_tokens,
        stage_2=stage_2,
        tiles=tiles,
        conv_vae=conv_vae,
    )
    load_s, denoise_s, decode_s = _compose_stages(system, load_mode, parts, machine_factor)
    s1_share = _stage_1_denoise_share(
        system=system,
        load_mode=load_mode,
        parts=parts,
        window_tokens=window_tokens,
        stage_2=stage_2,
        tiles=tiles,
        machine_factor=machine_factor,
    )
    return WallTimeBreakdown(
        load_s=load_s, denoise_s=denoise_s, decode_s=decode_s, stage_1_share=s1_share
    )


def implied_machine_factor(
    *,
    wall_s: float,
    width: int,
    height: int,
    frames: int,
    load_mode: EtaLoadMode = "streaming_models_loading",
    offload_mode: EtaOffloadMode = "DISK",
    system: EtaSystem = "Windows",
    stage_2: bool = True,
    ic_lora: bool = False,
    ic_lora_id: str | None = None,
    stage_2_ic_lora: bool = False,
    source_frames: int = 0,
    chunk_pixel_frames: int | None = None,
    carry_frames: int = CARRY_FRAMES,
    tiling: IcLoraTiling | None = None,
    conv_vae: bool = False,
    reject_clamped: bool = False,
) -> float | None:
    """Derive this computer's GPU compute factor from one completed run's wall time."""
    if wall_s <= 0 or width <= 0 or height <= 0 or not 0 < frames <= _MAX_FRAMES:
        return None
    win_list = _window_frames_list(width, height, frames, chunk_pixel_frames, carry_frames)
    s1_w = width // 2 if stage_2 else width
    s1_h = height // 2 if stage_2 else height
    tiles = spatial_tile_count(s1_w, s1_h, tiling)
    pass_s1_w, pass_s1_h = tiling.tile_size(s1_w, s1_h) if tiling is not None else (s1_w, s1_h)
    src_tokens = _latent_tokens(height, width, source_frames) if source_frames > 0 else 0.0
    ref_mult = _ic_lora_ref_multiplier(ic_lora_id)
    window_tokens = [
        (
            ref_mult * _latent_tokens(pass_s1_h, pass_s1_w, wf)
            if ic_lora
            else _latent_tokens(pass_s1_h, pass_s1_w, wf),
            (
                ref_mult * _latent_tokens(height, width, wf)
                if stage_2_ic_lora
                else _latent_tokens(height, width, wf)
            )
            if stage_2
            else 0.0,
            _latent_tokens(height, width, wf),
        )
        for wf in win_list
    ]
    parts = _stage_time_parts(
        system=system,
        load_mode=load_mode,
        offload_mode=offload_mode,
        window_tokens=window_tokens,
        src_tokens=src_tokens,
        stage_2=stage_2,
        tiles=tiles,
        conv_vae=conv_vae,
    )
    return _implied_factor(system, load_mode, parts, wall_s, reject_clamped=reject_clamped)


def _envelope_tokens(
    *,
    feature: str,
    adapter_id: str,
    stage_mode: str,
    seq: float,
    canvas_px: int,
    windows: int,
) -> tuple[list[tuple[float, float, float]], float, bool]:
    ic_lora = feature == "iclora"
    stage_2 = stage_mode != "single_stage" and (feature != "fast" or canvas_px > 640 * 368)
    stage_2_ic_lora = stage_mode == "stage_2_lora"
    ref_mult = _ic_lora_ref_multiplier(adapter_id or None)
    if ic_lora and (stage_mode == "single_stage" or stage_2_ic_lora):
        out_tok = seq / ref_mult
    else:
        out_tok = seq
    s1_tok = out_tok / 4.0 if stage_2 else out_tok
    seq1 = ref_mult * s1_tok if ic_lora else s1_tok
    seq2 = (ref_mult * out_tok if stage_2_ic_lora else out_tok) if stage_2 else 0.0
    src_tokens = out_tok * windows if feature in ("edit", "iclora") else 0.0
    return [(seq1, seq2, out_tok)] * windows, src_tokens, stage_2


def estimate_job_shape_seconds(
    shape: JobShape,
    load_mode: EtaLoadMode,
    *,
    system: EtaSystem = "Windows",
    machine_factor: float | None = 1.0,
    offload_mode: EtaOffloadMode | None = None,
    conv_vae: bool = False,
) -> int | None:
    """Whole-run estimated seconds from a priced ``JobShape``."""
    if machine_factor is None or machine_factor <= 0 or shape.seq <= 0 or shape.canvas_px <= 0 or not 1 <= shape.windows <= _MAX_WINDOWS:
        return None
    resolved_offload: EtaOffloadMode = (
        offload_mode
        if offload_mode is not None
        else ("NONE" if load_mode == "full_models_loading" else "DISK")
    )
    window_tokens, src_tokens, stage_2 = _envelope_tokens(
        feature=shape.feature,
        adapter_id=shape.adapter_id,
        stage_mode=shape.stage_mode,
        seq=shape.seq,
        canvas_px=shape.canvas_px,
        windows=shape.windows,
    )
    tiles = max(1, shape.tiles) if shape.tiled else 1
    parts = _stage_time_parts(
        system=system,
        load_mode=load_mode,
        offload_mode=resolved_offload,
        window_tokens=window_tokens,
        src_tokens=src_tokens,
        stage_2=stage_2,
        tiles=tiles,
        conv_vae=conv_vae,
    )
    return sum(_compose_stages(system, load_mode, parts, machine_factor))


def implied_envelope_factor(
    system: EtaSystem,
    bucket: CalibrationBucket,
    point: EnvelopePoint,
    wall_s: float | None,
    *,
    offload_mode: EtaOffloadMode | None = None,
    conv_vae: bool = False,
    reject_clamped: bool = False,
) -> float | None:
    """Derive ``machine_factor`` from a recorded calibration frontier point."""
    if wall_s is None or wall_s <= 0 or point.seq <= 0 or point.canvas_px <= 0 or not 1 <= point.windows <= _MAX_WINDOWS:
        return None
    resolved_offload: EtaOffloadMode = (
        offload_mode
        if offload_mode is not None
        else ("NONE" if bucket.load_mode == "full_models_loading" else "DISK")
    )
    window_tokens, src_tokens, stage_2 = _envelope_tokens(
        feature=bucket.feature,
        adapter_id=bucket.adapter_id,
        stage_mode=bucket.stage_mode,
        seq=point.seq,
        canvas_px=point.canvas_px,
        windows=point.windows,
    )
    tiles = max(1, point.tiles) if bucket.tiled else 1
    parts = _stage_time_parts(
        system=system,
        load_mode=bucket.load_mode,
        offload_mode=resolved_offload,
        window_tokens=window_tokens,
        src_tokens=src_tokens,
        stage_2=stage_2,
        tiles=tiles,
        conv_vae=conv_vae,
    )
    return _implied_factor(system, bucket.load_mode, parts, wall_s, reject_clamped=reject_clamped)


def estimate_envelope_wall_time(
    envelope: JobEnvelope,
    *,
    system: EtaSystem = "Windows",
    machine_factor: float | None = 1.0,
    offload_mode: EtaOffloadMode | None = None,
    conv_vae: bool = False,
) -> WallTimeEstimate | None:
    """Three-part wall-time estimate from a priced ``JobEnvelope``."""
    if machine_factor is None or machine_factor <= 0:
        return None
    bucket = envelope.bucket
    point = envelope.point
    if point.seq <= 0 or point.canvas_px <= 0 or not 1 <= point.windows <= _MAX_WINDOWS:
        return None
    resolved_offload: EtaOffloadMode = (
        offload_mode
        if offload_mode is not None
        else ("NONE" if bucket.load_mode == "full_models_loading" else "DISK")
    )
    window_tokens, src_tokens, stage_2 = _envelope_tokens(
        feature=bucket.feature,
        adapter_id=bucket.adapter_id,
        stage_mode=bucket.stage_mode,
        seq=point.seq,
        canvas_px=point.canvas_px,
        windows=point.windows,
    )
    tiles = max(1, point.tiles) if bucket.tiled else 1
    parts = _stage_time_parts(
        system=system,
        load_mode=bucket.load_mode,
        offload_mode=resolved_offload,
        window_tokens=window_tokens,
        src_tokens=src_tokens,
        stage_2=stage_2,
        tiles=tiles,
        conv_vae=conv_vae,
    )
    load_s, denoise_s, decode_s = _compose_stages(system, bucket.load_mode, parts, machine_factor)
    s1_share = _stage_1_denoise_share(
        system=system,
        load_mode=bucket.load_mode,
        parts=parts,
        window_tokens=window_tokens,
        stage_2=stage_2,
        tiles=tiles,
        machine_factor=machine_factor,
    )
    return WallTimeEstimate(
        seconds=load_s + denoise_s + decode_s,
        basis="scaled",
        t2v_reuse=False,
        load_s=load_s,
        denoise_s=denoise_s,
        decode_s=decode_s,
        stage_1_share=s1_share,
    )


def _denoise_progress_fraction(
    current_step: int, total_steps: int, stage_1_share: float | None = None
) -> float:
    """Cost-weighted denoise progress fraction using the stage-1 cost share from `_stage_time_parts`."""
    per_win = _STAGE_1_STEPS + _STAGE_2_STEPS
    if (
        stage_1_share is not None
        and 0.0 < stage_1_share < 1.0
        and total_steps > 0
        and total_steps % per_win == 0
    ):
        windows = total_steps // per_win
        done_wins = min(windows, current_step // per_win)
        in_win = current_step - done_wins * per_win
        if in_win <= _STAGE_1_STEPS:
            win_frac = stage_1_share * (in_win / _STAGE_1_STEPS)
        else:
            win_frac = stage_1_share + (1.0 - stage_1_share) * (
                (in_win - _STAGE_1_STEPS) / _STAGE_2_STEPS
            )
        return min(1.0, (done_wins + win_frac) / windows)
    return current_step / total_steps


def remaining_wall_time_s(
    estimated: WallTimeEstimate | None,
    *,
    elapsed_s: float,
    current_step: int | None,
    total_steps: int | None,
    decode_elapsed_s: float | None = None,
) -> int | None:
    """Live remaining seconds from the 3-part estimate, step progress, and elapsed time."""
    if estimated is None or estimated.seconds <= 0:
        return None
    load_s = estimated.load_s if estimated.load_s > 0 else max(1, round(estimated.seconds * 0.2))
    denoise_s = (
        estimated.denoise_s if estimated.denoise_s > 0 else max(1, round(estimated.seconds * 0.6))
    )
    decode_s = (
        estimated.decode_s
        if estimated.decode_s > 0
        else max(1, estimated.seconds - load_s - denoise_s)
    )
    elapsed = max(0.0, elapsed_s)
    if current_step is None or total_steps is None or total_steps <= 1 or current_step <= 0:
        return max(0, round(estimated.seconds - elapsed))
    if current_step >= total_steps:
        dec_elapsed = (
            max(0.0, decode_elapsed_s)
            if decode_elapsed_s is not None
            else max(0.0, elapsed - load_s - denoise_s)
        )
        return max(0, round(decode_s - dec_elapsed))
    frac = _denoise_progress_fraction(current_step, total_steps, estimated.stage_1_share)
    prior_rem_denoise = denoise_s * (1.0 - frac)
    obs_denoise = max(1.0, elapsed - load_s)
    paced_rem_denoise = obs_denoise * (1.0 - frac) / frac
    rem_denoise = (1.0 - frac) * prior_rem_denoise + frac * paced_rem_denoise
    return max(1, round(rem_denoise + decode_s))


def median_machine_factor(factors: Sequence[float]) -> float | None:
    """Cautious upper-quartile (75th percentile) of observed machine factors, or None when empty."""
    valid = sorted(f for f in factors if _MIN_MACHINE_FACTOR <= f <= _MAX_MACHINE_FACTOR)
    if not valid:
        return None
    if len(valid) < 4:
        return round(float(statistics.median(valid)), 3)
    idx = (3 * (len(valid) - 1)) // 4
    return round(float(valid[idx]), 3)


def _scale_breakdown(breakdown: WallTimeBreakdown, target_seconds: int) -> WallTimeBreakdown:
    """Scale denoise and decode so the 3 parts sum to ``target_seconds`` while keeping ``load_s``."""
    if breakdown.seconds == target_seconds:
        return breakdown
    load_s = min(breakdown.load_s, max(1, target_seconds - 2))
    rem = max(2, target_seconds - load_s)
    gpu_total = max(1, breakdown.denoise_s + breakdown.decode_s)
    denoise_s = max(1, round(rem * breakdown.denoise_s / gpu_total))
    decode_s = max(1, rem - denoise_s)
    return WallTimeBreakdown(
        load_s=load_s,
        denoise_s=denoise_s,
        decode_s=decode_s,
        stage_1_share=breakdown.stage_1_share,
    )


def estimate_fast_wall_time_s(
    *,
    width: int,
    height: int,
    frames: int,
    load_mode: EtaLoadMode,
    offload_mode: EtaOffloadMode,
    feature: EtaFeature,
    machine_factor: float | None,
    system: EtaSystem = "Windows",
    chunk_pixel_frames: int | None = None,
    stage_2: bool = True,
    conv_vae: bool = False,
) -> WallTimeEstimate | None:
    """Cold whole-run seconds and 3-part breakdown for a Fast cell, or None when unknown."""
    breakdown = estimate_stage_breakdown_s(
        width=width,
        height=height,
        frames=frames,
        load_mode=load_mode,
        offload_mode=offload_mode,
        system=system,
        machine_factor=machine_factor,
        stage_2=stage_2,
        chunk_pixel_frames=chunk_pixel_frames,
        conv_vae=conv_vae,
    )
    if breakdown is None or machine_factor is None:
        return None

    t2v_reuse = feature != "t2v"
    if system == "Darwin":
        mac_ref = _MAC_TABLE.get((width, height, frames)) if stage_2 and chunk_pixel_frames is None else None
        if mac_ref is not None:
            fixed_s = breakdown.load_s + 16 * len(_window_frames_list(width, height, frames, None))
            gpu_ref = max(1, mac_ref - fixed_s)
            seconds = round(fixed_s + gpu_ref * machine_factor)
            basis: EtaBasis = "measured" if machine_factor == 1.0 else "scaled"
            scaled = _scale_breakdown(breakdown, seconds)
            return WallTimeEstimate(
                seconds=seconds,
                basis=basis,
                t2v_reuse=t2v_reuse,
                load_s=scaled.load_s,
                denoise_s=scaled.denoise_s,
                decode_s=scaled.decode_s,
                stage_1_share=scaled.stage_1_share,
            )
        return WallTimeEstimate(
            seconds=breakdown.seconds,
            basis="scaled",
            t2v_reuse=t2v_reuse,
            load_s=breakdown.load_s,
            denoise_s=breakdown.denoise_s,
            decode_s=breakdown.decode_s,
            stage_1_share=breakdown.stage_1_share,
        )

    # Table cells are reference-machine wall times, so they apply at factor 1.0 only. A
    # calibrated factor is relative to the stage law, and scaling a table cell by it
    # double-counts wherever the cell and the law disagree at the reference (540p 5s DISK:
    # table 169 s, law 121 s).
    cell = (
        _TABLE.get((width, height, frames))
        if stage_2 and chunk_pixel_frames is None and machine_factor == 1.0
        else None
    )
    if cell is not None:
        if load_mode == "full_models_loading":
            if cell.full_cold_s is None:
                return None
            base_s = cell.full_cold_s
            basis = "measured"
        elif offload_mode == "CPU":
            base_s = cell.stream_cpu_cold_s
            basis = "measured"
        else:
            base_s = round(cell.stream_cpu_cold_s * cell.disk_ratio)
            basis = "measured" if cell.disk_ratio_measured else "scaled"
        if conv_vae:
            # Cells were measured with the CUDA default DiffVAE decode. Swap in ConvVAE decode.
            diff_vae = estimate_stage_breakdown_s(
                width=width,
                height=height,
                frames=frames,
                load_mode=load_mode,
                offload_mode=offload_mode,
                system=system,
                machine_factor=machine_factor,
                stage_2=stage_2,
            )
            if diff_vae is not None:
                base_s = max(1, base_s - (diff_vae.decode_s - breakdown.decode_s))
                basis = "scaled"
        scaled = _scale_breakdown(breakdown, base_s)
        return WallTimeEstimate(
            seconds=base_s,
            basis=basis,
            t2v_reuse=t2v_reuse,
            load_s=scaled.load_s,
            denoise_s=scaled.denoise_s,
            decode_s=scaled.decode_s,
            stage_1_share=scaled.stage_1_share,
        )

    if load_mode == "full_models_loading" and is_chunked_canvas(width, height):
        return None
    return WallTimeEstimate(
        seconds=breakdown.seconds,
        basis="scaled",
        t2v_reuse=t2v_reuse,
        load_s=breakdown.load_s,
        denoise_s=breakdown.denoise_s,
        decode_s=breakdown.decode_s,
        stage_1_share=breakdown.stage_1_share,
    )

