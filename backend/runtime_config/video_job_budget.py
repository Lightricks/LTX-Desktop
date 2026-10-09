"""Token/VRAM estimator for local video jobs.

Tool-agnostic: Fast t2v/i2v/a2v sizes advertised stage-2 H×W (not half-res
stage-1). IC-LoRA wraps this with a ×2 reference sequence and a measured
spatial No.

Calibrated from RTX 5090 32 GB Sage-on logs (desktop sessions 379ffcbf /
41952c05 / 715d94b):

- Full 5s 1024×576×121: seq ≈ 18432 (IC-LoRA ×2), peak alloc ≈ 22.5 GiB.
- Fast 720p 20s stage-2 1280×704×481: seq ≈ 53680. Full reserved 32.99 /
  peak 30.76 / free=0 (paged). Stream reserved 19.54 / peak 14.15.
- Fast 1080p 10s stage-2 seq ≈ 63240 → full ~33.4 GiB, stream instead.
- 5s 720p IC-LoRA hung in tiled VAE encode in both modes → spatial No.

Linear ``estimated_giB`` is intentionally a bit high so we 422 / stream early,
not late. Hang on a borderline mis-judge is acceptable.

Stream was recalibrated to sit above measured reserved (~19.5 GiB at Fast
720p/20s) so 15–16 GB CUDA 422s that cell instead of paging. 540p/5s stream
still fits the 15 GB floor. Full-mode comparison keeps 2 GiB of total VRAM as
headroom so a card that reports 32 does not full-load 720p/20s (reserved ~33).
IC-LoRA uses 7 GiB of full-mode headroom so 8s/10s 540p streams on 32 GB;
5s stays full.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from frame_math import compute_num_frames
from runtime_config.runtime_policy import LocalGenerationMode

LoadMode = Literal["full_models_loading", "streaming_models_loading"]

VIDEO_JOB_TOO_LARGE = "VIDEO_JOB_TOO_LARGE"
VIDEO_JOB_TOO_LARGE_MESSAGE = (
    "This job is larger than this machine can run locally right now. "
    "Lower the resolution or duration."
)
# Retake pads context back up to the encode cap, so a shorter selection often
# leaves the same window. Extend on a long source does the same. Resolution is
# the change that shrinks the job.
EDIT_JOB_TOO_LARGE_MESSAGE = (
    "This edit is larger than this machine can run locally right now. "
    "Lower the resolution."
)
LOCAL_GENERATION_UNSUPPORTED = "LOCAL_GENERATION_UNSUPPORTED"
LOCAL_GENERATION_UNSUPPORTED_MESSAGE = "This machine cannot run local generation."

# Full 5s 1024×576: intercept 18 + 0.000244 × 18432 ≈ 22.5 GiB.
_FULL_INTERCEPT_GIB = 18.0
_FULL_GIB_PER_TOKEN = 0.000244
# Stream: 540p/5s seq 9216 ≈ 6.0 GiB (15 GB floor still fits); 720p/20s seq 53680 ≈ 20.0 GiB
# (above measured reserved ~19.5, so 16 GB 422s).
_STREAM_INTERCEPT_GIB = 3.1
_STREAM_GIB_PER_TOKEN = 0.000315
# Full 720p/20s estimate ~31.10 GiB. A GPU that reports 32 would otherwise pick full
# and page (reserved ~32.99). Keep this off the full-mode comparison only.
_FULL_VRAM_HEADROOM_GIB = 2.0
# Retake/extend 720p on a 5090 (1280×704×337, full estimate 27.2 GiB) reserved
# 31.91 GiB and paged: free=0, ~130s/step, while the same card streamed 1080p at
# ~23s/step. 5 GiB so that clip streams on a card that reports 31 or 32. Fast
# video keeps the 2 GiB headroom. Do not drop this to 4: a 32-report would
# full-load the clip again (27.2 fits in 32−4).
_EDIT_FULL_HEADROOM_GIB = 5.0
# Fast 1080p/10s stream ≈ 23.02 GiB vs a 24 GB report: ~4% on a three-session fit.
# Do not round the stream intercept/slope without re-logging 16 GB and 24 GB cards.


def video_tokens(height: int, width: int, frames: int) -> float:
    """LTX-2 latent tokens: 8-frame temporal, 32 px spatial."""
    temporal = (max(1, frames) - 1) / 8.0 + 1.0
    return temporal * (height / 32.0) * (width / 32.0)


def estimated_giB(mode: LoadMode, seq: float) -> float:
    if mode == "full_models_loading":
        return _FULL_INTERCEPT_GIB + _FULL_GIB_PER_TOKEN * seq
    return _STREAM_INTERCEPT_GIB + _STREAM_GIB_PER_TOKEN * seq


def fits(seq: float, mode: LoadMode, memory_gb: float) -> bool:
    return estimated_giB(mode, seq) <= memory_gb


def _full_memory_gb(memory_gb: float, headroom_gib: float = _FULL_VRAM_HEADROOM_GIB) -> float:
    return max(0.0, memory_gb - headroom_gib)


def memory_gb_for_job(
    *,
    vram_gb: float | None,
    available_ram_gb: float | None,
    darwin: bool,
) -> float | None:
    """Discrete VRAM on CUDA; free RAM on Darwin (already not total)."""
    if darwin:
        return available_ram_gb
    return vram_gb


@dataclass(frozen=True, slots=True)
class OfferingCell:
    """One advertised res×duration cell (t2v/i2v grid, or any similar table)."""

    resolution: str
    duration_seconds: int
    fps: int
    width: int
    height: int


def enumerate_envelope(
    grid: Sequence[OfferingCell],
    mode: LoadMode,
    memory_gb: float,
    *,
    measured_no: Callable[[OfferingCell], bool] | None = None,
) -> list[OfferingCell]:
    """Keep cells that fit ``mode`` at ``memory_gb``. Measured No still wins."""

    kept: list[OfferingCell] = []
    for cell in grid:
        if measured_no is not None and measured_no(cell):
            continue
        frames = _frames_for_duration(cell.duration_seconds, cell.fps)
        seq = video_tokens(cell.height, cell.width, frames)
        if fits(seq, mode, memory_gb):
            kept.append(cell)
    return kept


def _frames_for_duration(duration_seconds: int, fps: int) -> int:
    return compute_num_frames(duration_seconds, fps)


@dataclass(frozen=True, slots=True)
class VideoJobLoad:
    mode: LoadMode
    seq: float
    full_gib: float
    stream_gib: float


@dataclass(frozen=True, slots=True)
class VideoJobReject:
    seq: float
    full_gib: float
    stream_gib: float
    reason: Literal["stream_over", "unsupported"] = "stream_over"


VideoJobDecision = VideoJobLoad | VideoJobReject


def decide_load_mode(
    seq: float,
    *,
    memory_gb: float | None,
    process_mode: LocalGenerationMode,
    darwin: bool = False,
    full_headroom_gib: float = _FULL_VRAM_HEADROOM_GIB,
) -> VideoJobDecision:
    """422 definite-over jobs; per-job stream when full will not fit.

    Does not change process ``local_generations_mode``. ``full_headroom_gib`` is
    subtracted only from the full-vs-stream pick, not from the stream-over 422.

    Two Darwin policies (do not fold them together later):

    - Never full: MPS full-resident is only ~8% faster at ~3.5x the shared
      unified-memory pool, and cannot serve smaller Macs at all
      (see ``runtime_policy``). The green/gray corruption that first ruled
      full out is fixed by ``ltx_core.devices.allow_async_transfer``.
      Independent of the token curve.
    - Never stream-over 422: the 5090 reserved curve is not unified memory, and
      ``available_ram_gb`` is already free-at-launch. Comparing them 422'd M4
      48 GB jobs that streamed. There is no Darwin-native token ceiling yet.
    """
    full_gib = estimated_giB("full_models_loading", seq)
    stream_gib = estimated_giB("streaming_models_loading", seq)

    if process_mode == "unsupported":
        return VideoJobReject(
            seq=seq, full_gib=full_gib, stream_gib=stream_gib, reason="unsupported"
        )

    if darwin:
        return VideoJobLoad(
            mode="streaming_models_loading",
            seq=seq,
            full_gib=full_gib,
            stream_gib=stream_gib,
        )

    if memory_gb is None:
        return VideoJobLoad(
            mode="streaming_models_loading",
            seq=seq,
            full_gib=full_gib,
            stream_gib=stream_gib,
        )

    if not fits(seq, "streaming_models_loading", memory_gb):
        return VideoJobReject(
            seq=seq, full_gib=full_gib, stream_gib=stream_gib, reason="stream_over"
        )

    if process_mode != "full_models_loading" or not fits(
        seq, "full_models_loading", _full_memory_gb(memory_gb, full_headroom_gib)
    ):
        return VideoJobLoad(
            mode="streaming_models_loading",
            seq=seq,
            full_gib=full_gib,
            stream_gib=stream_gib,
        )
    return VideoJobLoad(
        mode="full_models_loading", seq=seq, full_gib=full_gib, stream_gib=stream_gib
    )


def decide_video_job(
    width: int,
    height: int,
    frames: int,
    *,
    memory_gb: float | None,
    process_mode: LocalGenerationMode,
    darwin: bool = False,
) -> VideoJobDecision:
    """Fast t2v/i2v/a2v: advertised (stage-2) pixels, no ×2, no spatial cap."""
    return decide_load_mode(
        video_tokens(height, width, frames),
        memory_gb=memory_gb,
        process_mode=process_mode,
        darwin=darwin,
    )


def decide_edit_job(
    width: int,
    height: int,
    frames: int,
    *,
    memory_gb: float | None,
    process_mode: LocalGenerationMode,
    darwin: bool = False,
) -> VideoJobDecision:
    """Retake/extend. Same curve as Fast, with more full-mode headroom.

    A 5090 that full-loaded 1280×704×337 reserved the whole card. Short 720p
    still fits and stays full.
    """
    return decide_load_mode(
        video_tokens(height, width, frames),
        memory_gb=memory_gb,
        process_mode=process_mode,
        darwin=darwin,
        full_headroom_gib=_EDIT_FULL_HEADROOM_GIB,
    )


def max_edit_frames_that_load(
    width: int,
    height: int,
    *,
    frame_cap: int,
    memory_gb: float | None,
    process_mode: LocalGenerationMode,
    darwin: bool = False,
) -> int:
    """Largest frame count at or below ``frame_cap`` that ``decide_edit_job`` loads.

    1920×1088×505 streams at about 43 GiB and 422s on 31 GB. That canvas is the
    letterbox of a 1080p picture. Home passes this cap into the encode window
    so the job runs at the longest window that fits. When nothing fits, return
    ``frame_cap`` and let the loader 422.
    """
    if frame_cap < 1:
        return frame_cap

    def loads(frames: int) -> bool:
        decision = decide_edit_job(
            width,
            height,
            frames,
            memory_gb=memory_gb,
            process_mode=process_mode,
            darwin=darwin,
        )
        return isinstance(decision, VideoJobLoad)

    if loads(frame_cap):
        return frame_cap
    lo = 1
    hi = frame_cap - 1
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if loads(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best if best > 0 else frame_cap


def advertised_fast_durations(
    width: int,
    height: int,
    fps: int,
    durations: Sequence[int],
    *,
    memory_gb: float | None,
    process_mode: LocalGenerationMode,
    darwin: bool = False,
) -> list[int]:
    """Keep Fast cells that ``decide_video_job`` would load. Not a full-mode envelope.

    Filtering with ``fits(..., "full_models_loading")`` would hide 720p/20s on a
    5090 that streams it. Darwin never compares free RAM to the CUDA stream curve.
    """
    if process_mode == "unsupported":
        return []
    kept: list[int] = []
    for duration in durations:
        decision = decide_video_job(
            width,
            height,
            compute_num_frames(duration, fps),
            memory_gb=memory_gb,
            process_mode=process_mode,
            darwin=darwin,
        )
        if isinstance(decision, VideoJobLoad):
            kept.append(duration)
    return kept
