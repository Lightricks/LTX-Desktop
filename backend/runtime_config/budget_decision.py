"""Priced verdicts for one local job: load it (and how) or refuse it.

Leaf module shared by the pricing layer (``video_job_budget``,
``ic_lora_job_budget``) and the admission layer (``budget_admission``), so the
admission policy can build and downgrade verdicts without importing pricing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from runtime_config.reject_reasons import UnmeasuredBasis, VideoJobRejectReason

LoadMode = Literal["full_models_loading", "streaming_models_loading"]


@dataclass(frozen=True, slots=True)
class VideoJobLoad:
    mode: LoadMode
    seq: float
    full_gib: float
    stream_gib: float
    # Window length in pixel frames when the job runs in windows, else None (one pass).
    # ``seq`` and the GiB figures then describe one window. The pipeline runs this value.
    chunk_pixel_frames: int | None = None


@dataclass(frozen=True, slots=True)
class VideoJobReject:
    seq: float
    full_gib: float
    stream_gib: float
    reason: VideoJobRejectReason = "stream_over"
    # RAM floor the gate compared, set only on ram_shortfall (its sole
    # producer is _streaming_ram_reject, which holds the figure). Message
    # builders read this — never re-derive the floor from the profile, or a
    # wrong property prints a number the gate did not compare.
    required_ram_bytes: int | None = None
    # Set by IC-LoRA and by a Fast canvas that runs in windows, to the window the
    # pricing tried. A downgraded over-budget job must run that window: ``None`` is one
    # unchunked pass, which costs more than the over-budget figure the user
    # confirmed. Edit leaves it None.
    chunk_pixel_frames: int | None = None
    # Which lab measurement is missing, set only on ``unmeasured_cell`` (produced
    # by admission, never by pricing).
    unmeasured_basis: UnmeasuredBasis | None = None


VideoJobDecision = VideoJobLoad | VideoJobReject


@dataclass(frozen=True, slots=True)
class IcLoraJobLoad:
    mode: LoadMode
    seq: float
    full_gib: float
    stream_gib: float
    # Window length when the job runs chunked, else None. The pipeline runs this value.
    chunk_pixel_frames: int | None = None


@dataclass(frozen=True, slots=True)
class IcLoraJobReject:
    reason: Literal["spatial", "unsupported", "refine_decode_floor", "window_over_cap"]
    seq: float
    full_gib: float
    stream_gib: float
    # Uniform with VideoJobReject; always None (these verdicts never compare
    # RAM — spatial is a canvas fence, unsupported short-circuits first, and
    # the decode floor refuses before any curve runs).
    required_ram_bytes: int | None = None
    # Window the pricing tried; None is one unchunked pass.
    chunk_pixel_frames: int | None = None


IcLoraJobDecision = IcLoraJobLoad | IcLoraJobReject | VideoJobReject
