"""Token/VRAM estimator for local video jobs.

Tool-agnostic: Fast t2v/i2v/a2v sizes advertised stage-2 H×W (not half-res
stage-1). IC-LoRA wraps this with a ×2 reference sequence and a measured
spatial No.

Linear ``estimated_giB`` is intentionally a bit high so we 422 / stream early,
not late. Hang on a borderline mis-judge is acceptable.

Windows Fast (LTX 2.3 and 2.5; both ids are ``is_windows_fast``) gates on
max() of two continuous upper bounds: the allocated OOM law (capped runs can
only fit allocated — ``cuda_max_reserved`` expands to fill a virtual cap
while ``cuda_max_allocated`` is invariant) and the reserved paging law
(fitted to native reserved — on physical cards the reverse holds and
allocated-fitting cells page). Full vs stream selection still uses the
generic full curve with 2 GiB headroom.
Attention backend is deliberately not an availability input (measured no
difference). Do not gate on it without new data.

Mac (Darwin) LTX 2.5 Fast gates on the same continuous shape with Mac-native
coefficients. Compared against the sampled MPS pool, never full resident, no
RAM gate (rationales at decide_mac_job). Darwin 2.3 rides the same stream on
Windows-measured family equivalence (see is_mac_fast); unknown model ids fail
closed in is_mac_fast's assert-never until priced. Darwin additionally fences
single-forward sequences past the MPS kernel's element limit, a hard refusal
independent of the pool (see _MPS_TRANSFORMER_TOKEN_FENCE).
"""
# Darwin 2.3 routing evidence (for the module contract above): the Windows
# pair method plus the archived parity re-proofs.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final, Literal, assert_never

from api_types import LTXLocalModelId
from frame_math import compute_num_frames
from runtime_config.budget_admission import (
    Admission,
    Admitted,
    PricedJob,
    Refused,
    RiskAllowance,
    admit,
    priced_job,
    unmeasured_refusal_message,
)
from runtime_config.budget_decision import (
    LoadMode,
    VideoJobDecision,
    VideoJobLoad,
    VideoJobReject,
)
from runtime_config.budget_evidence import (
    Extrapolated,
    Fitted,
    GenericCurve,
    JobShape,
    PriceEvidence,
)
from runtime_config.device_calibration import (
    EMPTY_CALIBRATION,
    DeviceCalibration,
    envelope_for,
)
from runtime_config.fast_chunking import (
    MAX_WINDOW_TOKENS,
    PEAK_PLATEAU_WINDOWS,
    is_chunked_canvas,
    job_window_count,
    last_window_frames,
    window_candidates,
    window_count,
)
from runtime_config.ic_lora_decode_window import DECODE_MAX_UNTILED_WINDOW
from runtime_config.measured_cells import (
    MEASURED_MAX_DECODE_TOKENS,
    BudgetFeature,
    BudgetRoute,
    is_unmeasured_high_res,
    is_unmeasured_model,
    is_unmeasured_windowed_below_1440p,
)
from runtime_config.model_download_specs import ALL_LTX_LOCAL_MODEL_IDS, get_ltx_model_spec
from runtime_config.reject_reasons import (
    BudgetMode,
    PassShape,
    VideoJobRejectReason,
)
from runtime_config.runtime_policy import (
    CUDA_CPU_OFFLOAD_RAM_FLOOR_GB,
    LocalGenerationMode,
    windows_checkpoint_streams_from_disk,
)
from services.storage_info.storage_info import DriveType

if TYPE_CHECKING:
    # Runtime import would cycle (local_runtime imports this module).
    from runtime_config.runtime_config import RuntimeConfig

VIDEO_JOB_TOO_LARGE = "VIDEO_JOB_TOO_LARGE"
VIDEO_JOB_TOO_LARGE_MESSAGE = (
    "This job is larger than this machine can run locally right now. "
    "Lower the resolution, duration, frame rate, or aspect width."
)
LOCAL_GENERATION_UNSUPPORTED: Final = "LOCAL_GENERATION_UNSUPPORTED"
LOCAL_GENERATION_UNSUPPORTED_MESSAGE = "This machine cannot run local generation."
LOCAL_GENERATION_RAM_SHORTFALL = "LOCAL_GENERATION_RAM_SHORTFALL"
LOCAL_GENERATION_RAM_SHORTFALL_MESSAGE = (
    "This machine does not have enough system RAM for local generation."
)
# Fixed copy for the MPS sequence fence: nothing was compared, so no figures.
# Rides VIDEO_JOB_TOO_LARGE like the other fence verdicts.
MPS_SEQUENCE_OVER_MESSAGE = (
    "This job is too long for this Mac's GPU to run all at once. "
    "Lower the resolution, duration, frame rate, or aspect width."
)
# Fallback copy for an untested job the mode does not allow, when no basis rides
# the reject. Admission always sets one; ``unmeasured_refusal_message`` names it.
UNMEASURED_CELL_MESSAGE = (
    "This job hasn't been tested on this machine, so Cautious mode blocks it. "
    "Turn off Cautious mode to try it anyway, or make the job smaller."
)
ONE_PASS_UNAVAILABLE = "ONE_PASS_UNAVAILABLE"
ONE_PASS_UNAVAILABLE_MESSAGE = "1440p and 4K clips longer than one part are always made in parts."
# Total over VideoJobRejectReason: a new reason must add a row here. Codes come
# from this table; messages come from reject_message_and_code (enriched with the
# measured numbers) with these rows as the incoherent/unknown fallback — never
# print a number the budget did not actually compare. All budget raise sites map
# rejects through that builder — no per-handler chains.
VIDEO_JOB_REJECT_ERRORS: dict[VideoJobRejectReason, tuple[str, str]] = {
    "unsupported": (LOCAL_GENERATION_UNSUPPORTED_MESSAGE, LOCAL_GENERATION_UNSUPPORTED),
    "ram_shortfall": (LOCAL_GENERATION_RAM_SHORTFALL_MESSAGE, LOCAL_GENERATION_RAM_SHORTFALL),
    "stream_over": (VIDEO_JOB_TOO_LARGE_MESSAGE, VIDEO_JOB_TOO_LARGE),
    "mps_sequence_over": (MPS_SEQUENCE_OVER_MESSAGE, VIDEO_JOB_TOO_LARGE),
    "unmeasured_cell": (UNMEASURED_CELL_MESSAGE, VIDEO_JOB_TOO_LARGE),
    "one_pass_unavailable": (ONE_PASS_UNAVAILABLE_MESSAGE, ONE_PASS_UNAVAILABLE),
}

# Default shrink-job advice for stream_over total-short and Darwin pool
# messages. Home edit callers pass the resolution-only variant instead —
# queued Home pads context back up to the cap, so a shorter duration does
# not shrink that job, while GenSpace encodes the whole source, so it does.
_FULL_ACTION = "Lower the resolution, duration, frame rate, or aspect width."
# Default free-short advice: closing GPU apps frees VRAM and a retry can
# succeed. Queued Home callers override it — a Home re-pad makes retry
# futile, so the advice is the same resolution-only shrink as total-short.
_FREE_SHORT_ACTION = "Close other heavy apps (like games, browsers, or video editors) and retry."


def reject_message_and_code(
    decision: VideoJobReject,
    job_filter: LocalJobFilter,
    *,
    action: str = _FULL_ACTION,
    free_short_action: str = _FREE_SHORT_ACTION,
) -> tuple[str, str]:
    """Enriched (message, code) for a budget reject, naming the binding constraint.

    Reads the reason and compared figures off ``decision``: its ``stream_gib``
    is the figure the gate actually compared; its ``required_ram_bytes`` is
    the floor, set only on ram_shortfall — never re-derived from a profile
    property. Unsupported short-circuits before any comparison; a reject
    whose budget figures are unknown falls back to the table message rather
    than printing a lie.

    ``stream_over`` splits free-short from total-short because min(total, free)
    gating makes those different user actions: free-short can retry after
    closing GPU apps, total-short must shrink the job. On Darwin the gate is
    the MPS pool (recommended_max, not free memory), so closing apps is not
    the advice there either. ``action`` is the shrink-job advice, chosen
    caller-side; ``free_short_action`` is the free-short advice, also chosen
    caller-side (queued Home overrides it: context re-pads, so retry is
    futile and the job must shrink instead).
    """
    reason = decision.reason
    required_gib = decision.stream_gib
    required_ram_bytes = decision.required_ram_bytes
    legacy, code = VIDEO_JOB_REJECT_ERRORS[reason]
    if reason == "unsupported" or reason == "one_pass_unavailable":
        return (legacy, code)
    if reason == "mps_sequence_over":
        # Fence verdict: the pool figures were never compared, so fixed copy.
        return (legacy, code)
    if reason == "unmeasured_cell":
        basis = decision.unmeasured_basis
        return (unmeasured_refusal_message(basis) if basis is not None else legacy, code)
    if reason == "ram_shortfall":
        detected = job_filter.budget.total_ram_bytes
        if required_ram_bytes is None or detected is None:
            return (legacy, code)
        return (
            "This job needs "
            f"{required_ram_bytes // _GIB} GB of system RAM but this machine has "
            f"{detected // _GIB} GB. Close heavy programs; on desktops, more RAM "
            "is the real fix.",
            code,
        )
    budget = job_filter.budget
    if job_filter.profile.system == "Darwin":
        pool = budget.mps_pool_gib
        if pool is None:
            return (legacy, code)
        return (
            f"This job needs {required_gib:.1f} GB of GPU memory but this Mac allows "
            f"{pool:.1f} GB. {action}",
            code,
        )
    total = budget.usable_vram_gib
    available = budget.available_vram_gib
    if total is None or available is None:
        return (legacy, code)
    if required_gib > total:
        return (
            f"This job needs {required_gib:.1f} GB of GPU memory but this card has "
            f"{total:.1f} GB. {action}",
            code,
        )
    if required_gib > available:
        # available < total here (total-short ruled out above), so available is
        # the free-VRAM figure, not the total fallback.
        return (
            f"This job needs {required_gib:.1f} GB of GPU memory but only "
            f"{available:.1f} GB is free right now (of {total:.1f} GB total). "
            f"{free_short_action}",
            code,
        )
    return (legacy, code)


def vram_snapshot_suffix(
    *,
    allocated_bytes: int | None,
    reserved_bytes: int | None,
    free_bytes: int | None,
) -> str:
    """Gate-time VRAM triple for budget log lines.

    ``allocated``/``reserved`` are live torch probes; ``free`` is the figure
    the gate actually compared (the filter's sample, never re-sampled for
    logging). Reserved minus allocated is dead torch cache: the next triage
    reads reclaimable occupancy off that gap instead of guessing.
    """
    def _gib_or_unknown(value_bytes: int | None) -> str:
        return f"{value_bytes / _GIB:.1f}GiB" if value_bytes is not None else "unknown"

    return (
        f" (vram alloc={_gib_or_unknown(allocated_bytes)}"
        f" reserved={_gib_or_unknown(reserved_bytes)}"
        f" free={_gib_or_unknown(free_bytes)})"
    )

# Legacy full curve: predates the measured rows below. Re-derive rather than trust.
_FULL_INTERCEPT_GIB = 18.0
_FULL_GIB_PER_TOKEN = 0.000244
# Legacy stream curve: small cells fit small cards, large cells 422 instead of
# paging. Same basis as the full curve above. Re-derive rather than trust.
_STREAM_INTERCEPT_GIB = 3.1
_STREAM_GIB_PER_TOKEN = 0.000315
# A GPU that reports 32 would otherwise pick full and page. Keep this off the
# full-mode comparison only.
_FULL_VRAM_HEADROOM_GIB = 2.0
# Edit full-loads page where Fast fits: a 720p retake reserved the whole card
# and paged while the same card streamed 1080p. 5 GiB so that clip streams on a
# card that reports 31 or 32. Do not drop this to 4: a 32-report would
# full-load the clip again.
_EDIT_FULL_HEADROOM_GIB = 5.0
# Do not round the stream intercept/slope without re-logging 16 GB and 24 GB cards.

_GIB = 1024**3
# VRAM floors sit below nominal because cards report low; RAM floors are nominal,
# so the comparison carries the headroom instead. 1 GiB covers the observed
# ~0.5 GiB Windows under-report with margin. Messages still print nominal —
# only the verdict compares with headroom.
_RAM_REPORTING_HEADROOM_BYTES = 1 * _GIB
# Streamed jobs pin tens of GiB of transformer in host RAM. Shared with the
# Linux CPU-offload floor: same working set, same number — only the consequence
# differs (Linux streams from disk below it; Windows declines the cell on the
# CPU path and streams from disk on NVMe).
WINDOWS_FAST_STREAMING_RAM_BYTES = CUDA_CPU_OFFLOAD_RAM_FLOOR_GB * _GIB
# Disk streaming on 1.6.0 reads blocks into four block-sized host buffers plus
# two 64 MiB staging buckets (~5–6 GiB working set on Fast, zero swap at 24 GB).
# Even though positional reads cycle the checkpoint through the OS file cache at
# 32 and 24 GB (4–6x physical NVMe reads), NVMe prefetch holds wall time to
# 1.16–1.30x of native 64 GB. 24 GB is the lowest figure measured.
WINDOWS_FAST_DISK_STREAMING_RAM_GB = 24
# Edit RAM floors: the CPU warm pin exceeds what a 48 GB machine can hold plus
# the OS, so the CPU floor is 64; DISK holds down to 24 GB on 1.6.0 (E3 working
# set ~11 GiB, 1.15x wall time at 24 GB).
WINDOWS_EDIT_CPU_STREAMING_RAM_GB = 64
WINDOWS_EDIT_DISK_STREAMING_RAM_GB = 24
# IC-LoRA RAM floors: reference latents lift the CPU warm pin over Fast's, so
# the CPU floor is 64; DISK holds at 32 GB on 1.6.0 (L5 working set ~9.5–11.4 GiB,
# 1.33–1.44x wall time at 32 GB with flat swap).
WINDOWS_ICLORA_CPU_STREAMING_RAM_GB = 64
WINDOWS_ICLORA_DISK_STREAMING_RAM_GB = 32


@dataclass(frozen=True, slots=True)
class _StreamModel:
    """One floored-linear stream requirement: max(floor, intercept + slope*tokens).

    The floor covers the resident/encode region where the line does not apply;
    the slope folds working set plus the per-token transient.
    """

    floor_gib: float
    intercept_gib: float
    gib_per_token: float

    def required_gib(self, tokens: float) -> float:
        return max(
            self.floor_gib, self.intercept_gib + self.gib_per_token * tokens
        )


# Windows Fast stream requirement: the transient is a per-token allocation, not
# a per-resolution-class constant, so the slope folds it into the allocated fit.
# An upper bound, not a floor. This row predicts OOM crashes; the gate takes
# max() with the reserved row below, which predicts paging.
_WINDOWS_FAST_STREAM = _StreamModel(
    floor_gib=8.00,
    intercept_gib=1.552,
    gib_per_token=2.6481e-4,
)
# Windows Fast RESERVED row: native streamed reserved breaches small cards the
# allocated row offers. Same paging failure as edit. Fitted on native runs
# only — caps force reserved to equal the cap and destroy the signal. Floor is
# the encode-region ceiling; 10.00 would refuse idle 10 GB cards.
# Near-twin of the edit reserved law: same allocator physics, fitted
# independently per envelope — do not merge the rows.
_WINDOWS_FAST_RESERVED_STREAM = _StreamModel(
    floor_gib=9.90,
    intercept_gib=5.00,
    gib_per_token=2.80e-4,
)
# Windows Fast FULL-mode reserved row: the legacy full ask under-calls native
# full-mode reserved on every measured cell — and allocated too (allocated sits
# flat while reserved grows) — so the full pick admitted paging jobs. Fitted on
# native full-mode runs only. The law dominates the legacy ask at every token
# count, so it always binds.
_WINDOWS_FAST_FULL_RESERVED = _StreamModel(
    floor_gib=23.70,
    intercept_gib=20.90,
    gib_per_token=3.10e-4,
)
# Edit (retake/extend) gates on native reserved: an allocated-fitting retake
# paged past physical VRAM. Allocated predicts crashes; reserved predicts paging.
# Only the floor keeps its old basis: small-canvas edit fails where Fast passes
# (fragmentation from resident source latents, not bigger tensors).
# Unlike Fast's max(allocated, reserved) form, this row replaces: reserved
# strictly dominates the allocated line everywhere above the floor, so max()
# would be the same function — one row, less machinery. Revisit if a refit
# un-dominates.
_EDIT_STREAM = _StreamModel(
    floor_gib=8.75,
    intercept_gib=5.00,
    gib_per_token=2.77e-4,
)
# Windows edit VAE decode reserved law: DiffVAE splits the frame axis past 80 frames
# (largest 8k+1 untiled window: DECODE_MAX_UNTILED_WINDOW = 73), so a single temporal
# decode tile processes ``decode_tokens = video_tokens(H, W, min(F, 73))`` tokens.
# Taking ``max(_EDIT_STREAM(tokens), _EDIT_DECODE_STREAM(decode_tokens))`` across both
# phases prices the whole-run reserved peak continuously with no resolution-tier flags:
# the two lines cross at 40,929 decode tokens (~4.2 Mpx at 73 frames), so <=1440p
# (<=36,800 decode tokens) and 4K 81f+ (denoise 89,760t > decode 81,600t) stay on
# _EDIT_STREAM while untiled 4K 49-73f (57,120-81,600 decode tokens) prices on decode.
_EDIT_DECODE_STREAM = _StreamModel(
    floor_gib=8.75,
    intercept_gib=3.15,
    gib_per_token=3.222e-4,
)
# Mac (Darwin) Fast stream requirement, same shape as Windows, Mac-native fit.
# The per-token transient transfers from CUDA; the Mac surcharge is working set
# only. An upper bound, not a floor.
_MAC_FAST_STREAM = _StreamModel(
    floor_gib=9.75,
    intercept_gib=3.56,
    gib_per_token=3.349e-4,
)
# Mac (Darwin) edit stream requirement, Mac-native fit: retake == extend, one
# envelope serves both. Fragmentation transfers from Windows; only
# floor/intercept shift on MPS.
_MAC_EDIT_STREAM = _StreamModel(
    floor_gib=9.27,
    intercept_gib=7.10,
    gib_per_token=2.669e-4,
)
# Public scalar for matrix prose. Single source: the row above.
MAC_FAST_FLOOR_GIB = _MAC_FAST_STREAM.floor_gib
# Nominal MPS pool for compat-matrix Darwin bands: min observed ratio. The ratio
# is SKU-dependent, so the min is the conservative display pick.
# Display-only: the runtime gate probes the real pool per request.
NOMINAL_MPS_POOL_RATIO = 0.78


def video_tokens(height: int, width: int, frames: int) -> float:
    """LTX-2 latent tokens: 8-frame temporal, 32 px spatial."""
    temporal = (max(1, frames) - 1) / 8.0 + 1.0
    return temporal * (height / 32.0) * (width / 32.0)


def decode_tile_tokens(height: int, width: int, frames: int) -> float:
    """Tokens in a single DiffVAE temporal decode tile (frame axis untiled up to 73 frames)."""
    return video_tokens(height, width, min(frames, DECODE_MAX_UNTILED_WINDOW))


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
class HardwareProfile:
    """What machine + model + mode a budget decision is evaluated for.

    Profile facts for budget dispatch. The fused route lives in
    ``budget_route`` (keyed by feature, system, model);
    ``is_windows_fast`` / ``is_mac_fast`` are the Fast model predicates it
    consults (and handlers consult for preflight gating). Nothing downstream
    compares ``system`` / ``model_id`` strings directly.
    """

    system: Literal["Windows", "Linux", "Darwin"]
    model_id: LTXLocalModelId | None
    process_mode: LocalGenerationMode
    # Drive class of the checkpoint volume, probed once at startup. None means
    # unknown (unimplemented platform, probe failure) — never "slow".
    drive_type: DriveType | None = None
    # Native fp8 (compute capability >= 8.9). The measured laws were fit on
    # fp8; without it the budget takes the legacy curve.
    fp8_capable: bool = True
    device_name: str | None = None

    @property
    def darwin(self) -> bool:
        """Whether this profile runs on Darwin (derived from ``system``)."""
        return self.system == "Darwin"

    @property
    def is_windows_fast(self) -> bool:
        """Windows Fast routes through the measured bands whenever a model is set.

        Measured models (2.5 and 2.3) are Fitted; any new model id still uses
        these bands and is tagged Extrapolated(basis="unmeasured_model").
        """
        return self.system == "Windows" and self.model_id is not None

    # Darwin 2.3 routing evidence (for is_mac_fast below): the Windows pair
    # method plus the archived parity re-proofs.
    @property
    def is_mac_fast(self) -> bool:
        """Darwin Fast routes through _MAC_FAST_STREAM whenever a model is set.

        Measured models (2.5 and 2.3) are Fitted; any new model id still prices
        on _MAC_FAST_STREAM and is tagged Extrapolated(basis="unmeasured_model")
        instead of crashing or falling through to the legacy never-reject branch.
        """
        return self.system == "Darwin" and self.model_id is not None

    @property
    def streams_from_disk(self) -> bool:
        """Whether Windows + NVMe can fall back to disk streaming below the CPU floor.

        Reads the shared rule, so the lower NVMe RAM floors below open only
        when the drive can actually take the disk path.
        """
        return windows_checkpoint_streams_from_disk(self.system, self.drive_type)

    @property
    def windows_fast_required_ram_bytes(self) -> int:
        """RAM floor gating this profile's streamed Windows Fast jobs.

        Single source of truth shared by the budget gate and the reported
        ``local_unavailable_required_ram_gb``: both read the floor from the
        profile that produced the verdict. Mirrors the offload routing —
        Windows + NVMe falls back to DISK below the 40 GB CPU floor down to
        this 24 GB floor, while everything else requires the 40 GB CPU floor.
        """
        if self.streams_from_disk:
            return WINDOWS_FAST_DISK_STREAMING_RAM_GB * _GIB
        return WINDOWS_FAST_STREAMING_RAM_BYTES

    @property
    def windows_edit_required_ram_bytes(self) -> int:
        """RAM floor gating this profile's streamed Windows edit jobs.

        See the WINDOWS_EDIT_* constants: Windows + NVMe streams from disk,
        everything else uses CPU offload. Read only by decide_edit_job's Windows
        gate.
        """
        if self.streams_from_disk:
            return WINDOWS_EDIT_DISK_STREAMING_RAM_GB * _GIB
        return WINDOWS_EDIT_CPU_STREAMING_RAM_GB * _GIB

    @property
    def windows_iclora_required_ram_bytes(self) -> int:
        """RAM floor gating this profile's streamed Windows IC-LoRA jobs.

        See the WINDOWS_ICLORA_* constants: Windows + NVMe falls back to disk
        below the 64 GB CPU floor down to 32 GB, while everything else requires
        the 64 GB CPU floor. Read only by decide_ic_lora_job's Windows gate.
        """
        if self.streams_from_disk:
            return WINDOWS_ICLORA_DISK_STREAMING_RAM_GB * _GIB
        return WINDOWS_ICLORA_CPU_STREAMING_RAM_GB * _GIB


def budget_route(feature: BudgetFeature, profile: HardwareProfile) -> BudgetRoute:
    """Fused measured/legacy dispatch keyed by (feature, system, model).

    Single place that selects the stream curve. Route names carry the
    measured flag explicitly: ``mac_measured`` / ``windows_measured`` take
    fitted curves, ``legacy`` takes the generic path. Fast on Darwin and
    Windows consults ``is_mac_fast`` / ``is_windows_fast`` (known models are
    Fitted; any new model id still prices on the curve and is tagged
    ``unmeasured_model`` via ``is_unmeasured_model``); edit and IC-LoRA
    are measured on Windows and Darwin for any model. Process floors
    (``CUDA_VRAM_FLOOR_GB``, ``CUDA_FULL_VRAM_GB``, ``DARWIN_*``) and the
    per-feature RAM floors stay where they are — this route only selects
    the curve, never the floor.
    """
    if feature == "fast":
        if profile.system == "Darwin":
            return "mac_measured" if profile.is_mac_fast else "legacy"
        if profile.system == "Windows":
            if not profile.fp8_capable:
                return "legacy"
            return "windows_measured" if profile.is_windows_fast else "legacy"
        if profile.system == "Linux":
            return "legacy"
        assert_never(profile.system)
    if feature == "edit":
        if profile.system == "Windows":
            return "legacy" if not profile.fp8_capable else "windows_measured"
        if profile.system == "Darwin":
            return "mac_measured"
        if profile.system == "Linux":
            return "legacy"
        assert_never(profile.system)
    if feature == "iclora":
        if profile.system == "Windows":
            return "legacy" if not profile.fp8_capable else "windows_measured"
        if profile.system == "Darwin":
            return "mac_measured"
        if profile.system == "Linux":
            return "legacy"
        assert_never(profile.system)
    assert_never(feature)


def _floored_gib(memory_gb: float | None) -> float | None:
    """Coarse integer-GB figure as GiB, floored to byte granularity."""
    if memory_gb is None:
        return None
    return int(memory_gb * _GIB) / _GIB


def _total_vram_gib(
    total_vram_bytes: int | None, memory_gb: float | None
) -> float | None:
    """Total VRAM figure: exact bytes win, the integer falls back.

    Both-None fails closed (None) — the call site refuses as unsupported.
    """
    if total_vram_bytes is not None:
        return total_vram_bytes / _GIB
    return _floored_gib(memory_gb)


def _available_vram_gib(
    total_gib: float | None, free_vram_bytes: int | None
) -> float | None:
    """Consumable VRAM figure: min(total, free).

    Free-None fails open to total (unmeasured free never shrinks the offer);
    total-None fails closed (None).
    """
    if total_gib is None:
        return None
    if free_vram_bytes is None:
        return total_gib
    return min(total_gib, free_vram_bytes / _GIB)


def _mps_pool_gib(mps_pool_bytes: int | None) -> float | None:
    """MPS allocator pool in GiB, exact bytes.

    Pool-None fails closed (None) with no integer fallback — unlike free
    VRAM, there is no total to fall back to.
    """
    if mps_pool_bytes is None:
        return None
    return mps_pool_bytes / _GIB


@dataclass(frozen=True, slots=True)
class MemoryBudget:
    """VRAM + RAM figures plus their precedence, evaluated once per use.

    Sampling: VRAM total is startup-frozen device bytes; ``usable_vram_bytes`` is
    free VRAM sampled once at backend startup before any model loads; ``free_vram_bytes``
    is sampled fresh per request via ``GpuInfo.get_free_cuda_vram_bytes``; RAM total is the
    startup-frozen machine total via ``psutil.virtual_memory().total``; the
    MPS pool is sampled fresh per request via
    ``GpuInfo.get_mps_pool_bytes``. ``memory_gb`` is the coarse floored
    figure (CUDA integer GB / Darwin free RAM) backing the integer fallback.
    Precedence lives here so no caller re-derives it: the constructors above
    own the typed fail-mode asymmetry (free-None opens to total, pool-None
    closes, both-None closes), and unknown-everything fails closed at the
    call site. ``total_only`` gates CUDA availability on ``usable_vram_gib``
    (``min(total, startup_free)``, "will it ever fit") instead of ``min(total, free)``:
    queued creates and list-time display admit usable-fitting cells while free is
    momentarily depressed, and the execute-time re-check — still free-gated — refuses
    genuine drift. It is not free-None (unmeasured): the sample is still taken, and the
    flag names the policy. Mac pool gating is unchanged.
    """

    total_vram_bytes: int | None
    free_vram_bytes: int | None
    total_ram_bytes: int | None
    memory_gb: float | None
    mps_pool_bytes: int | None
    total_only: bool = False
    usable_vram_bytes: int | None = None

    @property
    def total_vram_gib(self) -> float | None:
        """Exact total VRAM, else the integer fallback. None only if both missing."""
        return _total_vram_gib(self.total_vram_bytes, self.memory_gb)

    @property
    def usable_vram_gib(self) -> float | None:
        """Idle-machine consumable ceiling: min(total, startup_free), or total when unprobed."""
        return _available_vram_gib(self.total_vram_gib, self.usable_vram_bytes)

    @property
    def available_vram_gib(self) -> float | None:
        """What a new job may actually consume: min(total, free), or usable when total_only."""
        if self.total_only:
            return self.usable_vram_gib
        return _available_vram_gib(self.total_vram_gib, self.free_vram_bytes)

    def streaming_ram_state(
        self, *, required_bytes: int
    ) -> Literal["ok", "shortfall", "unknown"]:
        """Tri-state RAM verdict: only known-insufficient claims shortfall."""
        if self.total_ram_bytes is None:
            return "unknown"
        return "ok" if self.total_ram_bytes >= required_bytes - _RAM_REPORTING_HEADROOM_BYTES else "shortfall"

    @property
    def mps_pool_gib(self) -> float | None:
        """MPS allocator pool in GiB, exact bytes. None off-MPS / unprobed."""
        return _mps_pool_gib(self.mps_pool_bytes)


@dataclass(frozen=True, slots=True)
class LocalJobFilter:
    """Profile + budget for list-time Fast specs and request-time job budget.

    Built once per request (one free-VRAM sample) and passed whole; nothing
    re-derives memory precedence or profile dispatch downstream.
    """

    profile: HardwareProfile
    budget: MemoryBudget
    # Live budget mode at build time. Admission reads it (never global state)
    # through RiskAllowance; pricing stays mode-blind.
    budget_mode: BudgetMode = "cautious"
    # What this device has run cleanly. Empty keeps pricing on the lab tables.
    calibration: DeviceCalibration = EMPTY_CALIBRATION
    # How a Fast job is cut: the live "Never split" setting for request-time
    # gates and specs, or the shape a queued job stored at create. Every range
    # and refusal priced from this filter follows it, so a flip never re-cuts or
    # re-sizes a queued job.
    pass_shape: PassShape = "auto"
    # Settings -> Fast decode resolved for this device. Only the wall-time estimate reads it.
    conv_vae: bool = False


def resolve_pass_shape(requested: PassShape, *, never_split: bool) -> PassShape:
    """An explicit shape wins, even over the switch; ``auto`` follows the switch."""
    if requested != "auto":
        return requested
    return "one_pass" if never_split else "auto"


def build_local_job_filter(
    config: RuntimeConfig,
    *,
    model_id: LTXLocalModelId | None,
    free_vram_bytes: int | None,
    mps_pool_bytes: int | None,
    total_only: bool = False,
    budget_mode: BudgetMode = "cautious",
    calibration: DeviceCalibration = EMPTY_CALIBRATION,
    pass_shape: PassShape = "auto",
    conv_vae: bool = False,
) -> LocalJobFilter:
    """One construction idiom for request filters. Callers sample probes first.

    ``None`` means unmeasured. Callers pass ``GpuInfo.get_free_cuda_vram_bytes()``
    and ``GpuInfo.get_mps_pool_bytes()`` sampled once per request; generic-curve
    paths pass None rather than paying for samples they ignore.

    ``total_only`` gates CUDA availability on total VRAM ("will it ever fit")
    instead of min(total, free): queued creates and list-time display admit
    total-fitting cells while free is momentarily depressed, and the
    execute-time re-check — still free-gated — refuses genuine drift with
    JOB_TOO_LARGE. It is not free-None (unmeasured): the sample is still
    taken, and the flag names the policy. GenSpace direct, estimate, and all
    execute checks keep the default free gate — branch at the call site, never
    here. Mac pool gating is unchanged.
    """
    # Unknown systems take the generic path, same as Linux: no measured
    # profile dispatches on them.
    if config.system == "Windows":
        system: Literal["Windows", "Linux", "Darwin"] = "Windows"
    elif config.system == "Darwin":
        system = "Darwin"
    else:
        system = "Linux"
    darwin = system == "Darwin"
    return LocalJobFilter(
        profile=HardwareProfile(
            system=system,
            model_id=model_id,
            process_mode=config.local_generations_mode,
            drive_type=config.checkpoint_drive_type,
            fp8_capable=config.fp8_capable,
            device_name=config.gpu_device_name,
        ),
        budget=MemoryBudget(
            total_vram_bytes=config.vram_bytes,
            free_vram_bytes=free_vram_bytes,
            total_ram_bytes=config.total_ram_bytes,
            memory_gb=memory_gb_for_job(
                vram_gb=config.vram_gb,
                available_ram_gb=config.available_ram_gb,
                darwin=darwin,
            ),
            mps_pool_bytes=mps_pool_bytes,
            total_only=total_only,
            usable_vram_bytes=config.usable_vram_bytes,
        ),
        budget_mode=budget_mode,
        calibration=calibration,
        pass_shape=pass_shape,
        conv_vae=conv_vae,
    )


def decide_from_fits(
    seq: float,
    full_gib: float,
    stream_gib: float,
    *,
    stream_fits: bool,
    full_fits: bool,
    process_mode: LocalGenerationMode,
    legacy_darwin_never_reject: bool,
) -> VideoJobDecision:
    """Single decision engine shared by the generic and Windows Fast paths.

    Callers evaluate their own fits (token curve vs allocated model) and the
    engine applies process viability, the legacy Darwin never-422 policy, and
    the full-vs-stream pick exactly once. Unknown budgets never reach here:
    callers map unknown to stream (generic) or fail closed (Windows) before
    calling. Measured Darwin profiles pass legacy_darwin_never_reject=False:
    the legacy branch is the absence of a ceiling, which they replace.
    """
    if process_mode == "unsupported":
        return VideoJobReject(
            seq=seq, full_gib=full_gib, stream_gib=stream_gib, reason="unsupported"
        )

    if legacy_darwin_never_reject:
        return VideoJobLoad(
            mode="streaming_models_loading",
            seq=seq,
            full_gib=full_gib,
            stream_gib=stream_gib,
        )

    if not stream_fits:
        return VideoJobReject(
            seq=seq, full_gib=full_gib, stream_gib=stream_gib, reason="stream_over"
        )

    if process_mode != "full_models_loading" or not full_fits:
        return VideoJobLoad(
            mode="streaming_models_loading",
            seq=seq,
            full_gib=full_gib,
            stream_gib=stream_gib,
        )
    return VideoJobLoad(
        mode="full_models_loading", seq=seq, full_gib=full_gib, stream_gib=stream_gib
    )


# Full-resident policy evidence (for the never-full rationale in
# decide_load_mode below): interleaved DISK-vs-NONE runs.
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

    - Never full: MPS full-resident is only slightly faster at multiples of the
      shared unified-memory pool, and cannot serve smaller Macs at all
      (see ``runtime_policy``; ``decide_mac_job`` pins full_fits=False).
      The green/gray corruption that first ruled full out is fixed by
      ``ltx_core.devices.allow_async_transfer``.
      Independent of the token curve.
    - Never stream-over 422: the 5090 reserved curve is not unified memory, and
      ``available_ram_gb`` is already free-at-launch. Comparing them 422'd M4
      48 GB jobs that streamed. Fast 2.5/2.3 now has a Darwin-native pool ceiling
      (see ``is_mac_fast``); the legacy curve still never compares.

    Unknown memory streams rather than guessing full.
    """
    full_gib = estimated_giB("full_models_loading", seq)
    stream_gib = estimated_giB("streaming_models_loading", seq)

    if memory_gb is None:
        stream_fits, full_fits = True, False
    else:
        stream_fits = fits(seq, "streaming_models_loading", memory_gb)
        full_fits = fits(seq, "full_models_loading", _full_memory_gb(memory_gb, full_headroom_gib))
    return decide_from_fits(
        seq,
        full_gib,
        stream_gib,
        stream_fits=stream_fits,
        full_fits=full_fits,
        process_mode=process_mode,
        legacy_darwin_never_reject=darwin,
    )


def decide_video_job(
    width: int,
    height: int,
    frames: int,
    *,
    job_filter: LocalJobFilter,
    allowance: RiskAllowance | None = None,
) -> Admission:
    """Fast t2v/i2v/a2v: price the job, then admit it under the allowance.

    ``allowance=None`` follows the filter's live mode (request-time gates);
    executors pass the stored allowance so a mode flip never re-gates a queued job.
    """
    allowance = allowance if allowance is not None else RiskAllowance.for_mode(job_filter.budget_mode)
    return admit(
        price_video_job(width, height, frames, job_filter=job_filter, allowance=allowance),
        allowance,
    )


def price_video_job(
    width: int,
    height: int,
    frames: int,
    *,
    job_filter: LocalJobFilter,
    allowance: RiskAllowance | None = None,
) -> PricedJob:
    """Pick the window and price it in the filter's ``pass_shape``. Under ``auto``,
    ``allowance`` decides whether a one-pass job past the fitted window size runs as one
    pass (it may take an unmeasured bet) or in windows. ``one_pass`` and ``parts`` price
    that shape whatever it costs, so admission turns an over one pass into a Brave over
    bet and a Cautious refusal."""
    allowance = allowance if allowance is not None else RiskAllowance.for_mode(job_filter.budget_mode)
    decision = _decide_priced_video_job(
        width,
        height,
        frames,
        job_filter=job_filter,
        allow_unfitted_pass=allowance.unmeasured,
        shape=job_filter.pass_shape,
    )
    return _priced_canvas_job(
        "fast",
        decision,
        width=width,
        height=height,
        windows=job_window_count(frames, decision.chunk_pixel_frames),
        job_filter=job_filter,
    )


def _priced_canvas_job(
    feature: Literal["fast", "edit"],
    decision: VideoJobDecision,
    *,
    width: int,
    height: int,
    windows: int,
    decode_tokens: float | None = None,
    job_filter: LocalJobFilter,
) -> PricedJob:
    """Tag a Fast or edit verdict: extrapolated in an unmeasured high-res or windowed cell."""
    route = budget_route(feature, job_filter.profile)
    evidence: PriceEvidence
    if is_unmeasured_high_res(
        feature,
        route,
        width,
        height,
        windows=windows,
        decode_tokens=decode_tokens,
    ):
        evidence = Extrapolated(basis="high_res_cell")
    elif is_unmeasured_windowed_below_1440p(
        feature,
        route,
        width,
        height,
        windows=windows,
        window_tokens=decision.seq,
    ):
        evidence = Extrapolated(basis="windowed_range")
    elif feature == "fast" and _is_pass_past_fitted_tokens(decision, windows=windows, route=route):
        evidence = Extrapolated(basis="pass_tokens")
    elif route == "legacy":
        evidence = GenericCurve()
    elif is_unmeasured_model(job_filter.profile.model_id):
        evidence = Extrapolated(basis="unmeasured_model")
    else:
        evidence = Fitted()
    return priced_job(
        feature,
        decision,
        evidence,
        _canvas_job_shape(feature, decision, width=width, height=height, windows=windows, job_filter=job_filter),
        job_filter=job_filter,
    )


def _canvas_job_shape(
    feature: Literal["fast", "edit"],
    decision: VideoJobDecision,
    *,
    width: int,
    height: int,
    windows: int,
    job_filter: LocalJobFilter,
) -> JobShape:
    return JobShape(
        feature=feature,
        model_id=job_filter.profile.model_id or "",
        adapter_id="",
        stage_mode="",
        tiled=False,
        chunked=decision.chunk_pixel_frames is not None,
        seq=decision.seq,
        canvas_px=width * height,
        windows=windows,
    )


def _is_pass_past_fitted_tokens(decision: VideoJobDecision, *, windows: int, route: BudgetRoute) -> bool:
    """A one-pass load larger than any window the laws were fitted on, on a measured route."""
    return (
        isinstance(decision, VideoJobLoad)
        and windows == 1
        and route != "legacy"
        and decision.seq > MAX_WINDOW_TOKENS
    )


def _decide_priced_video_job(
    width: int,
    height: int,
    frames: int,
    *,
    job_filter: LocalJobFilter,
    allow_unfitted_pass: bool,
    shape: PassShape = "auto",
) -> VideoJobDecision:
    """Fast t2v/i2v/a2v: advertised (stage-2) pixels, no ×2, no spatial cap.

    A clip runs in one pass when the machine loads it, and in overlapping windows when it
    does not. A 1440p or 4K clip longer than one window always runs in windows. Any other
    canvas windows once one pass cannot fit the machine at all (``_windows_help``), so a long
    or high-fps clip at 720p or 540p runs in as many windows as it needs, with no cap on the
    count: the budget prices one window, so the window count never grows the ask past the
    plateau. The window is the longest that loads, as ``VideoJobLoad.chunk_pixel_frames``.
    When none loads, the verdict is the reject of the shortest window, so the figures name the
    smallest ask and a Brave downgrade runs that window.

    A one-pass load past the largest fitted window is an unmeasured bet (``pass_tokens``). When the
    allowance takes no unmeasured bet and this device has not run that pass cleanly, the clip
    runs in windows at the ceiling instead.

    ``one_pass`` skips the window fallback; a 1440p or 4K clip longer than one window is
    ``one_pass_unavailable``. ``parts`` skips one pass whenever the clip is longer than the
    shortest window.
    """
    match shape:
        case "one_pass":
            single_pass = _decide_fast_pass(width, height, frames, windows=1, job_filter=job_filter)
            if is_chunked_canvas(width, height) and frames > window_candidates(width, height)[0]:
                return VideoJobReject(
                    seq=single_pass.seq,
                    full_gib=single_pass.full_gib,
                    stream_gib=single_pass.stream_gib,
                    reason="one_pass_unavailable",
                )
            return single_pass
        case "parts":
            windowed = _decide_windowed_video_job(width, height, frames, job_filter=job_filter)
            if windowed is not None:
                return windowed
            return _decide_fast_pass(width, height, frames, windows=1, job_filter=job_filter)
        case "auto":
            pass
        case _ as unreachable:
            assert_never(unreachable)
    if not is_chunked_canvas(width, height):
        single_pass = _decide_fast_pass(width, height, frames, windows=1, job_filter=job_filter)
        if isinstance(single_pass, VideoJobLoad):
            if allow_unfitted_pass or not _is_unproven_pass_past_fit(
                single_pass, width, height, job_filter
            ):
                return single_pass
            windowed = _decide_windowed_video_job(width, height, frames, job_filter=job_filter)
            return windowed if isinstance(windowed, VideoJobLoad) else single_pass
        if not _windows_help(single_pass, job_filter):
            return single_pass
        windowed = _decide_windowed_video_job(width, height, frames, job_filter=job_filter)
        return windowed if windowed is not None else single_pass
    decision: VideoJobDecision | None = None
    if frames <= window_candidates(width, height)[0]:
        decision = _decide_fast_pass(width, height, frames, windows=1, job_filter=job_filter)
        if isinstance(decision, VideoJobLoad):
            return decision
    windowed = _decide_windowed_video_job(width, height, frames, job_filter=job_filter)
    if windowed is not None:
        return windowed
    assert decision is not None  # window_candidates is never empty
    return decision


def _is_unproven_pass_past_fit(
    single_pass: VideoJobLoad, width: int, height: int, job_filter: LocalJobFilter
) -> bool:
    """A one-pass load past the fitted window size that this device has not run cleanly."""
    route = budget_route("fast", job_filter.profile)
    if not _is_pass_past_fitted_tokens(single_pass, windows=1, route=route):
        return False
    shape = _canvas_job_shape("fast", single_pass, width=width, height=height, windows=1, job_filter=job_filter)
    envelope = envelope_for(shape, single_pass.mode, single_pass.stream_gib)
    return job_filter.calibration.covering(envelope) is None


def _windows_help(single_pass: VideoJobDecision, job_filter: LocalJobFilter) -> bool:
    """A one-pass refusal that windows can lift: the pass is past the machine's ceiling.

    ``mps_sequence_over`` is a per-forward fence, so a window under it clears it. A
    ``stream_over`` qualifies only when its ask is over the whole card (the MPS pool on a Mac):
    an ask that fits an empty card is a busy or warm card, which frees up, so it keeps its
    refusal and the evict-and-retry and close-other-apps messages stay in charge. A RAM
    shortfall or an unsupported process is not shortened by a window.
    """
    if isinstance(single_pass, VideoJobLoad):
        return False
    if single_pass.reason == "mps_sequence_over":
        return True
    if single_pass.reason != "stream_over":
        return False
    budget = job_filter.budget
    ceiling = budget.mps_pool_gib if job_filter.profile.darwin else budget.usable_vram_gib
    return ceiling is not None and single_pass.stream_gib > ceiling


def _decide_windowed_video_job(
    width: int,
    height: int,
    frames: int,
    *,
    job_filter: LocalJobFilter,
) -> VideoJobDecision | None:
    """The longest window that loads, as ``chunk_pixel_frames``.

    When none loads, the reject of the shortest window, so a Brave downgrade runs that
    window and the figures name the smallest ask. None when the clip is no longer than the
    shortest window, so there is nothing to chunk.
    """
    decision: VideoJobDecision | None = None
    chunk_pixel_frames: int | None = None
    for window in window_candidates(width, height):
        if frames <= window:
            continue  # One pass already covered every window at least this long.
        chunk_pixel_frames = window
        decision = _decide_fast_pass(
            width,
            height,
            window,
            windows=window_count(frames, window),
            last_frames=last_window_frames(frames, window),
            job_filter=job_filter,
        )
        if isinstance(decision, VideoJobLoad):
            return replace(decision, chunk_pixel_frames=chunk_pixel_frames)
    if decision is None:
        return None
    # A Brave downgrade of this reject runs the shortest window, not one pass.
    return replace(decision, chunk_pixel_frames=chunk_pixel_frames)


def _decide_fast_pass(
    width: int,
    height: int,
    frames: int,
    *,
    windows: int,
    last_frames: int | None = None,
    job_filter: LocalJobFilter,
) -> VideoJobDecision:
    """One pass of ``frames`` frames, inside a job of ``windows`` windows."""
    seq = video_tokens(height, width, frames)
    last_tokens = (
        seq
        if last_frames is None or last_frames == frames
        else video_tokens(height, width, last_frames)
    )
    route = budget_route("fast", job_filter.profile)
    if route == "mac_measured":
        return decide_mac_job(
            seq,
            fast_job_stream_gib(seq, windows=windows, darwin=True, last_tokens=last_tokens),
            job_filter=job_filter,
        )
    if route == "windows_measured":
        return _decide_windows_fast_job(
            seq, windows=windows, last_tokens=last_tokens, job_filter=job_filter
        )
    return decide_load_mode(
        seq,
        memory_gb=job_filter.budget.memory_gb,
        process_mode=job_filter.profile.process_mode,
        darwin=job_filter.profile.darwin,
    )


@dataclass(frozen=True, slots=True)
class FastDurationOutcome:
    """Per-cell admissions in duration order, so the specs layer can message,
    tier, and figure every cell without re-deciding."""

    admitted: dict[int, Admitted] = field(default_factory=dict[int, Admitted])
    refused: dict[int, Refused] = field(default_factory=dict[int, Refused])

    @property
    def kept(self) -> list[int]:
        return list(self.admitted)


def decide_fast_durations(
    width: int,
    height: int,
    fps: int,
    durations: Sequence[int],
    *,
    job_filter: LocalJobFilter,
) -> FastDurationOutcome:
    """Admit every cell under the filter's live mode."""
    outcome = FastDurationOutcome()
    for duration in durations:
        admission = decide_video_job(
            width,
            height,
            compute_num_frames(duration, fps),
            job_filter=job_filter,
        )
        if isinstance(admission, Admitted):
            outcome.admitted[duration] = admission
        else:
            outcome.refused[duration] = admission
    return outcome


def price_alternative_pass(
    width: int,
    height: int,
    frames: int,
    *,
    job_filter: LocalJobFilter,
    auto: PricedJob,
) -> PricedJob | None:
    """The other shape a Brave user may pick for this clip, priced as create would run it.

    None wherever the "Make it" toggle stays hidden: Cautious (it picks for the user), the
    "Never split" switch or any other fixed shape (never means never), 1440p and 4K (always
    windows past one), a one pass inside the fitted range (windows buy nothing there), a clip
    too short to split, and a shape that a wall refuses even in Brave (the Mac sequence fence).
    """
    if job_filter.budget_mode != "brave" or job_filter.pass_shape != "auto" or is_chunked_canvas(width, height):
        return None
    route = budget_route("fast", job_filter.profile)
    if auto.chunk_pixel_frames is None:
        decision = auto.decision
        if not isinstance(decision, VideoJobLoad) or not _is_pass_past_fitted_tokens(
            decision, windows=1, route=route
        ):
            return None
        other: PassShape = "parts"
    else:
        other = "one_pass"
    brave = RiskAllowance.for_mode("brave")
    priced = price_video_job(
        width, height, frames, job_filter=replace(job_filter, pass_shape=other), allowance=brave
    )
    if other == "parts" and priced.chunk_pixel_frames is None:
        return None
    if isinstance(admit(priced, brave), Refused):
        return None
    return priced


def decide_edit_job(
    width: int,
    height: int,
    frames: int,
    *,
    job_filter: LocalJobFilter,
    allowance: RiskAllowance | None = None,
) -> Admission:
    """Retake/extend: price the job, then admit it under the allowance."""
    return admit(
        price_edit_job(width, height, frames, job_filter=job_filter),
        allowance if allowance is not None else RiskAllowance.for_mode(job_filter.budget_mode),
    )


def price_edit_job(
    width: int,
    height: int,
    frames: int,
    *,
    job_filter: LocalJobFilter,
) -> PricedJob:
    return _priced_canvas_job(
        "edit",
        _decide_priced_edit_job(width, height, frames, job_filter=job_filter),
        width=width,
        height=height,
        windows=1,
        decode_tokens=decode_tile_tokens(height, width, frames),
        job_filter=job_filter,
    )


def video_job_reject(refused: Refused) -> VideoJobReject:
    """The reject of a Fast or edit refusal: their pricing never yields an IC-LoRA reject."""
    reject = refused.reject
    assert isinstance(reject, VideoJobReject)
    return reject


def _decide_priced_edit_job(
    width: int,
    height: int,
    frames: int,
    *,
    job_filter: LocalJobFilter,
) -> VideoJobDecision:
    """Retake/extend. Reserved-basis stream law on Windows, with more full-mode headroom.

    The gate prices native cuda_max_reserved (see _EDIT_STREAM): allocated no
    longer gates because an allocated-fitting window paged past physical VRAM.
    Budget the footage processed — extend denoises source frames plus new ones,
    and one token function serves retake and extend (the retake/extend delta is
    denoise length, not the mask).

    Stays out of the Windows Fast profile: that envelope was measured for
    T2V/I2V/A2V, and both the edit floor and the edit law basis differ. On
    Darwin the stream gate is Mac-native (retake == extend, one envelope).
    Off Windows-and-Darwin the legacy generic curve is untouched (it errs
    conservative there). Full-vs-stream keeps the 5.0 headroom: a 5090 that full-loaded 1280x704x337 reserved the whole
    card, and full behavior on current builds is unmeasured either way.
    This gate also enables Mac window-shortening: ``max_edit_frames_that_load``
    trims 1080p windows to the pool instead of approving the full cap.
    """
    seq = video_tokens(height, width, frames)
    decode_tokens = decode_tile_tokens(height, width, frames)
    route = budget_route("edit", job_filter.profile)
    if route == "windows_measured":
        return decide_measured_job(
            required_gib=_edit_stream_required_gib(seq, decode_tokens=decode_tokens),
            required_ram_bytes=job_filter.profile.windows_edit_required_ram_bytes,
            seq=seq,
            full_headroom_gib=_EDIT_FULL_HEADROOM_GIB,
            job_filter=job_filter,
        )
    if route == "mac_measured":
        return decide_mac_job(
            seq, _mac_edit_stream_required_gib(seq), job_filter=job_filter
        )
    return decide_load_mode(
        seq,
        memory_gb=job_filter.budget.memory_gb,
        process_mode=job_filter.profile.process_mode,
        darwin=job_filter.profile.darwin,
        full_headroom_gib=_EDIT_FULL_HEADROOM_GIB,
    )


def max_edit_frames_that_load(
    width: int,
    height: int,
    *,
    frame_cap: int,
    min_frames: int = 1,
    job_filter: LocalJobFilter,
) -> int:
    """Largest frame count at or below ``frame_cap`` that edit pricing loads.

    1920x1088x505 asks about 41 GiB on the measured reserved curve and 422s
    everywhere at or under 40 GB.
    That canvas is the letterbox of a 1080p picture. Home passes this cap
    into the encode window so that 1080p job runs at the longest window that
    fits. Single-tile decode tokens stay within ``MEASURED_MAX_DECODE_TOKENS``
    so automatic context padding never grows a short 4K retake into the
    73-frame WDDM paging zone; in Brave mode an explicit selection above that
    ceiling (``min_frames``) can still reach the VRAM cap as an unmeasured bet.
    When nothing fits, return ``frame_cap`` and let the loader 422.
    """
    if frame_cap < 1:
        return frame_cap
    route = budget_route("edit", job_filter.profile)
    max_decode_tokens = MEASURED_MAX_DECODE_TOKENS.get(("edit", route))

    def loads(frames: int) -> bool:
        if (
            max_decode_tokens is not None
            and (job_filter.budget_mode == "cautious" or frames > min_frames)
            and decode_tile_tokens(height, width, frames) > max_decode_tokens
        ):
            return False
        # Priced, not admitted: shortening a window answers memory.
        decision = _decide_priced_edit_job(
            width,
            height,
            frames,
            job_filter=job_filter,
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
    job_filter: LocalJobFilter,
) -> list[int]:
    """Keep Fast cells that ``decide_video_job`` admits. Not a full-mode envelope.

    Filtering with ``fits(..., "full_models_loading")`` would hide 720p/20s on a
    5090 that streams it. Darwin never compares free RAM to the CUDA stream curve
    (measured Mac Fast gates on the MPS pool instead).
    """
    # Intentional fast path: unsupported means no list — skip N cell evaluations.
    if job_filter.profile.process_mode == "unsupported":
        return []
    return decide_fast_durations(
        width, height, fps, durations, job_filter=job_filter
    ).kept


# Windows (CUDA) windowed peak above the single-pass law, per token of one window. The
# carry and the earlier windows leave allocator state behind, and that state grows with the
# window's tokens: the peak climbs for two windows past the first, then holds flat for every
# window count measured. A fixed figure fits one window size only: the one fitted at a 4K
# window over-asked the small 41-frame windows, which refuses jobs a 16 GB card loads, and
# under-asked 1440p's 129-frame windows. The coefficients keep a small margin over the 4K
# points; lowering either leaves the 4K cells at zero margin.
WINDOWED_PLATEAU_GIB_PER_TOKEN = 1.05e-4
_WINDOWED_PLATEAU_GIB_PER_TOKEN = WINDOWED_PLATEAU_GIB_PER_TOKEN

# Two windows pay most of the plateau step when the second window is full (1440p at 10 s and
# 24 fps is 129 frames then 112 new + 17 carry = 129: G1 peaked at 25.58 GiB vs 26.58 ask).
# When the second window is shorter (G2a 4K 73f has a 41f second window and peaked at 21.80
# GiB; G2b 1440p 145f has a 33f second window and peaked at 21.88 GiB, with window 2 at 18.52
# GiB), window 2's own working set scales with its tokens ``last_tokens`` while still carrying
# window 1's overlap state (``_WINDOWED_TWO_WINDOW_GIB_PER_TOKEN * tokens``), and the job peak
# is ``max(base_law(t1), base_law(t2) + c_2w * t1)``.
WINDOWED_TWO_WINDOW_GIB_PER_TOKEN = 6.5e-5
_WINDOWED_TWO_WINDOW_GIB_PER_TOKEN = WINDOWED_TWO_WINDOW_GIB_PER_TOKEN


def windows_windowed_extra_gib(windows: int, tokens: float) -> float:
    if windows >= PEAK_PLATEAU_WINDOWS:
        return WINDOWED_PLATEAU_GIB_PER_TOKEN * tokens
    if windows == 2:
        return WINDOWED_TWO_WINDOW_GIB_PER_TOKEN * tokens
    return 0.0


_windows_windowed_extra_gib = windows_windowed_extra_gib


# Mac (MPS) windowed price above the single-pass law, per token of one window, on the same
# basis as the law itself: the smallest allocator cap that still passes, not the native
# peak, which holds cache the cap would force out. The carry and the earlier windows leave
# allocator state behind that grows with the window's tokens, and the peak climbs for two
# windows past the first and then holds. The Mac price sits above the Windows one because
# the 4K and 1440p cap runs landed above where the Windows coefficient would put them.
# A fixed figure under-asks the 4K cap edge and the 1440p windows; the plateau price
# (1.15e-4) clears the 4K 7-window edge in (28.4, 29.2] by +0.06..+0.86 GiB (29.26 ask;
# +0.19 above the 29.07 peak at cap 29.2) and the 1440p 5-window edge in (30.7, 31.7] by
# +0.01..+1.01 GiB (31.71 ask; +0.41 above the 31.30 peak at cap 31.7).
_MAC_WINDOWED_PLATEAU_GIB_PER_TOKEN = 1.15e-4

# Two windows pay ``max(base_law(t1), base_law(t2) + _MAC_WINDOWED_TWO_WINDOW_GIB_PER_TOKEN * t1)``,
# matching the Windows rule: a full second window at 1440p (10 s at 24 fps, 241f) asks 29.52 GiB
# (+0.27 above its 29.25 passing cap; 7.5e-5 sat 0.05 under it; 4K 81f full last asks 27.26 vs
# <= 26.0 passing cap), while a short second window asks 24.53 GiB at 4K 73f (41f last, +1.13
# above its 23.4 passing cap) and 24.51 GiB at 1440p 145f (33f last, where window 1 binds above
# its 23.2 passing cap).
_MAC_WINDOWED_TWO_WINDOW_GIB_PER_TOKEN = 8.0e-5


def _mac_windowed_extra_gib(windows: int, tokens: float) -> float:
    if windows >= PEAK_PLATEAU_WINDOWS:
        return _MAC_WINDOWED_PLATEAU_GIB_PER_TOKEN * tokens
    if windows == 2:
        return _MAC_WINDOWED_TWO_WINDOW_GIB_PER_TOKEN * tokens
    return 0.0


def _fast_base_stream_gib(tokens: float, *, darwin: bool) -> float:
    if darwin:
        return _mac_fast_stream_required_gib(tokens)
    return max(
        _windows_fast_stream_required_gib(tokens),
        _windows_fast_reserved_required_gib(tokens),
    )


def fast_job_stream_gib(
    tokens: float,
    *,
    windows: int,
    darwin: bool,
    last_tokens: float | None = None,
) -> float:
    """The stream ask of one Fast pass inside a job of ``windows`` windows. Run logs print it
    next to the measured peak, so a refit has both numbers."""
    base_t1 = _fast_base_stream_gib(tokens, darwin=darwin)
    if windows == 2:
        t2 = tokens if last_tokens is None else last_tokens
        base_t2 = _fast_base_stream_gib(t2, darwin=darwin)
        extra = (
            _mac_windowed_extra_gib(2, tokens)
            if darwin
            else _windows_windowed_extra_gib(2, tokens)
        )
        return max(base_t1, base_t2 + extra)
    if darwin:
        return base_t1 + _mac_windowed_extra_gib(windows, tokens)
    return max(
        _windows_fast_stream_required_gib(tokens),
        _windows_fast_reserved_required_gib(tokens) + _windows_windowed_extra_gib(windows, tokens),
    )


def edit_job_stream_gib(
    tokens: float, *, darwin: bool, decode_tokens: float | None = None
) -> float:
    """The stream ask of one Retake or Extend window. Run logs print it next to the peak."""
    return (
        _mac_edit_stream_required_gib(tokens)
        if darwin
        else _edit_stream_required_gib(tokens, decode_tokens=decode_tokens)
    )


def _windows_fast_stream_required_gib(tokens: float) -> float:
    """Continuous stream requirement: finite for every resolution, no lookup."""
    return _WINDOWS_FAST_STREAM.required_gib(tokens)


def _windows_fast_reserved_required_gib(tokens: float) -> float:
    """Reserved-basis twin: what the card must back with physical pages, no lookup."""
    return _WINDOWS_FAST_RESERVED_STREAM.required_gib(tokens)


def _windows_fast_full_reserved_required_gib(tokens: float) -> float:
    """Full-mode reserved ask: resident full models plus token growth, no lookup."""
    return _WINDOWS_FAST_FULL_RESERVED.required_gib(tokens)


def _mac_fast_stream_required_gib(tokens: float) -> float:
    """Continuous Mac stream requirement against the MPS pool, no lookup."""
    return _MAC_FAST_STREAM.required_gib(tokens)


def _edit_stream_required_gib(
    tokens: float, *, decode_tokens: float | None = None
) -> float:
    """Edit stream requirement: max of denoise and single-tile decode reserved laws."""
    base = _EDIT_STREAM.required_gib(tokens)
    if decode_tokens is None:
        return base
    return max(base, _EDIT_DECODE_STREAM.required_gib(decode_tokens))


def _mac_edit_stream_required_gib(tokens: float) -> float:
    """Mac edit requirement against the MPS pool."""
    return _MAC_EDIT_STREAM.required_gib(tokens)


def _streaming_ram_reject(
    seq: float,
    full_gib: float,
    stream_gib: float,
    *,
    budget: MemoryBudget,
    required_ram_bytes: int,
) -> VideoJobReject | None:
    """Shared streamed-RAM gate: the reject, or None when RAM suffices.

    ram_shortfall on known-insufficient RAM, unsupported when unknown.
    Callers pick the floor (Fast vs interim measured); the verdict shape
    is identical, so it lives here instead of pasted per path.
    """
    match budget.streaming_ram_state(required_bytes=required_ram_bytes):
        case "ok":
            return None
        case "shortfall":
            return VideoJobReject(
                seq=seq,
                full_gib=full_gib,
                stream_gib=stream_gib,
                reason="ram_shortfall",
                required_ram_bytes=required_ram_bytes,
            )
        case "unknown":
            return VideoJobReject(
                seq=seq, full_gib=full_gib, stream_gib=stream_gib, reason="unsupported"
            )
        case _ as unreachable:
            assert_never(unreachable)


def _decide_windows_fast_job(
    seq: float,
    *,
    windows: int = 1,
    last_tokens: float | None = None,
    job_filter: LocalJobFilter,
) -> VideoJobDecision:
    """Windows Fast (2.3 and 2.5) availability from the continuous model, then full-vs-stream pick.

    The stream ask is max(allocated, reserved): allocated predicts OOM
    crashes, reserved predicts WDDM paging, and the gate must refuse both
    failures. Reserved dominates everywhere today (floor 9.90 > 8.00 and a
    steeper line), so it binds; the max keeps both failures gated if a future
    refit ever un-dominates. Evaluates Windows-specific fits (continuous
    requirement against ``min(total, free)`` VRAM; full ask against exact
    total) and runs the shared engine.
    The full ask is max(legacy curve, full-mode reserved law): native full-mode
    reserved exceeds the legacy ask by ~3-4 GiB everywhere measured, so the law
    binds; the max keeps the legacy ask as the floor if a future refit ever
    un-dominates.
    Free can only restrict, never expand, and the full-vs-stream pick stays on
    total for stability. A 16 GB card has ~15.0 GiB free on a working desktop —
    the exact cap that OOM'd after ~150 s in stage-2 denoise, so free-VRAM
    gating is load-bearing. Missing exact bytes with a viable integer figure
    falls back to the integer rather than emptying the grid — floor rounding only
    hides, and only a missing integer too fails closed. The streamed-RAM gate
    reports ``ram_shortfall`` on known-insufficient RAM, ``unsupported`` when
    unknown.
    """
    full_gib = max(
        estimated_giB("full_models_loading", seq),
        _windows_fast_full_reserved_required_gib(seq),
    )
    stream_gib = fast_job_stream_gib(
        seq, windows=windows, darwin=False, last_tokens=last_tokens
    )

    available = job_filter.budget.available_vram_gib
    total = job_filter.budget.total_vram_gib
    if available is None or total is None:
        return VideoJobReject(
            seq=seq, full_gib=full_gib, stream_gib=stream_gib, reason="unsupported"
        )
    decision = decide_from_fits(
        seq,
        full_gib,
        stream_gib,
        stream_fits=(available >= stream_gib),
        full_fits=(full_gib <= _full_memory_gb(total)),
        process_mode=job_filter.profile.process_mode,
        # Windows-only path: profile.darwin is always False here.
        legacy_darwin_never_reject=False,
    )
    if (
        isinstance(decision, VideoJobLoad)
        and decision.mode == "streaming_models_loading"
    ):
        ram_reject = _streaming_ram_reject(
            seq,
            full_gib,
            stream_gib,
            budget=job_filter.budget,
            required_ram_bytes=job_filter.profile.windows_fast_required_ram_bytes,
        )
        if ram_reject is not None:
            return ram_reject
    return decision


# MPS transformer sequence fence: a per-forward token ceiling, not a memory
# curve. Past it the distilled stage-2 forward aborts the process — Metal
# asserts `[MPSNDArrayMatrixMultiplication encodeToCommandBuffer:] destination
# is too large for kernel` and the backend dies with SIGABRT, mid-queue, with
# no catchable error. The failing op is the per-token-timestep AdaLN
# projection: [tokens, 36864]. At 257,040 tokens that is 9.48B elements,
# past the uint32 index range (2^32 = 4.29B); the exact bound at width 36864
# is 116,508 tokens, and the fence sits at 116,000, an epsilon under it.
# AdaLN binds first by construction — at 6-9x the inner width it is the
# widest per-token linear in the arch (FFN is 4x, QKV 3x, patchify 1x), so
# the rest of the family still has headroom at the fence. No extra margin:
# the fence is the hardware's limit, and anything the hardware can run stays
# admitted; an ltx-core bump that widens AdaLN moves the bound and must
# re-derive this constant like any other fitted figure. Bracket: 1080p/10s@24
# (63,240 tokens) passes, 1080p/20s@50fps (257,040 tokens) SIGABRTs. The fence
# binds one forward, never a clip: a windowed Fast job or a tiled IC-LoRA job
# runs one window per forward, so its whole-clip token count (253k at 4K/10s)
# may sit far past the fence while every forward stays under it. The pool
# gate below already refuses most of what the fence admits on small Macs —
# the fence only binds on huge pools that would otherwise crash.
# Every decide_mac_job caller passes the heaviest single forward's tokens
# (Fast: stage-2 canvas; edit: canvas window; IC-LoRA: per-window/per-tile
# pass), so one fence here guards all three features. CUDA has no such
# limit (64-bit indexing), so the fence is Darwin-only by construction.
_MPS_TRANSFORMER_TOKEN_FENCE = 116_000

# ltx-core's disk-streaming read buffers: host_buffers + prefetch_depth (2 + 2 at the
# StreamingConfig defaults Desktop runs), each sized to the largest streamed block, plus
# DeviceStager's two 64 MiB buckets. On MPS they are ordinary host memory outside
# recommended_max, so the Mac ask carries them. Re-check both on an ltx-core bump.
_MAC_DISK_STREAM_HOST_BUFFERS: Final = 4
_MAC_DEVICE_STAGER_BYTES: Final = 2 * (64 << 20)


def mac_disk_stream_host_gib(model_id: LTXLocalModelId | None) -> float:
    """Host memory the Mac disk-streaming loader holds outside the MPS pool.

    An unset or future model id (budget pricing outlives the spec table) prices the
    largest known block, so it is never under-charged and never crashes the lookup.
    """
    if model_id in ALL_LTX_LOCAL_MODEL_IDS:
        unit_bytes = get_ltx_model_spec(model_id).disk_stream_unit_bytes
    else:
        unit_bytes = max(
            get_ltx_model_spec(known).disk_stream_unit_bytes
            for known in ALL_LTX_LOCAL_MODEL_IDS
        )
    return (_MAC_DISK_STREAM_HOST_BUFFERS * unit_bytes + _MAC_DEVICE_STAGER_BYTES) / _GIB


# Full-resident policy evidence (for decide_mac_job below): interleaved
# DISK-vs-NONE runs. Per-feature bracket evidence lives with each curve
# (_MAC_FAST_STREAM, _MAC_EDIT_STREAM, and the IC-LoRA constants).
def decide_mac_job(
    seq: float,
    stream_gib: float,
    *,
    job_filter: LocalJobFilter,
) -> VideoJobDecision:
    """Darwin availability from the Mac allocator-bracket pool model. Always streams.

    Compares the caller's continuous stream requirement against the sampled
    MPS pool (recommended_max) and runs the shared engine with the legacy
    Darwin branch opted out: that branch never refuses, which was correct
    while no Darwin-native ceiling existed and is exactly what measured Mac
    profiles replace. Full resident stays unselectable (interleaved
    DISK-vs-NONE comparison: full resident only modestly faster at several
    times the pool — never the better pick on shared unified memory).
    Unknown pool fails closed to unsupported. The ask adds the
    disk-streaming loader's host read buffers (``mac_disk_stream_host_gib``,
    about 3 GiB for a 22B bf16 transformer): they sit in ordinary memory
    outside the pool, and recommended_max leaves the rest of RAM to the OS, so
    the pool pays for them. There is no separate RAM gate. Process mode flows
    through, so unsupported still rejects first.

    Callers supply the feature's MPS stream requirement; this core adds the
    host buffers once, then owns the pool comparison and the never-full
    policy. Every verdict carries the charged ask, so readouts, tight-fit and
    over figures compare the same number the gate did. The MPS sequence fence
    runs before the pool comparison (a fence verdict never prints pool
    figures), after unsupported (a dead machine reports unsupported, not the
    fence).
    """
    stream_gib += mac_disk_stream_host_gib(job_filter.profile.model_id)
    full_gib = estimated_giB("full_models_loading", seq)

    if (
        job_filter.profile.process_mode != "unsupported"
        and seq > _MPS_TRANSFORMER_TOKEN_FENCE
    ):
        return VideoJobReject(
            seq=seq,
            full_gib=full_gib,
            stream_gib=stream_gib,
            reason="mps_sequence_over",
        )

    pool = job_filter.budget.mps_pool_gib
    if pool is None:
        return VideoJobReject(
            seq=seq, full_gib=full_gib, stream_gib=stream_gib, reason="unsupported"
        )
    return decide_from_fits(
        seq,
        full_gib,
        stream_gib,
        stream_fits=(pool >= stream_gib),
        full_fits=False,
        process_mode=job_filter.profile.process_mode,
        legacy_darwin_never_reject=False,
    )


def decide_measured_job(
    *,
    required_gib: float,
    required_ram_bytes: int,
    seq: float,
    full_headroom_gib: float,
    job_filter: LocalJobFilter,
    force_stream: bool = False,
) -> VideoJobDecision:
    """Measured stream requirement + legacy full pick + Windows RAM gate.

    Twin of _decide_windows_fast_job for edit/IC-LoRA on Windows: the stream
    requirement is measured (not the generic curve), compared against
    min(total, free) with the integer fallback; the full-vs-stream pick stays
    on the legacy headroom (full behavior on current builds is unmeasured —
    conservative by design); streamed loads face the
    caller's per-feature RAM floor. Unknown bytes fail closed like Fast. Off
    Windows the callers keep the legacy generic path untouched. ``force_stream``
    skips the full pick for jobs whose full-resident shape is unmeasured (the
    legacy curve on a small seq would full-load cards the true resident
    working set does not fit).
    """
    full_gib = estimated_giB("full_models_loading", seq)
    available = job_filter.budget.available_vram_gib
    total = job_filter.budget.total_vram_gib
    if available is None or total is None:
        return VideoJobReject(
            seq=seq, full_gib=full_gib, stream_gib=required_gib, reason="unsupported"
        )
    decision = decide_from_fits(
        seq,
        full_gib,
        required_gib,
        stream_fits=(available >= required_gib),
        full_fits=not force_stream
        and fits(
            seq, "full_models_loading", _full_memory_gb(total, full_headroom_gib)
        ),
        process_mode=job_filter.profile.process_mode,
        # Windows-only path: profile.darwin is always False here.
        legacy_darwin_never_reject=False,
    )
    if (
        isinstance(decision, VideoJobLoad)
        and decision.mode == "streaming_models_loading"
        and job_filter.profile.system == "Windows"
    ):
        ram_reject = _streaming_ram_reject(
            seq,
            full_gib,
            required_gib,
            budget=job_filter.budget,
            required_ram_bytes=required_ram_bytes,
        )
        if ram_reject is not None:
            return ram_reject
    return decision
