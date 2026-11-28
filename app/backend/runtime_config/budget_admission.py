"""The one admission policy: may this priced job run, and as what bet.

Pricing (``video_job_budget``, ``ic_lora_job_budget``) is mode-blind and
returns a ``PricedJob``. ``admit`` turns it into ``Admitted`` or ``Refused``
from the verdict, its evidence, and a ``RiskAllowance``. The budget mode
enters only through the allowance, so no caller re-reads the mode, the
reject reasons, or the evidence to decide whether a job may run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict, TypeGuard, assert_never

from pydantic import ValidationError

from api_types import BudgetReadout, GenerationRisk, LocalUnavailableReason
from runtime_config.budget_decision import (
    IcLoraJobDecision,
    IcLoraJobLoad,
    IcLoraJobReject,
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
from runtime_config.device_calibration import JobEnvelope, envelope_for
from runtime_config.measured_cells import BudgetFeature
from runtime_config.reject_reasons import (
    BudgetMode,
    JobRiskTier,
    ShortfallKind,
    UnmeasuredBasis,
    VideoJobRejectReason,
    WarnedCellTier,
)

if TYPE_CHECKING:
    # Runtime import would cycle (video_job_budget imports this module).
    from runtime_config.video_job_budget import LocalJobFilter

_GIB = 1024**3
TIGHT_FIT_HEADROOM_GIB = 0.5
TIGHT_FIT_HEADROOM_RATIO = 0.05


def _is_str_object_dict(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict)


def stored_generation_risk(spec: Mapping[str, object]) -> GenerationRisk | None:
    """The stored bet, or None when it is missing or junk.

    Junk fails closed: execute must not waive a limit off a corrupt row.
    """
    risk = spec.get("_risk")
    if not _is_str_object_dict(risk):
        return None
    try:
        return GenerationRisk.model_validate(risk)
    except ValidationError:
        return None


@dataclass(frozen=True, slots=True)
class PricedJob:
    """A mode-blind verdict plus everything admission needs to apply the mode.

    ``covered``, ``tight``, and ``over_risk`` are attached by ``priced_job``:
    ``covered`` marks an extrapolated load this device already ran cleanly at
    an equal or larger shape; ``tight`` marks a load inside max(0.5 GiB, 5% of
    the ask); ``over_risk`` is the bet a downgradable reject would run as,
    None for loads, hard walls, and uncompared figures.
    """

    feature: BudgetFeature
    decision: VideoJobDecision | IcLoraJobDecision
    evidence: PriceEvidence
    shape: JobShape
    covered: bool = False
    tight: bool = False
    over_risk: GenerationRisk | None = None
    # The carry an IC-LoRA job on Auto carry runs, picked with its window. None otherwise.
    carry_frames: int | None = None

    @property
    def seq(self) -> float:
        return self.decision.seq

    @property
    def full_gib(self) -> float:
        return self.decision.full_gib

    @property
    def stream_gib(self) -> float:
        return self.decision.stream_gib

    @property
    def chunk_pixel_frames(self) -> int | None:
        return self.decision.chunk_pixel_frames


@dataclass(frozen=True, slots=True)
class RiskAllowance:
    """Which bets may run: an over-budget downgrade, an untested load, or both."""

    over: bool
    unmeasured: bool

    @staticmethod
    def none() -> RiskAllowance:
        return RiskAllowance(over=False, unmeasured=False)

    @staticmethod
    def for_mode(mode: BudgetMode) -> RiskAllowance:
        match mode:
            case "brave":
                return RiskAllowance(over=True, unmeasured=True)
            case "cautious":
                return RiskAllowance(over=False, unmeasured=False)
            case _ as unreachable:
                assert_never(unreachable)

    @staticmethod
    def from_risk(risk: GenerationRisk | None) -> RiskAllowance:
        """Allowance for the bet a queued job already confirmed.

        A stored ``over`` bet covers the over-budget downgrade (plus the
        extrapolated load when ``unmeasured_basis`` was part of the bet). A
        stored ``unmeasured`` bet covers the untested load it was taken for,
        never a later VRAM or RAM miss.
        """
        if risk is None:
            return RiskAllowance.none()
        match risk.tier:
            case "over":
                return RiskAllowance(
                    over=True,
                    unmeasured=risk.unmeasured_basis is not None,
                )
            case "unmeasured":
                return RiskAllowance(over=False, unmeasured=True)
            case _ as unreachable:
                assert_never(unreachable)

    @staticmethod
    def from_stored(spec: Mapping[str, object]) -> RiskAllowance:
        """Parse the stored ``_risk`` annotation on a queued generation spec."""
        return RiskAllowance.from_risk(stored_generation_risk(spec))

    def without_over(self) -> RiskAllowance:
        """Keep ``unmeasured`` while probing for a free-VRAM dip that can shrink."""
        return RiskAllowance(over=False, unmeasured=self.unmeasured)


@dataclass(frozen=True, slots=True)
class Admitted:
    priced: PricedJob
    load: VideoJobLoad | IcLoraJobLoad
    # Confirmable bet, stored as spec["_risk"]. None for a clean admit.
    risk: GenerationRisk | None
    # Passive tight-fit note; never confirms.
    tight: bool

    @property
    def envelope(self) -> JobEnvelope:
        """The shape this admission runs, as device calibration records it."""
        return envelope_for(self.priced.shape, self.load.mode, self.load.stream_gib)


@dataclass(frozen=True, slots=True)
class Refused:
    priced: PricedJob
    reject: VideoJobReject | IcLoraJobReject
    # The bet the job would run as if the allowance covered it. None for a hard wall.
    bet: GenerationRisk | None


Admission = Admitted | Refused


def priced_job(
    feature: BudgetFeature,
    decision: VideoJobDecision | IcLoraJobDecision,
    evidence: PriceEvidence,
    shape: JobShape,
    *,
    job_filter: LocalJobFilter,
) -> PricedJob:
    """Attach the compared figures admission needs: local coverage, tight fit, or the over bet.

    An extrapolated load this device already ran cleanly at an equal or larger
    shape is ``covered``. An over bet on an extrapolated verdict names the
    missing measurement, but keeps tier ``over``: a real memory miss is never
    relabelled untested, and a device proof never waives it.
    """
    if isinstance(decision, (VideoJobLoad, IcLoraJobLoad)):
        covered = isinstance(evidence, Extrapolated) and (
            job_filter.calibration.covering(
                envelope_for(shape, decision.mode, decision.stream_gib)
            )
            is not None
        )
        return PricedJob(
            feature=feature,
            decision=decision,
            evidence=evidence,
            shape=shape,
            covered=covered,
            tight=is_tight_fit(decision, job_filter),
        )
    over_risk = (
        risk_for_reject(decision, job_filter)
        if isinstance(decision, VideoJobReject)
        else None
    )
    if over_risk is not None and isinstance(evidence, Extrapolated):
        over_risk = over_risk.model_copy(update={"unmeasured_basis": evidence.basis})
    return PricedJob(
        feature=feature,
        decision=decision,
        evidence=evidence,
        shape=shape,
        over_risk=over_risk,
    )


def admit(priced: PricedJob, allowance: RiskAllowance) -> Admission:
    """Apply the admission table to one priced job.

    Hard walls refuse in every mode. A fitted load admits (tight when the
    headroom is small); a generic-curve load or a locally covered extrapolation
    admits silently. An uncovered extrapolated load admits with an unmeasured
    bet when the allowance covers it, else refuses as ``unmeasured_cell`` with
    the priced figures. A downgradable reject runs streaming at its own window
    with an over bet when the allowance covers it: a real memory miss is never
    relabelled untested.
    """
    decision = priced.decision
    if isinstance(decision, (VideoJobLoad, IcLoraJobLoad)):
        match priced.evidence:
            case Fitted():
                return Admitted(priced=priced, load=decision, risk=None, tight=priced.tight)
            case GenericCurve():
                return Admitted(priced=priced, load=decision, risk=None, tight=False)
            case Extrapolated(basis=basis):
                if priced.covered:
                    return Admitted(priced=priced, load=decision, risk=None, tight=False)
                risk = unmeasured_risk(basis, need_gib=decision.stream_gib)
                if allowance.unmeasured:
                    return Admitted(priced=priced, load=decision, risk=risk, tight=False)
                return Refused(
                    priced=priced,
                    reject=VideoJobReject(
                        seq=decision.seq,
                        full_gib=decision.full_gib,
                        stream_gib=decision.stream_gib,
                        reason="unmeasured_cell",
                        chunk_pixel_frames=decision.chunk_pixel_frames,
                        unmeasured_basis=basis,
                    ),
                    bet=risk,
                )
            case _ as unreachable:
                assert_never(unreachable)
    if isinstance(decision, IcLoraJobReject) or not is_downgradable(decision):
        return Refused(priced=priced, reject=decision, bet=None)
    risk = priced.over_risk
    if risk is not None and allowance.over:
        return Admitted(
            priced=priced,
            load=_streaming_load(priced.feature, decision),
            risk=risk,
            tight=False,
        )
    return Refused(priced=priced, reject=decision, bet=risk)


def admits_within_device_memory(priced: PricedJob, mode: BudgetMode) -> bool:
    """Whether ``admit`` accepts ``priced`` in ``mode`` without a device-memory miss.

    Used by the IC-LoRA recipe fit preview: Cautious requires a clean admit;
    Brave admits untested loads and host-RAM shortfalls whose GPU ask fits, but
    still bounds the preview by the device's VRAM / MPS pool (``stream_over``).
    """
    if isinstance(priced.decision, VideoJobReject) and priced.decision.reason == "stream_over":
        return False
    return isinstance(admit(priced, RiskAllowance.for_mode(mode)), Admitted)


def is_downgradable(decision: VideoJobReject | IcLoraJobReject) -> bool:
    """A memory miss (VRAM or RAM) that may run as an over bet.

    ``unsupported`` (no pipeline), ``mps_sequence_over`` (the Metal kernel
    aborts past the fence), and the IC-LoRA canvas and decode walls never run.
    """
    if isinstance(decision, IcLoraJobReject):
        return False
    match decision.reason:
        case "stream_over" | "ram_shortfall":
            return True
        case "unsupported" | "mps_sequence_over" | "unmeasured_cell" | "one_pass_unavailable":
            return False
        case _ as unreachable:
            assert_never(unreachable)


def is_clean(priced: PricedJob) -> bool:
    """A load whose evidence needs no bet."""
    return isinstance(priced.decision, (VideoJobLoad, IcLoraJobLoad)) and (
        not isinstance(priced.evidence, Extrapolated) or priced.covered
    )


_SafetyRank = Literal[0, 1, 2, 3]


def safety_rank(priced: PricedJob) -> _SafetyRank:
    decision = priced.decision
    if isinstance(decision, (VideoJobLoad, IcLoraJobLoad)):
        if isinstance(priced.evidence, Extrapolated) and not priced.covered:
            return 2
        return 3
    return 1 if is_downgradable(decision) else 0


def safer(first: PricedJob, chunked: PricedJob) -> PricedJob:
    """The safer of an unchunked and a chunked price for one job.

    Clean load > extrapolated load > downgradable reject > hard wall. A tie
    keeps the unchunked job, so chunking only escapes risk and never adds
    seams to a job that already fits; between two downgradable rejects the
    smaller ask wins, so a downgrade runs the cheaper window.
    """
    first_rank = safety_rank(first)
    chunked_rank = safety_rank(chunked)
    if chunked_rank > first_rank:
        return chunked
    if (
        chunked_rank == first_rank == 1
        and chunked.decision.stream_gib < first.decision.stream_gib
    ):
        return chunked
    return first


def budget_readout_for_reject(
    decision: VideoJobReject, job_filter: LocalJobFilter
) -> BudgetReadout | None:
    """Need and have for the same comparison ``reject_message_and_code`` prints.

    None for a hard wall or when the gate compared no figure.
    """
    reason = decision.reason
    if reason == "unmeasured_cell":
        return untested_readout(decision.stream_gib)
    if reason in ("unsupported", "mps_sequence_over", "one_pass_unavailable"):
        return None
    if reason == "ram_shortfall":
        required = decision.required_ram_bytes
        detected = job_filter.budget.total_ram_bytes
        if required is None or detected is None:
            return None
        return BudgetReadout(
            kind="ram",
            need_gb=float(required // _GIB),
            have_gb=float(detected // _GIB),
        )
    required_gib = decision.stream_gib
    if job_filter.profile.system == "Darwin":
        pool = job_filter.budget.mps_pool_gib
        if pool is None:
            return None
        return BudgetReadout(kind="mac", need_gb=required_gib, have_gb=pool)
    total = job_filter.budget.usable_vram_gib
    available = job_filter.budget.available_vram_gib
    if total is None or available is None:
        return None
    if required_gib > total:
        return BudgetReadout(kind="gpu", need_gb=required_gib, have_gb=total)
    if required_gib > available:
        return BudgetReadout(kind="gpu_busy", need_gb=required_gib, have_gb=available)
    return None


def tight_readout(ask_gib: float, available_gib: float) -> BudgetReadout:
    return BudgetReadout(kind="tight", need_gb=ask_gib, have_gb=available_gib)


def compared_available_gib(job_filter: LocalJobFilter) -> float | None:
    """The figure the gate compared the ask against.

    Single home for the figure-selection rule: the MPS pool on Darwin,
    min(total, free) — or usable_vram_gib under total_only — on CUDA. None
    means the gate compared nothing (callers stay silent).
    """
    if job_filter.profile.system == "Darwin":
        return job_filter.budget.mps_pool_gib
    if job_filter.budget.total_vram_bytes is None:
        return None
    return job_filter.budget.available_vram_gib


def load_headroom_gib(
    decision: VideoJobLoad | IcLoraJobLoad, job_filter: LocalJobFilter
) -> float | None:
    """Available minus ask on the compared figure. None when uncompared."""
    available = compared_available_gib(job_filter)
    if available is None:
        return None
    return available - decision.stream_gib


def is_tight_fit(
    decision: VideoJobLoad | IcLoraJobLoad, job_filter: LocalJobFilter
) -> bool:
    """A load inside max(0.5 GiB, 5% of the ask) on the compared figure.

    Unknown figures stay silent.
    """
    headroom = load_headroom_gib(decision, job_filter)
    if headroom is None:
        return False
    threshold = max(
        TIGHT_FIT_HEADROOM_GIB, TIGHT_FIT_HEADROOM_RATIO * decision.stream_gib
    )
    return headroom < threshold


def reject_risk_tier(reason: VideoJobRejectReason) -> JobRiskTier | None:
    """Measurement rejects are over-budget bets, an unmeasured cell is an
    unmeasured bet; hard walls return None.

    Allow-list: a new ``VideoJobRejectReason`` fails typecheck here until it
    picks a tier. ``unsupported`` (no pipeline) and ``mps_sequence_over`` (the
    Metal kernel SIGABRTs past the fence — uncatchable, not a bet) never
    downgrade.
    """
    match reason:
        case "stream_over" | "ram_shortfall":
            return "over"
        case "unmeasured_cell":
            return "unmeasured"
        case "unsupported" | "mps_sequence_over" | "one_pass_unavailable":
            return None
        case _ as unreachable:
            assert_never(unreachable)


def reject_over_gib(
    decision: VideoJobReject, job_filter: LocalJobFilter
) -> float | None:
    """VRAM shortfall on a stream_over reject. None for any other reason —
    RAM verdicts passed VRAM, so subtracting would print a negative lie."""
    if decision.reason != "stream_over":
        return None
    available = compared_available_gib(job_filter)
    if available is None:
        return None
    return decision.stream_gib - available


def reject_over_ram_gb(
    decision: VideoJobReject, job_filter: LocalJobFilter
) -> int | None:
    """RAM shortfall in whole GB, message-consistent (required//GiB minus
    detected//GiB, the same ints the 422 prints)."""
    if decision.reason != "ram_shortfall":
        return None
    required = decision.required_ram_bytes
    detected = job_filter.budget.total_ram_bytes
    if required is None or detected is None:
        return None
    return required // _GIB - detected // _GIB


def reject_shortfall_kind(
    decision: VideoJobReject, job_filter: LocalJobFilter
) -> ShortfallKind | None:
    """Which figure bound a CUDA stream_over reject.

    "free" only when free strictly binds (free < total on exact floats —
    min() returns its input, so equality with total is exact). Total-gated
    lists, missing samples, and Darwin (pool gate, never free advice) all
    report "total"/None respectively: Darwin returns None and the frontend
    keys Darwin copy off platform, not kind.
    """
    if decision.reason != "stream_over":
        return None
    if job_filter.profile.system == "Darwin":
        return None
    total = job_filter.budget.usable_vram_gib
    available = job_filter.budget.available_vram_gib
    if total is None or available is None:
        return None
    return "total" if available == total or decision.stream_gib > total else "free"


def tight_fit_message(ask_gib: float, available_gib: float, *, darwin: bool) -> str:
    if darwin:
        return (
            f"Tight fit: this needs ~{ask_gib:.1f} GB of the "
            f"{available_gib:.1f} GB available for video on this Mac. It should run."
        )
    return (
        f"Tight fit: this needs ~{ask_gib:.1f} GB and about "
        f"{available_gib:.1f} GB is available. It should run, but close "
        "other heavy apps (like games, browsers, or video editors) first."
    )


def over_budget_message(
    ask_gib: float,
    available_gib: float,
    *,
    darwin: bool,
    shortfall_kind: ShortfallKind | None,
) -> str:
    if darwin:
        return (
            f"Needs too much memory: this needs {ask_gib:.1f} GB but this Mac allows "
            f"{available_gib:.1f} GB. It may run out of memory or stall — "
            "lower the resolution or duration to fit."
        )
    if shortfall_kind == "free":
        return (
            f"Needs too much memory right now: this needs {ask_gib:.1f} GB but only "
            f"{available_gib:.1f} GB is free. Close other heavy apps (like games, "
            "browsers, or video editors) and retry, or lower the resolution or duration."
        )
    return (
        f"Needs too much memory: this needs {ask_gib:.1f} GB but this machine's graphics "
        f"card has {available_gib:.1f} GB. It may run out of memory or stall — "
        "lower the resolution or duration to fit."
    )


def over_ram_message(required_gb: int, detected_gb: int) -> str:
    return (
        f"Needs too much memory: this needs {required_gb} GB of system RAM but this machine "
        f"has {detected_gb} GB. Close heavy programs; on desktops, more RAM "
        "is the real fix."
    )


def risk_for_reject(
    decision: VideoJobReject, job_filter: LocalJobFilter
) -> GenerationRisk | None:
    """Over-budget annotation for a measurement reject, if downgradable.

    Returns None for unsupported (hard wall) and for uncompared figures.
    Legacy-route rejects tier as over: the curve is rough, but a reject is
    conservative by construction — the unmeasured bet is the legacy *load*.
    """
    tier = reject_risk_tier(decision.reason)
    if tier is None:
        return None
    if tier == "unmeasured":
        basis = decision.unmeasured_basis
        if basis is None:
            return None
        return unmeasured_risk(basis, need_gib=decision.stream_gib)
    darwin = job_filter.profile.system == "Darwin"
    if decision.reason == "ram_shortfall":
        required = decision.required_ram_bytes
        detected = job_filter.budget.total_ram_bytes
        if required is None or detected is None:
            return None
        required_gb = required // _GIB
        detected_gb = detected // _GIB
        return GenerationRisk(
            tier="over",
            shortfall_kind=None,
            over_gib=None,
            over_ram_gb=required_gb - detected_gb,
            message=over_ram_message(required_gb, detected_gb),
            readout=budget_readout_for_reject(decision, job_filter),
        )
    available = compared_available_gib(job_filter)
    if available is None:
        return None
    kind = reject_shortfall_kind(decision, job_filter)
    over = decision.stream_gib - available
    return GenerationRisk(
        tier="over",
        shortfall_kind=kind,
        over_gib=over,
        over_ram_gb=None,
        message=over_budget_message(
            decision.stream_gib, available, darwin=darwin, shortfall_kind=kind
        ),
        readout=budget_readout_for_reject(decision, job_filter),
    )


def unmeasured_message(basis: UnmeasuredBasis) -> str:
    """The bet line for an admitted untested job."""
    match basis:
        case "high_res_cell":
            return (
                "Untested: this resolution hasn't been run on hardware like yours, "
                "so the memory estimate is extrapolated. It may fail or stall."
            )
        case "two_stage_range":
            return (
                "Untested: this two-step upscale is longer than we have measured at "
                "this photo size, so the memory estimate is extrapolated. It may "
                "fail or stall."
            )
        case "mac_two_stage_cell":
            return (
                "Untested: two-step upscaling at this size isn't measured on Mac yet. "
                "It may fail or stall."
            )
        case "windowed_range":
            return (
                "Untested: this clip runs in more sections than we have tested on "
                "hardware like yours, so the memory estimate is extrapolated. It may "
                "take a long time, and it may fail or stall."
            )
        case "pass_tokens":
            return (
                "Untested: this clip is longer than the longest clip we have "
                "tested all at once, so the memory estimate is extrapolated. It may fail or stall."
            )
        case "unmeasured_model":
            return (
                "Untested: this model hasn't been measured on hardware like yours yet, "
                "so the memory estimate is extrapolated. It may fail or stall."
            )
        case _ as unreachable:
            assert_never(unreachable)


def unmeasured_refusal_message(basis: UnmeasuredBasis) -> str:
    """The refusal line for an untested job the mode does not allow."""
    match basis:
        case "high_res_cell":
            return (
                "This resolution hasn't been tested on this machine, so Cautious "
                "mode blocks it. Turn off Cautious mode to try it anyway, or lower "
                "the resolution."
            )
        case "two_stage_range":
            return (
                "This two-step upscale is longer than we have measured at this photo "
                "size, so Cautious mode blocks it. Try a shorter clip, a smaller "
                "photo, or single-stage, or turn off Cautious mode to try it anyway."
            )
        case "mac_two_stage_cell":
            return (
                "Two-step upscaling at this size isn't measured on Macs yet, so "
                "Cautious mode blocks it. Use single-stage, or turn off Cautious "
                "mode to try it anyway."
            )
        case "windowed_range":
            return (
                "This clip runs in more sections than we have tested on this machine, "
                "so Cautious mode blocks it. Try a shorter clip, a lower frame rate or "
                "a smaller resolution, or turn off Cautious mode to try it anyway."
            )
        case "pass_tokens":
            return (
                "This clip is longer than the longest clip we have tested all at once, so "
                "Cautious mode blocks it. Try a shorter clip or a lower frame rate, or "
                "turn off Cautious mode to try it anyway."
            )
        case "unmeasured_model":
            return (
                "This model hasn't been tested on this machine yet, so Cautious "
                "mode blocks it. Turn off Cautious mode to try it anyway."
            )
        case _ as unreachable:
            assert_never(unreachable)


def untested_readout(need_gib: float | None = None) -> BudgetReadout:
    return BudgetReadout(kind="untested", need_gb=need_gib)


def unmeasured_risk(basis: UnmeasuredBasis, *, need_gib: float) -> GenerationRisk:
    return GenerationRisk(
        tier="unmeasured",
        message=unmeasured_message(basis),
        readout=untested_readout(need_gib),
        unmeasured_basis=basis,
    )


def _streaming_load(
    feature: BudgetFeature, reject: VideoJobReject
) -> VideoJobLoad | IcLoraJobLoad:
    """The streamed load an over bet runs: the reject's own window and figures."""
    if feature == "iclora":
        return IcLoraJobLoad(
            mode="streaming_models_loading",
            seq=reject.seq,
            full_gib=reject.full_gib,
            stream_gib=reject.stream_gib,
            chunk_pixel_frames=reject.chunk_pixel_frames,
        )
    return VideoJobLoad(
        mode="streaming_models_loading",
        seq=reject.seq,
        full_gib=reject.full_gib,
        stream_gib=reject.stream_gib,
        chunk_pixel_frames=reject.chunk_pixel_frames,
    )


class DroppedCellFieldsPayload(TypedDict):
    reason: LocalUnavailableReason
    message: str
    over_gib: float | None
    over_ram_gb: int | None
    shortfall_kind: ShortfallKind | None
    readout: BudgetReadout | None
    unmeasured_basis: UnmeasuredBasis | None


def dropped_cell_fields(
    reject: VideoJobReject,
    evidence: PriceEvidence,
    job_filter: LocalJobFilter,
    *,
    message: str,
) -> DroppedCellFieldsPayload:
    """Refusal copy and figures for one withheld cell.

    Figures come from one ``risk_for_reject`` call. When it declines (hard
    wall, or no compared exact bytes) the per-figure helpers still name an
    integer-fallback total binding, so the fallback keeps those rows
    byte-identical.
    """
    risk = risk_for_reject(reject, job_filter)
    return {
        "reason": reject.reason,
        "message": message,
        "over_gib": (risk.over_gib if risk is not None else reject_over_gib(reject, job_filter)),
        "over_ram_gb": (
            risk.over_ram_gb if risk is not None else reject_over_ram_gb(reject, job_filter)
        ),
        "shortfall_kind": (
            risk.shortfall_kind
            if risk is not None
            else reject_shortfall_kind(reject, job_filter)
        ),
        "readout": budget_readout_for_reject(reject, job_filter),
        "unmeasured_basis": (evidence.basis if isinstance(evidence, Extrapolated) else None),
    }


def warned_cell_parts(
    admitted: Admitted, job_filter: LocalJobFilter
) -> tuple[WarnedCellTier, float | None, str, BudgetReadout | None] | None:
    """Warning tier, headroom, and tooltip for a kept cell, if it warns.

    Single home for warned-cell emission (base cells and aspect cells): a bet
    the mode admitted (over or unmeasured) carries its own line and readout;
    tight cells carry available-minus-ask headroom. None when the cell fits
    cleanly.
    """
    risk = admitted.risk
    if risk is not None:
        return (risk.tier, None, risk.message, risk.readout)
    if not admitted.tight:
        return None
    load = admitted.load
    headroom = load_headroom_gib(load, job_filter)
    assert headroom is not None  # a tight fit implies compared figures exist
    available = headroom + load.stream_gib
    darwin = job_filter.profile.system == "Darwin"
    return (
        "tight",
        headroom,
        tight_fit_message(load.stream_gib, available, darwin=darwin),
        tight_readout(load.stream_gib, available),
    )

