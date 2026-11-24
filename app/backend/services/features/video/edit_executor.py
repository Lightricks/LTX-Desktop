"""Template-method base for queued Home/Remote Retake and Extend."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Generic, Literal, Protocol, Self, TypeVar

from api_types import LTXLocalModelId, OfferingId, PromptProvenance
from _routes._errors import HTTPError
from runtime_config.budget_admission import RiskAllowance
from runtime_config.device_calibration import JobEnvelope
from runtime_config.ltx_capabilities import LtxCapabilityFeature
from runtime_config.video_job_budget import (
    LOCAL_GENERATION_RAM_SHORTFALL,
    VIDEO_JOB_TOO_LARGE,
)
from server_utils.heartbeat import log_heartbeat
from services.features.video.job_common import (
    AssetLookup,
    capability_failed_from_http,
    require_capable_local_model,
    require_existing_asset,
    resolve_generation_seed,
    spec_risk_allowance,
)
from services.features.video.text_to_video import (
    VIDEO_OUTPUT_PLAN,
    require_supported_contract_version,
    require_video_output,
)
from services.generation_interrupt import GenerationCancelledError, is_cancel_exception
from services.fast_video_pipeline.nag_config import (
    DEFAULT_SCALE as DEFAULT_NAG_SCALE,
    NagConfig,
    NagDecision,
    decide_nag,
    log_nag_decision,
)
from services.generation_queue.sessions import JobSession
from services.generation_queue.types import ExecutionOutcome, OutputAllocation, OutputPlan
from services.prompt_enhancement.source_video import SourceVideo
from services.records import (
    AssetRecord,
    CapabilityFailedError,
    GenerationRecord,
    MediaError,
)
from state.app_state_types import RetakePipelineState

logger = logging.getLogger(__name__)

# Widest total-cap/free-cap gap the shrink retry bridges. Shrink exists for
# small structural gaps (a Windows desktop holds ~1.7GB ≈ 24 frames at
# 1080p; this allows 2x that, ≈3.5GB): under heavy external pressure the
# salvaged window would run on slivers, so fail with the max named instead
# of generating mush. Single choke point for retake and extend.
MAX_SHRINK_GAP_FRAMES = 48


@dataclass(frozen=True)
class _AfterProbe:
    kind: Literal["shrink", "bet"]
    free_cap: int | None = None


class GpuJobSlot(Protocol):
    def start_generation(
        self, generation_id: str, *, ran: JobEnvelope | None = None
    ) -> None: ...

    def complete_generation(
        self, result: str | list[str] | None = None, *, ran: JobEnvelope | None = None
    ) -> None: ...

    def fail_generation(self, error: str) -> None: ...


class RetakePipelineLoader(Protocol):
    def load_retake_pipeline(
        self,
        *,
        distilled: bool = True,
        ltx_model_id: LTXLocalModelId | None = None,
        width: int | None = None,
        height: int | None = None,
        frames: int | None = None,
        queued_home: bool = False,
        allowance: RiskAllowance | None = None,
    ) -> tuple[RetakePipelineState, JobEnvelope | None]: ...

    def edit_encode_frame_cap(
        self, width: int, height: int, *, total_only: bool = False
    ) -> int: ...

    def high_res_video_enabled(self) -> bool: ...


class TextEncodingPrep(Protocol):
    def prepare_text_encoding(
        self,
        prompt: str,
        enhance_prompt: bool,
        model_id: LTXLocalModelId | None = None,
    ) -> None: ...

    def clear_api_embeddings(self) -> None: ...

    def should_use_local_encoding(
        self, model_id: LTXLocalModelId | None = None
    ) -> bool: ...


class _ResolvedPrompt(Protocol):
    @property
    def prompt(self) -> str: ...


class SourceVideoPromptResolver(Protocol):
    """The app's one auto-enhance policy, fed the job's source video for scene context."""

    def resolve_for_generation(
        self,
        prompt: str,
        *,
        provenance: PromptProvenance,
        generation_seed: int,
        explicit_generation_seed: bool,
        local_model_id: LTXLocalModelId | None,
        explore_generation: bool,
        source_video: SourceVideo | None,
        negative_prompt: str = "",
    ) -> _ResolvedPrompt: ...


class VideoEditParams(Protocol):
    model: OfferingId
    seed: int | None
    prompt: str
    negativePrompt: str
    nagScale: float | None
    nagAlpha: float | None
    nagTau: float | None
    promptProvenance: PromptProvenance

    def model_copy(self, *, update: Mapping[str, object]) -> Self: ...


class _VideoRef(Protocol):
    # Pydantic model fields are descriptors; a Protocol attribute does not match.
    @property
    def assetId(self) -> str: ...


class VideoEditInputs(Protocol):
    @property
    def video(self) -> _VideoRef: ...


class PreparedEdit(Protocol):
    @property
    def ran(self) -> JobEnvelope | None: ...


PParams = TypeVar("PParams", bound=VideoEditParams)
PInputs = TypeVar("PInputs", bound=VideoEditInputs)
PPrepared = TypeVar("PPrepared", bound=PreparedEdit)


class VideoEditExecutor(ABC, Generic[PParams, PInputs, PPrepared]):
    capability: ClassVar[LtxCapabilityFeature]
    unsupported_code: ClassVar[str]
    unsupported_message: ClassVar[str]
    heartbeat_label: ClassVar[str]

    def __init__(
        self,
        pipelines: RetakePipelineLoader,
        assets: AssetLookup,
        text: TextEncodingPrep,
        models_dir: Callable[[], Path],
        default_negative_prompt: str,
        generation: GpuJobSlot,
        prompt_enhancement: SourceVideoPromptResolver,
        default_nag_scale: Callable[[], float] | None = None,
    ) -> None:
        self._pipelines = pipelines
        self._assets = assets
        self._text = text
        self._models_dir = models_dir
        self._default_negative_prompt = default_negative_prompt
        self._generation = generation
        self._prompt_enhancement = prompt_enhancement
        self._default_nag_scale = default_nag_scale or (lambda: DEFAULT_NAG_SCALE)

    def validate_params(
        self, spec: Mapping[str, object], *, contract_version: int
    ) -> None:
        require_supported_contract_version(contract_version)
        self.parse_params(spec)
        self.parse_inputs(spec)

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]:
        self.validate_params(
            generation.spec, contract_version=generation.contract_version
        )
        return (VIDEO_OUTPUT_PLAN,)

    def _load_pipeline(
        self,
        *,
        local_model_id: LTXLocalModelId,
        width: int,
        height: int,
        frames: int,
        allowance: RiskAllowance,
    ) -> tuple[RetakePipelineState, JobEnvelope | None]:
        return self._pipelines.load_retake_pipeline(
            distilled=True,
            ltx_model_id=local_model_id,
            width=width,
            height=height,
            frames=frames,
            queued_home=True,
            allowance=allowance,
        )

    def _after_probe(
        self,
        exc: HTTPError,
        *,
        width: int,
        height: int,
        total_cap: int,
        allowance: RiskAllowance,
    ) -> _AfterProbe:
        """What to do after the no-bet probe refuses.

        Re-raises ``exc`` when that refusal stands. A free gap past
        ``MAX_SHRINK_GAP_FRAMES`` always stands, stored over or not: the guard
        exists so that window does not run. A smaller free dip returns the cap
        to re-window. A RAM shortfall, or a VRAM miss where free is not below
        total, reloads the original window with the bet.
        """
        if exc.code == LOCAL_GENERATION_RAM_SHORTFALL:
            if not allowance.over:
                raise exc
            return _AfterProbe(kind="bet")
        if exc.code != VIDEO_JOB_TOO_LARGE:
            raise exc
        free_cap = self._pipelines.edit_encode_frame_cap(width, height)
        if free_cap >= total_cap:
            if not allowance.over:
                raise exc
            return _AfterProbe(kind="bet")
        if total_cap - free_cap > MAX_SHRINK_GAP_FRAMES:
            logger.info(
                "Edit shrink skipped: free cap %d too far below total cap %d for %dx%d "
                "(gap %d > %d); failing instead of running on slivers",
                free_cap,
                total_cap,
                width,
                height,
                total_cap - free_cap,
                MAX_SHRINK_GAP_FRAMES,
            )
            raise exc
        return _AfterProbe(kind="shrink", free_cap=free_cap)

    def _load_edit_window(
        self,
        *,
        local_model_id: LTXLocalModelId,
        width: int,
        height: int,
        frames: int,
        total_cap: int,
        allowance: RiskAllowance,
        generation_id: str,
        log_name: str,
        rewindow: Callable[[int], int],
    ) -> tuple[RetakePipelineState, JobEnvelope | None]:
        """Probe without the over bet, then shrink or cover what shrink cannot.

        The first load passes ``allowance.without_over()`` so a free-gate refusal
        is visible while any stored unmeasured bet still carries through. A
        re-windowable dip calls ``rewindow`` and loads that frame count with the
        real allowance. A mask ``rewindow`` cannot fit raises there and does not
        fall through to the original window.
        """
        try:
            return self._load_pipeline(
                local_model_id=local_model_id,
                width=width,
                height=height,
                frames=frames,
                allowance=allowance.without_over(),
            )
        except HTTPError as exc:
            decision = self._after_probe(
                exc,
                width=width,
                height=height,
                total_cap=total_cap,
                allowance=allowance,
            )
            if decision.kind == "bet":
                logger.info(
                    "%s %s stored over covers a refusal shrink cannot re-window (%s); loading %d frames",
                    log_name,
                    generation_id,
                    exc.code,
                    frames,
                )
                try:
                    return self._load_pipeline(
                        local_model_id=local_model_id,
                        width=width,
                        height=height,
                        frames=frames,
                        allowance=allowance,
                    )
                except HTTPError as bet_exc:
                    raise bet_exc from exc
            assert decision.free_cap is not None
            return self._load_pipeline(
                local_model_id=local_model_id,
                width=width,
                height=height,
                frames=rewindow(decision.free_cap),
                allowance=allowance,
            )

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
        session: JobSession,
    ) -> ExecutionOutcome | None:
        self.validate_params(
            generation.spec, contract_version=generation.contract_version
        )
        params = self.parse_params(generation.spec)
        inputs = self.parse_inputs(generation.spec)
        video = require_existing_asset(
            self._assets,
            inputs.video.assetId,
            media_kind="video",
            unavailable="video is no longer available",
        )
        try:
            local_model_id = require_capable_local_model(
                self._models_dir(),
                params.model,
                self.capability,
                unsupported_code=self.unsupported_code,
                unsupported_message=self.unsupported_message,
            )
        except HTTPError as exc:
            raise capability_failed_from_http(exc) from exc
        output = require_video_output(outputs)
        seed = resolve_generation_seed(params.seed)
        # The video gives scene context only: the typed prompt keeps the most weight. Runs
        # before the retake pipeline loads and before start_generation, as the enhancer needs
        # the VRAM a resident pipeline holds. The server-side API flag stays off (it cannot see
        # the video), so enhance_prompt=False below is unchanged.
        typed_prompt = params.prompt
        nag_decision = self.decide_nag(
            params,
            local_encoding=self._text.should_use_local_encoding(local_model_id),
        )
        enhanced_prompt = self._prompt_enhancement.resolve_for_generation(
            typed_prompt,
            provenance=params.promptProvenance,
            generation_seed=seed,
            explicit_generation_seed=params.seed is not None,
            local_model_id=local_model_id,
            explore_generation=True,
            source_video=SourceVideo(
                path=video.path,
                replaced_window=self.replaced_window(params),
                writes_blank_prompt=True,
            ),
            negative_prompt=nag_decision.enhancer_negative,
        ).prompt
        params = params.model_copy(update={"prompt": enhanced_prompt})
        try:
            self._text.prepare_text_encoding(
                params.prompt, enhance_prompt=False, model_id=local_model_id
            )
        except HTTPError as exc:
            raise capability_failed_from_http(exc) from exc
        except RuntimeError as exc:
            raise CapabilityFailedError(str(exc)) from exc
        dest = Path(output.dest_path)
        windowed_path = str(dest.with_name(f"{dest.stem}.window{dest.suffix}"))
        try:
            prepared = self.prepare(
                generation,
                params,
                video,
                local_model_id,
                seed,
                dest,
                windowed_path,
                allowance=spec_risk_allowance(generation.spec),
            )

            def _infer() -> None:
                try:
                    with log_heartbeat(self.heartbeat_label):
                        self.generate(prepared)
                    self.stitch(prepared)
                finally:
                    Path(windowed_path).unlink(missing_ok=True)

            run_gpu_job(self._generation, generation.id, str(dest), _infer, ran=prepared.ran)
            if not typed_prompt.strip() and enhanced_prompt.strip():
                return ExecutionOutcome(written_prompt=enhanced_prompt)
            # Modes unknown: the retake pipeline does not surface its load/offload
            # decisions. Nulls mean unknown — a follow-up can thread them through.
            return ExecutionOutcome()
        except Exception as exc:
            if is_cancel_exception(exc):
                raise GenerationCancelledError() from exc
            if isinstance(exc, HTTPError):
                raise capability_failed_from_http(exc) from exc
            if isinstance(exc, CapabilityFailedError):
                raise
            if isinstance(exc, MediaError):
                raise CapabilityFailedError(str(exc)) from exc
            raise
        finally:
            self._text.clear_api_embeddings()

    @abstractmethod
    def parse_params(self, spec: Mapping[str, object]) -> PParams: ...

    @abstractmethod
    def parse_inputs(self, spec: Mapping[str, object]) -> PInputs: ...

    def replaced_window(self, params: PParams) -> tuple[float, float] | None:
        """``(start, end)`` seconds of the source video that this job replaces, if it is a part."""
        del params
        return None

    def decide_nag(
        self, params: PParams, *, local_encoding: bool
    ) -> NagDecision:
        return decide_nag(
            params.negativePrompt,
            os.environ,
            scale=params.nagScale,
            alpha=params.nagAlpha,
            tau=params.nagTau,
            default_scale=self._default_nag_scale(),
            local_encoding=local_encoding,
            compiled=False,
        )

    def resolve_nag(
        self, params: PParams, pipeline_state: RetakePipelineState
    ) -> NagConfig | None:
        return log_nag_decision(
            logger,
            self.capability,
            self.decide_nag(
                params,
                local_encoding=(
                    self._text.should_use_local_encoding(pipeline_state.ltx_model_id)
                    and pipeline_state.gemma_root is not None
                ),
            ),
        )

    @abstractmethod
    def prepare(
        self,
        generation: GenerationRecord,
        params: PParams,
        video: AssetRecord,
        local_model_id: LTXLocalModelId,
        seed: int,
        dest: Path,
        windowed_path: str,
        *,
        allowance: RiskAllowance = RiskAllowance.none(),
    ) -> PPrepared: ...

    @abstractmethod
    def generate(self, prepared: PPrepared) -> None: ...

    @abstractmethod
    def stitch(self, prepared: PPrepared) -> None: ...


def run_gpu_job(
    slot: GpuJobSlot,
    generation_id: str,
    result_path: str,
    run: Callable[[], None],
    *,
    ran: JobEnvelope | None = None,
) -> None:
    slot.start_generation(generation_id, ran=ran)
    try:
        run()
    except Exception as exc:
        if not is_cancel_exception(exc):
            slot.fail_generation(str(exc))
        raise
    slot.complete_generation(result_path, ran=ran)
