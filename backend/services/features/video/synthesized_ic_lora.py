"""Queued Home IC-LoRA recipes whose control clip the job makes (a sheet, a track).

The user supplies pictures, not a clip. A subclass says how to turn them into a control file
and the request that runs on it. This base owns the rest: the spec checks, the output plan,
the cancel and error mapping, and the removal of the control file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from _routes._errors import HTTPError
from api_types import (
    IcLoraGenerateCancelledResponse,
    IcLoraGenerateRequest,
    IcLoraImageInput,
    IcLoraSynthesizedStoredParams,
    TargetResolution,
)
from services.features.ic_lora_recipes import IcLoraRecipe
from services.features.lora_recipes import apply_recipe_prompt_template
from services.features.video.ic_lora_recipe import ReservedIcLoraGenerator
from services.features.video.job_common import (
    UNNAMED_RECIPE,
    AssetLookup,
    capability_failed_from_http,
    denoise_canvas,
    model_from_spec,
    resolve_generation_seed,
    resolve_job_local_model_id,
    spec_risk_allowance,
)
from services.features.video.text_to_video import (
    VIDEO_OUTPUT_PLAN,
    require_supported_contract_version,
    require_video_output,
)
from services.generation_interrupt import GenerationCancelledError, is_cancel_exception
from services.generation_queue.sessions import JobSession
from services.generation_queue.types import (
    ExecutionOutcome,
    OutputAllocation,
    OutputPlan,
)
from services.records import CapabilityFailedError, GenerationRecord

TInputs = TypeVar("TInputs", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class SynthesizedJob:
    request: IcLoraGenerateRequest
    # Automatic enhancement shows the rewrite these images, not only the control file.
    panel_paths: Sequence[str] = ()
    # Set when the job writes its control file at a shared path (such as a cached sheet).
    control_path: Path | None = None


class SynthesizedIcLoraExecutor(ABC, Generic[TInputs]):
    inputs_model: type[TInputs]
    # The name of the control file beside the output. The executor removes it when the job ends.
    control_suffix: str
    # True when the control file is a still image that the pipeline loops for ``duration`` seconds.
    loops_control_file: bool = False

    def __init__(
        self,
        ic_lora: ReservedIcLoraGenerator,
        assets: AssetLookup,
        models_dir: Callable[[], Path],
        *,
        recipe: IcLoraRecipe,
    ) -> None:
        self._ic_lora = ic_lora
        self._assets = assets
        self._models_dir = models_dir
        self._recipe = recipe

    @classmethod
    def generate_request(
        cls,
        recipe: IcLoraRecipe,
        params: IcLoraSynthesizedStoredParams,
        canvas: tuple[int, int],
        *,
        input_path: str | None = None,
        images: list[IcLoraImageInput] | None = None,
    ) -> IcLoraGenerateRequest:
        """The generate request of this recipe. Create-time preflight and the executor both build it."""
        return IcLoraGenerateRequest(
            ic_lora_id=recipe.catalog_id,
            input_path=input_path,
            prompt=apply_recipe_prompt_template(
                recipe, params.prompt.strip() if recipe.prompt_template else params.prompt
            ),
            negative_prompt=params.negativePrompt,
            nag_scale=params.nagScale,
            nag_alpha=params.nagAlpha,
            nag_tau=params.nagTau,
            prompt_provenance=params.promptProvenance,
            conditioning_type="custom",
            resolution=TargetResolution(width=canvas[0], height=canvas[1]),
            audio_mode=params.audioMode,
            lora_strength=params.loras[0].scale,
            variant_id=params.loras[0].variantId,
            loop_fps=int(params.fps) if cls.loops_control_file else None,
            loop_seconds=params.duration if cls.loops_control_file else None,
            images=images or [],
        )

    @abstractmethod
    def prepare(
        self,
        params: IcLoraSynthesizedStoredParams,
        inputs: TInputs,
        canvas: tuple[int, int],
        control: Path,
    ) -> SynthesizedJob:
        """Write the control file at ``control``, sized to ``canvas``, and return the job."""

    def validate_params(
        self, spec: Mapping[str, object], *, contract_version: int
    ) -> None:
        require_supported_contract_version(contract_version)
        try:
            model_from_spec(spec, "params", IcLoraSynthesizedStoredParams)
            model_from_spec(spec, "inputs", self.inputs_model)
        except (ValidationError, ValueError) as exc:
            raise CapabilityFailedError(
                str(exc), code="INVALID_GENERATION_SPEC"
            ) from exc

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]:
        self.validate_params(
            generation.spec, contract_version=generation.contract_version
        )
        return (VIDEO_OUTPUT_PLAN,)

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
        session: JobSession,
    ) -> ExecutionOutcome:
        self.validate_params(
            generation.spec, contract_version=generation.contract_version
        )
        params = model_from_spec(generation.spec, "params", IcLoraSynthesizedStoredParams)
        inputs = model_from_spec(generation.spec, "inputs", self.inputs_model)
        local_model_id = resolve_job_local_model_id(self._models_dir(), params.model)
        dest = Path(require_video_output(outputs).dest_path)
        control = dest.with_name(f"{dest.stem}-{self.control_suffix}")
        job: SynthesizedJob | None = None
        try:
            job = self.prepare(
                params,
                inputs,
                denoise_canvas(local_model_id, params, name=UNNAMED_RECIPE),
                control,
            )
            result = self._ic_lora.generate_local_reserved(
                job.request,
                generation_id=generation.id,
                output_path=dest,
                local_model_id=local_model_id,
                seed=resolve_generation_seed(params.seed),
                explicit_generation_seed=params.seed is not None,
                allowance=spec_risk_allowance(generation.spec),
                panel_paths=job.panel_paths,
            )
        except Exception as exc:
            if is_cancel_exception(exc):
                raise GenerationCancelledError() from exc
            if isinstance(exc, HTTPError):
                raise capability_failed_from_http(exc) from exc
            raise
        finally:
            control.unlink(missing_ok=True)
            if job is not None and job.control_path is not None:
                job.control_path.unlink(missing_ok=True)
        if isinstance(result, IcLoraGenerateCancelledResponse):
            raise GenerationCancelledError()
        return ExecutionOutcome()
