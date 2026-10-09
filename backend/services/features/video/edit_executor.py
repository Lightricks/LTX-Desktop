"""Template-method base for queued Home/Remote Retake and Extend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import ClassVar, Generic, Protocol, TypeVar

from api_types import LTXLocalModelId, OfferingId
from _routes._errors import HTTPError
from runtime_config.ltx_capabilities import LtxCapabilityFeature
from server_utils.heartbeat import log_heartbeat
from services.features.video.job_common import (
    AssetLookup,
    capability_failed_from_http,
    require_capable_local_model,
    require_existing_asset,
    resolve_generation_seed,
)
from services.features.video.text_to_video import (
    VIDEO_OUTPUT_PLAN,
    require_supported_contract_version,
    require_video_output,
)
from services.generation_interrupt import GenerationCancelledError, is_cancel_exception
from services.generation_queue.types import OutputAllocation, OutputPlan
from services.records import (
    AssetRecord,
    CapabilityFailedError,
    GenerationRecord,
    MediaError,
)
from state.app_state_types import RetakePipelineState


class GpuJobSlot(Protocol):
    def start_generation(self, generation_id: str) -> None: ...

    def complete_generation(self, result: str | list[str] | None = None) -> None: ...

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
    ) -> RetakePipelineState: ...

    def edit_encode_frame_cap(self, width: int, height: int) -> int: ...


class TextEncodingPrep(Protocol):
    def prepare_text_encoding(
        self,
        prompt: str,
        enhance_prompt: bool,
        model_id: LTXLocalModelId | None = None,
    ) -> None: ...

    def clear_api_embeddings(self) -> None: ...


class VideoEditParams(Protocol):
    model: OfferingId
    seed: int | None
    prompt: str


class _VideoRef(Protocol):
    # Pydantic model fields are descriptors; a Protocol attribute does not match.
    @property
    def assetId(self) -> str: ...


class VideoEditInputs(Protocol):
    @property
    def video(self) -> _VideoRef: ...


PParams = TypeVar("PParams", bound=VideoEditParams)
PInputs = TypeVar("PInputs", bound=VideoEditInputs)
PPrepared = TypeVar("PPrepared")


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
    ) -> None:
        self._pipelines = pipelines
        self._assets = assets
        self._text = text
        self._models_dir = models_dir
        self._default_negative_prompt = default_negative_prompt
        self._generation = generation

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

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
    ) -> None:
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
            )

            def _infer() -> None:
                try:
                    with log_heartbeat(self.heartbeat_label):
                        self.generate(prepared)
                    self.stitch(prepared)
                finally:
                    Path(windowed_path).unlink(missing_ok=True)

            run_gpu_job(self._generation, generation.id, str(dest), _infer)
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
) -> None:
    slot.start_generation(generation_id)
    try:
        run()
    except Exception as exc:
        if not is_cancel_exception(exc):
            slot.fail_generation(str(exc))
        raise
    slot.complete_generation(result_path)
