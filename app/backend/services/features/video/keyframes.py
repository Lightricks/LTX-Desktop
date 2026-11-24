"""Home Keyframes: several stills at chosen frames guide one local clip."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from api_types import (
    GenerateVideoCancelledResponse,
    GenerateVideoRequest,
    KeyframeRef,
    KeyframesInputs,
    KeyframesParams,
)
from _routes._errors import HTTPError
from keyframe_request import validate_keyframe_frames
from services.features.video.image_to_video import AssetResolver, resolve_image_aspect_ratio
from services.features.video.job_common import (
    capability_failed_from_http,
    model_from_spec,
    require_existing_asset,
    resolve_job_local_model_id,
    spec_pass_shape,
    spec_risk_allowance,
)
from services.features.video.text_to_video import (
    ReservedVideoGenerator,
    VIDEO_OUTPUT_PLAN,
    offering_payload_from_params,
    require_video_output,
    require_supported_contract_version,
)
from services.generation_interrupt import GenerationCancelledError, is_cancel_exception
from services.generation_queue.sessions import JobSession
from services.generation_queue.types import ExecutionOutcome, OutputAllocation, OutputPlan
from services.records import AssetRecord, GenerationRecord

FEATURE = "keyframes"
_INPUT_ASSET_UNAVAILABLE = "keyframe image is no longer available"


class KeyframesExecutor:
    def __init__(self, video_generation: ReservedVideoGenerator, assets: AssetResolver) -> None:
        self._video_generation = video_generation
        self._assets = assets

    def validate_params(self, spec: Mapping[str, object], *, contract_version: int) -> None:
        require_supported_contract_version(contract_version)
        params = model_from_spec(spec, "params", KeyframesParams)
        inputs = model_from_spec(spec, "inputs", KeyframesInputs)
        if params.duration is None:
            raise ValueError("Keyframes cannot be combined with automatic duration")
        error = validate_keyframe_frames(
            [keyframe.frameIndex for keyframe in inputs.keyframes],
            duration=params.duration,
            fps=params.fps,
        )
        if error is not None:
            raise ValueError(error)

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]:
        self.validate_params(generation.spec, contract_version=generation.contract_version)
        return (VIDEO_OUTPUT_PLAN,)

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
        session: JobSession,
    ) -> ExecutionOutcome:
        self.validate_params(generation.spec, contract_version=generation.contract_version)
        params = model_from_spec(generation.spec, "params", KeyframesParams)
        inputs = model_from_spec(generation.spec, "inputs", KeyframesInputs)
        stills = [(keyframe, self._resolve_image(keyframe)) for keyframe in inputs.keyframes]
        request_payload, offering = offering_payload_from_params(params)
        request_payload["keyframes"] = [
            {"imagePath": still.path, "frameIndex": keyframe.frameIndex, "strength": keyframe.strength}
            for keyframe, still in stills
        ]
        output = require_video_output(outputs)
        try:
            # Auto follows the first still on the timeline, the one the prompt enhancer sees.
            _, first_still = min(stills, key=lambda pair: pair[0].frameIndex)
            request_payload["aspectRatio"] = resolve_image_aspect_ratio(
                params.aspectRatio, params.resolution, first_still
            )
            req = GenerateVideoRequest.model_validate(request_payload)
            result = self._video_generation.generate_local_reserved(
                req,
                generation_id=generation.id,
                output_path=Path(output.dest_path),
                prompt_wrap=None,
                local_model_id=resolve_job_local_model_id(
                    self._video_generation.models_dir, offering
                ),
                allowance=spec_risk_allowance(generation.spec),
                pass_shape=spec_pass_shape(generation.spec),
            )
        except Exception as exc:
            if is_cancel_exception(exc):
                raise GenerationCancelledError() from exc
            if isinstance(exc, HTTPError):
                raise capability_failed_from_http(
                    exc, unavailable=_INPUT_ASSET_UNAVAILABLE
                ) from exc
            raise
        if isinstance(result, GenerateVideoCancelledResponse):
            raise GenerationCancelledError()
        return ExecutionOutcome(load_mode=result.load_mode, offload_mode=result.offload_mode)

    def _resolve_image(self, keyframe: KeyframeRef) -> AssetRecord:
        return require_existing_asset(
            self._assets,
            keyframe.assetId,
            media_kind="image",
            unavailable=_INPUT_ASSET_UNAVAILABLE,
        )
