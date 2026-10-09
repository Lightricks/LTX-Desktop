from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from api_types import (
    AudioToVideoInputs,
    AudioToVideoParams,
    GenerateVideoCancelledResponse,
    GenerateVideoRequest,
)
from _routes._errors import HTTPError
from services.features.video.image_to_video import (
    AssetResolver,
    find_closest_aspect_ratio_option,
)
from services.features.video.job_common import (
    capability_failed_from_http,
    model_from_spec,
    require_existing_asset,
    resolve_job_local_model_id,
)
from services.features.video.text_to_video import (
    ReservedVideoGenerator,
    VIDEO_OUTPUT_PLAN,
    offering_payload_from_params,
    require_supported_contract_version,
    require_video_output,
)
from runtime_config.model_download_specs import local_aspect_ratios
from services.generation_interrupt import GenerationCancelledError, is_cancel_exception
from services.generation_queue.types import OutputAllocation, OutputPlan
from services.records import AssetRecord, GenerationRecord

FEATURE = "audio-to-video"
# A2V always has audio and optionally a start image, and a path-leaking pipeline
# error does not say which one it choked on, so this stays slot-neutral.
_INPUT_ASSET_UNAVAILABLE = "input media is no longer available"

class AudioToVideoExecutor:
    def __init__(
        self,
        video_generation: ReservedVideoGenerator,
        assets: AssetResolver,
    ) -> None:
        self._video_generation = video_generation
        self._assets = assets

    def validate_params(
        self, spec: Mapping[str, object], *, contract_version: int
    ) -> None:
        require_supported_contract_version(contract_version)
        _params_from_spec(spec)
        _inputs_from_spec(spec)

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
        params = _params_from_spec(generation.spec)
        inputs = _inputs_from_spec(generation.spec)
        audio = self._resolve_audio(inputs.audio.assetId)
        start = (
            None
            if inputs.startFrame is None
            else self._resolve_image(inputs.startFrame.assetId, slot="startFrame")
        )
        payload, offering = offering_payload_from_params(params)
        payload.pop("numFrames", None)
        payload["audioPath"] = audio.path
        payload["audio"] = True
        if start is not None:
            payload["imagePath"] = start.path
            payload["aspectRatio"] = _resolve_aspect_ratio(params, start)
        elif params.aspectRatio == "auto":
            payload["aspectRatio"] = "16:9"
        req = GenerateVideoRequest.model_validate(payload)
        output = require_video_output(outputs)
        try:
            self._video_generation.validate_local_a2v_request(
                params, audio_duration_ms=_audio_duration_ms(audio)
            )
            result = self._video_generation.generate_local_reserved(
                req,
                generation_id=generation.id,
                output_path=Path(output.dest_path),
                a2v_num_frames=params.numFrames,
                a2v_audio_duration_seconds=_audio_duration_ms(audio) / 1000,
                local_model_id=resolve_job_local_model_id(
                    self._video_generation.models_dir, offering
                ),
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

    def _resolve_audio(self, asset_id: str) -> AssetRecord:
        return require_existing_asset(
            self._assets,
            asset_id,
            media_kind="audio",
            unavailable="audio is no longer available",
        )

    def _resolve_image(self, asset_id: str, *, slot: str) -> AssetRecord:
        return require_existing_asset(
            self._assets,
            asset_id,
            media_kind="image",
            unavailable=f"{slot} image is no longer available",
        )


def _audio_duration_ms(asset: AssetRecord) -> int:
    metadata = asset.metadata
    if metadata.mediaType != "audio":
        raise ValueError("audio asset has non-audio metadata")
    return metadata.metadata.durationMs


def _resolve_aspect_ratio(params: AudioToVideoParams, start: AssetRecord) -> str:
    if params.aspectRatio != "auto":
        return params.aspectRatio
    metadata = start.metadata
    if metadata.mediaType != "image":
        raise ValueError("startFrame must have image metadata")
    return find_closest_aspect_ratio_option(
        metadata.metadata.width,
        metadata.metadata.height,
        local_aspect_ratios(params.resolution, is_a2v=True),
    )


def _params_from_spec(spec: Mapping[str, object]) -> AudioToVideoParams:
    return model_from_spec(spec, "params", AudioToVideoParams)


def _inputs_from_spec(spec: Mapping[str, object]) -> AudioToVideoInputs:
    return model_from_spec(spec, "inputs", AudioToVideoInputs)
