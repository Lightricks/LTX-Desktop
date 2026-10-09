from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from api_types import (
    GenerateVideoCancelledResponse,
    GenerateVideoRequest,
    ImageToVideoInputs,
    ImageToVideoParams,
    ImageToVideoRecipeParams,
    LoraEntry,
)
from _routes._errors import HTTPError
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
    require_video_output,
    require_supported_contract_version,
)
from runtime_config.model_download_specs import local_aspect_ratios
from services.generation_interrupt import GenerationCancelledError, is_cancel_exception
from services.generation_queue.types import OutputAllocation, OutputPlan
from services.records import AssetRecord, GenerationRecord

FEATURE = "image-to-video"
_INPUT_ASSET_UNAVAILABLE = "image input is no longer available"


class AssetResolver(Protocol):
    def get_asset(self, asset_id: str) -> AssetRecord | None: ...


class RebindableAssetResolver:
    def __init__(self, assets: AssetResolver) -> None:
        self._assets = assets

    def bind(self, assets: AssetResolver) -> None:
        self._assets = assets

    def get_asset(self, asset_id: str) -> AssetRecord | None:
        return self._assets.get_asset(asset_id)


class ImageToVideoExecutor:
    def __init__(
        self,
        video_generation: ReservedVideoGenerator,
        assets: AssetResolver,
        prompt_wrap: Callable[[str], str] | None = None,
        lora_resolver: Callable[[list[LoraEntry]], list[LoraEntry]] | None = None,
    ) -> None:
        self._video_generation = video_generation
        self._assets = assets
        # Recipe-scoped: post-enhancement prompt scaffold. None for plain i2v.
        self._prompt_wrap = prompt_wrap
        # Recipe-scoped: fills the LoRA ref from its catalog id at execution. None for plain i2v.
        self._lora_resolver = lora_resolver

    def validate_params(
        self, spec: Mapping[str, object], *, contract_version: int
    ) -> None:
        require_supported_contract_version(contract_version)
        self._params_from_spec(spec)
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
        params = self._params_from_spec(generation.spec)
        inputs = _inputs_from_spec(generation.spec)
        start = self._resolve_image(inputs.startFrame.assetId, slot="startFrame")
        end = (
            None
            if inputs.endFrame is None
            else self._resolve_image(inputs.endFrame.assetId, slot="endFrame")
        )
        request_payload, offering = offering_payload_from_params(params)
        request_payload["imagePath"] = start.path
        if end is not None:
            request_payload["lastImagePath"] = end.path
        output = require_video_output(outputs)
        try:
            # Inside the try so a resolution with no local cell becomes a capability
            # failure, the same as generate_local_reserved's pixel lookup.
            request_payload["aspectRatio"] = _resolve_aspect_ratio(params, start)
            req = GenerateVideoRequest.model_validate(request_payload)
            if self._lora_resolver is not None:
                # Inside the try so a missing/removed adapter becomes a capability failure.
                req = req.model_copy(update={"loras": self._lora_resolver(req.loras)})
            result = self._video_generation.generate_local_reserved(
                req,
                generation_id=generation.id,
                output_path=Path(output.dest_path),
                prompt_wrap=self._prompt_wrap,
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

    def _params_from_spec(
        self, spec: Mapping[str, object]
    ) -> ImageToVideoParams | ImageToVideoRecipeParams:
        if self._lora_resolver is not None:
            return model_from_spec(spec, "params", ImageToVideoRecipeParams)
        return model_from_spec(spec, "params", ImageToVideoParams)

    def _resolve_image(self, asset_id: str, *, slot: str) -> AssetRecord:
        return require_existing_asset(
            self._assets,
            asset_id,
            media_kind="image",
            unavailable=f"{slot} image is no longer available",
        )


def find_closest_aspect_ratio_option(
    width: int, height: int, options: Sequence[str]
) -> str:
    if height <= 0:
        raise ValueError("startFrame height must be greater than 0")
    target = width / height
    closest: str | None = None
    closest_diff = float("inf")
    for option in options:
        ratio = parse_aspect_ratio(option)
        if ratio is None:
            continue
        diff = abs(ratio - target)
        if diff < closest_diff:
            closest_diff = diff
            closest = option
    if closest is None:
        raise ValueError("no valid aspect ratio options")
    return closest


def parse_aspect_ratio(option: str) -> float | None:
    parts = option.split(":")
    if len(parts) != 2:
        return None
    try:
        option_width = float(parts[0])
        option_height = float(parts[1])
    except ValueError:
        return None
    if option_width <= 0 or option_height <= 0:
        return None
    return option_width / option_height


def _resolve_aspect_ratio(
    params: ImageToVideoParams, start: AssetRecord
) -> str:
    if params.aspectRatio != "auto":
        return params.aspectRatio
    metadata = start.metadata
    if metadata.mediaType != "image":
        raise ValueError("startFrame must have image metadata")
    try:
        options = local_aspect_ratios(params.resolution, is_a2v=False)
    except KeyError as exc:
        raise HTTPError(400, "INVALID_LOCAL_RESOLUTION") from exc
    return find_closest_aspect_ratio_option(
        metadata.metadata.width,
        metadata.metadata.height,
        options,
    )


def _inputs_from_spec(spec: Mapping[str, object]) -> ImageToVideoInputs:
    return model_from_spec(spec, "inputs", ImageToVideoInputs)
