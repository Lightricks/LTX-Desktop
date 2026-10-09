"""Queued Home IC-LoRA recipe. Runs the catalog IC-LoRA path for one recipe row."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, cast, get_args

from pydantic import ValidationError

from api_types import (
    IcLoraRecipeInputs,
    IcLoraRecipeStoredParams,
    IcLoraGenerateCancelledResponse,
    IcLoraGenerateRequest,
    IcLoraImageInput,
    LTXLocalModelId,
    LTXVideoGenAspectRatio,
    TargetResolution,
)
from _routes._errors import HTTPError
from runtime_config.ic_lora_local_envelope import fit_ic_lora_stage1
from runtime_config.ltx_capabilities import budget_pixels, local_caps
from services.media_probe import video_fps, video_frame_count
from services.features.video.alpha_cutout import (
    CUTOUT_OUTPUT_PLANS,
    bake_alpha_cutout,
    bake_alpha_gif_if_possible,
    require_cutout_outputs,
)
from services.features.video.cutout_alignment import (
    conform_matte_to_clip,
    fit_clip_to_canvas,
    prepare_clip_for_model,
)
from services.features.video.image_to_video import (
    find_closest_aspect_ratio_option,
    parse_aspect_ratio,
)
from services.features.video.job_common import (
    AssetLookup,
    capability_failed_from_http,
    model_from_spec,
    require_existing_asset,
    resolve_generation_seed,
    resolve_job_local_model_id,
)
from services.features.video.text_to_video import (
    VIDEO_OUTPUT_PLAN,
    require_supported_contract_version,
    require_video_output,
)
from services.ffmpeg import FFMPEG_PROTOCOL_WHITELIST, run_ffmpeg
from services.generation_interrupt import GenerationCancelledError, is_cancel_exception
from services.generation_queue.types import OutputAllocation, OutputPlan
from services.records import CapabilityFailedError, GenerationRecord, MediaError

_ASPECT_MATCH_TOLERANCE = 0.01


class ReservedIcLoraGenerator(Protocol):
    def generate_local_reserved(
        self,
        req: IcLoraGenerateRequest,
        *,
        generation_id: str,
        output_path: Path,
        local_model_id: LTXLocalModelId,
        seed: int,
        explore_generation: bool = True,
        explicit_generation_seed: bool = False,
    ) -> object: ...


def center_crop_box(
    width: int, height: int, aspect: str
) -> tuple[int, int, int, int] | None:
    """Even crop to ``aspect``, or None when the source is already that ratio."""
    target = parse_aspect_ratio(aspect)
    if target is None or height <= 0:
        return None
    current = width / height
    if abs(current - target) / target < _ASPECT_MATCH_TOLERANCE:
        return None
    if current > target:
        crop_w = int(height * target)
        crop_w -= crop_w % 2
        crop_h = height - height % 2
        x = (width - crop_w) // 2
        y = 0
    else:
        crop_h = int(width / target)
        crop_h -= crop_h % 2
        crop_w = width - width % 2
        x = 0
        y = (height - crop_h) // 2
    x -= x % 2
    y -= y % 2
    if crop_w < 2 or crop_h < 2 or (crop_w == width and crop_h == height):
        return None
    return crop_w, crop_h, x, y


def ic_lora_video_filters(
    box: tuple[int, int, int, int] | None, output_fps: float | None
) -> str | None:
    """ffmpeg -vf chain. ``fps`` picks the nearest source frame, so the file the
    catalog path reads is already at the selected rate."""
    filters: list[str] = []
    if output_fps is not None:
        filters.append(f"fps={output_fps:g}")
    if box is not None:
        crop_w, crop_h, x, y = box
        filters.append(f"crop={crop_w}:{crop_h}:{x}:{y}")
    if not filters:
        return None
    return ",".join(filters)


def center_crop_video(
    src: Path,
    dest: Path,
    width: int,
    height: int,
    aspect: str,
    *,
    keep_audio: bool,
    output_fps: float | None = None,
    crop: bool = True,
) -> Path:
    box = center_crop_box(width, height, aspect) if crop else None
    video_filter = ic_lora_video_filters(box, output_fps)
    if video_filter is None:
        return src
    args = [
        "-y",
        "-protocol_whitelist",
        FFMPEG_PROTOCOL_WHITELIST,
        "-i",
        str(src),
        "-vf",
        video_filter,
        "-pix_fmt",
        "yuv420p",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_range",
        "tv",
    ]
    # Re-encode. Stream-copy fails when the source audio (WebM Vorbis/Opus) cannot
    # live in the mp4 crop.
    args += ["-c:a", "aac"] if keep_audio else ["-an"]
    args += ["-protocol_whitelist", FFMPEG_PROTOCOL_WHITELIST, str(dest)]
    run_ffmpeg(args)
    return dest


class IcLoraRecipeExecutor:
    def __init__(
        self,
        ic_lora: ReservedIcLoraGenerator,
        assets: AssetLookup,
        models_dir: Callable[[], Path],
        *,
        catalog_id: str,
        cutout: bool = False,
    ) -> None:
        self._ic_lora = ic_lora
        self._assets = assets
        self._models_dir = models_dir
        self._catalog_id = catalog_id
        self._cutout = cutout

    def validate_params(
        self, spec: Mapping[str, object], *, contract_version: int
    ) -> None:
        require_supported_contract_version(contract_version)
        try:
            model_from_spec(spec, "params", IcLoraRecipeStoredParams)
            model_from_spec(spec, "inputs", IcLoraRecipeInputs)
        except (ValidationError, ValueError) as exc:
            raise CapabilityFailedError(
                str(exc), code="INVALID_GENERATION_SPEC"
            ) from exc

    def _reference_image_path(self, inputs: IcLoraRecipeInputs) -> str | None:
        if inputs.image is None:
            return None
        image = require_existing_asset(
            self._assets,
            inputs.image.assetId,
            media_kind="image",
            unavailable="reference image is no longer available",
        )
        return image.path

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]:
        self.validate_params(
            generation.spec, contract_version=generation.contract_version
        )
        return CUTOUT_OUTPUT_PLANS if self._cutout else (VIDEO_OUTPUT_PLAN,)

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
    ) -> None:
        self.validate_params(
            generation.spec, contract_version=generation.contract_version
        )
        params = model_from_spec(generation.spec, "params", IcLoraRecipeStoredParams)
        inputs = model_from_spec(generation.spec, "inputs", IcLoraRecipeInputs)
        video = require_existing_asset(
            self._assets,
            inputs.video.assetId,
            media_kind="video",
            unavailable="video is no longer available",
        )
        if video.metadata.mediaType != "video":
            raise CapabilityFailedError(
                "INVALID_VIDEO_ASSET", code="INVALID_GENERATION_SPEC"
            )
        reference_image_path = self._reference_image_path(inputs)
        meta = video.metadata.metadata
        local_model_id = resolve_job_local_model_id(self._models_dir(), params.model)
        seed = resolve_generation_seed(params.seed)
        # A cutout recipe keeps the model matte as the second output and bakes the
        # WebM with alpha and a GIF from it. Any other recipe writes its one output directly.
        cutout_dest: Path | None = None
        gif_dest: Path | None = None
        if self._cutout:
            cutout, matte, gif = require_cutout_outputs(outputs)
            cutout_dest = Path(cutout.dest_path)
            gif_dest = Path(gif.dest_path)
            dest = Path(matte.dest_path)
        else:
            dest = Path(require_video_output(outputs).dest_path)
        source = Path(video.path)
        cropped: Path | None = None
        padded: Path | None = None
        try:
            aspect = cast(
                LTXVideoGenAspectRatio,
                find_closest_aspect_ratio_option(
                    meta.width,
                    meta.height,
                    get_args(LTXVideoGenAspectRatio),
                ),
            )
            cropped = dest.with_name(f"{dest.stem}-crop.mp4")
            source_fps = meta.fps if meta.fps is not None else video_fps(Path(video.path))
            # A selected rate at or above the source is a no-op. 23.976 labelled
            # 24 must not be resampled up.
            output_fps = params.fps if params.fps + 1e-3 < source_fps else None
            source = center_crop_video(
                Path(video.path),
                cropped,
                meta.width,
                meta.height,
                aspect,
                keep_audio=params.audioMode == "source" and meta.audioStreamCount > 0,
                output_fps=output_fps,
                # A cutout keeps the whole frame. The clip is fitted into the canvas below.
                crop=not self._cutout,
            )
            # Inside the try so a resolution with no local cell becomes a capability
            # failure, the same as image-to-video's pixel lookup.
            width, height = budget_pixels(
                local_caps(local_model_id),
                params.resolution,
                aspect,
                mode="video",
            )
            model_input = source
            fit = None
            if cutout_dest is not None:
                # The model denoises one stage canvas, and the pipeline center-crops the clip
                # to fill it. Fit the clip inside that canvas, so the matte covers the same view
                # as `source`. Repeat the last frame, so the 8k+1 snap keeps every frame.
                fit = fit_clip_to_canvas(
                    meta.width, meta.height, *fit_ic_lora_stage1(width, height)
                )
                padded = dest.with_name(f"{dest.stem}-fit.mp4")
                model_input = prepare_clip_for_model(
                    source,
                    padded,
                    fit=fit,
                    keep_audio=params.audioMode == "source" and meta.audioStreamCount > 0,
                )
            request = IcLoraGenerateRequest(
                ic_lora_id=self._catalog_id,
                input_path=str(model_input),
                prompt=params.prompt,
                prompt_provenance=params.promptProvenance,
                conditioning_type="custom",
                resolution=TargetResolution(width=width, height=height),
                audio_mode=params.audioMode,
                lora_strength=params.loras[0].scale,
                variant_id=params.loras[0].variantId,
                # No request frame. The handler uses the catalog pin
                # (-1 for the Layout look still).
                images=(
                    [IcLoraImageInput(path=reference_image_path)]
                    if reference_image_path is not None
                    else []
                ),
            )
            result = self._ic_lora.generate_local_reserved(
                request,
                generation_id=generation.id,
                output_path=dest,
                local_model_id=local_model_id,
                seed=seed,
                explicit_generation_seed=params.seed is not None,
            )
            if isinstance(result, IcLoraGenerateCancelledResponse):
                raise GenerationCancelledError()
            if cutout_dest is not None and fit is not None and gif_dest is not None:
                conform_matte_to_clip(
                    dest, frames=video_frame_count(source), fps=video_fps(dest), fit=fit
                )
                # `source` is the clip the conformed matte is aligned with. Keep it until here.
                bake_alpha_cutout(source, dest, cutout_dest)
                bake_alpha_gif_if_possible(source, dest, gif_dest)
        except Exception as exc:
            if is_cancel_exception(exc):
                raise GenerationCancelledError() from exc
            if isinstance(exc, KeyError):
                raise CapabilityFailedError(
                    "INVALID_LOCAL_RESOLUTION", code="INVALID_LOCAL_RESOLUTION"
                ) from exc
            if isinstance(exc, HTTPError):
                raise capability_failed_from_http(exc) from exc
            if isinstance(exc, CapabilityFailedError):
                raise
            if isinstance(exc, MediaError):
                raise CapabilityFailedError(str(exc)) from exc
            if isinstance(exc, subprocess.CalledProcessError):
                raise CapabilityFailedError("crop failed") from exc
            raise
        finally:
            # Unlink the temp path itself. `source` is only reassigned after a
            # successful crop, so a failed ffmpeg would otherwise leave the file.
            if cropped is not None:
                cropped.unlink(missing_ok=True)
            if padded is not None:
                padded.unlink(missing_ok=True)
