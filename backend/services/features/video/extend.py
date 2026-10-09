from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from api_types import ExtendInputs, ExtendParams, LTXLocalModelId
from handlers.video_resolution import (
    correct_frame_count,
    generation_letterbox,
    read_source_metadata,
    resolve_home_edit_size,
)
from runtime_config.video_job_budget import (
    VIDEO_JOB_TOO_LARGE,
    VIDEO_JOB_TOO_LARGE_MESSAGE,
)
from services.features.video.edit_executor import VideoEditExecutor
from services.features.video.job_common import model_from_spec
from services.records import AssetRecord, CapabilityFailedError, GenerationRecord
from services.retake_pipeline.window import (
    MAX_INPUT_FRAMES,
    ExtendEncodeWindow,
    duration_to_extend_frames,
    window_for_extend,
)
from services.retake_pipeline.extend_stitch import stitch_extend_output
from state.app_state_types import RetakePipelineState

FEATURE = "extend"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PreparedExtend:
    params: ExtendParams
    video: AssetRecord
    pipeline_state: RetakePipelineState
    seed: int
    fps: float
    content_width: int
    content_height: int
    canvas_width: int
    canvas_height: int
    window: ExtendEncodeWindow
    extend_frames: int
    dest_path: Path
    windowed_path: str


class ExtendExecutor(VideoEditExecutor[ExtendParams, ExtendInputs, _PreparedExtend]):
    """Queued Home/Remote extend. Same zero-pad as GenSpace ``POST /api/extend``."""

    capability = "extend"
    unsupported_code = "UNSUPPORTED_EXTEND"
    unsupported_message = "Extend is not supported for the selected LTX model."
    heartbeat_label = "extend inference"

    def parse_params(self, spec: Mapping[str, object]) -> ExtendParams:
        return model_from_spec(spec, "params", ExtendParams)

    def parse_inputs(self, spec: Mapping[str, object]) -> ExtendInputs:
        return model_from_spec(spec, "inputs", ExtendInputs)

    def prepare(
        self,
        generation: GenerationRecord,
        params: ExtendParams,
        video: AssetRecord,
        local_model_id: LTXLocalModelId,
        seed: int,
        dest: Path,
        windowed_path: str,
    ) -> _PreparedExtend:
        fps, source_width, source_height, source_frames = read_source_metadata(
            video.path
        )
        # Min-length only. Window/stitch math uses the raw count so leftover
        # frames after the 8k+1 snap stay in the original half of the concat.
        correct_frame_count(source_frames)
        extend_frames = duration_to_extend_frames(params.duration, fps)
        requested = params.resolution
        content_width, content_height = resolve_home_edit_size(
            None if requested is None else requested.width,
            None if requested is None else requested.height,
            source_width,
            source_height,
        )
        box = generation_letterbox(content_width, content_height)
        # A long 1080p source fills the 505-frame window and 422s on 31 GB.
        # Use the longest window that still loads. A new section that is
        # already longer than that cannot be saved by more context.
        frame_cap = self._pipelines.edit_encode_frame_cap(box.canvas_width, box.canvas_height)
        if frame_cap < MAX_INPUT_FRAMES and extend_frames >= frame_cap:
            raise CapabilityFailedError(
                VIDEO_JOB_TOO_LARGE_MESSAGE, code=VIDEO_JOB_TOO_LARGE
            )
        try:
            window = window_for_extend(
                source_frames=source_frames,
                extend_frames=extend_frames,
                mode=params.mode,
                max_input_frames=frame_cap,
            )
        except ValueError as exc:
            raise CapabilityFailedError(str(exc), code="INVALID_GENERATION_SPEC") from exc
        if (
            abs(content_width - source_width) > 31
            or abs(content_height - source_height) > 31
        ):
            logger.info(
                "Downscaling source video from %dx%d to %dx%d; queued extend supports up to 1080p",
                source_width,
                source_height,
                content_width,
                content_height,
            )
        logger.info(
            "Extend %s started (%s, source %dx%d → %dx%d canvas %dx%d, %d source frames @ %g fps, %s +%d, encode %d, seed=%d)",
            generation.id,
            local_model_id,
            source_width,
            source_height,
            content_width,
            content_height,
            box.canvas_width,
            box.canvas_height,
            source_frames,
            fps,
            params.mode,
            extend_frames,
            window.context_frames,
            seed,
        )
        return _PreparedExtend(
            params=params,
            video=video,
            pipeline_state=self._pipelines.load_retake_pipeline(
                distilled=True,
                ltx_model_id=local_model_id,
                width=box.canvas_width,
                height=box.canvas_height,
                frames=window.context_frames + extend_frames,
                queued_home=True,
            ),
            seed=seed,
            fps=fps,
            content_width=content_width,
            content_height=content_height,
            canvas_width=box.canvas_width,
            canvas_height=box.canvas_height,
            window=window,
            extend_frames=extend_frames,
            dest_path=dest,
            windowed_path=windowed_path,
        )

    def generate(self, prepared: _PreparedExtend) -> None:
        prepared.pipeline_state.pipeline.extend(
            video_path=prepared.video.path,
            prompt=prepared.params.prompt,
            extend_frames=prepared.extend_frames,
            mode=prepared.params.mode,
            seed=prepared.seed,
            output_path=prepared.windowed_path,
            negative_prompt=self._default_negative_prompt,
            regenerate_audio=True,
            enhance_prompt=False,
            distilled=True,
            target_width=prepared.canvas_width,
            target_height=prepared.canvas_height,
            content_width=prepared.content_width,
            content_height=prepared.content_height,
            target_frames=prepared.window.context_frames,
            encode_start_time=prepared.window.start_time(prepared.fps),
            encode_max_duration=prepared.window.max_duration(prepared.fps),
        )

    def stitch(self, prepared: _PreparedExtend) -> None:
        stitch_extend_output(
            source_path=prepared.video.path,
            windowed_path=prepared.windowed_path,
            dest_path=str(prepared.dest_path),
            window=prepared.window,
            fps=prepared.fps,
            mode=prepared.params.mode,
            extend_frames=prepared.extend_frames,
            target_width=prepared.content_width,
            target_height=prepared.content_height,
        )
