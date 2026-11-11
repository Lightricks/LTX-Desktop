from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from api_types import LTXLocalModelId, RetakeInputs, RetakeParams
from handlers.video_resolution import (
    correct_frame_count,
    generation_letterbox,
    read_source_metadata,
    resolve_requested_home_edit_size,
)
from runtime_config.budget_admission import RiskAllowance
from runtime_config.device_calibration import JobEnvelope
from services.features.video.edit_executor import VideoEditExecutor
from services.features.video.job_common import model_from_spec
from services.fast_video_pipeline.windowed_run import measured_edit
from services.records import AssetRecord, CapabilityFailedError, GenerationRecord
from services.retake_pipeline.retake_mode import (
    InvalidRetakeModeError,
    resolve_retake_mode,
)
from services.retake_pipeline.retake_stitch import stitch_retake_output
from services.retake_pipeline.window import (
    EditRefusal,
    RetakeEncodeWindow,
    feathered_mask_times,
    prepare_retake_window,
)
from state.app_state_types import RetakePipelineState

FEATURE = "retake"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PreparedRetake:
    params: RetakeParams
    video: AssetRecord
    pipeline_state: RetakePipelineState
    seed: int
    fps: float
    content_width: int
    content_height: int
    canvas_width: int
    canvas_height: int
    window: RetakeEncodeWindow
    region_start: float
    region_end: float
    regenerate_video: bool
    regenerate_audio: bool
    dest_path: Path
    windowed_path: str
    ran: JobEnvelope | None


class RetakeExecutor(VideoEditExecutor[RetakeParams, RetakeInputs, _PreparedRetake]):
    """Queued Home/Remote retake. 505-frame window, MASK_DELTA, original-pixel stitch."""

    capability = "retake"
    unsupported_code = "UNSUPPORTED_RETAKE"
    unsupported_message = "Retake is not supported for the selected LTX model."
    heartbeat_label = "retake inference"

    def parse_params(self, spec: Mapping[str, object]) -> RetakeParams:
        return model_from_spec(spec, "params", RetakeParams)

    def parse_inputs(self, spec: Mapping[str, object]) -> RetakeInputs:
        return model_from_spec(spec, "inputs", RetakeInputs)

    def replaced_window(self, params: RetakeParams) -> tuple[float, float]:
        return params.startTime, params.startTime + params.duration

    def prepare(
        self,
        generation: GenerationRecord,
        params: RetakeParams,
        video: AssetRecord,
        local_model_id: LTXLocalModelId,
        seed: int,
        dest: Path,
        windowed_path: str,
        *,
        allowance: RiskAllowance = RiskAllowance.none(),
    ) -> _PreparedRetake:
        try:
            regenerate_video, regenerate_audio = resolve_retake_mode(params.mode)
        except InvalidRetakeModeError as exc:
            raise CapabilityFailedError(str(exc), code="INVALID_GENERATION_SPEC") from exc
        fps, source_width, source_height, source_frames = read_source_metadata(video.path)
        target_frames = correct_frame_count(source_frames)
        content_width, content_height = resolve_requested_home_edit_size(
            params.resolution,
            source_width,
            source_height,
            high_res=self._pipelines.high_res_video_enabled(),
        )
        box = generation_letterbox(content_width, content_height)
        # Cap on total: the resident pipeline is still loaded here, so free
        # would shorten the window before the load gate's evict-retry runs.
        frame_cap = self._pipelines.edit_encode_frame_cap(
            box.canvas_width,
            box.canvas_height,
            total_only=True,
        )
        prepared = prepare_retake_window(
            fps=fps,
            target_frames=target_frames,
            target_width=content_width,
            target_height=content_height,
            frame_cap=frame_cap,
            start_time=params.startTime,
            duration=params.duration,
        )
        if isinstance(prepared, EditRefusal):
            prepared.raise_capability_error()
        window, end_time = prepared.window, prepared.end_time
        region_start, region_end = feathered_mask_times(window, fps)
        if (
            abs(content_width - source_width) > 31
            or abs(content_height - source_height) > 31
        ):
            logger.info(
                "Downscaling source video from %dx%d to %dx%d; queued retake supports up to 1080p (4K with highResVideoToVideo)",
                source_width,
                source_height,
                content_width,
                content_height,
            )
        def rewindow(free_cap: int) -> int:
            nonlocal window, end_time, region_start, region_end
            original_frames = window.encode_frames
            shrunk = prepare_retake_window(
                fps=fps,
                target_frames=target_frames,
                target_width=content_width,
                target_height=content_height,
                frame_cap=free_cap,
                start_time=params.startTime,
                duration=params.duration,
            )
            if isinstance(shrunk, EditRefusal):
                shrunk.raise_capability_error()
            logger.info(
                "Retake %s free gate refused the total-fitting %d-frame window; shrinking to free-gated cap %d (encode %d)",
                generation.id,
                original_frames,
                free_cap,
                shrunk.window.encode_frames,
            )
            window, end_time = shrunk.window, shrunk.end_time
            region_start, region_end = feathered_mask_times(window, fps)
            return window.encode_frames

        pipeline_state, ran = self._load_edit_window(
            local_model_id=local_model_id,
            width=box.canvas_width,
            height=box.canvas_height,
            frames=window.encode_frames,
            total_cap=frame_cap,
            allowance=allowance,
            generation_id=generation.id,
            log_name="Retake",
            rewindow=rewindow,
        )
        logger.info(
            "Retake %s started (%s, source %dx%d → %dx%d canvas %dx%d, %d frames @ %g fps, %s %.2fs–%.2fs, encode %d, seed=%d)",
            generation.id,
            local_model_id,
            source_width,
            source_height,
            content_width,
            content_height,
            box.canvas_width,
            box.canvas_height,
            target_frames,
            fps,
            params.mode,
            params.startTime,
            end_time,
            window.encode_frames,
            seed,
        )
        return _PreparedRetake(
            params=params,
            video=video,
            pipeline_state=pipeline_state,
            seed=seed,
            fps=fps,
            content_width=content_width,
            content_height=content_height,
            canvas_width=box.canvas_width,
            canvas_height=box.canvas_height,
            window=window,
            region_start=region_start,
            region_end=region_end,
            regenerate_video=regenerate_video,
            regenerate_audio=regenerate_audio,
            dest_path=dest,
            windowed_path=windowed_path,
            ran=ran,
        )

    def generate(self, prepared: _PreparedRetake) -> None:
        with measured_edit(prepared.canvas_width, prepared.canvas_height, prepared.window.encode_frames):
            self._generate(prepared)

    def _generate(self, prepared: _PreparedRetake) -> None:
        nag = self.resolve_nag(prepared.params, prepared.pipeline_state)
        prepared.pipeline_state.pipeline.generate(
            video_path=prepared.video.path,
            prompt=prepared.params.prompt,
            start_time=prepared.region_start,
            end_time=prepared.region_end,
            seed=prepared.seed,
            output_path=prepared.windowed_path,
            negative_prompt=prepared.params.negativePrompt or self._default_negative_prompt,
            num_inference_steps=40,
            video_guider_params=None,
            audio_guider_params=None,
            regenerate_video=prepared.regenerate_video,
            regenerate_audio=prepared.regenerate_audio,
            enhance_prompt=False,
            distilled=True,
            target_width=prepared.canvas_width,
            target_height=prepared.canvas_height,
            content_width=prepared.content_width,
            content_height=prepared.content_height,
            target_frames=prepared.window.encode_frames,
            encode_start_time=prepared.window.encode_start_time(prepared.fps),
            encode_max_duration=prepared.window.encode_duration(prepared.fps),
            nag=nag,
        )

    def stitch(self, prepared: _PreparedRetake) -> None:
        stitch_retake_output(
            source_path=prepared.video.path,
            windowed_path=prepared.windowed_path,
            dest_path=str(prepared.dest_path),
            window=prepared.window,
            fps=prepared.fps,
            target_width=prepared.content_width,
            target_height=prepared.content_height,
        )
