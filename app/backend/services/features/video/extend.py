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
    resolve_requested_home_edit_size,
)
from runtime_config.budget_admission import RiskAllowance
from runtime_config.device_calibration import JobEnvelope
from services.features.video.edit_executor import VideoEditExecutor
from services.features.video.job_common import model_from_spec
from services.fast_video_pipeline.windowed_run import measured_edit
from services.records import AssetRecord, GenerationRecord
from services.retake_pipeline.window import (
    MIN_EXTEND_CONTEXT_FRAMES,
    EditRefusal,
    ExtendEncodeWindow,
    prepare_extend_window,
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
    ran: JobEnvelope | None


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
        *,
        allowance: RiskAllowance = RiskAllowance.none(),
    ) -> _PreparedExtend:
        fps, source_width, source_height, source_frames = read_source_metadata(
            video.path
        )
        # Min-length only. Window/stitch math uses the raw count so leftover
        # frames after the 8k+1 snap stay in the original half of the concat.
        correct_frame_count(source_frames)
        content_width, content_height = resolve_requested_home_edit_size(
            params.resolution,
            source_width,
            source_height,
            high_res=self._pipelines.high_res_video_enabled(),
        )
        box = generation_letterbox(content_width, content_height)
        # A long 1080p source fills the 505-frame window and 422s on 31 GB.
        # Use the longest window that still loads. Total-gated: the resident
        # pipeline is still loaded here, so free would shorten the window
        # before the load gate's evict-retry runs.
        frame_cap = self._pipelines.edit_encode_frame_cap(
            box.canvas_width,
            box.canvas_height,
            total_only=True,
        )
        prepared = prepare_extend_window(
            fps=fps,
            source_frames=source_frames,
            target_width=content_width,
            target_height=content_height,
            frame_cap=frame_cap,
            duration=params.duration,
            mode=params.mode,
        )
        if isinstance(prepared, EditRefusal):
            prepared.raise_capability_error()
        window, extend_frames = prepared.window, prepared.extend_frames
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
        def rewindow(free_cap: int) -> int:
            nonlocal window, extend_frames
            original_frames = window.context_frames + extend_frames
            # Salvage with reduced context: the job was admitted with the
            # full 121-frame floor, so a free dip shrinks the context
            # (far-side-trimmed by the window math) instead of failing.
            # Refuses only when the extension alone exceeds the free cap.
            shrunk = prepare_extend_window(
                fps=fps,
                source_frames=source_frames,
                target_width=content_width,
                target_height=content_height,
                frame_cap=free_cap,
                duration=params.duration,
                mode=params.mode,
                minimum_context_frames=1,
            )
            # Unreachable while MAX_SHRINK_GAP_FRAMES < 120 (a passing first
            # gate leaves extend <= total - 121, which always fits a free cap
            # within 48): kept so a wider future bound still fails cleanly
            # with the max named instead of crashing on the refusal below.
            if isinstance(shrunk, EditRefusal):
                shrunk.raise_capability_error()
            if shrunk.window.context_frames < MIN_EXTEND_CONTEXT_FRAMES:
                logger.warning(
                    "Extend %s running with %d context frames, below the %d-frame coherence floor (unmeasured regime)",
                    generation.id,
                    shrunk.window.context_frames,
                    MIN_EXTEND_CONTEXT_FRAMES,
                )
            logger.info(
                "Extend %s free gate refused the total-fitting %d-frame window; shrinking to free-gated cap %d (encode %d)",
                generation.id,
                original_frames,
                free_cap,
                shrunk.window.context_frames + shrunk.extend_frames,
            )
            window, extend_frames = shrunk.window, shrunk.extend_frames
            return window.context_frames + extend_frames

        pipeline_state, ran = self._load_edit_window(
            local_model_id=local_model_id,
            width=box.canvas_width,
            height=box.canvas_height,
            frames=window.context_frames + extend_frames,
            total_cap=frame_cap,
            allowance=allowance,
            generation_id=generation.id,
            log_name="Extend",
            rewindow=rewindow,
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
            pipeline_state=pipeline_state,
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
            ran=ran,
        )

    def generate(self, prepared: _PreparedExtend) -> None:
        frames = prepared.window.context_frames + prepared.extend_frames
        with measured_edit(prepared.canvas_width, prepared.canvas_height, frames):
            self._generate(prepared)

    def _generate(self, prepared: _PreparedExtend) -> None:
        nag = self.resolve_nag(prepared.params, prepared.pipeline_state)
        prepared.pipeline_state.pipeline.extend(
            video_path=prepared.video.path,
            prompt=prepared.params.prompt,
            extend_frames=prepared.extend_frames,
            mode=prepared.params.mode,
            seed=prepared.seed,
            output_path=prepared.windowed_path,
            negative_prompt=prepared.params.negativePrompt or self._default_negative_prompt,
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
            nag=nag,
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
