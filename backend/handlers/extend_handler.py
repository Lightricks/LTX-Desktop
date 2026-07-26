"""Extend API orchestration handler.

Mirrors ``RetakeHandler``'s dual API/local dispatch. Extend appends (``mode="end"``)
or prepends (``mode="start"``) freshly generated frames to a source video. The cloud
``/v1/extend`` endpoint and the local PyTorch wrapper share this entry point.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock

from api_types import (
    ExtendMode,
    ExtendRequest,
    ExtendResponse,
    RetakeCancelledResponse,
    RetakePayloadResponse,
    RetakeVideoResponse,
    TargetResolution,
)
from _routes._errors import HTTPError
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.text_handler import TextHandler
from handlers.video_resolution import (
    TIME_FACTOR,
    correct_frame_count,
    read_source_metadata,
    resolve_target_resolution,
    validate_source_video_path,
)
from runtime_config.runtime_config import RuntimeConfig
from services.ltx_api_client.ltx_api_client import LTXAPIClientError
from services.interfaces import LTXAPIClient
from state.app_state_types import AppState
from state.app_settings import should_video_generate_with_ltx_api

# Cloud caps a single extend at 20s; mirror it locally. Minimum is 2s (cloud min).
_MIN_DURATION = 2.0
_MAX_DURATION = 20.0


class ExtendHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        ltx_api_client: LTXAPIClient,
        config: RuntimeConfig,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        text_handler: TextHandler,
    ) -> None:
        super().__init__(state, lock, config)
        self._ltx_api_client = ltx_api_client
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._text = text_handler

    def run(self, req: ExtendRequest) -> ExtendResponse:
        video_path = req.video_path
        duration = req.duration
        prompt = req.prompt
        mode = req.mode

        if duration < _MIN_DURATION:
            raise HTTPError(400, f"duration must be at least {int(_MIN_DURATION)} seconds")
        if duration > _MAX_DURATION:
            raise HTTPError(400, f"duration must be at most {int(_MAX_DURATION)} seconds")

        video_file = validate_source_video_path(video_path)

        if should_video_generate_with_ltx_api(
            force_api_generations=self.config.force_api_generations,
            settings=self.state.app_settings,
        ):
            # The cloud preserves source resolution (no resolution param); resolution
            # selection is local-only.
            return self._run_api_extend(video_file=video_file, duration=duration, prompt=prompt, mode=mode)

        return self._run_local_extend(
            video_file=video_file, duration=duration, prompt=prompt, mode=mode, resolution=req.resolution
        )

    def _run_api_extend(
        self,
        *,
        video_file: Path,
        duration: float,
        prompt: str,
        mode: ExtendMode,
    ) -> ExtendResponse:
        api_key = self.state.app_settings.ltx_api_key
        if not api_key:
            raise HTTPError(400, "LTX API key not configured. Set it in Settings.")

        with self._generation.reserved_generation_start():

            # Drive the generation state machine so the result is recoverable via
            # /api/generation/progress if the page unmounts mid-generation (mirrors retake).
            try:
                self._generation.start_api_generation(uuid.uuid4().hex[:8])
            except RuntimeError as exc:
                # Lost a race with a concurrent generation between the check above and here;
                # surface the same 409 as the guard, not a bare 500 (the running generation
                # owns the state now, so don't fail_generation).
                raise HTTPError(409, str(exc)) from exc
            try:
                self._generation.update_progress("inference", 55, None, None)
                result = self._ltx_api_client.extend(
                    api_key=api_key,
                    video_path=str(video_file),
                    duration=duration,
                    prompt=prompt,
                    mode=mode,
                )

                if result.video_bytes is not None:
                    self._generation.update_progress("downloading_output", 85, None, None)
                    output = self.config.outputs_dir / f"extend_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
                    try:
                        with open(output, "wb") as out:
                            out.write(result.video_bytes)
                        self._generation.update_progress("complete", 100, None, None)
                        self._generation.complete_generation(str(output))
                    except Exception:
                        output.unlink(missing_ok=True)  # don't strand a half-written .mp4
                        raise
                    return RetakeVideoResponse(status="complete", video_path=str(output))

                if result.result_payload is not None:
                    self._generation.update_progress("complete", 100, None, None)
                    self._generation.complete_generation(None)
                    return RetakePayloadResponse(status="complete", result=result.result_payload)

                raise HTTPError(500, "Extend API returned no result")
            except LTXAPIClientError as exc:
                self._generation.fail_generation(exc.detail)
                raise HTTPError(exc.status_code, exc.detail) from exc
            except HTTPError as exc:
                self._generation.fail_generation(exc.detail)
                raise
            except Exception as exc:
                self._generation.fail_generation(str(exc))
                raise

    def _run_local_extend(
        self,
        *,
        video_file: Path,
        duration: float,
        prompt: str,
        mode: ExtendMode,
        resolution: TargetResolution | None,
    ) -> ExtendResponse:
        with self._generation.reserved_generation_start():

            fps, source_width, source_height, source_frames = read_source_metadata(str(video_file))
            target_frames = correct_frame_count(source_frames)
            extend_frames = self._duration_to_extend_frames(duration, fps)
            target_width, target_height = resolve_target_resolution(resolution, source_width, source_height)

            generation_id = uuid.uuid4().hex[:8]
            seed = self._resolve_seed()
            output_path = self.config.outputs_dir / f"extend_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{generation_id}.mp4"

            lease = None
            try:
                lease = self._generation.acquire_local_metal_lease(
                    generation_id=generation_id,
                    workload="extend",
                    reason="Torch local extend generation",
                )
                try:
                    self._text.prepare_text_encoding(prompt, enhance_prompt=False)
                except RuntimeError as exc:
                    raise HTTPError(400, str(exc)) from exc
                pipeline_state = self._pipelines.load_retake_pipeline(distilled=True)
                self._generation.start_generation(generation_id)
                self._generation.update_progress("loading_model", 5, 0, 1)
                self._generation.update_progress("inference", 15, 0, 1)

                pipeline_state.pipeline.extend(
                    video_path=str(video_file),
                    prompt=prompt,
                    extend_frames=extend_frames,
                    mode=mode,
                    seed=seed,
                    output_path=str(output_path),
                    negative_prompt=self.config.default_negative_prompt,
                    regenerate_audio=True,
                    enhance_prompt=False,
                    distilled=True,
                    target_width=target_width,
                    target_height=target_height,
                    target_frames=target_frames,
                )

                if self._generation.is_generation_cancelled():
                    output_path.unlink(missing_ok=True)
                    raise RuntimeError("Generation was cancelled")

                self._generation.update_progress("complete", 100, 1, 1)
                self._generation.complete_generation(str(output_path))
                return RetakeVideoResponse(status="complete", video_path=str(output_path))
            except HTTPError:
                self._generation.fail_generation("Extend generation failed")
                raise
            except Exception as exc:
                self._generation.fail_generation(str(exc))
                if "cancelled" in str(exc).lower():
                    return RetakeCancelledResponse(status="cancelled")
                raise HTTPError(500, f"Generation error: {exc}") from exc
            finally:
                self._text.clear_api_embeddings()
                if lease is not None:
                    self._pipelines.cleanup_runtime_caches()
                    lease.close()

    @staticmethod
    def _duration_to_extend_frames(duration: float, fps: float) -> int:
        # Snap up to the nearest multiple of TIME_FACTOR so the output stays 8k+1: the
        # source is 8k+1, so adding a multiple of 8 keeps the padded latent integer-sized.
        frames = round(duration * fps)
        return ((frames + TIME_FACTOR - 1) // TIME_FACTOR) * TIME_FACTOR
