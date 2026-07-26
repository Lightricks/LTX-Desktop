"""Retake API orchestration handler."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock

from api_types import (
    RetakeCancelledResponse,
    RetakeMode,
    RetakePayloadResponse,
    RetakeRequest,
    RetakeResponse,
    RetakeVideoResponse,
    TargetResolution,
)
from _routes._errors import HTTPError
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from handlers.text_handler import TextHandler
from handlers.video_resolution import (
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


class RetakeHandler(StateHandlerBase):
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

    def run(self, req: RetakeRequest) -> RetakeResponse:
        video_path = req.video_path
        start_time = req.start_time
        duration = req.duration
        prompt = req.prompt
        mode = req.mode

        if duration < 2:
            raise HTTPError(400, "duration must be at least 2 seconds")

        video_file = validate_source_video_path(video_path)

        if should_video_generate_with_ltx_api(
            force_api_generations=self.config.force_api_generations,
            settings=self.state.app_settings,
        ):
            return self._run_api_retake(
                video_file=video_file,
                start_time=start_time,
                duration=duration,
                prompt=prompt,
                mode=mode,
            )

        return self._run_local_retake(
            video_file=video_file,
            start_time=start_time,
            duration=duration,
            prompt=prompt,
            mode=mode,
            resolution=req.resolution,
        )

    def _run_api_retake(
        self,
        *,
        video_file: Path,
        start_time: float,
        duration: float,
        prompt: str,
        mode: RetakeMode,
    ) -> RetakeResponse:
        api_key = self.state.app_settings.ltx_api_key
        if not api_key:
            raise HTTPError(400, "LTX API key not configured. Set it in Settings.")

        with self._generation.reserved_generation_start():

            # Drive the generation state machine so the result is recoverable via
            # /api/generation/progress if the page unmounts mid-generation (mirrors the
            # forced-API video path). Without this, /progress stays idle and the finished
            # retake is stranded in outputs/.
            try:
                self._generation.start_api_generation(uuid.uuid4().hex[:8])
            except RuntimeError as exc:
                # A concurrent generation started between the line-97 check and here; surface
                # the same 409 as the guard above, not a bare 500 (don't fail_generation — the
                # running generation we lost the race to owns the state now).
                raise HTTPError(409, str(exc)) from exc
            try:
                self._generation.update_progress("inference", 55, None, None)
                result = self._ltx_api_client.retake(
                    api_key=api_key,
                    video_path=str(video_file),
                    start_time=start_time,
                    duration=duration,
                    prompt=prompt,
                    mode=mode,
                )

                if result.video_bytes is not None:
                    self._generation.update_progress("downloading_output", 85, None, None)
                    output = self.config.outputs_dir / f"retake_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
                    with open(output, "wb") as out:
                        out.write(result.video_bytes)
                    self._generation.update_progress("complete", 100, None, None)
                    self._generation.complete_generation(str(output))
                    return RetakeVideoResponse(status="complete", video_path=str(output))

                if result.result_payload is not None:
                    # Success, but a remote payload with no local file: complete the
                    # generation with no recoverable artifact (the caller already has the
                    # payload synchronously). Not fail_generation — the retake succeeded.
                    self._generation.update_progress("complete", 100, None, None)
                    self._generation.complete_generation(None)
                    return RetakePayloadResponse(status="complete", result=result.result_payload)

                raise HTTPError(500, "Retake API returned no result")
            except LTXAPIClientError as exc:
                self._generation.fail_generation(exc.detail)
                raise HTTPError(exc.status_code, exc.detail) from exc
            except HTTPError as exc:
                self._generation.fail_generation(exc.detail)
                raise
            except Exception as exc:
                self._generation.fail_generation(str(exc))
                raise

    def _run_local_retake(
        self,
        *,
        video_file: Path,
        start_time: float,
        duration: float,
        prompt: str,
        mode: RetakeMode,
        resolution: TargetResolution | None,
    ) -> RetakeResponse:
        with self._generation.reserved_generation_start():

            fps, source_width, source_height, source_frames = read_source_metadata(str(video_file))
            target_frames = correct_frame_count(source_frames)
            # The clip is trimmed to target_frames; clamp the selection to the corrected length
            # so region_end / the temporal mask can't reference frames past the latent.
            corrected_duration = target_frames / fps
            end_time = min(start_time + duration, corrected_duration)
            if start_time >= end_time:
                raise HTTPError(400, "Selection is outside the usable video range")

            target_width, target_height = resolve_target_resolution(resolution, source_width, source_height)

            generation_id = uuid.uuid4().hex[:8]
            seed = self._resolve_seed()
            output_path = self.config.outputs_dir / f"retake_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{generation_id}.mp4"
            regenerate_video, regenerate_audio = self._resolve_retake_mode(mode)

            lease = None
            try:
                lease = self._generation.acquire_local_metal_lease(
                    generation_id=generation_id,
                    workload="retake",
                    reason="Torch local retake generation",
                )
                try:
                    self._text.prepare_text_encoding(prompt, enhance_prompt=False)
                except RuntimeError as exc:
                    raise HTTPError(400, str(exc)) from exc
                pipeline_state = self._pipelines.load_retake_pipeline(distilled=True)
                self._generation.start_generation(generation_id)
                self._generation.update_progress("loading_model", 5, 0, 1)
                self._generation.update_progress("inference", 15, 0, 1)

                pipeline_state.pipeline.generate(
                    video_path=str(video_file),
                    prompt=prompt,
                    start_time=start_time,
                    end_time=end_time,
                    seed=seed,
                    output_path=str(output_path),
                    negative_prompt=self.config.default_negative_prompt,
                    num_inference_steps=40,
                    video_guider_params=None,
                    audio_guider_params=None,
                    regenerate_video=regenerate_video,
                    regenerate_audio=regenerate_audio,
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
                self._generation.fail_generation("Retake generation failed")
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
    def _resolve_retake_mode(mode: RetakeMode) -> tuple[bool, bool]:
        if mode == "replace_audio_and_video":
            return True, True
        if mode == "replace_video":
            return True, False
        if mode == "replace_audio":
            return False, True
        raise HTTPError(400, "INVALID_RETAKE_MODE")
