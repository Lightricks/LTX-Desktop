"""Queued text-to-image jobs: the local pipeline lane and the fal API lane."""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from _routes._errors import HTTPError
from api_types import TextToImageStoredParams, ZIT_DEFAULT_STEPS
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.image_generation_handler import text_to_image_on_pipeline
from handlers.image_prompt_plan import plan_fal_image_prompt
from handlers.pipelines_handler import PipelinesHandler
from handlers.prompt_enhancement_handler import PromptEnhancementHandler
from server_utils.heartbeat import log_heartbeat
from services.features.image.text_to_image import QueuedImageResult
from services.generation_queue.sessions import JobSession
from services.interfaces import ZitAPIClient
from services.zit_api_client.zit_api_client import ZitImageBlockedError
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


class QueuedImageHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        config: RuntimeConfig,
        zit_api_client: ZitAPIClient,
        prompt_enhancement_handler: PromptEnhancementHandler,
        lease_idle_gpu: Callable[[], AbstractContextManager[bool]],
    ) -> None:
        super().__init__(state, lock, config)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._zit_api_client = zit_api_client
        self._prompt_enhancement = prompt_enhancement_handler
        self._lease_idle_gpu = lease_idle_gpu

    def _gpu_lease_for(self, session: JobSession) -> Callable[[], AbstractContextManager[bool]]:
        """A job that holds the GPU slot (the kill switch runs fal on the local lane) needs no lease."""
        if session.holds_gpu_slot:
            return lambda: nullcontext(True)
        return self._lease_idle_gpu

    def cancel_fal_request(self, request_id: str) -> None:
        """The cancel of the fal lane for a job that a stopped app left at fal."""
        api_key = self.state.app_settings.fal_api_key.strip()
        if not api_key:
            raise RuntimeError("FAL API key is not configured")
        self._zit_api_client.cancel_queued(api_key=api_key, request_id=request_id)

    def generate_queued(
        self,
        params: TextToImageStoredParams,
        *,
        generation_id: str,
        output_path: Path,
        session: JobSession,
    ) -> QueuedImageResult:
        """Run one queued text-to-image job.

        Text-to-image only: the Z-Image edit path is never used. Raises on failure and on
        cancellation, and removes a partial output. The job reports through its session, so
        a job on the fal lane never touches the local generation state.
        """
        explicit_seed = params.seed is not None
        seed = params.seed if params.seed is not None else self._resolve_seed()
        try:
            if params.provider == "fal":
                image_bytes, result = self._via_api(
                    session, params, seed=seed, explicit_seed=explicit_seed
                )
            else:
                image_bytes = self._via_local(
                    session, generation_id, params, seed=seed, explicit_seed=explicit_seed
                )
                result = QueuedImageResult()
            session.raise_if_cancelled()
            output_path.write_bytes(image_bytes)
            session.set_progress("complete", 100, None, None)
            session.complete(str(output_path))
            return result
        except Exception as e:
            output_path.unlink(missing_ok=True)
            session.fail(e.detail if isinstance(e, HTTPError) else str(e))
            raise

    def _via_local(
        self,
        session: JobSession,
        generation_id: str,
        params: TextToImageStoredParams,
        *,
        seed: int,
        explicit_seed: bool,
    ) -> bytes:
        prompt = self._prompt_enhancement.rewrite_image_prompt(
            params.prompt,
            provenance=params.promptProvenance,
            generation_seed=seed,
            explicit_generation_seed=explicit_seed,
            raise_if_cancelled=session.raise_if_cancelled,
        ).prompt
        session.raise_if_cancelled()
        steps = ZIT_DEFAULT_STEPS
        session.set_progress("loading_model", 5, 0, steps)
        pipeline = self._pipelines.load_image_generation_pipeline_to_gpu()
        # The local pipeline owns the global GPU state. Only the local lane runs this path.
        self._generation.start_generation(generation_id)
        session.set_progress("inference", 15, 0, steps)
        with log_heartbeat(f"zit image {params.width}x{params.height}"):
            image = text_to_image_on_pipeline(
                pipeline,
                prompt=prompt,
                width=params.width,
                height=params.height,
                num_inference_steps=steps,
                seed=seed,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _via_api(
        self,
        session: JobSession,
        params: TextToImageStoredParams,
        *,
        seed: int,
        explicit_seed: bool,
    ) -> tuple[bytes, QueuedImageResult]:
        api_key = self.state.app_settings.fal_api_key.strip()
        session.start_api()
        session.set_progress("validating_request", 5, None, None)
        if not api_key:
            raise HTTPError(422, "FAL_API_KEY_NOT_CONFIGURED")
        session.raise_if_cancelled()
        planned = plan_fal_image_prompt(
            self._prompt_enhancement,
            params.prompt,
            provenance=params.promptProvenance,
            generation_seed=seed,
            explicit_generation_seed=explicit_seed,
            lease_idle_gpu=self._gpu_lease_for(session),
            raise_if_cancelled=session.raise_if_cancelled,
        )
        prompt = planned.prompt
        session.raise_if_cancelled()
        session.set_progress("inference", 15, None, None)
        try:
            queued = self._zit_api_client.generate_text_to_image_queued(
                api_key=api_key,
                prompt=prompt,
                width=params.width,
                height=params.height,
                seed=seed,
                num_inference_steps=ZIT_DEFAULT_STEPS,
                enable_prompt_expansion=planned.expand_on_provider,
                on_status=session.set_api_status,
                on_submitted=session.set_provider_job_id,
                wait_cancelled=session.wait_cancelled,
            )
        except ZitImageBlockedError as exc:
            raise HTTPError(422, "IMAGE_BLOCKED_BY_SAFETY_CHECKER") from exc
        # The provider's expansion replaces the typed prompt on the finished row. A rewrite
        # done here does not: the local provider keeps the typed prompt in the same case.
        written = (
            queued.prompt
            if planned.expand_on_provider and queued.prompt not in (None, prompt)
            else None
        )
        return queued.image_bytes, QueuedImageResult(
            written_prompt=written, prompt_rewrite=planned.rewrite_source
        )
