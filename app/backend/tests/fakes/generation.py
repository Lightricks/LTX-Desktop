"""Fake generation executor for queue runner tests."""

from __future__ import annotations

import time
import wave
from collections.abc import Mapping
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from api_types import (
    ApiActivityStatus,
    IcLoraRecipeInputs,
    IcLoraVideoRecipeStoredParams,
    QueueProgress,
    IcLoraSheetInputs,
    IcLoraSynthesizedStoredParams,
    IcLoraTrackInputs,
    ExtendInputs,
    ExtendParams,
    ImageToVideoInputs,
    ImageToVideoRecipeParams,
    RetakeInputs,
    RetakeParams,
    TextToVideoParams,
)
from services.features.ic_lora_recipes import get_ic_lora_recipe
from services.generation_interrupt import GenerationCancelledError, is_requested
from services.features.video import require_supported_contract_version
from services.generation_queue.control import QueueLiveState
from services.generation_queue.sessions import JobProgress, JobSession, LocalGenerationOps, LocalJobSession
from services.generation_queue.types import ExecutionOutcome, OutputAllocation, OutputPlan
from services.records import GenerationRecord

_VIDEO_PLAN = OutputPlan(
    slot="output",
    media_kind="video",
    mime_type="video/mp4",
    name="output.mp4",
)


def _spec_params(spec: Mapping[str, object]) -> Mapping[str, object]:
    params = spec.get("params")
    if not isinstance(params, Mapping):
        raise ValueError("generation spec requires a params object")
    return params


def _write_valid_mp4(path: Path) -> None:
    writer = imageio.get_writer(
        str(path), fps=8, codec="libx264", macro_block_size=None
    )
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    # 9 frames: the shortest clip extend accepts as a source.
    for _ in range(9):
        writer.append_data(frame)
    writer.close()


def _write_valid_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * 800)


def _write_valid_output(allocation: OutputAllocation) -> None:
    dest = Path(allocation.dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    match allocation.plan.media_kind:
        case "video":
            _write_valid_mp4(dest)
        case "image":
            Image.new("RGB", (16, 16), color=(1, 2, 3)).save(dest)
        case "audio":
            _write_valid_wav(dest)


class FakeGenerationExecutor:
    def __init__(
        self,
        seen_prompts: list[str] | None = None,
        error: Exception | None = None,
        block: bool = False,
        feature: str | None = None,
    ) -> None:
        self._seen_prompts = seen_prompts
        self._error = error
        self._block = block
        self._feature = feature

    def validate_params(
        self, spec: Mapping[str, object], *, contract_version: int
    ) -> None:
        require_supported_contract_version(contract_version)
        inputs = spec.get("inputs")
        params = _spec_params(spec)
        if isinstance(inputs, Mapping) and "images" in inputs:
            IcLoraSynthesizedStoredParams.model_validate(params)
            IcLoraSheetInputs.model_validate(inputs)
        elif isinstance(inputs, Mapping) and "paths" in inputs:
            IcLoraSynthesizedStoredParams.model_validate(params)
            IcLoraTrackInputs.model_validate(inputs)
        elif isinstance(inputs, Mapping) and "video" in inputs:
            if get_ic_lora_recipe(self._feature or "") is not None:
                IcLoraVideoRecipeStoredParams.model_validate(params)
                IcLoraRecipeInputs.model_validate(inputs)
            elif self._feature == "retake" or (
                self._feature is None and "startTime" in params
            ):
                RetakeParams.model_validate(params)
                RetakeInputs.model_validate(inputs)
            else:
                ExtendParams.model_validate(params)
                ExtendInputs.model_validate(inputs)
        elif isinstance(inputs, Mapping) and "startFrame" in inputs:
            # Recipe i2v specs carry `loras`; public I2V specs omit them. The
            # recipe params model is a superset so one fake covers both.
            ImageToVideoRecipeParams.model_validate(params)
            ImageToVideoInputs.model_validate(inputs)
        else:
            TextToVideoParams.model_validate(params)

    def plan_outputs(self, generation: GenerationRecord) -> tuple[OutputPlan, ...]:
        require_supported_contract_version(generation.contract_version)
        return (_VIDEO_PLAN,)

    def execute(
        self,
        generation: GenerationRecord,
        outputs: tuple[OutputAllocation, ...],
        session: JobSession,
    ) -> ExecutionOutcome:
        if self._seen_prompts is not None:
            prompt = _spec_params(generation.spec).get("prompt")
            self._seen_prompts.append(prompt if isinstance(prompt, str) else "")
        if self._error is not None:
            raise self._error
        if self._block:
            while not is_requested():
                time.sleep(0.01)
            raise GenerationCancelledError()
        for allocation in outputs:
            _write_valid_output(allocation)
        return ExecutionOutcome()


class FakeJobSession:
    """A JobSession that records cancellation and nothing else. For executors under test."""

    def __init__(self, generation_id: str = "test-generation") -> None:
        self._generation_id = generation_id
        self.cancelled = False
        self.progress_updates: list[tuple[str, int]] = []
        self.provider_job_ids: list[str] = []

    @property
    def generation_id(self) -> str:
        return self._generation_id

    @property
    def holds_gpu_slot(self) -> bool:
        return False

    def start_api(self) -> None:
        return None

    def set_progress(
        self,
        phase: str,
        progress: int,
        current_step: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        del current_step, total_steps
        self.progress_updates.append((phase, progress))

    def set_api_status(self, status: ApiActivityStatus) -> None:
        del status

    def set_provider_job_id(self, provider_job_id: str) -> bool:
        self.provider_job_ids.append(provider_job_id)
        return True

    def progress(self) -> JobProgress | None:
        return None

    def api_status(self) -> ApiActivityStatus | None:
        return None

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise GenerationCancelledError()

    def wait_cancelled(self, timeout: float) -> bool:
        del timeout
        return self.cancelled

    def cancel(self) -> None:
        self.cancelled = True

    def complete(self, result: str | list[str] | None) -> None:
        del result

    def fail(self, error: str) -> None:
        del error


class LocalLiveQueueControl:
    """A queue control for tests that run no runner: it reads progress from the shared slot."""

    def __init__(self, generation: LocalGenerationOps) -> None:
        self._generation = generation

    def wake(self) -> None:
        return None

    def interrupt(self, generation_id: str, attempt_count: int) -> None:
        del generation_id, attempt_count

    def live_state(self, generation_id: str) -> QueueLiveState:
        progress = LocalJobSession(self._generation, generation_id).progress()
        return QueueLiveState(
            progress=None
            if progress is None
            else QueueProgress(
                phase=progress.phase,
                progress=progress.progress,
                currentStep=progress.current_step,
                totalSteps=progress.total_steps,
                remainingSeconds=progress.remaining_seconds,
            ),
            api_status=None,
            cancellable=True,
        )
