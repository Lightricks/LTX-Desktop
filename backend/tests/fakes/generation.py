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
    IcLoraRecipeInputs,
    IcLoraRecipeStoredParams,
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
from services.generation_queue.types import OutputAllocation, OutputPlan
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
    for _ in range(8):
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
        if isinstance(inputs, Mapping) and "video" in inputs:
            if get_ic_lora_recipe(self._feature or "") is not None:
                IcLoraRecipeStoredParams.model_validate(params)
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
    ) -> None:
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
