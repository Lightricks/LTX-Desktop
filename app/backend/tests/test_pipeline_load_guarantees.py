"""Loader guarantees: a swap releases before it allocates, and cold loads hold liveness.

A same-kind cache miss must drop the resident pipeline state before the new
pipeline is created — a leaked frame reference would hold the old weights
alive through evict and re-create (transient double residency on a tight
card). And every cold load must run under the Electron proof-of-life hold:
a slow weights read or MPS graph build can stall the GIL past the kill
threshold, and without the hold it reads as hung instead of busy.
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from runtime_config.model_download_specs import (
    DEPTH_PROCESSOR_CP_ID,
    get_ltx_model_spec,
    resolve_model_path,
)
from server_utils.generation_liveness import GENERATION_HEARTBEAT_FILENAME
from tests.conftest import _IC_LORA_MODEL_ID
from tests.fakes.services import (
    FakeA2VPipeline,
    FakeFastVideoPipeline,
    FakeIcLoraPipeline,
    FakeImageGenerationPipeline,
    FakeRetakePipeline,
)

_SWAP_NAMES = ("fast", "a2v", "retake", "ic-lora")
_HOLD_NAMES = _SWAP_NAMES + ("image",)


@dataclass(frozen=True)
class _LoaderCase:
    fake_class: type[Any]
    first: Callable[[], object]
    swap: Callable[[], object]


def _swap_cases(test_state, lora_path: str, depth_path: str) -> dict[str, _LoaderCase]:
    return {
        "fast": _LoaderCase(
            fake_class=FakeFastVideoPipeline,
            first=lambda: test_state.pipelines.load_gpu_pipeline("fast"),
            swap=lambda: test_state.pipelines.load_gpu_pipeline(
                "fast", loras=[("other-lora.safetensors", 1.0)]
            ),
        ),
        "a2v": _LoaderCase(
            fake_class=FakeA2VPipeline,
            first=lambda: test_state.pipelines.load_a2v_pipeline(),
            swap=lambda: test_state.pipelines.load_a2v_pipeline(
                loras=[("other-lora.safetensors", 1.0)]
            ),
        ),
        "retake": _LoaderCase(
            fake_class=FakeRetakePipeline,
            first=lambda: test_state.pipelines.load_retake_pipeline(distilled=True)[0],
            swap=lambda: test_state.pipelines.load_retake_pipeline(distilled=False),
        ),
        "ic-lora": _LoaderCase(
            fake_class=FakeIcLoraPipeline,
            first=lambda: test_state.pipelines.load_ic_lora(lora_path, depth_path),
            swap=lambda: test_state.pipelines.load_ic_lora(
                lora_path, depth_path, lora_strength=0.5
            ),
        ),
    }


def _hold_loads(
    test_state, lora_path: str, depth_path: str
) -> dict[str, tuple[type[Any], Callable[[], object]]]:
    loads = {
        name: (case.fake_class, case.first)
        for name, case in _swap_cases(test_state, lora_path, depth_path).items()
    }
    loads["image"] = (
        FakeImageGenerationPipeline,
        test_state.pipelines.load_image_generation_pipeline_to_gpu,
    )
    return loads


def _ic_lora_paths(test_state) -> tuple[str, str]:
    model_spec = get_ltx_model_spec(_IC_LORA_MODEL_ID)
    assert model_spec.ic_loras_spec is not None
    lora_path = str(
        resolve_model_path(
            test_state.config.default_models_dir, model_spec.ic_loras_spec.canny_cp
        )
    )
    depth_path = str(
        resolve_model_path(test_state.config.default_models_dir, DEPTH_PROCESSOR_CP_ID)
    )
    return lora_path, depth_path


@pytest.mark.parametrize("name", _SWAP_NAMES)
def test_swap_load_releases_old_state_before_create(
    name: str, test_state, create_fake_model_files, create_fake_ic_lora_files, monkeypatch
) -> None:
    """A match-case capture keeps the old weights alive through evict+create (2x residency)."""
    create_fake_model_files()
    create_fake_model_files(model_id=_IC_LORA_MODEL_ID)
    create_fake_ic_lora_files()
    lora_path, depth_path = _ic_lora_paths(test_state)
    case = _swap_cases(test_state, lora_path, depth_path)[name]
    first = case.first()
    old = weakref.ref(first)
    del first
    original_create = case.fake_class.create
    observed: dict[str, bool] = {}

    def _spy(*args: Any, **kwargs: Any) -> Any:
        gc.collect()
        observed["old_alive"] = old() is not None
        return original_create(*args, **kwargs)

    monkeypatch.setattr(case.fake_class, "create", _spy)
    case.swap()
    assert observed.get("old_alive") is False


@pytest.mark.parametrize("name", _HOLD_NAMES)
def test_cold_load_holds_liveness_file_during_create(
    name: str,
    test_state,
    create_fake_model_files,
    create_fake_ic_lora_files,
    tmp_path,
    monkeypatch,
) -> None:
    """A slow weights read / MPS build outside the hold gets killed as hung (no grace)."""
    monkeypatch.setenv("LTX_APP_DATA_DIR", str(tmp_path))
    create_fake_model_files(include_zit=True)
    create_fake_model_files(model_id=_IC_LORA_MODEL_ID)
    create_fake_ic_lora_files()
    lora_path, depth_path = _ic_lora_paths(test_state)
    heartbeat = tmp_path / GENERATION_HEARTBEAT_FILENAME
    fake_class, load = _hold_loads(test_state, lora_path, depth_path)[name]
    original_create = fake_class.create
    observed: dict[str, bool] = {}

    def _spy(*args: Any, **kwargs: Any) -> Any:
        observed["held"] = heartbeat.exists()
        return original_create(*args, **kwargs)

    monkeypatch.setattr(fake_class, "create", _spy)
    load()
    assert observed.get("held") is True


def test_image_load_releases_a_resident_video_pipeline_before_create(
    test_state, create_fake_model_files, monkeypatch
) -> None:
    """A queued image job often follows a video job: both models must not be resident at once."""
    create_fake_model_files(include_zit=True)
    video = test_state.pipelines.load_gpu_pipeline("fast")
    old = weakref.ref(video)
    del video
    original_create = FakeImageGenerationPipeline.create
    observed: dict[str, bool] = {}

    def _spy(*args: Any, **kwargs: Any) -> Any:
        gc.collect()
        observed["video_alive"] = old() is not None
        return original_create(*args, **kwargs)

    monkeypatch.setattr(FakeImageGenerationPipeline, "create", _spy)
    test_state.pipelines.load_image_generation_pipeline_to_gpu()
    assert observed.get("video_alive") is False
