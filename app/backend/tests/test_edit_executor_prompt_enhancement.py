"""Queued Home Extend and Retake enhance the prompt once, from stills of their source video.

The real executors run under the real queue runner and store. Only their GPU stages are stubbed
out: those need real pipelines and are covered by the retake and extend suites.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from api_types import ExtendParams, RetakeParams
from runtime_config.budget_admission import RiskAllowance
from runtime_config.device_calibration import JobEnvelope
from services.features.video.edit_executor import VideoEditExecutor
from services.features.video.extend import ExtendExecutor
from services.features.video.retake import RetakeExecutor
from services.features.video.text_to_video import VIDEO_OUTPUT_PLAN
from services.generation_queue.registry import ExecutorRegistry
from services.generation_queue.runner import QueueRunner
from services.sqlite_store import SqliteStore
from tests.video_fixtures import write_mp4

_TYPED = "the character turns around"
_ENHANCED = "a long caption of the scene, the character turns around"


class _Slot:
    def start_generation(
        self, generation_id: str, *, ran: JobEnvelope | None = None
    ) -> None: ...

    def complete_generation(
        self, result: str | list[str] | None = None, *, ran: JobEnvelope | None = None
    ) -> None: ...

    def fail_generation(self, error: str) -> None: ...


class _Prepared(NamedTuple):
    dest: Path
    ran: JobEnvelope | None = None


class _Text:
    def __init__(self, *, local_encoding: bool = True) -> None:
        self.encoded: list[tuple[str, bool]] = []
        self.local_encoding = local_encoding

    def prepare_text_encoding(self, prompt: str, enhance_prompt: bool, model_id: Any = None) -> None:
        self.encoded.append((prompt, enhance_prompt))

    def clear_api_embeddings(self) -> None: ...

    def should_use_local_encoding(self, model_id: Any = None) -> bool:
        del model_id
        return self.local_encoding


class _StubbedGpuRetake(RetakeExecutor):
    prepared_prompts: list[str]
    fail_in_gpu_stage = False

    def prepare(
        self,
        generation: Any,
        params: RetakeParams,
        *rest: Any,
        allowance: RiskAllowance = RiskAllowance.none(),
    ) -> Any:
        del allowance
        self.prepared_prompts = [*getattr(self, "prepared_prompts", []), params.prompt]
        return _Prepared(rest[3])

    def generate(self, prepared: Any) -> None:
        if self.fail_in_gpu_stage:
            raise RuntimeError("gpu stage failed")

    def stitch(self, prepared: Any) -> None:
        write_mp4(prepared.dest, seconds=1)


class _StubbedGpuExtend(ExtendExecutor):
    prepared_prompts: list[str]
    fail_in_gpu_stage = False

    def prepare(
        self,
        generation: Any,
        params: ExtendParams,
        *rest: Any,
        allowance: RiskAllowance = RiskAllowance.none(),
    ) -> Any:
        del allowance
        self.prepared_prompts = [*getattr(self, "prepared_prompts", []), params.prompt]
        return _Prepared(rest[3])

    def generate(self, prepared: Any) -> None:
        if self.fail_in_gpu_stage:
            raise RuntimeError("gpu stage failed")

    def stitch(self, prepared: Any) -> None:
        write_mp4(prepared.dest, seconds=1)


_EXECUTORS: dict[str, tuple[type[VideoEditExecutor[Any, Any, Any]], dict[str, object]]] = {
    "retake": (_StubbedGpuRetake, {"startTime": 0, "duration": 2}),
    "extend": (_StubbedGpuExtend, {"duration": 4, "mode": "end"}),
}


class _Result(NamedTuple):
    executor: Any
    text: _Text
    store: SqliteStore
    generation_id: str


def _run(
    feature: str,
    *,
    provenance: str,
    client,
    test_state,
    create_fake_model_files,
    tmp_path: Path,
    typed_prompt: str = _TYPED,
    negative_prompt: str = "",
    local_encoding: bool = True,
    fail_in_gpu_stage: bool = False,
) -> _Result:
    create_fake_model_files(include_prompt_enhancer=True)
    test_state.state.app_settings.use_local_text_encoder = local_encoding
    asset = client.post(
        "/api/assets", json={"path": str(write_mp4(tmp_path / "source.mp4", seconds=6))}
    ).json()
    executor_class, feature_params = _EXECUTORS[feature]
    store = SqliteStore(test_state.config.app_data_dir)
    text = _Text(local_encoding=local_encoding)
    executor = executor_class(
        test_state.pipelines,
        store,
        text,
        lambda: test_state.video_generation.models_dir,
        "negative",
        _Slot(),
        test_state.prompt_enhancement,
    )
    executor.fail_in_gpu_stage = fail_in_gpu_stage  # type: ignore[attr-defined]
    generation = store.insert_generation(
        feature,
        {
            "params": {
                "prompt": typed_prompt,
                "negativePrompt": negative_prompt,
                "model": "ltx-2.5-fast",
                "promptProvenance": provenance,
                **feature_params,
            },
            "inputs": {"video": {"assetId": asset["id"]}},
        },
    )
    runner = QueueRunner(
        db=store,
        generation=test_state.generation,
        executors=ExecutorRegistry({feature: executor}),
    )
    runner.start()
    try:
        runner.wake()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            loaded = store.get_generation(generation.id)
            assert loaded is not None
            if loaded.status in ("succeeded", "failed"):
                break
            time.sleep(0.02)
        else:
            raise AssertionError("the queue runner did not finish the generation")
    finally:
        runner.stop()
    return _Result(executor, text, store, generation.id)


def _stored_prompt(result: _Result) -> object:
    generation = result.store.get_generation(result.generation_id)
    assert generation is not None
    params = generation.spec["params"]
    assert isinstance(params, dict)
    return params["prompt"]


@pytest.mark.parametrize("feature", ["retake", "extend"])
def test_a_typed_prompt_is_enhanced_once_from_the_source_video_and_not_by_the_server(
    feature, client, test_state, fake_services, create_fake_model_files, tmp_path
):
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = _ENHANCED

    result = _run(
        feature,
        provenance="typed",
        client=client,
        test_state=test_state,
        create_fake_model_files=create_fake_model_files,
        tmp_path=tmp_path,
    )

    calls = fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls
    assert len(calls) == 1
    assert calls[0]["prompt"] == _TYPED
    assert calls[0]["role"] == "scene"
    assert calls[0]["frames_existed"]
    assert result.executor.prepared_prompts == [_ENHANCED]  # type: ignore[attr-defined]
    # The server-side flag stays off: it cannot see the video, and the rewrite already ran.
    assert result.text.encoded == [(_ENHANCED, False)]


@pytest.mark.parametrize("feature", ["retake", "extend"])
def test_a_prompt_the_user_already_enhanced_by_hand_is_not_rewritten_again(
    feature, client, test_state, fake_services, create_fake_model_files, tmp_path
):
    result = _run(
        feature,
        provenance="enhanced",
        client=client,
        test_state=test_state,
        create_fake_model_files=create_fake_model_files,
        tmp_path=tmp_path,
    )

    assert fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls == []
    assert result.executor.prepared_prompts == [_TYPED]  # type: ignore[attr-defined]
    assert result.text.encoded == [(_TYPED, False)]


@pytest.mark.parametrize("feature", ["retake", "extend"])
def test_a_blank_prompt_is_written_from_the_video_and_saved_after_the_run_succeeds(
    feature, client, test_state, fake_services, create_fake_model_files, tmp_path
):
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = _ENHANCED

    result = _run(
        feature,
        provenance="typed",
        client=client,
        test_state=test_state,
        create_fake_model_files=create_fake_model_files,
        tmp_path=tmp_path,
        typed_prompt="",
    )

    assert result.executor.prepared_prompts == [_ENHANCED]
    assert _stored_prompt(result) == _ENHANCED


@pytest.mark.parametrize("feature", ["retake", "extend"])
def test_a_blank_prompt_stays_blank_in_the_store_when_the_run_fails_so_a_retry_starts_blank(
    feature, client, test_state, fake_services, create_fake_model_files, tmp_path
):
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = _ENHANCED

    result = _run(
        feature,
        provenance="typed",
        client=client,
        test_state=test_state,
        create_fake_model_files=create_fake_model_files,
        tmp_path=tmp_path,
        typed_prompt="",
        fail_in_gpu_stage=True,
    )

    assert _stored_prompt(result) == ""


@pytest.mark.parametrize("feature", ["retake", "extend"])
def test_a_typed_prompt_is_not_overwritten_in_the_store_by_its_rewrite(
    feature, client, test_state, fake_services, create_fake_model_files, tmp_path
):
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = _ENHANCED

    result = _run(
        feature,
        provenance="typed",
        client=client,
        test_state=test_state,
        create_fake_model_files=create_fake_model_files,
        tmp_path=tmp_path,
    )

    assert _stored_prompt(result) == _TYPED


def test_a_retake_samples_around_its_window_so_no_still_shows_what_it_replaces(
    client, test_state, fake_services, create_fake_model_files, tmp_path
):
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = _ENHANCED

    _run(
        "retake",  # replaces seconds 0 to 2 of the 6 second source
        provenance="typed",
        client=client,
        test_state=test_state,
        create_fake_model_files=create_fake_model_files,
        tmp_path=tmp_path,
    )

    times = fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls[0]["frame_times"]
    assert times
    assert min(times) >= 2


def test_an_extend_samples_the_whole_clip_because_it_replaces_no_part_of_it(
    client, test_state, fake_services, create_fake_model_files, tmp_path
):
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = _ENHANCED

    _run(
        "extend",
        provenance="typed",
        client=client,
        test_state=test_state,
        create_fake_model_files=create_fake_model_files,
        tmp_path=tmp_path,
    )

    times = fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls[0]["frame_times"]
    assert min(times) < 1


@pytest.mark.parametrize("feature", ["retake", "extend"])
def test_edit_executor_passes_negative_prompt_to_enhancer_only_when_local_encoding_is_active(
    feature, client, test_state, fake_services, create_fake_model_files, tmp_path
):
    fake_services.prompt_enhancer_pipeline.enhanced_prompt = _ENHANCED

    _run(
        feature,
        provenance="typed",
        client=client,
        test_state=test_state,
        create_fake_model_files=create_fake_model_files,
        tmp_path=tmp_path,
        negative_prompt="blurry, flicker",
        local_encoding=True,
    )
    assert (
        fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls[-1]["negative_prompt"]
        == "blurry, flicker"
    )

    _run(
        feature,
        provenance="typed",
        client=client,
        test_state=test_state,
        create_fake_model_files=create_fake_model_files,
        tmp_path=tmp_path,
        negative_prompt="blurry, flicker",
        local_encoding=False,
    )
    assert (
        fake_services.prompt_enhancer_pipeline.enhance_video_frames_calls[-1]["negative_prompt"]
        == ""
    )

