"""Unit tests for the process-wide generation interrupt token."""

from __future__ import annotations

import pytest

from services.generation_interrupt import (
    GenerationCancelledError,
    clear,
    diffusers_step_callback,
    is_cancel_exception,
    is_requested,
    raise_if_requested,
    request,
    wrap_denoiser,
    yield_and_check,
)


def test_request_sets_event_and_clear_unsets_it() -> None:
    clear()
    assert not is_requested()
    request()
    assert is_requested()
    clear()
    assert not is_requested()


def test_raise_if_requested_raises_generation_cancelled() -> None:
    request()
    with pytest.raises(GenerationCancelledError, match="cancelled"):
        raise_if_requested()


def test_wrap_denoiser_stops_after_request() -> None:
    calls: list[int] = []

    def denoiser(step: int) -> int:
        calls.append(step)
        if step == 2:
            request()
        return step

    wrapped = wrap_denoiser(denoiser)
    assert wrapped(1) == 1
    assert wrapped(2) == 2
    with pytest.raises(GenerationCancelledError):
        wrapped(3)
    assert calls == [1, 2]


def test_yield_and_check_raises_when_requested() -> None:
    request()
    with pytest.raises(GenerationCancelledError):
        yield_and_check()


def test_diffusers_step_callback_raises_when_requested() -> None:
    kwargs = {"latents": object()}
    assert diffusers_step_callback(object(), 0, None, kwargs) is kwargs
    request()
    with pytest.raises(GenerationCancelledError):
        diffusers_step_callback(object(), 1, None, kwargs)


def test_is_cancel_exception() -> None:
    assert is_cancel_exception(GenerationCancelledError())
    wrapped = RuntimeError("pipeline failed")
    wrapped.__cause__ = GenerationCancelledError()
    assert is_cancel_exception(wrapped)
    via_context = RuntimeError("pipeline failed")
    via_context.__context__ = GenerationCancelledError()
    assert is_cancel_exception(via_context)
    cyclic = RuntimeError("a")
    cyclic.__cause__ = cyclic
    assert not is_cancel_exception(cyclic)
    assert not is_cancel_exception(RuntimeError("cancelled"))
    assert not is_cancel_exception(RuntimeError("operation cancelled by peer"))
    assert not is_cancel_exception(RuntimeError("GPU OOM"))


def test_zit_pipeline_passes_step_end_callback() -> None:
    import inspect

    from services.image_generation_pipeline.zit_image_generation_pipeline import (
        ZitImageGenerationPipeline,
    )

    generate_src = inspect.getsource(ZitImageGenerationPipeline.generate)
    edit_src = inspect.getsource(ZitImageGenerationPipeline.edit)
    assert "callback_on_step_end=diffusers_step_callback" in generate_src
    assert "callback_on_step_end=diffusers_step_callback" in edit_src


def test_image_callback_stops_later_steps() -> None:
    from tests.fakes.services import FakeImageGenerationPipeline

    pipeline = FakeImageGenerationPipeline()
    pipeline.inference_steps = 6

    def callback(pipe: object, step_index: int, timestep: object, callback_kwargs: dict) -> dict:
        if step_index == 1:
            request()
        return diffusers_step_callback(pipe, step_index, timestep, callback_kwargs)

    with pytest.raises(GenerationCancelledError):
        pipeline.generate(callback_on_step_end=callback)
    assert pipeline.steps_completed == 1
