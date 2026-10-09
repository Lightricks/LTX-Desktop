"""Unit tests for the DiffusionStage.__call__ denoiser interrupt patch."""

from __future__ import annotations

import inspect

import pytest
from ltx_pipelines.utils.blocks import DiffusionStage

import services.patches.diffusion_interrupt as diffusion_interrupt
from services.generation_interrupt import GenerationCancelledError, request


def test_patch_rebinds_diffusion_stage_call() -> None:
    assert callable(getattr(DiffusionStage, "__call__", None))
    assert DiffusionStage.__call__ is diffusion_interrupt._call_with_interrupt
    assert "denoiser" in inspect.signature(diffusion_interrupt._original_call).parameters
    params = list(inspect.signature(diffusion_interrupt._original_call).parameters)
    assert params[0] == "self" and params[1] == "denoiser"


def test_call_wrap_raises_before_later_denoiser_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def dummy_call(self: object, denoiser: object, latents: object = None) -> object:
        del self
        return denoiser(latents)  # type: ignore[operator]

    monkeypatch.setattr(diffusion_interrupt, "_original_call", dummy_call)

    def denoiser(latents: object) -> object:
        calls.append(latents)
        return latents

    stage = object()
    assert diffusion_interrupt._call_with_interrupt(stage, denoiser, latents="step-1") == "step-1"
    request()
    with pytest.raises(GenerationCancelledError):
        diffusion_interrupt._call_with_interrupt(stage, denoiser, latents="step-2")
    assert calls == ["step-1"]


def test_call_wrap_hits_kwargs_denoiser(monkeypatch: pytest.MonkeyPatch) -> None:
    def dummy_call(self: object, denoiser: object, latents: object = None) -> object:
        del self
        return denoiser(latents)  # type: ignore[operator]

    monkeypatch.setattr(diffusion_interrupt, "_original_call", dummy_call)

    def denoiser(latents: object) -> object:
        return latents

    stage = object()
    assert diffusion_interrupt._call_with_interrupt(stage, denoiser=denoiser, latents="kw") == "kw"
    request()
    with pytest.raises(GenerationCancelledError):
        diffusion_interrupt._call_with_interrupt(stage, denoiser=denoiser, latents="kw-2")


def test_call_wrap_binds_denoiser_by_name_not_position(monkeypatch: pytest.MonkeyPatch) -> None:
    # A leading positional that is not denoiser must not be wrapped.
    def dummy_call(self: object, sigmas: object, denoiser: object, latents: object = None) -> object:
        del self, sigmas
        return denoiser(latents)  # type: ignore[operator]

    monkeypatch.setattr(diffusion_interrupt, "_original_call", dummy_call)

    def denoiser(latents: object) -> object:
        return latents

    stage = object()
    assert diffusion_interrupt._call_with_interrupt(stage, "sigmas", denoiser, latents="pos") == "pos"
    request()
    with pytest.raises(GenerationCancelledError):
        diffusion_interrupt._call_with_interrupt(stage, "sigmas", denoiser, latents="pos-2")


def test_loop_tqdm_iterations_count_as_one_step_even_when_denoiser_runs_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """res2s calls the denoiser twice per loop step; the bar must move once per step."""
    from services.denoising_progress import configure_sink, reset, track_denoising

    seen: list[tuple[int, int]] = []
    configure_sink(lambda current, total: seen.append((current, total)))

    def dummy_call(
        self: object,
        denoiser: object,
        sigmas: object,
        loop: object = None,
        **kwargs: object,
    ) -> str:
        del self, kwargs
        assert loop is not None
        loop(sigmas=sigmas, denoiser=denoiser)  # type: ignore[operator]
        return "ok"

    monkeypatch.setattr(diffusion_interrupt, "_original_call", dummy_call)

    def denoise_loop(*, sigmas: object, denoiser: object, **kwargs: object) -> None:
        del kwargs
        import ltx_pipelines.utils.samplers as samplers

        for _ in samplers.tqdm(range(len(sigmas) - 1)):  # type: ignore[arg-type]
            denoiser()  # type: ignore[operator]
            denoiser()  # type: ignore[operator]

    def denoiser() -> None:
        return None

    sigmas = list(range(9))  # 8 distilled stage-1 steps
    with track_denoising(11):
        diffusion_interrupt._call_with_interrupt(
            object(), denoiser, sigmas=sigmas, loop=denoise_loop
        )
    reset()
    step_reports = [pair for pair in seen if pair[0] > 0]
    assert len(step_reports) == 8
    assert step_reports[-1] == (8, 11)


def _patch_call_to_invoke_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    def dummy_call(
        self: object,
        denoiser: object,
        sigmas: object,
        loop: object = None,
        **kwargs: object,
    ) -> str:
        del self, kwargs
        assert loop is not None
        loop(sigmas=sigmas, denoiser=denoiser)  # type: ignore[operator]
        return "ok"

    monkeypatch.setattr(diffusion_interrupt, "_original_call", dummy_call)


def test_ancestral_terminal_break_still_counts_the_last_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """euler_ancestral_denoising_loop breaks when the next sigma is 0, so the
    tqdm consumer never resumes after that yield. The completed iteration must
    still count when the loop returns normally."""
    from services.denoising_progress import configure_sink, reset, track_denoising

    seen: list[tuple[int, int]] = []
    configure_sink(lambda current, total: seen.append((current, total)))
    _patch_call_to_invoke_loop(monkeypatch)

    def ancestral_loop(*, sigmas: object, denoiser: object, **kwargs: object) -> None:
        del kwargs
        import ltx_pipelines.utils.samplers as samplers

        schedule = list(sigmas)  # type: ignore[arg-type]
        for step_idx, _ in enumerate(samplers.tqdm(schedule[:-1])):
            denoiser()  # type: ignore[operator]
            if schedule[step_idx + 1] == 0:
                break

    sigmas = [1.0, 0.5, 0.25, 0.0]  # 3 denoise steps, last is terminal
    with track_denoising(3):
        diffusion_interrupt._call_with_interrupt(
            object(), lambda: None, sigmas=sigmas, loop=ancestral_loop
        )
    reset()
    step_reports = [pair for pair in seen if pair[0] > 0]
    assert len(step_reports) == 3
    assert step_reports[-1] == (3, 3)


def test_cancelled_loop_does_not_count_the_aborted_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from services.denoising_progress import configure_sink, reset, track_denoising

    seen: list[tuple[int, int]] = []
    configure_sink(lambda current, total: seen.append((current, total)))
    _patch_call_to_invoke_loop(monkeypatch)

    calls = 0

    def ancestral_loop(*, sigmas: object, denoiser: object, **kwargs: object) -> None:
        del kwargs
        import ltx_pipelines.utils.samplers as samplers

        schedule = list(sigmas)  # type: ignore[arg-type]
        for step_idx, _ in enumerate(samplers.tqdm(schedule[:-1])):
            denoiser()  # type: ignore[operator]
            if schedule[step_idx + 1] == 0:
                break

    def denoiser() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise GenerationCancelledError()

    sigmas = [1.0, 0.5, 0.25, 0.0]
    with track_denoising(3):
        with pytest.raises(GenerationCancelledError):
            diffusion_interrupt._call_with_interrupt(
                object(), denoiser, sigmas=sigmas, loop=ancestral_loop
            )
    reset()
    step_reports = [pair for pair in seen if pair[0] > 0]
    assert step_reports == [(1, 3)]
