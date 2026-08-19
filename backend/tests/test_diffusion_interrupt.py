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
