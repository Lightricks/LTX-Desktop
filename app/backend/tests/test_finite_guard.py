"""A denoise step with NaN or Inf must fail the job on MPS (see ``inference.finite_guard``)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest
import torch
from ltx_pipelines.utils.types import DenoisedLatentResult, VideoAudio

from inference.finite_guard import NonFiniteLatentError, raise_if_nonfinite, wrap_denoiser

needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="the guard only reads MPS tensors")


def _result(*, video: torch.Tensor | None = None, audio: torch.Tensor | None = None) -> VideoAudio[DenoisedLatentResult]:
    return VideoAudio(
        video=DenoisedLatentResult(denoised=video) if video is not None else None,
        audio=DenoisedLatentResult(denoised=audio) if audio is not None else None,
    )


def test_guard_reads_the_field_names_ltx_pipelines_uses() -> None:
    # The guard reads these by name. A rename upstream would turn it into a silent no-op.
    assert {"video", "audio"} <= {f.name for f in dataclasses.fields(VideoAudio)}
    assert "denoised" in {f.name for f in dataclasses.fields(DenoisedLatentResult)}


def test_nonfinite_values_off_mps_are_not_checked() -> None:
    raise_if_nonfinite(_result(video=torch.tensor([float("nan")])))


@needs_mps
@pytest.mark.parametrize(
    ("bad_side", "bad_value"),
    [("video", float("nan")), ("video", float("inf")), ("audio", float("nan"))],
)
def test_nonfinite_denoised_on_mps_fails_the_job(bad_side: str, bad_value: float) -> None:
    good = torch.ones(4, device="mps")
    bad = torch.tensor([1.0, bad_value], device="mps")
    result = _result(video=bad if bad_side == "video" else good, audio=bad if bad_side == "audio" else good)

    with pytest.raises(NonFiniteLatentError, match=bad_side):
        raise_if_nonfinite(result)


@needs_mps
def test_finite_denoised_on_mps_passes_through_unchanged() -> None:
    result = _result(video=torch.ones(4, device="mps"))
    seen: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def denoiser(*args: Any, **kwargs: Any) -> VideoAudio[DenoisedLatentResult]:
        seen.append((args, kwargs))
        return result

    wrapped: Callable[..., Any] = wrap_denoiser(denoiser)

    assert wrapped("transformer", step_index=3) is result
    assert seen == [(("transformer",), {"step_index": 3})]
