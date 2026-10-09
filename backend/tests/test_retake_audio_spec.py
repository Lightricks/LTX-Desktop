"""Retake/extend audio modality for sources with and without a soundtrack.

v1.4 needs the audio latent up front. A source with no audio gets a zero latent sized to the
video, and that latent must stay unfrozen and unmasked so the stage generates it freely.
"""

from __future__ import annotations

import torch
from ltx_core.conditioning.types.noise_mask_cond import TemporalRegionMask
from ltx_core.types import VIDEO_SCALE_FACTORS, AudioLatentShape, SpatioTemporalScaleFactors

from services.retake_pipeline.ltx_retake_pipeline import _audio_modality_spec

_FPS = 24.0
# 4 latent frames at time-scale 8 is 25 pixel frames, about 1.04 s.
_VIDEO_LATENT = torch.zeros(1, 128, 4, 2, 2)
_SOURCE_AUDIO = torch.ones(1, 8, 26, 16)


def _spec(*, source, regenerate_audio: bool, scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS):
    return _audio_modality_spec(
        scale_factors,
        source_latent=source,
        video_latent=_VIDEO_LATENT,
        context="ctx",
        fps=_FPS,
        region_start=0.25,
        region_end=0.75,
        regenerate_audio=regenerate_audio,
    )


def test_no_source_audio_is_fresh_unfrozen_and_unmasked() -> None:
    for regenerate_audio in (True, False):
        spec = _spec(source=None, regenerate_audio=regenerate_audio)

        assert spec.frozen is False
        assert spec.conditionings == []
        assert spec.context == "ctx"
        assert spec.conditioning_fps == _FPS
        assert torch.count_nonzero(spec.latent) == 0


def test_no_source_audio_covers_the_video_duration_using_transformer_scale() -> None:
    transformer = _spec(source=None, regenerate_audio=True)
    # A decoder-style 2x8x8 scale would read the same latent as a quarter of the duration.
    decoder = _spec(
        source=None,
        regenerate_audio=True,
        scale_factors=SpatioTemporalScaleFactors(time=2, height=8, width=8),
    )

    pixel_frames = (_VIDEO_LATENT.shape[2] - 1) * VIDEO_SCALE_FACTORS.time + 1
    expected = AudioLatentShape.from_duration(batch=1, duration=pixel_frames / _FPS).frames
    assert transformer.latent.shape[2] == expected
    assert decoder.latent.shape[2] < expected


def test_source_audio_kept_when_not_regenerating() -> None:
    spec = _spec(source=_SOURCE_AUDIO, regenerate_audio=False)

    assert spec.frozen is True
    assert spec.conditionings == []
    assert spec.latent is _SOURCE_AUDIO


def test_source_audio_regenerates_only_the_masked_region() -> None:
    spec = _spec(source=_SOURCE_AUDIO, regenerate_audio=True)

    assert spec.frozen is False
    assert spec.latent is _SOURCE_AUDIO
    (mask,) = spec.conditionings
    assert isinstance(mask, TemporalRegionMask)
    assert (mask.start_time, mask.end_time, mask.fps) == (0.25, 0.75, _FPS)
