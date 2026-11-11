"""Distilled A2V stage flow against the real v1.5 chunk operators.

A2V has no GPU-free coverage of its two ``DiffusionStage`` calls. This drives
``DistilledA2VPipeline.__call__`` (one window through ``stream_chunks``) with a
recording stage, a fake upsampler and decoder, and the real chunk planner, so a
mis-sized latent or a wrong stage argument changes what the stage sees here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from ltx_core.types import Audio, SpatioTemporalScaleFactors
from ltx_core.conditioning import VideoConditionByKeyframeIndex, VideoConditionByLatentIndex
from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.types import ImageConditioningInput, LoadedImageConditioning

from services.a2v_pipeline.distilled_a2v_pipeline import DistilledA2VPipeline

# The transformer latent geometry. Upstream sizes stills from the module default (8x32x32), not from the 2.5 DiffVAE decoder scale (2x8x8).
_STAGE_SCALE = SpatioTemporalScaleFactors(time=8, height=32, width=32)
_CHANNELS = 128
_FRAMES = 9
_HEIGHT, _WIDTH = 576, 1024
_SAMPLE_RATE = 16000


class _State:
    generated_keyframes = None
    generated_keyframe_layout = None

    def __init__(self, latent: torch.Tensor) -> None:
        self.latent = latent


class _Still:
    def tensor(self, height, width, dtype, device):
        return torch.zeros(1, 3, 1, height, width, dtype=dtype, device=device)


def _encoder(pixels: torch.Tensor) -> torch.Tensor:
    return torch.zeros(1, _CHANNELS, 1, pixels.shape[-2] // 32, pixels.shape[-1] // 32)


class _Conditioner:
    def load_conditionings(self, images, color_space=None):  # noqa: ARG002
        return [LoadedImageConditioning(still=_Still(), strength=i.strength, binding=i.binding) for i in images]

    def __call__(self, build):
        return build(_encoder)


def _pipeline(monkeypatch: pytest.MonkeyPatch) -> tuple[DistilledA2VPipeline, list[dict[str, Any]], list[torch.Tensor]]:
    stage_calls: list[dict[str, Any]] = []
    upsampler_inputs: list[torch.Tensor] = []

    monkeypatch.setattr(
        "ltx_pipelines.utils.media_io.decode_audio_from_file",
        lambda *_args, **_kwargs: Audio(torch.ones(1, 2, _SAMPLE_RATE * 2), _SAMPLE_RATE),
    )

    def stage(**kwargs: Any) -> tuple[_State, _State]:
        stage_calls.append(kwargs)
        modalities = kwargs["modalities"]
        return _State(modalities.video.latent), _State(modalities.audio.latent)

    stage.video_scale_factors = _STAGE_SCALE  # type: ignore[attr-defined]

    def upsampler(latent: torch.Tensor) -> torch.Tensor:
        upsampler_inputs.append(latent)
        return latent.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)

    def prompt_encoder(prompts: list[str]) -> tuple[Any]:
        assert len(prompts) == 1
        return (SimpleNamespace(video_encoding="video-ctx", audio_encoding="audio-ctx"),)

    pipe = cast(DistilledA2VPipeline, object.__new__(DistilledA2VPipeline))
    pipe.device = torch.device("cpu")
    pipe.dtype = torch.bfloat16
    pipe.use_ancestral_sampler = False
    pipe.prompt_encoder = prompt_encoder  # type: ignore[assignment]
    pipe.image_conditioner = _Conditioner()  # type: ignore[assignment]
    pipe.audio_conditioner = lambda _build: torch.zeros(1, 8, 4, 16)  # type: ignore[assignment]
    pipe.stage = stage  # type: ignore[assignment]
    pipe.upsampler = upsampler  # type: ignore[assignment]

    def video_decoder(latent: torch.Tensor, _tiling: Any, _generator: Any, **_kwargs: Any) -> Any:
        pixel_frames = (latent.shape[2] - 1) * _STAGE_SCALE.time + 1
        yield torch.zeros(pixel_frames, latent.shape[3] * _STAGE_SCALE.height, latent.shape[4] * _STAGE_SCALE.width, 3)

    pipe.video_decoder = video_decoder  # type: ignore[assignment]
    return pipe, stage_calls, upsampler_inputs


def _run(pipe: DistilledA2VPipeline, images: list[ImageConditioningInput] | None = None) -> tuple[Any, Audio]:
    return pipe(  # type: ignore[return-value]
        prompt="a clip",
        seed=3,
        height=_HEIGHT,
        width=_WIDTH,
        num_frames=_FRAMES,
        frame_rate=24.0,
        images=images or [],
        audio_path="speech.wav",
    )


def test_stage_1_latent_is_half_canvas_in_stage_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, calls, _ = _pipeline(monkeypatch)

    _run(pipe)

    video = calls[0]["modalities"].video
    frames = (_FRAMES - 1) // _STAGE_SCALE.time + 1
    expected = (1, _CHANNELS, frames, _HEIGHT // 2 // _STAGE_SCALE.height, _WIDTH // 2 // _STAGE_SCALE.width)
    assert tuple(video.latent.shape) == expected
    assert video.conditioning_fps == 24.0
    assert video.context == "video-ctx"


def test_stage_2_refines_the_upsampled_latent_at_the_first_stage_2_sigma(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, calls, upsampler_inputs = _pipeline(monkeypatch)

    _run(pipe)

    assert len(calls) == 2
    stage_1_latent = calls[0]["modalities"].video.latent
    assert upsampler_inputs[0].shape == stage_1_latent.shape
    stage_2 = calls[1]["modalities"].video
    assert tuple(stage_2.latent.shape[-2:]) == (stage_1_latent.shape[-2] * 2, stage_1_latent.shape[-1] * 2)
    assert stage_2.noise_scale == pytest.approx(STAGE_2_DISTILLED_SIGMA_VALUES[0])
    assert torch.equal(calls[1]["sigmas"].cpu(), torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES))


def test_audio_stays_frozen_and_unnoised_in_both_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, calls, _ = _pipeline(monkeypatch)

    _run(pipe)

    for call in calls:
        audio = call["modalities"].audio
        assert audio.frozen is True
        assert audio.noise_scale == 0.0
        assert audio.context == "audio-ctx"
    # Both stages read the same encoded audio latent.
    assert calls[0]["modalities"].audio.latent is calls[1]["modalities"].audio.latent


def test_returns_the_source_audio_trimmed_to_the_video_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, _, _ = _pipeline(monkeypatch)

    audio = _run(pipe).audio

    assert audio.sampling_rate == _SAMPLE_RATE
    assert audio.waveform.shape[-1] == round(_FRAMES / 24.0 * _SAMPLE_RATE)
    assert torch.all(audio.waveform == 1)


def _target_audio_frames() -> int:
    from ltx_core.types import AudioLatentShape

    return AudioLatentShape.from_duration(batch=1, duration=_FRAMES / 24.0, channels=8, mel_bins=16).frames


@pytest.mark.parametrize("delta", [-1, 0, 2], ids=["padded", "exact", "trimmed"])
def test_audio_latent_is_padded_or_trimmed_to_the_video_duration(monkeypatch: pytest.MonkeyPatch, delta: int) -> None:
    target = _target_audio_frames()
    encoded_frames = target + delta
    assert encoded_frames >= 1
    pipe, calls, _ = _pipeline(monkeypatch)
    pipe.audio_conditioner = lambda _build: torch.ones(1, 8, encoded_frames, 16)  # type: ignore[assignment]

    _run(pipe)

    latent = calls[0]["modalities"].audio.latent
    assert tuple(latent.shape) == (1, 8, target, 16)
    kept = min(encoded_frames, target)
    assert torch.all(latent[:, :, :kept] == 1)
    assert torch.all(latent[:, :, kept:] == 0)


def test_audio_window_arguments_reach_the_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    pipe, _, _ = _pipeline(monkeypatch)
    seen: list[tuple[Any, ...]] = []

    def recording_decode(*args: Any, **_kwargs: Any) -> Audio:
        seen.append(args)
        return Audio(torch.ones(1, 2, _SAMPLE_RATE * 2), _SAMPLE_RATE)

    monkeypatch.setattr("ltx_pipelines.utils.media_io.decode_audio_from_file", recording_decode)

    pipe(
        prompt="a clip",
        seed=3,
        height=_HEIGHT,
        width=_WIDTH,
        num_frames=_FRAMES,
        frame_rate=24.0,
        images=[],
        audio_path="speech.wav",
        audio_start_time=1.5,
        audio_max_duration=4.0,
    )

    assert len(seen) == 1
    path, _device, start, max_duration = seen[0]
    assert (path, start, max_duration) == ("speech.wav", 1.5, 4.0)


def test_first_and_last_pixel_frame_stills_condition_both_stages_at_their_canvas(monkeypatch: pytest.MonkeyPatch) -> None:
    """The last pixel frame is past the latent count. It must stay a keyframe guide and the first frame a replace."""
    pipe, calls, _ = _pipeline(monkeypatch)
    last = _FRAMES - 1

    _run(
        pipe,
        [ImageConditioningInput.at("start.png", 0, 1.0), ImageConditioningInput.at("end.png", last, 1.0)],
    )

    for call, divisor in zip(calls, (2, 1), strict=True):
        first, end = call["modalities"].video.conditionings
        assert isinstance(first, VideoConditionByLatentIndex)
        assert isinstance(end, VideoConditionByKeyframeIndex)
        assert end.frame_idx == last
        assert tuple(end.keyframes.shape[-2:]) == (_HEIGHT // divisor // 32, _WIDTH // divisor // 32)


@pytest.mark.parametrize("ancestral", [True, False], ids=["ltx_2_5", "ltx_2_3"])
def test_both_stages_use_the_ancestral_loop_only_on_ancestral_checkpoints(
    monkeypatch: pytest.MonkeyPatch, ancestral: bool
) -> None:
    """A2V ran plain Euler on LTX 2.5 while DistilledPipeline and retake ran the ancestral loop."""
    pipe, calls, _ = _pipeline(monkeypatch)
    pipe.use_ancestral_sampler = ancestral

    _run(pipe)

    assert [call["loop"] is not None for call in calls] == [ancestral, ancestral]


def test_a_still_past_the_snapped_clip_is_rejected_before_any_denoising(monkeypatch: pytest.MonkeyPatch) -> None:
    """10 frames snap to 9, so a still at frame 9 passed the unsnapped check and was dropped later."""
    pipe, calls, _ = _pipeline(monkeypatch)

    with pytest.raises(ValueError):
        pipe(
            prompt="a clip",
            seed=3,
            height=_HEIGHT,
            width=_WIDTH,
            num_frames=_FRAMES + 1,
            frame_rate=24.0,
            images=[ImageConditioningInput.at("end.png", _FRAMES, 1.0)],
            audio_path="speech.wav",
        )

    assert calls == []


def test_a_clip_longer_than_the_default_window_still_runs_as_one_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a window sized to the clip, 121 frames split at the default 97 and ran stage 1 and 2 twice."""
    pipe, calls, _ = _pipeline(monkeypatch)

    output = pipe(
        prompt="a clip",
        seed=3,
        height=_HEIGHT,
        width=_WIDTH,
        num_frames=121,
        frame_rate=24.0,
        images=[],
        audio_path="speech.wav",
    )

    assert len(calls) == 2
    assert sum(frames.shape[0] for frames in output.video) == 121
