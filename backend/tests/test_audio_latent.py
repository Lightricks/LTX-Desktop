"""Mono sources must reach the audio encoder as two channels."""

from __future__ import annotations

from pathlib import Path

import torch
from ltx_core.types import VideoPixelShape
from ltx_pipelines.utils import helpers as library
from ltx_pipelines.utils.helpers import decode_audio_from_file

from inference.retake_pipeline.audio_latent import as_stereo, audio_latent_from_file
from tests.video_fixtures import write_clip_with_tone_spans


class _FakeAudioEncoder(torch.nn.Module):
    sample_rate = 16000
    mel_bins = 64
    mel_hop_length = 160
    n_fft = 1024

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return mel.mean(dim=-1, keepdim=True)


def test_a_mono_source_reaches_the_encoder_as_two_equal_channels(tmp_path: Path) -> None:
    # Bug: the audio encoder reads two channels, so a mono phone clip fails the whole extend
    # with a conv shape error.
    mono = write_clip_with_tone_spans(tmp_path / "mono.mp4", frames=48, spans=[(0.0, 2.0)], channels=1)
    audio = decode_audio_from_file(str(mono), torch.device("cpu"), 0.0, 2.0)
    assert audio is not None
    assert audio.waveform.shape[1] == 1
    stereo = as_stereo(audio)
    assert stereo.waveform.shape[1] == 2
    assert torch.equal(stereo.waveform[:, 0], stereo.waveform[:, 1])


def test_a_stereo_source_is_not_changed(tmp_path: Path) -> None:
    stereo_clip = write_clip_with_tone_spans(tmp_path / "stereo.mp4", frames=48, spans=[(0.0, 2.0)], channels=2)
    audio = decode_audio_from_file(str(stereo_clip), torch.device("cpu"), 0.0, 2.0)
    assert audio is not None
    assert as_stereo(audio).waveform is audio.waveform


def test_a_surround_source_reaches_the_encoder_as_two_channels(tmp_path: Path) -> None:
    # Bug: the audio encoder reads two channels, so a 5.1 clip fails the whole extend.
    surround = write_clip_with_tone_spans(tmp_path / "surround.mp4", frames=48, spans=[(0.0, 2.0)], channels=6)
    audio = decode_audio_from_file(str(surround), torch.device("cpu"), 0.0, 2.0)
    assert audio is not None
    assert audio.waveform.shape[1] == 6
    stereo = as_stereo(audio)
    assert stereo.waveform.shape[1] == 2
    assert torch.equal(stereo.waveform, audio.waveform[:, :2])


def test_the_local_latent_matches_the_library_one_for_a_stereo_source(tmp_path: Path) -> None:
    # Our copy of ltx_pipelines.audio_latent_from_file only adds the mono fix. If a library
    # bump changes the original, this fails so the copy gets updated with it.
    clip = write_clip_with_tone_spans(tmp_path / "stereo.mp4", frames=48, spans=[(0.0, 2.0)], channels=2)
    shape = VideoPixelShape(batch=1, frames=48, height=64, width=64, fps=24.0)
    kwargs = {
        "audio_encoder": _FakeAudioEncoder(),
        "file_path": str(clip),
        "output_shape": shape,
        "device": torch.device("cpu"),
        "dtype": torch.float32,
    }
    assert torch.equal(audio_latent_from_file(**kwargs), library.audio_latent_from_file(**kwargs))
