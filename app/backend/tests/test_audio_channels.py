from __future__ import annotations

import torch

from services.audio_channels import stereo_waveform


def test_stereo_waveform_upmixes_batched_mono() -> None:
    mono = torch.tensor([[[0.1, -0.2, 0.3]]])
    stereo = stereo_waveform(mono)
    assert stereo.shape == (1, 2, 3)
    assert torch.equal(stereo[0, 0], stereo[0, 1])
    assert torch.equal(stereo[0, 0], mono[0, 0])


def test_stereo_waveform_keeps_batched_stereo() -> None:
    original = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
    assert torch.equal(stereo_waveform(original), original)


def test_stereo_waveform_keeps_first_two_batched_channels() -> None:
    multi = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    stereo = stereo_waveform(multi)
    assert stereo.shape == (1, 2, 3)
    assert torch.equal(stereo, multi[:, :2])


def test_stereo_waveform_upmixes_planar_mono() -> None:
    mono = torch.tensor([[0.5, -0.5]])
    stereo = stereo_waveform(mono)
    assert stereo.shape == (2, 2)
    assert torch.equal(stereo[0], stereo[1])
