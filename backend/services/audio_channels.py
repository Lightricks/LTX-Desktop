"""Normalize decoded audio waveforms to the stereo layout the audio VAE expects."""

from __future__ import annotations

import torch


def stereo_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """Return a waveform with exactly two channels.

    `decode_audio_from_file` yields `(batch, channels, samples)` or
    `(channels, samples)`. The DistilledA2V audio VAE conv is stereo
    (`in_channels=2`); mono STFT otherwise fails with CAPABILITY_FAILED.
    """
    if waveform.ndim == 3:
        channels = int(waveform.shape[1])
        if channels == 1:
            return waveform.repeat(1, 2, 1)
        if channels > 2:
            return waveform[:, :2]
        return waveform
    if waveform.ndim == 2:
        channels = int(waveform.shape[0])
        if channels == 1:
            return waveform.repeat(2, 1)
        if channels > 2:
            return waveform[:2]
        return waveform
    raise ValueError(f"expected waveform with 2 or 3 dims, got {waveform.ndim}")
