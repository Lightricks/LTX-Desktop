"""Audio latent for a source file whose sound may be mono.

``ltx_pipelines`` ``audio_latent_from_file`` passes the channels of the file to the audio
encoder. The encoder reads two channels, so a mono source fails with a conv shape error.
This function repeats the steps of the library call and makes the sound stereo (see
``inference.audio_channels``).
"""

from __future__ import annotations

from typing import cast

import torch
from ltx_core.model.audio_vae.audio_vae import AudioEncoder
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_pipelines.utils.helpers import (
    _conform_latent_length,  # pyright: ignore[reportPrivateUsage]
    decode_audio_from_file,
    encode_audio,
    is_exr_dir,
)

from inference.audio_channels import stereo_waveform


def as_stereo(audio: Audio) -> Audio:
    """Mono plays on both channels, and a layout with more than two keeps its first two."""
    return Audio(waveform=stereo_waveform(audio.waveform), sampling_rate=audio.sampling_rate)


def audio_latent_from_file(
    *,
    audio_encoder: torch.nn.Module,
    file_path: str,
    output_shape: VideoPixelShape,
    device: torch.device,
    dtype: torch.dtype,
    start_time: float = 0.0,
    max_duration: float | None = None,
) -> torch.Tensor | None:
    """The result of the library call of the same name. A mono source plays on both channels."""
    if is_exr_dir(file_path):
        return None
    max_duration = max_duration or output_shape.frames / output_shape.fps
    audio = decode_audio_from_file(file_path, device, start_time, max_duration)
    if audio is None:
        return None
    latents = encode_audio(as_stereo(audio), cast(AudioEncoder, audio_encoder), None).to(device, dtype)
    return _conform_latent_length(
        latents, AudioLatentShape.from_video_pixel_shape(output_shape).frames
    )
