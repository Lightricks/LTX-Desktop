"""Types, chunk planning and the streamed encode for one IC-LoRA run (see ``ltx_ic_lora_pipeline``)."""

from __future__ import annotations

import logging
from collections.abc import Generator, Iterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypedDict

import torch

from runtime_config.ic_lora_chunking import IC_LORA_CHUNK_CARRY_FRAMES
from services.audio_channels import stereo_waveform
from services.ltx_pipeline_common import encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, AudioType, PipelineTilingType, TilingConfigType

if TYPE_CHECKING:
    from ltx_pipelines.chunks import ChunkConfig
    from ltx_pipelines.utils.types import ImageConditioningInput as LtxImageInput

logger = logging.getLogger(__name__)


class PipelineKwargs(TypedDict):
    """Arguments shared by ``pipeline(...)`` and ``pipeline.stream_chunks(...)``."""

    prompt: str
    seed: int
    height: int
    width: int
    num_frames: int
    frame_rate: float
    images: list[LtxImageInput]
    video_conditioning: list[tuple[str, float]]
    tiling_config: PipelineTilingType
    skip_stage_2: bool
    conditioning_attention_mask: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """What one IC-LoRA run hands to the encoder.

    A chunked job streams: ``video`` (and ``audio``, when the model made it) are iterators
    that run the windows while the caller reads them. The caller must encode inside the
    denoising progress scope. ``audio_sampling_rate`` is set only for that streamed audio.
    ``streams`` lists the generators behind a streamed run. Call ``close_streams`` when the
    encode ends, also on failure or cancel, so the reference video is released.
    """

    video: torch.Tensor | Iterator[torch.Tensor]
    audio: AudioOrNone | Iterator[AudioType]
    num_frames: int
    tiling_config: TilingConfigType | None
    audio_sampling_rate: int | None = None
    streams: tuple[Iterator[object], ...] = ()


def close_streams(result: InferenceResult) -> None:
    """Close the generators of a streamed run. A stream already finished closes cleanly."""
    for stream in result.streams:
        if isinstance(stream, Generator):
            stream.close()


def encode_streamed_output(
    video: torch.Tensor | Iterator[torch.Tensor],
    audio: AudioOrNone | Iterator[AudioType],
    audio_sampling_rate: int | None,
    *,
    fps: int,
    output_path: str,
) -> None:
    """Encode a chunked job. The windows are one video stream, like the upstream CLI."""
    from ltx_pipelines.utils.media_io import encode_video

    encode_video(
        video=video,
        fps=fps,
        audio=audio,
        output_path=output_path,
        video_chunks_number=1,
        audio_sampling_rate=audio_sampling_rate,
    )


def plan_chunks(
    chunk_pixel_frames: int | None,
    *,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: float,
) -> tuple[ChunkConfig | None, int]:
    """(chunk config, window count) for the window the job budget chose.

    ``None`` means one pass. The budget decides, so the rule is not run again here."""
    if chunk_pixel_frames is None:
        return None, 1
    from ltx_core.types import VideoPixelShape
    from ltx_pipelines.chunks import ChunkConfig
    from ltx_pipelines.chunks.layout import uniform_chunk_layouts

    config = ChunkConfig(
        chunk_pixel_frames=chunk_pixel_frames, next_video_carry_frames=IC_LORA_CHUNK_CARRY_FRAMES
    )
    layouts = uniform_chunk_layouts(
        target_pixel_shape=VideoPixelShape(
            batch=1, frames=num_frames, height=height, width=width, fps=frame_rate
        ),
        config=config,
    )
    return config, len(layouts)


def _load_source_audio(path: str, device: torch.device) -> AudioOrNone:
    from ltx_pipelines.utils.media_io import decode_audio_from_file

    src = decode_audio_from_file(path, device)
    if src is None:
        return None
    # decode_audio_from_file returns (1, channels, samples), but encode_video's writer
    # expects a 2D (channels, samples) tensor with exactly 2 channels. Feeding the 3D
    # tensor flattens it in planar order (LL..RR) and it gets misread as interleaved
    # stereo (LRLR..) -> the audio plays sped up and loops. Drop the batch dim and
    # normalize to stereo.
    waveform = stereo_waveform(src.waveform)
    if waveform.ndim == 3:
        waveform = waveform.squeeze(0)
    return replace(src, waveform=waveform)


def resolve_output_audio(
    result: InferenceResult,
    *,
    mute_audio: bool,
    source_audio_path: str | None,
    num_frames: int,
    frame_rate: float,
    device: torch.device,
) -> tuple[AudioOrNone | Iterator[AudioType], int | None]:
    """Audio and sample rate to encode.

    "off" drops it entirely; "source" muxes the input clip's soundtrack and discards the
    prompt-synthesized model audio; otherwise keep the generated audio.
    """
    if mute_audio:
        return None, None
    if source_audio_path is None:
        return result.audio, result.audio_sampling_rate
    audio = _load_source_audio(source_audio_path, device)
    if audio is None:
        return None, None
    v_dur = num_frames / frame_rate
    # Trim the source soundtrack to the generated video's length. A longer
    # source (e.g. an 8s clip rendered to 5s) would otherwise keep playing
    # over a frozen last frame.
    max_samples = int(round(v_dur * audio.sampling_rate))
    if audio.waveform.shape[-1] > max_samples:
        audio = replace(audio, waveform=audio.waveform[..., :max_samples])
    a_dur = audio.waveform.shape[-1] / audio.sampling_rate
    logger.info(
        "[ic-lora] source audio: %dHz %dch %.2fs (video %.2fs)%s",
        audio.sampling_rate, audio.waveform.shape[0], a_dur, v_dur,
        "" if abs(a_dur - v_dur) < 0.1 else "  MISMATCH",
    )
    return audio, None


def encode_result(
    result: InferenceResult,
    *,
    output_path: str,
    num_frames: int,
    frame_rate: float,
    mute_audio: bool,
    source_audio_path: str | None,
    streamed: bool,
    device: torch.device,
) -> None:
    """Mux the audio and encode the video. The streams close on success, failure and cancel.

    That releases the reference video, which a temp file delete needs on Windows.
    """
    try:
        audio, audio_rate = resolve_output_audio(
            result,
            mute_audio=mute_audio,
            source_audio_path=source_audio_path,
            num_frames=num_frames,
            frame_rate=frame_rate,
            device=device,
        )
        # round(), not int(): avoids truncating e.g. 23.976 -> 23 (encode_video int()s again).
        fps = round(frame_rate)
        if streamed:
            encode_streamed_output(result.video, audio, audio_rate, fps=fps, output_path=output_path)
        else:
            assert not isinstance(audio, Iterator)  # a one-pass job never streams audio
            encode_video_output(
                video=result.video,
                audio=audio,
                fps=fps,
                output_path=output_path,
                video_chunks_number_value=video_chunks_number(result.num_frames, result.tiling_config),
            )
    finally:
        close_streams(result)
