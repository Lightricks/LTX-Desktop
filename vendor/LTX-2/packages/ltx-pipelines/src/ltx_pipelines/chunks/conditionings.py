"""Per-item builders that turn a Chunk's window into ConditioningItem lists.
The chunk operators do not import this module. Each function produces one item
type. Pipelines close over ImageConditioner, color space, and clip-level inputs,
and compose the builders they need into ``make_video_conditionings`` /
``make_audio_conditionings``. Point items map through
:meth:`ChunkLayout.local_frame_index`; span items apply to every window.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ltx_core.conditioning import (
    AudioConditionByReferenceLatent,
    ConditioningItem,
    VideoGeneratedKeyframeSlots,
)
from ltx_core.model.video_vae import TilingConfig, VideoEncoder
from ltx_core.types import VIDEO_SCALE_FACTORS
from ltx_pipelines.chunks.export import SequentialVideoFrameSource
from ltx_pipelines.chunks.layout import audio_latent_for_layout
from ltx_pipelines.chunks.state import Chunk
from ltx_pipelines.iclora_utils import (
    append_ic_lora_reference_video_conditionings,
    patchify_audio_reference_latent,
)
from ltx_pipelines.utils.helpers import image_conditionings
from ltx_pipelines.utils.media_io import is_exr_dir
from ltx_pipelines.utils.media_io.color_config import EXRColorSpace
from ltx_pipelines.utils.types import LoadedImageConditioning


def assert_image_frames_in_clip(images: Sequence[LoadedImageConditioning], num_frames: int) -> None:
    """Raise if any image binding index lies outside the planned stitched clip."""
    for image in images:
        image.assert_in_clip(num_frames)


def images_for_chunk(chunk: Chunk, images: Sequence[LoadedImageConditioning]) -> list[LoadedImageConditioning]:
    """Remap clip-level image inputs onto ``chunk``'s stitched kept window."""
    local_images: list[LoadedImageConditioning] = []
    for image in images:
        local_image = image.map_frame_index(chunk.layout.local_frame_index)
        if local_image is not None:
            local_images.append(local_image)
    return local_images


def image_conditionings_for_chunk(
    chunk: Chunk,
    *,
    images: Sequence[LoadedImageConditioning],
    video_encoder: VideoEncoder,
) -> list[ConditioningItem]:
    """Encode image conditionings that land in ``chunk``, at its current latent spatial size."""
    local = images_for_chunk(chunk, images)
    if not local or chunk.video is None:
        return []
    height, width = _pixel_hw(chunk)
    return image_conditionings(
        images=list(local),
        height=height,
        width=width,
        video_encoder=video_encoder,
        dtype=chunk.video.dtype,
        device=chunk.video.device,
    )


def generated_keyframe_slots_for_chunk(
    chunk: Chunk,
    *,
    keyframe_indices: Sequence[int],
) -> list[ConditioningItem]:
    """Build generated-keyframe slots whose clip-level indices land in ``chunk``."""
    local = [local for idx in keyframe_indices if (local := chunk.layout.local_frame_index(idx)) is not None]
    if not local:
        return []
    return [
        VideoGeneratedKeyframeSlots(
            pixel_frame_indices=local,
            initial_keyframes=None if chunk.generated_keyframes is None else chunk.generated_keyframes.latent,
        )
    ]


def reference_video_conditionings_for_chunk(
    chunk: Chunk,
    *,
    video_conditioning: Sequence[tuple[str, float]],
    video_encoder: VideoEncoder,
    frame_sources: Mapping[str, SequentialVideoFrameSource],
    downscale_factor: int = 1,
    reference_temporal_scale_factor: int = 1,
    conditioning_attention_strength: float = 1.0,
    conditioning_attention_mask: torch.Tensor | None = None,
    encode_tiling: TilingConfig | None = None,
    color_space: EXRColorSpace | None = None,
) -> list[ConditioningItem]:
    """Encode reference video(s) sliced to ``chunk``'s pixel window, one item per pair.
    ``frame_sources`` must hold one :class:`~ltx_pipelines.chunks.export.SequentialVideoFrameSource`
    per reference path so multi-chunk runs decode each file once in order.
    ``encode_tiling``: pass ``None`` for a single-pass untiled encode or a concrete
    config for tiled encode.
    """
    if chunk.video is None or not video_conditioning:
        return []
    height, width = _pixel_hw(chunk)
    items: list[ConditioningItem] = []
    mask_slice = (
        _slice_conditioning_attention_mask_for_chunk(conditioning_attention_mask, chunk)
        if conditioning_attention_mask is not None
        else None
    )
    start = chunk.layout.start_pixel_frame
    count = chunk.layout.pixel_frames
    for path, strength in video_conditioning:
        if is_exr_dir(path):
            frames = None
        else:
            if path not in frame_sources:
                raise KeyError(f"frame_sources missing SequentialVideoFrameSource for reference path {path!r}")
            frames = frame_sources[path].take(start, count)
        append_ic_lora_reference_video_conditionings(
            items,
            [(path, strength)],
            height=height,
            width=width,
            num_frames=count,
            starting_frame=start,
            video_encoder=video_encoder,
            dtype=chunk.video.dtype,
            device=chunk.video.device,
            reference_downscale_factor=downscale_factor,
            reference_temporal_scale_factor=reference_temporal_scale_factor,
            conditioning_attention_strength=conditioning_attention_strength,
            conditioning_attention_mask=mask_slice,
            tiling_config=encode_tiling,
            color_space=color_space,
            frames=frames,
            reference_prefix_frames=chunk.layout.prev_video_carry_frames,
        )
    return items


def audio_reference_conditionings_for_chunk(
    chunk: Chunk,
    *,
    latent: torch.Tensor | None,
    fps: float,
    strength: float = 1.0,
    negative_positions: bool = True,
) -> list[ConditioningItem]:
    """Build audio-reference tokens for ``chunk``, sliced to its layout window."""
    source = chunk.audio if latent is None else audio_latent_for_layout(latent, chunk.layout, fps)
    if source is None:
        raise ValueError("audio reference with latent=None requires chunk.audio")
    ref_patch, ref_pos = patchify_audio_reference_latent(
        source,
        negative_positions=negative_positions,
        device=source.device,
    )
    return [AudioConditionByReferenceLatent(ref_patch, ref_pos, strength=strength)]


def _pixel_hw(chunk: Chunk) -> tuple[int, int]:
    assert chunk.video is not None
    return chunk.video.shape[-2] * VIDEO_SCALE_FACTORS.height, chunk.video.shape[-1] * VIDEO_SCALE_FACTORS.width


def _slice_conditioning_attention_mask_for_chunk(
    mask: torch.Tensor,
    chunk: Chunk,
) -> torch.Tensor:
    start = chunk.layout.start_pixel_frame
    end = start + chunk.layout.pixel_frames
    if mask.shape[2] < end:
        raise ValueError(f"conditioning_attention_mask has {mask.shape[2]} frames but chunk needs [{start}, {end})")
    return mask[:, :, start:end, :, :]
