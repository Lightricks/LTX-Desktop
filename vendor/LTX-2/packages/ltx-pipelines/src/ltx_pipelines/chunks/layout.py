"""Chunk window layout: sizing, carry, and temporal windows."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from ltx_core.types import VIDEO_SCALE_FACTORS, AudioLatentShape, VideoPixelShape
from ltx_pipelines.utils.helpers import snap_frames_to_grid

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkConfig:
    """Window size, carry, and decoded overlap blending for uniform chunk planning."""

    chunk_pixel_frames: int = 97
    next_video_carry_frames: int = 25
    overlap_blend_frames: int | None = None

    def __post_init__(self) -> None:
        blend = self.next_video_carry_frames if self.overlap_blend_frames is None else self.overlap_blend_frames
        object.__setattr__(self, "overlap_blend_frames", blend)
        if blend < 0:
            raise ValueError(f"overlap_blend_frames ({blend}) must be >= 0")
        if blend > self.next_video_carry_frames:
            raise ValueError(
                f"overlap_blend_frames ({blend}) must be <= next_video_carry_frames ({self.next_video_carry_frames})"
            )


@dataclass(frozen=True)
class ChunkLayout:
    """Temporal window for one chunk on the stitched **video pixel** timeline.
    All ``*_frames`` counts are pixel frames on the stitched video timeline.
    Attributes:
        pixel_frames: Full generation window for this chunk, previous carry included.
        start_pixel_frame: Global pixel-frame index where this window begins (previous
            carry included). Kept decoded output starts at
            ``start_pixel_frame + prev_video_carry_frames``.
        prev_video_carry_frames: Leading pixel frames this chunk re-renders in overlap
            with the previous chunk. These frames are dropped after decode because the
            prior chunk already emitted them (0 on chunk 0).
        next_video_carry_frames: Pixel duration this chunk passes to the next chunk (0 on
            the last chunk). After denoise, the matching tail is sliced from the end of
            the video (and audio) latent and pinned at index 0 for continuity.
        next_video_blend_frames: Number of decoded tail frames to retain and crossfade
            with the next chunk's overlap (0 on the last chunk).
    """

    pixel_frames: int
    start_pixel_frame: int
    prev_video_carry_frames: int
    next_video_carry_frames: int
    next_video_blend_frames: int = 0

    def __post_init__(self) -> None:
        t = VIDEO_SCALE_FACTORS.time
        if self.pixel_frames != 0 and (self.pixel_frames - 1) % t != 0:
            raise ValueError(
                f"pixel_frames ({self.pixel_frames}) must be on the causal grid ((n - 1) % {t} == 0, e.g. 97), or 0"
            )
        if self.prev_video_carry_frames != 0 and (self.prev_video_carry_frames - 1) % t != 0:
            raise ValueError(
                f"prev_video_carry_frames ({self.prev_video_carry_frames}) must be on the causal grid "
                f"((n - 1) % {t} == 0, e.g. 97), or 0"
            )
        if self.next_video_carry_frames != 0 and (self.next_video_carry_frames - 1) % t != 0:
            raise ValueError(
                f"next_video_carry_frames ({self.next_video_carry_frames}) must be on the causal grid "
                f"((n - 1) % {t} == 0, e.g. 97), or 0"
            )
        if self.prev_video_carry_frames < 0:
            raise ValueError(f"prev_video_carry_frames ({self.prev_video_carry_frames}) must be >= 0")
        if self.next_video_carry_frames < 0:
            raise ValueError(f"next_video_carry_frames ({self.next_video_carry_frames}) must be >= 0")
        if self.prev_video_carry_frames >= self.pixel_frames:
            raise ValueError(
                f"prev_video_carry_frames ({self.prev_video_carry_frames}) must be < pixel_frames ({self.pixel_frames})"
            )
        if self.next_video_carry_frames >= self.pixel_frames:
            raise ValueError(
                f"next_video_carry_frames ({self.next_video_carry_frames}) must be < pixel_frames ({self.pixel_frames})"
            )
        if not 0 <= self.next_video_blend_frames <= self.next_video_carry_frames:
            raise ValueError(
                f"next_video_blend_frames ({self.next_video_blend_frames}) must be between 0 and "
                f"next_video_carry_frames ({self.next_video_carry_frames})"
            )
        kept_frames = self.pixel_frames - self.prev_video_carry_frames
        if self.next_video_blend_frames >= kept_frames:
            raise ValueError(
                f"next_video_blend_frames ({self.next_video_blend_frames}) must be < kept pixel frames ({kept_frames})"
            )

    def local_frame_index(self, global_idx: int) -> int | None:
        """Map a stitched-timeline pixel index onto this chunk, or ``None`` if another chunk owns it.
        The owning window is the kept range after dropping the previous chunk's carry
        (``start_pixel_frame + prev_video_carry_frames`` … ``start_pixel_frame + pixel_frames``).
        The local index is ``global_idx - start_pixel_frame``, which places a seam frame at the carry.
        """
        kept_start = self.start_pixel_frame + self.prev_video_carry_frames
        kept_end = self.start_pixel_frame + self.pixel_frames
        if kept_start <= global_idx < kept_end:
            return global_idx - self.start_pixel_frame
        return None

    @property
    def owned_pixel_range(self) -> tuple[int, int]:
        """Half-open stitched-timeline range owned after the pinned incoming carry."""
        start = self.start_pixel_frame + self.prev_video_carry_frames
        end = self.start_pixel_frame + self.pixel_frames
        return start, end


def _split_target_pixel_frames_into_chunk_lengths(
    target_pixel_frames: int, chunk_pixel_frames: int, next_video_carry_frames: int
) -> list[int]:
    """Per-chunk pixel lengths; stitched output floors ``target_pixel_frames`` to the causal grid."""
    t = VIDEO_SCALE_FACTORS.time
    if chunk_pixel_frames != 0 and (chunk_pixel_frames - 1) % t != 0:
        raise ValueError(
            f"chunk_pixel_frames ({chunk_pixel_frames}) must be on the causal grid ((n - 1) % {t} == 0, e.g. 97), or 0"
        )
    if next_video_carry_frames != 0 and (next_video_carry_frames - 1) % t != 0:
        raise ValueError(
            f"next_video_carry_frames ({next_video_carry_frames}) must be on the causal grid "
            f"((n - 1) % {t} == 0, e.g. 97), or 0"
        )

    chunk_latent_frames = (chunk_pixel_frames - 1) // t + 1
    target_latent_frames = (target_pixel_frames - 1) // t + 1
    if target_latent_frames <= chunk_latent_frames:
        return [(target_latent_frames - 1) * t + 1]

    carry_latent_frames = (next_video_carry_frames - 1) // t + 1 if next_video_carry_frames != 0 else 0
    if carry_latent_frames < 3:
        raise ValueError(
            f"next_video_carry_frames ({next_video_carry_frames}) must be >= "
            f"{(3 - 1) * t + 1} on the causal grid when chunking"
        )
    if carry_latent_frames >= chunk_latent_frames:
        raise ValueError(
            f"next_video_carry_frames ({next_video_carry_frames}) must be < chunk pixel frames ({chunk_pixel_frames})"
        )

    new_latent_frames = chunk_latent_frames - carry_latent_frames
    extra_full, remainder = divmod(target_latent_frames - chunk_latent_frames, new_latent_frames)
    sizes = [chunk_pixel_frames] * (1 + extra_full)
    if remainder:
        sizes.append((carry_latent_frames + remainder - 1) * t + 1)
    return sizes


def uniform_chunk_layouts(
    *,
    target_pixel_shape: VideoPixelShape,
    config: ChunkConfig,
) -> list[ChunkLayout]:
    """Uniform fixed-size chunking plan for the target pixel shape.
    The stitched clip length floors ``target_pixel_shape.frames`` to the causal grid
    (``(frames - 1) % 8 == 0``), matching single-shot ``VideoLatentShape.from_pixel_shape``.
    When ``chunk_pixel_frames`` covers the snapped clip, the plan is a single chunk with no
    carry.
    """
    target_frames = target_pixel_shape.frames
    snapped_frames = snap_frames_to_grid(target_frames)
    if snapped_frames != target_frames:
        logger.info(
            "Target %d pixel frames floors to %d on the causal grid",
            target_frames,
            snapped_frames,
        )
    chunk_lengths = _split_target_pixel_frames_into_chunk_lengths(
        target_frames, config.chunk_pixel_frames, config.next_video_carry_frames
    )

    stitched_start = 0
    layouts: list[ChunkLayout] = []
    last = len(chunk_lengths) - 1
    for i, size in enumerate(chunk_lengths):
        prev_carry = 0 if i == 0 else layouts[i - 1].next_video_carry_frames
        next_carry = 0 if i == last else config.next_video_carry_frames
        layouts.append(
            ChunkLayout(
                pixel_frames=size,
                start_pixel_frame=stitched_start - prev_carry,
                prev_video_carry_frames=prev_carry,
                next_video_carry_frames=next_carry,
                next_video_blend_frames=0 if i == last else config.overlap_blend_frames,
            )
        )
        stitched_start += size - prev_carry
    return layouts


def audio_latent_for_layout(
    full: torch.Tensor,
    layout: ChunkLayout,
    fps: float,
) -> torch.Tensor:
    """Audio latent covering ``layout``'s pixel window at ``fps``.
    A source shorter than the window is zero-padded on the right so the slice always
    matches the chunk's audio latent length (same as one-shot A2Vid feeding a short
    frozen latent into a longer video).
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    start = AudioLatentShape.from_duration(batch=full.shape[0], duration=float(layout.start_pixel_frame) / fps).frames
    window = VideoPixelShape(
        batch=full.shape[0],
        frames=layout.pixel_frames,
        height=1,
        width=1,
        fps=fps,
    )
    n_frames = AudioLatentShape.from_video_pixel_shape(window).frames
    end = start + n_frames
    if start < 0:
        raise ValueError(
            f"audio latent slice [{start}:{end}] starts before the source "
            f"for start_pixel_frame={layout.start_pixel_frame} pixel_frames={layout.pixel_frames} fps={fps}"
        )
    available = full[:, :, start:end]
    if available.shape[2] == n_frames:
        return available
    pad = n_frames - available.shape[2]
    return torch.nn.functional.pad(available, (0, 0, 0, pad))
