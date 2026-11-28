"""Pure DFR helpers: fps snap, Lanczos, audio tiling, keyframe bags, canvas trim.
No ``DiffusionStage`` / Gemma / ``ModelPaths``. Imported by ``dfr_stages`` and ``dfr_pipeline``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch
from PIL import Image

from ltx_core.conditioning import ConditioningItem, VideoConditionByKeyframeIndex, VideoConditionByLatentIndex
from ltx_core.tiling import DimensionTilingConfig, TileCountConfig
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_pipelines.dfr_helpers.lanczos import lanczos_x2_u8_hwc, rgb_to_uint8_fhwc
from ltx_pipelines.utils.types import LoadedImageConditioning

logger = logging.getLogger(__name__)

# Anchor keyframes carried between temporal rounds are ours, pinned just short of fully clean so a
# tile can still settle its seam frame.
ANCHOR_KEYFRAME_STRENGTH = 0.95
TEMPORAL_ANCESTRAL_ETA = 0.5
# Transformer fps is independent of playback fps. RoPE time is ``pixel_frame / fps``, so rates the
# model never saw (48, 50, 120, ...) cannot lay out the 8 pixel frames inside one latent token --
# they decode as a motion spike at each latent border followed by a stall. Anything above 30 snaps
# to 60; playback fps is used for decoding only.
MAX_CONDITIONING_FPS = 60.0
SNAP_CONDITIONING_FPS_ABOVE = 30.0


def conditioning_fps(playback_fps: float) -> float:
    """RoPE/token fps for the transformer. Values above 30 snap to 60; playback fps is unchanged."""
    return MAX_CONDITIONING_FPS if playback_fps > SNAP_CONDITIONING_FPS_ABOVE else playback_fps


# Spatial epilogue tiling. Temporal windows are sequential and pinned and are not blended; spatial
# tiles inside a time window blend after every Euler step. Every step uses one grid of tiles at most
# EPILOGUE_TILE_CELLS latent cells per side, overlapping by EPILOGUE_SPATIAL_OVERLAP cells: the blend
# only has to hide two tiles' per-step disagreement on the shared cells, which 2 cells (64 px) keep
# within ~0.5 dB of the rest of the frame.
EPILOGUE_SPATIAL_OVERLAP = 2
EPILOGUE_TILE_CELLS = 32
EPILOGUE_KEYFRAME_STRENGTH = 1.0
DETAILING_LORA_STRENGTH = 0.5


def epilogue_tile_count(cells: int) -> int:
    """Tiles along a latent axis of ``cells`` cells: the fewest whose overlapping tiles fit ``EPILOGUE_TILE_CELLS``."""
    stride = EPILOGUE_TILE_CELLS - EPILOGUE_SPATIAL_OVERLAP
    return max(1, math.ceil((cells - EPILOGUE_SPATIAL_OVERLAP) / stride))


def lanczos_x2_fhwc(frames: torch.Tensor) -> torch.Tensor:
    """Stretch each RGB frame 2x with Lanczos. ``frames`` is ``(F, H, W, C)`` in ``[0, 1]``.
    CUDA frames are resampled on their device by :mod:`ltx_pipelines.dfr_helpers.lanczos`, CPU
    frames by Pillow; both quantize to bytes first and return the same pixels.
    """
    if frames.ndim != 4:
        raise ValueError(f"Expected (F, H, W, C) RGB, got shape {tuple(frames.shape)}")
    if frames.shape[0] < 1:
        raise ValueError("Need at least one frame to Lanczos-upsample")
    if frames.shape[-1] not in (1, 3):
        raise ValueError(f"Lanczos x2 expects 1 or 3 channels, got {frames.shape[-1]}")
    if frames.is_cuda:
        resized = torch.stack([lanczos_x2_u8_hwc(frame) for frame in rgb_to_uint8_fhwc(frames)]).float()
        # A tensor divisor keeps this an IEEE division, as on the host (a Python scalar becomes a reciprocal multiply).
        return resized / torch.full((), 255.0, device=resized.device)
    out: list[torch.Tensor] = []
    for frame in frames:
        height, width, channels = frame.shape
        array = rgb_to_uint8_fhwc(frame).numpy()
        image = Image.fromarray(array[..., 0], mode="L") if channels == 1 else Image.fromarray(array, mode="RGB")
        image = image.resize((width * 2, height * 2), resample=Image.Resampling.LANCZOS)
        resized = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
        if resized.ndim == 2:
            resized = resized.unsqueeze(-1)
        out.append(resized)
    return torch.stack(out, dim=0)


def clamp_dim_tiling(cfg: DimensionTilingConfig, dim_size: int, axis: str) -> DimensionTilingConfig:
    """Clamp a single dim's tile count and overlap to the latent's extent.
    ``split_by_count`` requires ``overlap < tile_size``; with
    ``tile_size = (dim_size + overlap*(n-1)) // n`` this reduces to
    ``overlap <= dim_size - n``. When the configured overlap exceeds this
    bound it is clamped; if the latent is too small to hold ``n`` tiles
    at all, tiling falls back to a single tile on this axis.
    """
    n = cfg.num_tiles
    if n <= 1:
        return cfg
    if dim_size < n:
        logger.warning(
            "%s tiling: dim_size=%d < num_tiles=%d; falling back to 1 tile on this axis.",
            axis,
            dim_size,
            n,
        )
        return DimensionTilingConfig(1, 0)
    max_overlap = dim_size - n
    if cfg.overlap <= max_overlap:
        return cfg
    logger.warning(
        "%s tiling: overlap=%d exceeds latent bound (%d); clamping to %d.",
        axis,
        cfg.overlap,
        max_overlap,
        max_overlap,
    )
    return DimensionTilingConfig(n, max_overlap)


def clamp_tile_to_latent(tiling: TileCountConfig, latent_shape: tuple[int, int, int]) -> TileCountConfig:
    """Clamp frame, height, and width tilings to the latent's extents.
    ``latent_shape`` is ``(F, H, W)`` in latent units.
    """
    frames, height, width = latent_shape
    return replace(
        tiling,
        frames=clamp_dim_tiling(tiling.frames, frames, "Frame"),
        height=clamp_dim_tiling(tiling.height, height, "Height"),
        width=clamp_dim_tiling(tiling.width, width, "Width"),
    )


def resample_audio_time(
    audio_latent: torch.Tensor,
    src_start: float,
    src_end: float,
    out_frames: int,
) -> torch.Tensor:
    """Linearly sample ``audio_latent`` along T over ``[src_start, src_end)`` latent cells."""
    if out_frames < 1:
        raise ValueError(f"out_frames must be >= 1, got {out_frames}")
    full_t = audio_latent.shape[2]
    if full_t < 1:
        raise ValueError("Cannot resample an empty audio latent")
    span = src_end - src_start
    if span <= 0:
        raise ValueError(f"Audio window is empty: [{src_start}, {src_end})")
    step = span / out_frames
    positions = src_start + step * torch.arange(out_frames, device=audio_latent.device, dtype=torch.float32)
    positions = positions.clamp(0, full_t - 1)
    lo = positions.floor().long()
    hi = (lo + 1).clamp(max=full_t - 1)
    weight = (positions - lo.to(torch.float32)).to(dtype=audio_latent.dtype).view(1, 1, -1, 1)
    return audio_latent[:, :, lo] * (1 - weight) + audio_latent[:, :, hi] * weight


def audio_latent_for_tile(
    audio_latent: torch.Tensor,
    *,
    pixel_start: int,
    local_frames: int,
    playback_fps: float,
    source_duration: float,
    cond_fps: float,
) -> torch.Tensor:
    """Stage-1 audio for one temporal tile: playback window, resampled to the stage's token count.
    Source bounds are wall-clock: ``pixel_start / playback_fps`` through
    ``(pixel_start + local_frames) / playback_fps``, as a fraction of ``source_duration`` (stage 1's
    ``N / fps``). Temporal upsample uses ``N -> 2(N-1)+1`` while fps doubles, so the new canvas is
    slightly shorter than ``2x``; a fraction of *canvas frames* would therefore pull audio from past
    the tile's playback. ``cond_fps`` sizes only the *output* token count DiffusionStage allocates.
    """
    if local_frames <= 0:
        raise ValueError(f"local_frames must be >= 1, got {local_frames}")
    if playback_fps <= 0:
        raise ValueError(f"playback_fps must be > 0, got {playback_fps}")
    if source_duration <= 0:
        raise ValueError(f"source_duration must be > 0, got {source_duration}")
    full_t = audio_latent.shape[2]
    if full_t <= 0:
        raise ValueError("Cannot slice audio for a tile with an empty audio latent")
    src_start = pixel_start / playback_fps / source_duration * full_t
    src_end = (pixel_start + local_frames) / playback_fps / source_duration * full_t
    target_frames = AudioLatentShape.from_video_pixel_shape(
        VideoPixelShape(batch=1, frames=local_frames, height=1, width=1, fps=cond_fps)
    ).frames
    return resample_audio_time(audio_latent, src_start, src_end, target_frames)


def keyframe_conditionings_from_latents(
    keyframes: torch.Tensor,
    positions: Sequence[int],
    strength: float,
) -> list[ConditioningItem]:
    """Build ``VideoConditionByKeyframeIndex`` guides from already-encoded keyframe latents."""
    if keyframes.ndim != 5:
        raise ValueError(f"Expected keyframes (B, C, K, H, W), got {tuple(keyframes.shape)}")
    if keyframes.shape[2] != len(positions):
        raise ValueError(f"Expected {len(positions)} keyframe latents, got K={keyframes.shape[2]}")
    return [
        VideoConditionByKeyframeIndex(
            keyframes=keyframes[:, :, index : index + 1],
            frame_idx=int(frame_idx),
            strength=strength,
        )
        for index, frame_idx in enumerate(positions)
    ]


def slot_initials_from_video(
    video_latent: torch.Tensor,
    positions: Sequence[int],
    temporal_scale: int,
) -> torch.Tensor:
    """Stack the nearest video latent frames as ``(B, C, K, H, W)`` slot seeds."""
    frames = []
    for position in positions:
        index = min(max(round(int(position) / temporal_scale), 0), video_latent.shape[2] - 1)
        frames.append(video_latent[:, :, index : index + 1])
    return torch.cat(frames, dim=2)


@dataclass(frozen=True)
class TilePrefix:
    """Where a non-first tile starts, and how much of it is handed over rather than denoised.
    A tile used to begin on a mid-canvas 8-frame cell while being denoised as its own clip, whose
    cell 0 the model reads as a *single* pixel frame (causal convention). Content and shape
    disagreed there, which is why the lead-in had to be thrown away and why RoPE time inside a tile
    ran 7 frames ahead of the canvas.
    So a tile now begins on a real one-frame latent: the keyframe plane at the last plane position
    before the seam. Cell 0 is that plane, cells ``1 .. (seam - keyframe) / scale`` are the video
    between it and the seam, and the tile resumes on the cell after the seam. The frame accounting
    closes exactly -- cell ``k`` covers ``keyframe + scale*(k-1) + 1 ... keyframe + scale*k``, so the
    last pinned cell ends *on* the seam and the seam keyframe needs no conditioning of its own: it is
    absorbed into that cell. The tile then looks like an i2v clip, which is in distribution.
    Attributes:
        keyframe_position: pixel frame of the plane that becomes cell 0.
        video_start_cell: first canvas cell of the pinned video run (right after the plane).
        cells: pinned latent cells, the plane included.
        resume_pixel: first pixel frame this tile actually denoises.
    """

    keyframe_position: int
    video_start_cell: int
    cells: int
    resume_pixel: int


def tile_prefix(seam_pixel: int, plane_positions: Iterable[int], temporal_scale: int) -> TilePrefix:
    """Resolve a non-first tile's pinned prefix from the planes available before its seam."""
    before = [position for position in plane_positions if position < seam_pixel]
    if not before:
        raise RuntimeError(f"no keyframe plane before seam {seam_pixel} to start the tile on")
    keyframe = max(before)
    if keyframe % temporal_scale or seam_pixel % temporal_scale:
        raise RuntimeError(f"keyframe {keyframe} and seam {seam_pixel} must both sit on the x{temporal_scale} border")
    return TilePrefix(
        keyframe_position=keyframe,
        video_start_cell=keyframe // temporal_scale + 1,
        cells=1 + (seam_pixel - keyframe) // temporal_scale,
        resume_pixel=seam_pixel + 1,
    )


def lead_in_carryover(
    previous: tuple[int, torch.Tensor],
    prefix: TilePrefix,
    plane: torch.Tensor,
) -> ConditioningItem:
    """Pin a tile's prefix: its cell-0 plane, then the video between that plane and the seam.
    The video half used to be re-denoised from the upsampled input with the shared keyframe at the
    seam left to make the two tiles agree. This makes them agree instead: strength 1 sets
    ``denoise_mask = 0``, and ``GaussianNoiser`` does ``lerp(clean, noised, mask)``, so those cells
    *are* the previous tile's result at every step -- the mechanism A2Vid uses to freeze audio.
    ``previous`` is ``(canvas cell the previous tile's cell 0 stands for, its output)``. The lookup
    has to go through the canvas because the two tiles start in different places: a prefixed tile's
    cell 0 is its *plane*, so its cell ``k`` is canvas cell ``base + k``, while the first tile of a
    round starts on the canvas itself. Passing the video-run start for both and adding one is how
    this got the first hand-over wrong -- tile 0 has no plane to skip.
    """
    base_cell, previous_latent = previous
    video_cells = prefix.cells - 1
    offset = prefix.video_start_cell - base_cell
    if offset < 0 or offset + video_cells > previous_latent.shape[2]:
        raise RuntimeError(
            f"pinned canvas cells [{prefix.video_start_cell}, {prefix.video_start_cell + video_cells}) fall "
            f"outside the previous tile, whose cell 0 is canvas cell {base_cell} and which is "
            f"{previous_latent.shape[2]} cells long"
        )
    carried = torch.cat([plane, previous_latent[:, :, offset : offset + video_cells]], dim=2)
    if carried.shape[2] != prefix.cells:
        raise RuntimeError(f"pinned prefix is {carried.shape[2]} cells, expected {prefix.cells}")
    return VideoConditionByLatentIndex(latent=carried, strength=1.0, latent_idx=0)


def rebase_image_conditionings(
    images: Sequence[LoadedImageConditioning],
    *,
    pixel_scale: int,
    pixel_start: int = 0,
    pixel_end: int | None = None,
) -> list[LoadedImageConditioning]:
    """Map user ``frame_idx`` values from the requested canvas onto a temporally-upsampled grid.
    After ``r`` temporal rounds the same moment sits at ``frame_idx * 2**r``. When ``pixel_end`` is
    set, only images inside ``[pixel_start, pixel_end]`` are kept and indices become tile-local.
    """

    def map_frame(frame_idx: int) -> int | None:
        scaled = frame_idx * pixel_scale
        if pixel_end is not None and not (pixel_start <= scaled <= pixel_end):
            return None
        return scaled - pixel_start

    rebased: list[LoadedImageConditioning] = []
    for image in images:
        rebased_image = image.map_frame_index(map_frame)
        if rebased_image is not None:
            rebased.append(rebased_image)
    return rebased


def trim_video_to_requested_frames(
    latent: torch.Tensor,
    *,
    canvas_frames: int,
    requested_frames: int,
    temporal_upscalings: int,
    temporal_scale: int,
) -> tuple[torch.Tensor, int]:
    """Drop canvas tail padding so the caller gets ``(requested - 1) * 2**rounds + 1`` frames."""
    target_frames = (requested_frames - 1) * 2**temporal_upscalings + 1
    if target_frames > canvas_frames:
        raise RuntimeError(f"Target {target_frames} frames exceeds the generated canvas {canvas_frames}")
    if target_frames == canvas_frames:
        return latent, canvas_frames
    keep_latents = (target_frames - 1) // temporal_scale + 1
    return latent[:, :, :keep_latents], target_frames


def trim_audio_to_video_duration(
    decoded_audio: Audio,
    *,
    num_frames: int,
    playback_fps: float,
) -> Audio:
    """Cut decoded audio to the shipped video duration so the muxed container does not outlast it."""
    video_seconds = num_frames / playback_fps
    audio_samples = min(decoded_audio.waveform.shape[-1], round(video_seconds * decoded_audio.sampling_rate))
    if audio_samples != decoded_audio.waveform.shape[-1]:
        return replace(decoded_audio, waveform=decoded_audio.waveform[..., :audio_samples])
    return decoded_audio
