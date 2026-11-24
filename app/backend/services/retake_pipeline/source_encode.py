"""VAE-encode a source clip without assembling pixels on CUDA.

``ltx_pipelines.utils.helpers.video_latent_from_file`` decodes every frame onto
the GPU and ``torch.cat``s them. On Windows WDDM that reserved ~67 GiB on a
32 GB 5090 (``free=0``) during retake encode. The first job still returned; the
next encode after DiffVAE neighborhood-attention decode never did.

``VideoEncoder.tiled_encode`` already copies each tile to the model device, so
the pixel tensor can stay on CPU. Callers pass a probed ``output_shape`` (fps,
size, frame count) and a non-EXR path.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from ltx_core.model.video_vae import DimensionSizeConfig, TileSizeConfig, VideoEncoder
from ltx_core.tiling import TileCountConfig
from ltx_core.types import VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.helpers import decode_video_from_file, video_preprocess
from ltx_pipelines.utils.media_io.range_map import normalize_images

from handlers.video_resolution import generation_letterbox
from services.retake_pipeline.letterbox import fit_and_pad


# CUDA keeps small tiles: one encoder forward must stay small on a 32 GB card.
# MPS uses the library default (80 frames × 768). The 24×256 grid turns one
# 1080p extend into hundreds of forwards.
_CUDA_ENCODE_TILING = TileSizeConfig(
    frames=DimensionSizeConfig(tile_size=24, overlap=16),
    height=DimensionSizeConfig(tile_size=256, overlap=64),
    width=DimensionSizeConfig(tile_size=256, overlap=64),
)
# 540p snaps to 960×512. One spatial tile stayed near 8 GiB on an M4 Pro.
# 720p is 1280 and climbed to 17 GiB, so it keeps the 768 default. A square
# 960×960 has the same long edge and about twice the pixels; that stays on
# the default tiles until it is measured.
_MPS_SINGLE_TILE_MAX_PIXELS = 960 * 512


def source_encode_tiling(
    device: torch.device, *, height: int = 0, width: int = 0
) -> TileSizeConfig:
    if device.type != "mps":
        return _CUDA_ENCODE_TILING
    default = TileSizeConfig.default()
    if height <= 0 or width <= 0 or height * width > _MPS_SINGLE_TILE_MAX_PIXELS:
        return default
    return TileSizeConfig(
        frames=default.frames,
        height=DimensionSizeConfig(
            tile_size=max(default.height.tile_size, height),
            overlap=default.height.overlap,
        ),
        width=DimensionSizeConfig(
            tile_size=max(default.width.tile_size, width),
            overlap=default.width.overlap,
        ),
    )


def extend_freeze_pad_frames(device: torch.device, extend_frames: int) -> int:
    """How many held frames to VAE-encode beyond the source.

    CUDA pixel-freezes the whole tail (cloud ``pad_video``). Those copies are
    cheap there and the seam tiles see a held frame. MPS encodes the source
    only and repeats the edge latent: each Mac tile is ~30s, and the new
    frames are fully noised at sigma 1.
    """
    if device.type == "mps":
        return 0
    return extend_frames


def _match_latent_length(
    latent: torch.Tensor, expected_frames: int, *, at: str = "end"
) -> torch.Tensor:
    actual = latent.shape[2]
    if actual > expected_frames:
        return latent[:, :, :expected_frames]
    if actual < expected_frames:
        # Repeat the edge rather than zeros: zeros at the source tail sit in the
        # extend/retake seam and decode as sludge. Cloud pad_video freezes pixels.
        edge = latent[:, :, :1] if at == "start" else latent[:, :, -1:]
        extra = edge.repeat(1, 1, expected_frames - actual, *([1] * (latent.ndim - 3)))
        return torch.cat([extra, latent] if at == "start" else [latent, extra], dim=2)
    return latent


def _freeze_pad_pixels(
    frames: torch.Tensor,
    target_frames: int,
    *,
    at: str,
) -> torch.Tensor:
    """Repeat first/last pixel frame so the VAE encodes a freeze-frame tail/lead."""
    have = frames.shape[2]
    if have == target_frames:
        return frames
    if have > target_frames:
        return frames[:, :, :target_frames]
    pad = target_frames - have
    edge = frames[:, :, :1] if at == "start" else frames[:, :, -1:]
    extra = edge.repeat(1, 1, pad, 1, 1)
    return torch.cat([extra, frames] if at == "start" else [frames, extra], dim=2)


def _letterbox_preprocess(
    frames: Iterable[torch.Tensor],
    content_width: int,
    content_height: int,
    canvas_width: int,
    canvas_height: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Fit each frame to the picture and center it on the black ÷32 canvas.

    ``video_preprocess`` center-crops instead, which shifts rows relative to a
    stitch that stretches. The canvas must already be the letterbox of the picture.
    """
    box = generation_letterbox(content_width, content_height)
    if box.canvas_width != canvas_width or box.canvas_height != canvas_height:
        raise ValueError(
            f"Encode canvas {canvas_width}x{canvas_height} is not the letterbox of "
            f"{content_width}x{content_height} ({box.canvas_width}x{box.canvas_height})"
        )
    result: torch.Tensor | None = None
    for frame in frames:
        canvas = fit_and_pad(frame.reshape(frame.shape[-3:]).numpy(), box)
        image = torch.as_tensor(canvas).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        tensor = normalize_images(image.unsqueeze(2), device, dtype)
        result = tensor if result is None else torch.cat([result, tensor], dim=2)
    if result is None:
        raise ValueError("letterbox preprocess received no frames")
    return result


def encode_source_video_latent(
    video_encoder: VideoEncoder,
    file_path: str,
    output_shape: VideoPixelShape,
    *,
    dtype: torch.dtype,
    tiling_config: TileSizeConfig | TileCountConfig | None,
    start_time: float = 0.0,
    max_duration: float | None = None,
    freeze_pad_frames: int = 0,
    freeze_at: str = "end",
    content_width: int | None = None,
    content_height: int | None = None,
) -> torch.Tensor:
    cpu = torch.device("cpu")
    fps = output_shape.fps
    duration = max_duration or output_shape.frames / fps
    frame_gen = decode_video_from_file(
        path=file_path, device=cpu, start_time=start_time, max_duration=duration
    )
    if content_width is not None and content_height is not None:
        frames = _letterbox_preprocess(
            frame_gen,
            content_width,
            content_height,
            output_shape.width,
            output_shape.height,
            dtype,
            cpu,
        )
    else:
        frames = video_preprocess(
            frame_gen, output_shape.height, output_shape.width, dtype, cpu
        )
    if freeze_pad_frames > 0:
        frames = _freeze_pad_pixels(frames, output_shape.frames, at=freeze_at)
    latents = video_encoder.tiled_encode(frames, tiling_config)
    required_latent_frames = VideoLatentShape.from_pixel_shape(
        output_shape, scale_factors=video_encoder.video_scale_factors
    ).frames
    return _match_latent_length(latents, required_latent_frames, at=freeze_at)
