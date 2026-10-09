"""ICLoraPipeline subclass that tiles the reference-video VAE encode.

Upstream encodes an untiled stage's guide in one pass (``encode_tiling`` stays
``None`` unless the stage itself uses transformer tiles). Catalog IC-LoRAs
default to ``skip_stage_2`` and do not use those tiles, so a large reference
still OOMs. This override keeps Desktop's rule: ``TileSizeConfig.default()``
only when the post-downscale encode canvas is larger than one tile.

Encode size matches ltxv-api tia2v: VAE-align the stage canvas to
``VIDEO_SCALE_FACTORS * downscale``, then the encode is that divided by
``downscale``. Tiling is decided on the post-downscale tensor.

Stills at a negative ``frame_idx`` (``REFERENCE_STILL_FRAME_IDX``, -1 for Layout To Render) are 1-frame IC-LoRA
reference tokens. Upstream rejects a negative index and would drop it. They are
appended after the reference video, as ``[target | video ref | still]``.
"""

from __future__ import annotations

import logging

from ltx_core.conditioning import ConditioningItem
from ltx_core.model.video_vae import TileSizeConfig, VideoEncoder
from ltx_core.types import VIDEO_SCALE_FACTORS
from ltx_pipelines.chunks import Chunk, MakeVideoConditionings, SequentialVideoFrameSource
from ltx_pipelines.ic_lora import ICLoraPipeline, ICLoraStageConfig, _ICLoraRunContext  # pyright: ignore[reportPrivateUsage]
from ltx_pipelines.utils.helpers import combined_image_conditionings

from api_types import REFERENCE_STILL_FRAME_IDX

logger = logging.getLogger(__name__)


def reference_encode_canvas(
    stage1_height: int, stage1_width: int, downscale_factor: int
) -> tuple[int, int]:
    """VAE-aligned (height, width) the reference encode is sized from."""
    if downscale_factor < 1:
        raise ValueError(f"downscale_factor must be >= 1, got {downscale_factor}")
    align_h = VIDEO_SCALE_FACTORS.height * downscale_factor
    align_w = VIDEO_SCALE_FACTORS.width * downscale_factor
    return stage1_height // align_h * align_h, stage1_width // align_w * align_w


def reference_encode_tiling(height: int, width: int, num_frames: int) -> TileSizeConfig | None:
    """``None`` when the canvas fits in one default tile; otherwise tile.

    Time stays tiled (80 frames, overlap 24). An untiled 217-frame encode at 1024x576
    spilled a 32 GB RTX 5090 into shared memory in stage 2. The fal CG-aligned worker
    untiled time on a B200 after a 1920x1024x201 job collapsed at frame 136.
    """
    tile = TileSizeConfig.default()
    if (
        num_frames <= tile.frames.tile_size
        and height <= tile.height.tile_size
        and width <= tile.width.tile_size
    ):
        return None
    return tile


def encode_tiling_for_canvas(
    height: int, width: int, num_frames: int, downscale_factor: int
) -> TileSizeConfig | None:
    """Tile choice for a stage canvas after VAE alignment and reference downscale."""
    pass_height, pass_width = reference_encode_canvas(height, width, downscale_factor)
    encode_height = pass_height // downscale_factor
    encode_width = pass_width // downscale_factor
    return reference_encode_tiling(encode_height, encode_width, num_frames)


class TiledICLoraPipeline(ICLoraPipeline):
    def _video_conditionings(
        self,
        ctx: _ICLoraRunContext,
        stage_config: ICLoraStageConfig,
        frame_sources: dict[str, SequentialVideoFrameSource],
    ) -> MakeVideoConditionings:
        """Same closure as upstream, with Desktop's reference-encode tiling.

        A stage that runs without the IC-LoRA (stage 2 of the default recipe) attaches no
        reference video upstream, so it takes the parent closure and skips the VAE encode.
        """
        if not stage_config.apply_ic_lora:
            return super()._video_conditionings(ctx, stage_config, frame_sources)
        from ltx_pipelines.chunks.conditionings import (
            assert_image_frames_in_clip,
            image_conditionings_for_chunk,
            reference_video_conditionings_for_chunk,
        )

        frame_images = [img for img in ctx.images if img.frame_idx >= 0]
        stills = [img for img in ctx.images if img.frame_idx < 0]
        if any(img.frame_idx != REFERENCE_STILL_FRAME_IDX for img in stills):
            raise ValueError(f"A reference still must use frame_idx {REFERENCE_STILL_FRAME_IDX}")
        if stills and self.reference_downscale_factor != 1:
            # A still is placed at full stage size. A LoRA with a reference downscale
            # scales reference positions, which this path does not do for stills.
            raise ValueError(
                "A reference still at a negative frame index needs a LoRA with "
                f"reference_downscale_factor 1, got {self.reference_downscale_factor}"
            )
        assert_image_frames_in_clip(frame_images, ctx.num_frames)

        def make(chunk: Chunk) -> list[ConditioningItem]:
            if chunk.video is None:
                return []
            height = int(chunk.video.shape[-2]) * VIDEO_SCALE_FACTORS.height
            width = int(chunk.video.shape[-1]) * VIDEO_SCALE_FACTORS.width
            dtype = chunk.video.dtype
            device = chunk.video.device
            count = chunk.layout.pixel_frames
            downscale = self.reference_downscale_factor
            encode_tiling = encode_tiling_for_canvas(height, width, count, downscale)
            logger.info(
                "[ic-lora] reference VAE encode tiling %s (stage %dx%d, %d frames, downscale=%d)",
                encode_tiling,
                height,
                width,
                count,
                downscale,
            )

            def encode(enc: VideoEncoder) -> list[ConditioningItem]:
                return [
                    *image_conditionings_for_chunk(
                        chunk,
                        images=frame_images,
                        video_encoder=enc,
                        color_space=ctx.color_space,
                    ),
                    *reference_video_conditionings_for_chunk(
                        chunk,
                        video_conditioning=ctx.video_conditioning,
                        video_encoder=enc,
                        frame_sources=frame_sources,
                        downscale_factor=downscale,
                        reference_temporal_scale_factor=self.reference_temporal_scale_factor,
                        conditioning_attention_strength=ctx.conditioning_attention_strength,
                        conditioning_attention_mask=ctx.conditioning_attention_mask,
                        encode_tiling=encode_tiling,
                        color_space=ctx.color_space,
                    ),
                    # frame_idx != 0 becomes a VideoConditionByKeyframeIndex. Its time
                    # offset puts a -1 still at [-1, 0) / fps, as a 1-frame token.
                    *combined_image_conditionings(
                        images=stills,
                        height=height,
                        width=width,
                        video_encoder=enc,
                        dtype=dtype,
                        device=device,
                        color_space=ctx.color_space,
                    ),
                ]

            return self.image_conditioner(encode)

        return make
