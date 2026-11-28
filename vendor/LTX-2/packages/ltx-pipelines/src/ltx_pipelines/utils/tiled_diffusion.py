"""Shared single-GPU transformer tiling for diffusion stages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ltx_core.modality_tiling import VideoModalityTilingHelper
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.video_vae import TilingConfig
from ltx_core.tiling import DEFAULT_SPLIT_OPERATION, SplitOperation, Tile, split_by_size_pinned
from ltx_core.tools import LatentTools, VideoLatentTools
from ltx_core.types import SpatioTemporalScaleFactors
from ltx_pipelines.utils.blocks import DiffusionStage

if TYPE_CHECKING:
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig


@dataclass(frozen=True)
class FixedSizeSpatialTiling:
    """Fixed transformer windows with a minimum overlap in pixel units.
    Every tile remains exactly ``height`` by ``width`` unless the full canvas is
    smaller on that axis. First and last tile origins are pinned to the canvas;
    interior origins are redistributed so the requested overlap is a minimum.
    """

    height: int
    width: int
    height_overlap: int
    width_overlap: int

    def to_splitters(
        self,
        scale_factors: SpatioTemporalScaleFactors,
        min_tile_size: tuple[int, int, int] | None = None,
        *,
        causal_temporal: bool = False,
    ) -> tuple[SplitOperation, SplitOperation, SplitOperation]:
        """Resolve fixed pixel windows onto the transformer's latent grid."""
        del min_tile_size, causal_temporal
        values = (
            ("height", self.height, self.height_overlap, scale_factors.height),
            ("width", self.width, self.width_overlap, scale_factors.width),
        )
        for axis, size, overlap, scale in values:
            if size % scale or overlap % scale:
                raise ValueError(f"{axis} tile size and overlap must be divisible by {scale}")
        return (
            DEFAULT_SPLIT_OPERATION,
            split_by_size_pinned(self.height // scale_factors.height, self.height_overlap // scale_factors.height),
            split_by_size_pinned(self.width // scale_factors.width, self.width_overlap // scale_factors.width),
        )


DiffusionTilingConfig = TilingConfig | FixedSizeSpatialTiling


def diffusion_stage_with_tiling(
    stage: DiffusionStage,
    tiling: DiffusionTilingConfig | None,
) -> DiffusionStage:
    """Return a diffusion stage that tiles each transformer call when ``tiling`` is set."""
    if tiling is None:
        return stage

    def wrap(model: torch.nn.Module, tools: LatentTools | None) -> torch.nn.Module:
        if not isinstance(tools, VideoLatentTools):
            raise TypeError(f"Spatial tiling requires video tools, got {type(tools).__name__}")
        return TiledDiffusionModel(model, video_tools=tools, tiling=tiling)

    return stage.with_model_wrapper(wrap)


class TiledDiffusionModel(torch.nn.Module):
    """Run one transformer call over sequential video tiles and blend its prediction.
    The single-GPU counterpart of
    :class:`~ltx_core.multigpu.transformer.tiled_data_parallel.TiledDataParallelModelWrapper`:
    same :class:`VideoModalityTilingHelper`, same trapezoidal blend masks, same
    conditioning-token filtering. Tiles run here rather than round-robin across
    ranks. This wraps the ``X0Model`` instead of the bare velocity model, so it
    blends x0 predictions; ``to_denoised`` is affine in the velocity and every
    tile sees the same per-token timesteps, so with blend weights summing to 1
    the two are the same value.
    The sampler still owns one full-canvas latent and advances it once per diffusion
    step. This wrapper only changes how that step's model prediction is evaluated:
    each tile runs sequentially, and overlapping predictions are accumulated with
    the tiling subsystem's separable trapezoidal masks.
    ``seams`` are optional known temporal handover cells. They are used by DFR to
    denoise overlap for context and then discard it instead of blending. Ordinary
    spatial IC-LoRA tiling leaves them empty.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        video_tools: VideoLatentTools,
        tiling: DiffusionTilingConfig,
        seams: Sequence[int] = (),
        normalize_positions: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self._normalize_positions = normalize_positions
        self._helper = VideoModalityTilingHelper(tiling, video_tools, seams=seams)  # type: ignore[arg-type]

    @property
    def num_blocks(self) -> int:
        """Number of transformer blocks in the wrapped model."""
        return self.model.num_blocks

    @property
    def tiles(self) -> list[Tile]:
        """Tiles walked by every transformer call, in blend order."""
        return self._helper.tiles

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Evaluate and blend one full-canvas model prediction."""
        if video is None:
            return self.model(video, audio, perturbations)

        denoised_video: torch.Tensor | None = None
        denoised_audio: torch.Tensor | None = None
        for tile in self._helper.tiles:
            tiled_video, context = self._helper.tile_modality(
                video,
                tile,
                normalize_positions=self._normalize_positions,
            )
            tile_video, tile_audio = self.model(tiled_video, audio, perturbations)
            if tile_video is None:
                raise RuntimeError("Transformer returned no video prediction for a video tile")
            denoised_video = self._helper.blend(tile_video, tile, context, denoised_video)
            if tile_audio is not None:
                denoised_audio = tile_audio if denoised_audio is None else denoised_audio + tile_audio

        if denoised_video is None:
            raise RuntimeError("Tiled transformer produced no video prediction")
        if denoised_audio is not None:
            # Every tile sees the full audio stream with a different video context.
            denoised_audio = denoised_audio / len(self._helper.tiles)
        return denoised_video, denoised_audio
