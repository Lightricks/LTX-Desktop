"""DiffVAE decode must drop the resident transformer before building the VAE."""

from __future__ import annotations

import pytest
import torch
from ltx_core.model.video_vae import DimensionSizeConfig, TileSizeConfig
from ltx_core.tiling import _validate_overlap

from services.patches import diffusion_stage_cache as dsc
from services.patches import diffvae_decode_vram as patch


def _tiles(*, frames: int, height: int, width: int, overlap_t: int = 40, overlap_hw: int = 160) -> TileSizeConfig:
    return TileSizeConfig(
        frames=DimensionSizeConfig(tile_size=frames, overlap=overlap_t if frames else 0),
        height=DimensionSizeConfig(tile_size=height, overlap=overlap_hw if height else 0),
        width=DimensionSizeConfig(tile_size=width, overlap=overlap_hw if width else 0),
    )


class _FakeModel:
    def __init__(self) -> None:
        self.freed_to: str | None = None

    def to(self, device: str) -> "_FakeModel":
        self.freed_to = device
        return self


class _FakeDecoder:
    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._checkpoint_path = "ltx-2.5-video-vae-bf16.safetensors"
        self.diffvae_optimization = "chunked_eager"

    @property
    def checkpoint_path(self) -> str:
        return self._checkpoint_path


@pytest.fixture(autouse=True)
def _reset_cache_state() -> None:
    dsc.set_enabled(True)
    dsc.evict()
    yield
    dsc.set_enabled(True)
    dsc.evict()


def test_patch_rebinds_video_decoder_call() -> None:
    from ltx_pipelines.utils.blocks import VideoDecoder

    assert VideoDecoder.__call__ is patch._patched_video_decoder_call


def test_video_decoder_call_evicts_cached_transformer(monkeypatch) -> None:
    model = _FakeModel()
    dsc._cached_model = model
    dsc._cached_key = ("planted",)
    monkeypatch.setattr(patch, "_orig_video_decoder_call", lambda *args, **kwargs: "ok")

    assert patch._patched_video_decoder_call(object()) == "ok"
    assert model.freed_to == "meta"
    assert dsc._cached_model is None


def test_video_decoder_call_cleans_allocator_even_when_cache_empty(monkeypatch) -> None:
    cleaned: list[bool] = []
    dsc.set_enabled(False)
    dsc.evict()
    monkeypatch.setattr(patch, "_orig_video_decoder_call", lambda *args, **kwargs: "ok")
    monkeypatch.setattr(patch, "cleanup_memory", lambda: cleaned.append(True))

    assert patch._patched_video_decoder_call(object()) == "ok"
    assert cleaned == [True]


def test_pixel_shape_from_transformer_latent_is_540p() -> None:
    # 20s 540p: (481-1)/8+1=61 latent frames, 576/32=18, 1024/32=32.
    assert patch._pixel_shape_from_latent(torch.zeros(1, 128, 61, 18, 32)) == (576, 1024, 481)


def test_pixel_shape_from_5s_latent_is_not_stage4_scale() -> None:
    # DiffVAE pixel_scale is 8 spatial and would report this as 144x256.
    assert patch._pixel_shape_from_latent(torch.zeros(1, 128, 16, 18, 32)) == (576, 1024, 121)


def test_cuda_diffvae_reresolves_tiling_after_evict(monkeypatch) -> None:
    captured: dict[str, object] = {}
    recommended = _tiles(frames=80, height=576, width=608)
    latent = torch.zeros(1, 4, 61, 18, 32)

    def _recommend(checkpoint: str, **kwargs: object) -> object:
        captured["checkpoint"] = checkpoint
        captured["kwargs"] = kwargs
        return recommended

    monkeypatch.setattr(patch, "is_diffusion_video_vae", lambda _path: True)
    monkeypatch.setattr(patch, "cuda_activation_budget_bytes", lambda _device: 6 * 1024**3)
    monkeypatch.setattr(patch, "tiling_config_for_vae", _recommend)
    monkeypatch.setattr(patch, "_orig_video_decoder_call", lambda _self, *args, **kwargs: (args, kwargs))

    decoder = _FakeDecoder(torch.device("cuda"))
    old_tiling = object()
    args, kwargs = patch._patched_video_decoder_call(decoder, latent, old_tiling)

    assert args[0] is latent
    tiling = args[1]
    assert tiling.frames.tile_size == 80
    assert tiling.height.tile_size == 0
    assert tiling.width.tile_size == 608
    assert kwargs == {}
    assert captured["checkpoint"] == decoder.checkpoint_path
    rec = captured["kwargs"]
    assert rec["height"] == 576
    assert rec["width"] == 1024
    assert rec["num_frames"] == 481
    assert rec["free_bytes"] == 6 * 1024**3
    assert rec["device"] == decoder._device


def test_cuda_diffvae_forwards_29gib_budget_to_recommend(monkeypatch) -> None:
    captured: dict[str, object] = {}
    measured = int(29.5 * 1024**3)
    monkeypatch.setattr(patch, "is_diffusion_video_vae", lambda _path: True)
    monkeypatch.setattr(patch, "cuda_activation_budget_bytes", lambda _device: measured)
    monkeypatch.setattr(
        patch,
        "tiling_config_for_vae",
        lambda _checkpoint, **kwargs: captured.update(kwargs) or _tiles(frames=80, height=576, width=608),
    )
    monkeypatch.setattr(patch, "_orig_video_decoder_call", lambda *_args, **_kwargs: "ok")

    patch._patched_video_decoder_call(
        _FakeDecoder(torch.device("cuda")),
        torch.zeros(1, 128, 61, 18, 32),
        object(),
    )
    assert captured["free_bytes"] == measured
    assert captured["width"] == 1024
    assert captured["height"] == 576
    assert captured["num_frames"] == 481


def test_clamp_splits_full_frame_540p_20s() -> None:
    clamped = patch._clamp_diffvae_tiles(_tiles(frames=481, height=576, width=1024), 576, 1024, 481)
    assert clamped.width.tile_size <= patch._MAX_DIFFVAE_TILE_SPATIAL
    assert clamped.frames.tile_size <= patch._MAX_DIFFVAE_TILE_FRAMES
    assert clamped.height.tile_size <= 576


def test_clamp_splits_full_frame_1080p_10s() -> None:
    clamped = patch._clamp_diffvae_tiles(
        _tiles(frames=241, height=1088, width=1920), 1088, 1920, 241
    )
    assert clamped.width.tile_size <= patch._MAX_DIFFVAE_TILE_SPATIAL
    assert clamped.height.tile_size <= patch._MAX_DIFFVAE_TILE_SPATIAL
    assert clamped.frames.tile_size <= patch._MAX_DIFFVAE_TILE_FRAMES


def test_clamp_untiled_axis_is_forced_split() -> None:
    clamped = patch._clamp_diffvae_tiles(_tiles(frames=0, height=0, width=0), 576, 1024, 481)
    assert clamped.frames.tile_size == patch._MAX_DIFFVAE_TILE_FRAMES
    assert clamped.width.tile_size == patch._MAX_DIFFVAE_TILE_SPATIAL
    assert clamped.height == DimensionSizeConfig()
    assert clamped.frames.overlap == patch._DEFAULT_DIFFVAE_FRAME_OVERLAP
    assert clamped.width.overlap == patch._DEFAULT_DIFFVAE_SPATIAL_OVERLAP


@pytest.mark.parametrize(
    ("frames", "height", "width"),
    [
        (9, 256, 512),
        (33, 256, 512),
        (41, 256, 512),  # IC-LoRA output: snapping 41 down to 40 must not tile it
        (49, 544, 960),
        (81, 576, 1024),
        (121, 544, 960),
        (481, 576, 1024),
        (241, 1088, 1920),
    ],
)
@pytest.mark.parametrize("upstream_frames", [0, 48, 80, 481])
def test_clamp_output_passes_upstream_overlap_floors(
    frames: int, height: int, width: int, upstream_frames: int
) -> None:
    """ltx_core 1.4.0 rejects any axis with tile_size > 0 whose overlap is below the floor."""
    upstream_tile = upstream_frames if upstream_frames < frames else 0
    upstream = _tiles(frames=upstream_tile, height=0, width=0)
    clamped = patch._clamp_diffvae_tiles(upstream, height, width, frames)
    _validate_overlap(
        clamped,
        min_overlap_frames=patch._DEFAULT_DIFFVAE_FRAME_OVERLAP,
        min_overlap_pixels=patch._DEFAULT_DIFFVAE_SPATIAL_OVERLAP,
    )


@pytest.mark.parametrize("frames", [9, 33, 41, 49, 65, 73, 80])
def test_clamp_leaves_short_clips_untiled(frames: int) -> None:
    clamped = patch._clamp_diffvae_tiles(_tiles(frames=0, height=0, width=0), 256, 512, frames)
    assert clamped == TileSizeConfig(
        frames=DimensionSizeConfig(), height=DimensionSizeConfig(), width=DimensionSizeConfig()
    )


def test_clamp_keeps_smaller_upstream_tiles_when_they_force_a_split() -> None:
    clamped = patch._clamp_diffvae_tiles(_tiles(frames=48, height=0, width=0), 256, 512, 121)
    assert clamped.frames == DimensionSizeConfig(tile_size=48, overlap=40)


def test_cuda_diffvae_clamps_full_frame_recommend_for_540p(monkeypatch) -> None:
    monkeypatch.setattr(patch, "is_diffusion_video_vae", lambda _path: True)
    monkeypatch.setattr(patch, "cuda_activation_budget_bytes", lambda _device: int(29.5 * 1024**3))
    monkeypatch.setattr(
        patch,
        "tiling_config_for_vae",
        lambda _checkpoint, **_kwargs: _tiles(frames=481, height=576, width=1024),
    )
    monkeypatch.setattr(patch, "_orig_video_decoder_call", lambda _self, *args, **kwargs: (args, kwargs))

    args, _kwargs = patch._patched_video_decoder_call(
        _FakeDecoder(torch.device("cuda")),
        torch.zeros(1, 128, 61, 18, 32),
        object(),
    )
    tiling = args[1]
    assert tiling.width.tile_size <= patch._MAX_DIFFVAE_TILE_SPATIAL
    assert tiling.frames.tile_size <= patch._MAX_DIFFVAE_TILE_FRAMES


def test_cuda_diffvae_clamps_full_frame_recommend_for_1080p_10s(monkeypatch) -> None:
    monkeypatch.setattr(patch, "is_diffusion_video_vae", lambda _path: True)
    monkeypatch.setattr(patch, "cuda_activation_budget_bytes", lambda _device: int(29.5 * 1024**3))
    monkeypatch.setattr(
        patch,
        "tiling_config_for_vae",
        lambda _checkpoint, **_kwargs: _tiles(frames=241, height=1088, width=1920),
    )
    monkeypatch.setattr(patch, "_orig_video_decoder_call", lambda _self, *args, **kwargs: (args, kwargs))

    # 10s 1080p: 1920×1088×241 → latent 60×34×31
    args, _kwargs = patch._patched_video_decoder_call(
        _FakeDecoder(torch.device("cuda")),
        torch.zeros(1, 128, 31, 34, 60),
        object(),
    )
    tiling = args[1]
    assert tiling.width.tile_size <= patch._MAX_DIFFVAE_TILE_SPATIAL
    assert tiling.height.tile_size <= patch._MAX_DIFFVAE_TILE_SPATIAL
    assert tiling.frames.tile_size <= patch._MAX_DIFFVAE_TILE_FRAMES


def test_non_cuda_decoder_keeps_pipeline_tiling(monkeypatch) -> None:
    monkeypatch.setattr(patch, "tiling_config_for_vae", lambda *_args, **_kwargs: pytest.fail("should not re-resolve"))
    monkeypatch.setattr(patch, "_orig_video_decoder_call", lambda _self, *args, **kwargs: (args, kwargs))
    latent = torch.zeros(1, 4, 61, 18, 32)
    old_tiling = object()
    args, _kwargs = patch._patched_video_decoder_call(_FakeDecoder(torch.device("cpu")), latent, old_tiling)
    assert args[1] is old_tiling


def test_mps_diffvae_clamps_untiled_1080p_10s(monkeypatch) -> None:
    monkeypatch.setattr(patch, "is_diffusion_video_vae", lambda _path: True)
    monkeypatch.setattr(patch, "tiling_config_for_vae", lambda *_args, **_kwargs: pytest.fail("should not re-resolve"))
    monkeypatch.setattr(patch, "_orig_video_decoder_call", lambda _self, *args, **kwargs: (args, kwargs))
    # 10s 1080p: 1920×1088×241 → latent 60×34×31
    args, _kwargs = patch._patched_video_decoder_call(
        _FakeDecoder(torch.device("mps")),
        torch.zeros(1, 128, 31, 34, 60),
        _tiles(frames=0, height=0, width=0),
    )
    tiling = args[1]
    assert tiling.width.tile_size <= patch._MAX_DIFFVAE_TILE_SPATIAL
    assert tiling.height.tile_size <= patch._MAX_DIFFVAE_TILE_SPATIAL
    assert tiling.frames.tile_size <= patch._MAX_DIFFVAE_TILE_FRAMES


def test_conv_vae_keeps_pipeline_tiling(monkeypatch) -> None:
    monkeypatch.setattr(patch, "is_diffusion_video_vae", lambda _path: False)
    monkeypatch.setattr(patch, "tiling_config_for_vae", lambda *_args, **_kwargs: pytest.fail("should not re-resolve"))
    monkeypatch.setattr(patch, "_orig_video_decoder_call", lambda _self, *args, **kwargs: (args, kwargs))
    latent = torch.zeros(1, 4, 61, 18, 32)
    old_tiling = object()
    args, _kwargs = patch._patched_video_decoder_call(_FakeDecoder(torch.device("cuda")), latent, old_tiling)
    assert args[1] is old_tiling
