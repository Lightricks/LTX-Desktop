"""DiffVAE decode must drop the resident transformer before building the VAE."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from services.patches import diffusion_stage_cache as dsc
from services.patches import diffvae_decode_vram as patch


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
    sentinel = SimpleNamespace(
        frames=SimpleNamespace(tile_size=80),
        height=SimpleNamespace(tile_size=576),
        width=SimpleNamespace(tile_size=608),
    )
    latent = torch.zeros(1, 4, 61, 18, 32)

    def _recommend(checkpoint: str, **kwargs: object) -> object:
        captured["checkpoint"] = checkpoint
        captured["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(patch, "is_diffusion_video_vae", lambda _path: True)
    monkeypatch.setattr(patch, "cuda_activation_budget_bytes", lambda _device: 6 * 1024**3)
    monkeypatch.setattr(patch, "tiling_config_for_vae", _recommend)
    monkeypatch.setattr(patch, "_orig_video_decoder_call", lambda _self, *args, **kwargs: (args, kwargs))

    decoder = _FakeDecoder(torch.device("cuda"))
    old_tiling = object()
    args, kwargs = patch._patched_video_decoder_call(decoder, latent, old_tiling)

    assert args[0] is latent
    assert args[1] is sentinel
    assert kwargs == {}
    assert captured["checkpoint"] == decoder.checkpoint_path
    rec = captured["kwargs"]
    assert rec["height"] == 576
    assert rec["width"] == 1024
    assert rec["num_frames"] == 481
    assert rec["free_bytes"] == 6 * 1024**3
    assert rec["device"] == decoder._device


def test_cuda_tile_budget_caps_optimistic_mem_get_info() -> None:
    assert patch._cuda_tile_budget_bytes(int(29.5 * 1024**3)) == patch._CUDA_DIFFVAE_TILE_BUDGET_CAP_BYTES
    assert patch._cuda_tile_budget_bytes(6 * 1024**3) == 6 * 1024**3


def test_cuda_diffvae_caps_29gib_budget_before_recommend(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(patch, "is_diffusion_video_vae", lambda _path: True)
    monkeypatch.setattr(patch, "cuda_activation_budget_bytes", lambda _device: int(29.5 * 1024**3))
    monkeypatch.setattr(
        patch,
        "tiling_config_for_vae",
        lambda _checkpoint, **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(patch, "_orig_video_decoder_call", lambda *_args, **_kwargs: "ok")

    patch._patched_video_decoder_call(
        _FakeDecoder(torch.device("cuda")),
        torch.zeros(1, 128, 61, 18, 32),
        object(),
    )
    assert captured["free_bytes"] == patch._CUDA_DIFFVAE_TILE_BUDGET_CAP_BYTES
    assert captured["width"] == 1024
    assert captured["height"] == 576
    assert captured["num_frames"] == 481


def test_non_cuda_decoder_keeps_pipeline_tiling(monkeypatch) -> None:
    monkeypatch.setattr(patch, "tiling_config_for_vae", lambda *_args, **_kwargs: pytest.fail("should not re-resolve"))
    monkeypatch.setattr(patch, "_orig_video_decoder_call", lambda _self, *args, **kwargs: (args, kwargs))
    latent = torch.zeros(1, 4, 61, 18, 32)
    old_tiling = object()
    args, _kwargs = patch._patched_video_decoder_call(_FakeDecoder(torch.device("cpu")), latent, old_tiling)
    assert args[1] is old_tiling


def test_conv_vae_keeps_pipeline_tiling(monkeypatch) -> None:
    monkeypatch.setattr(patch, "is_diffusion_video_vae", lambda _path: False)
    monkeypatch.setattr(patch, "tiling_config_for_vae", lambda *_args, **_kwargs: pytest.fail("should not re-resolve"))
    monkeypatch.setattr(patch, "_orig_video_decoder_call", lambda _self, *args, **kwargs: (args, kwargs))
    latent = torch.zeros(1, 4, 61, 18, 32)
    old_tiling = object()
    args, _kwargs = patch._patched_video_decoder_call(_FakeDecoder(torch.device("cuda")), latent, old_tiling)
    assert args[1] is old_tiling
