"""Source encode must not assemble the pixel tensor on CUDA.

Windows WDDM retake: first tiled encode reserved ~67 GiB with free=0 and still
returned; the next encode after DiffVAE decode never did. Keeping pixels on CPU
until each tile copies in is the observable contract.
"""

from __future__ import annotations

import inspect

import torch
from ltx_core.types import VIDEO_SCALE_FACTORS, VideoPixelShape

from services.retake_pipeline.source_encode import (
    _freeze_pad_pixels,
    _match_latent_length,
    encode_source_video_latent,
    extend_freeze_pad_frames,
    source_encode_tiling,
)


def test_mps_encode_uses_default_tiles_cuda_stays_small() -> None:
    from ltx_core.model.video_vae import TileSizeConfig

    default = TileSizeConfig.default()
    assert source_encode_tiling(torch.device("mps")) == default
    # 720p stays split. 540p (long edge 960) is one spatial tile.
    assert source_encode_tiling(torch.device("mps"), height=704, width=1280) == default
    small = source_encode_tiling(torch.device("mps"), height=512, width=960)
    assert small.width.tile_size >= 960
    assert small.height.tile_size >= 512
    assert small.frames == default.frames
    # Same measured cell, portrait. A square or a taller portrait is unmeasured.
    portrait = source_encode_tiling(torch.device("mps"), height=960, width=512)
    assert portrait.height.tile_size >= 960
    assert portrait.width.tile_size >= 512
    assert source_encode_tiling(torch.device("mps"), height=960, width=960) == default
    assert source_encode_tiling(torch.device("mps"), height=960, width=800) == default
    cuda = source_encode_tiling(torch.device("cuda"), height=512, width=960)
    assert cuda.frames.tile_size == 24
    assert cuda.height.tile_size == 256
    assert cuda.width.tile_size == 256


def test_mps_extend_skips_pixel_freeze_other_devices_keep_it() -> None:
    assert extend_freeze_pad_frames(torch.device("mps"), 152) == 0
    assert extend_freeze_pad_frames(torch.device("cuda"), 152) == 152
    assert extend_freeze_pad_frames(torch.device("cpu"), 152) == 152


class _FakeEncoder:
    def __init__(self) -> None:
        self.video_scale_factors = VIDEO_SCALE_FACTORS
        self.seen: torch.Tensor | None = None
        self.tiling = None

    def tiled_encode(self, video: torch.Tensor, tiling_config: object) -> torch.Tensor:
        self.seen = video
        self.tiling = tiling_config
        frames = 1 + (video.shape[2] - 1) // VIDEO_SCALE_FACTORS.time
        return torch.zeros(1, 4, frames, 2, 2)


def test_encode_source_video_latent_keeps_pixels_on_cpu(monkeypatch) -> None:
    encoder = _FakeEncoder()
    frames = [torch.zeros(1, 64, 64, 3, dtype=torch.uint8) for _ in range(9)]
    monkeypatch.setattr(
        "services.retake_pipeline.source_encode.decode_video_from_file",
        lambda **_kwargs: iter(frames),
    )
    shape = VideoPixelShape(batch=1, frames=9, height=64, width=64, fps=24.0)
    encode_source_video_latent(
        encoder,
        "clip.mp4",
        shape,
        dtype=torch.float32,
        tiling_config="tiles",
    )
    assert encoder.seen is not None
    assert encoder.seen.device.type == "cpu"
    assert encoder.tiling == "tiles"


def test_extend_and_generate_both_encode_source_from_cpu() -> None:
    """Home/GenSpace extend shares ``_run`` with retake; that is the CUDA hang path."""
    from services.retake_pipeline.ltx_retake_pipeline import LTXRetakePipeline

    run_src = inspect.getsource(LTXRetakePipeline._run)
    assert "encode_source_video_latent(" in run_src
    assert "lambda enc: encode_source_video_latent" in run_src
    assert "freeze_pad_frames=freeze_pad_frames" in run_src
    assert "pad_video_frames" not in run_src
    assert "_run(" in inspect.getsource(LTXRetakePipeline.generate)
    assert "_run(" in inspect.getsource(LTXRetakePipeline.extend)


def test_freeze_pad_pixels_repeats_last_frame_at_end() -> None:
    frames = torch.arange(3, dtype=torch.float32).view(1, 1, 3, 1, 1)
    padded = _freeze_pad_pixels(frames, 5, at="end")
    assert padded.shape[2] == 5
    assert padded[0, 0, :3, 0, 0].tolist() == [0.0, 1.0, 2.0]
    assert padded[0, 0, 3:, 0, 0].tolist() == [2.0, 2.0]


def test_freeze_pad_pixels_repeats_first_frame_at_start() -> None:
    frames = torch.arange(3, dtype=torch.float32).view(1, 1, 3, 1, 1)
    padded = _freeze_pad_pixels(frames, 5, at="start")
    assert padded[0, 0, :2, 0, 0].tolist() == [0.0, 0.0]
    assert padded[0, 0, 2:, 0, 0].tolist() == [0.0, 1.0, 2.0]


def test_match_latent_length_repeats_edge_instead_of_zeros() -> None:
    latent = torch.arange(3, dtype=torch.float32).view(1, 1, 3, 1, 1)
    matched = _match_latent_length(latent, 5)
    assert matched[0, 0, 3:, 0, 0].tolist() == [2.0, 2.0]


def test_match_latent_length_repeats_first_frame_at_start() -> None:
    latent = torch.arange(3, dtype=torch.float32).view(1, 1, 3, 1, 1)
    matched = _match_latent_length(latent, 5, at="start")
    assert matched[0, 0, :2, 0, 0].tolist() == [0.0, 0.0]
    assert matched[0, 0, 2:, 0, 0].tolist() == [0.0, 1.0, 2.0]


def test_encode_source_video_latent_freeze_pads_pixels_before_vae(monkeypatch) -> None:
    encoder = _FakeEncoder()
    frames = [
        torch.full((1, 64, 64, 3), fill_value=i, dtype=torch.uint8) for i in range(9)
    ]
    monkeypatch.setattr(
        "services.retake_pipeline.source_encode.decode_video_from_file",
        lambda **_kwargs: iter(frames),
    )
    shape = VideoPixelShape(batch=1, frames=17, height=64, width=64, fps=24.0)
    encode_source_video_latent(
        encoder,
        "clip.mp4",
        shape,
        dtype=torch.float32,
        tiling_config="tiles",
        freeze_pad_frames=8,
        freeze_at="end",
        max_duration=9 / 24,
    )
    assert encoder.seen is not None
    assert encoder.seen.shape[2] == 17
    frozen = encoder.seen[:, :, 8:9]
    assert torch.equal(encoder.seen[:, :, 9:], frozen.expand_as(encoder.seen[:, :, 9:]))


def test_encode_without_freeze_pad_repeats_the_latent_edge(monkeypatch) -> None:
    encoder = _FakeEncoder()
    frames = [
        torch.full((1, 64, 64, 3), fill_value=i, dtype=torch.uint8) for i in range(9)
    ]
    monkeypatch.setattr(
        "services.retake_pipeline.source_encode.decode_video_from_file",
        lambda **_kwargs: iter(frames),
    )
    shape = VideoPixelShape(batch=1, frames=17, height=64, width=64, fps=24.0)
    latent = encode_source_video_latent(
        encoder,
        "clip.mp4",
        shape,
        dtype=torch.float32,
        tiling_config="tiles",
        freeze_pad_frames=0,
        max_duration=9 / 24,
    )
    assert encoder.seen is not None
    assert encoder.seen.shape[2] == 9
    assert latent.shape[2] == 3
