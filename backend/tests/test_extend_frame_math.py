"""CPU-only checks for the extend latent/frame math.

The local extend path runs on CUDA/MPS and isn't exercised by the integration
suite (which uses a fake pipeline). These guard the two pure pieces of logic that
the real wrapper depends on: temporal zero-padding and seconds -> frame snapping.
"""

from __future__ import annotations

import torch

from ltx_core.types import VideoLatentShape, VideoPixelShape
from handlers.extend_handler import ExtendHandler
from services.retake_pipeline.ltx_retake_pipeline import LTXRetakePipeline


def test_pad_latent_frames_appends_for_end():
    latent = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(1, 2, 3, 4, 5)
    out = LTXRetakePipeline._pad_latent_frames(latent, 4, "end")
    assert out.shape == (1, 2, 7, 4, 5)
    # Source frames preserved at the front, zeros appended at the back.
    assert torch.equal(out[:, :, :3], latent)
    assert torch.count_nonzero(out[:, :, 3:]) == 0


def test_pad_latent_frames_prepends_for_start():
    latent = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(1, 2, 3, 4, 5)
    out = LTXRetakePipeline._pad_latent_frames(latent, 4, "start")
    assert out.shape == (1, 2, 7, 4, 5)
    # Zeros prepended at the front, source frames preserved at the back.
    assert torch.count_nonzero(out[:, :, :4]) == 0
    assert torch.equal(out[:, :, 4:], latent)


def test_pad_latent_frames_noop_when_nonpositive():
    latent = torch.ones(1, 1, 2, 2, 2)
    assert LTXRetakePipeline._pad_latent_frames(latent, 0, "end") is latent


def test_extend_latent_frame_count_matches_pixel_padding():
    # Source 8k+1 clip; extend by a multiple of 8 keeps the total 8k+1, and the
    # latent-frame delta equals extend_frames // time_scale (the wrapper's pad size).
    source = VideoPixelShape(batch=1, frames=97, height=64, width=64, fps=24.0)
    extend_frames = 96
    target = source._replace(frames=source.frames + extend_frames)
    assert (target.frames - 1) % 8 == 0
    src_latent = VideoLatentShape.from_pixel_shape(source).frames
    tot_latent = VideoLatentShape.from_pixel_shape(target).frames
    assert tot_latent - src_latent == extend_frames // 8


def test_duration_to_extend_frames_snaps_up_to_multiple_of_8():
    # 3s * 24fps = 72 (already 8k); 3.1s * 24 = 74.4 -> 74 -> snaps to 80.
    assert ExtendHandler._duration_to_extend_frames(3.0, 24.0) == 72
    assert ExtendHandler._duration_to_extend_frames(3.1, 24.0) == 80
    assert ExtendHandler._duration_to_extend_frames(4.0, 24.0) == 96
    assert ExtendHandler._duration_to_extend_frames(4.0, 24.0) % 8 == 0
