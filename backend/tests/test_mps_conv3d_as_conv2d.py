"""Per-tap conv2d MPS Conv3d must match stock Conv3d. Needs an Apple Silicon GPU."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import services.patches.mps_conv3d_as_conv2d as mps_conv3d_as_conv2d

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


def _assert_matches_stock(conv: nn.Conv3d, x: torch.Tensor, atol: float) -> None:
    with torch.inference_mode():
        expected = mps_conv3d_as_conv2d._original_forward(conv, x)
        actual = mps_conv3d_as_conv2d._patched_forward(conv, x)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=0)


def test_channel_sliced_channels_last_input_matches_stock_conv3d() -> None:
    # MPS conv2d misreads strided input views; the decoder feeds channel slices of its workspace.
    torch.manual_seed(0)
    channels = 128
    conv = nn.Conv3d(channels, channels, 3, padding=(0, 1, 1)).to("mps", torch.bfloat16)
    conv = conv.to(memory_format=torch.channels_last_3d)
    workspace = torch.randn(1, 2 * channels, 10, 18, 32, device="mps", dtype=torch.bfloat16)
    workspace = workspace.contiguous(memory_format=torch.channels_last_3d)
    _assert_matches_stock(conv, workspace[:, :channels, 2:8], atol=0.1)


@pytest.mark.parametrize(
    ("stride", "padding"),
    [((2, 2, 2), (0, 1, 1)), ((1, 1, 1), (1, 1, 1))],
    ids=["strided", "temporally_padded"],
)
def test_strided_or_temporally_padded_conv3d_uses_original_forward(
    stride: tuple[int, int, int], padding: tuple[int, int, int]
) -> None:
    torch.manual_seed(0)
    conv = nn.Conv3d(16, 16, 3, stride=stride, padding=padding).to("mps")
    x = torch.randn(1, 16, 8, 12, 12, device="mps")
    with torch.inference_mode():
        expected = mps_conv3d_as_conv2d._original_forward(conv, x)
        actual = mps_conv3d_as_conv2d._patched_forward(conv, x)
    assert torch.equal(actual, expected)


def test_last_partial_chunk_matches_stock_conv3d(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    channels, height, width = 32, 8, 8
    frame_bytes = channels * height * width * 4
    # 3 frames per chunk over 8 output frames: chunks of 3, 3, 2.
    monkeypatch.setattr(mps_conv3d_as_conv2d, "_CHUNK_BYTES", 3 * frame_bytes)
    conv = nn.Conv3d(channels, channels, 3, padding=(0, 1, 1)).to("mps")
    x = torch.randn(1, channels, 10, height, width, device="mps")
    _assert_matches_stock(conv, x, atol=1e-4)
