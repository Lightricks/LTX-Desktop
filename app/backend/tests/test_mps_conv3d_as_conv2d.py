"""Per-tap conv2d MPS Conv3d must match stock Conv3d. The numeric tests need an Apple Silicon GPU.

Credit: tests first written by Sabbagh Fares (@Sabmit) in
https://github.com/Lightricks/LTX-Desktop/pull/189
"""

from __future__ import annotations

import sys

import pytest
import torch
from torch import nn

import services.patches.mps_conv3d_as_conv2d as mps_conv3d_as_conv2d

needs_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")


@pytest.fixture(autouse=True)
def _patch_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mps_conv3d_as_conv2d, "_enabled", True)


def _assert_matches_stock(conv: nn.Conv3d, x: torch.Tensor, atol: float) -> None:
    with torch.inference_mode():
        expected = mps_conv3d_as_conv2d._original_forward(conv, x)
        actual = mps_conv3d_as_conv2d._forward(conv, x)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=0)


@needs_mps
def test_channel_sliced_channels_last_input_matches_stock_conv3d() -> None:
    # MPS conv2d misreads strided input views; the decoder feeds channel slices of its workspace.
    torch.manual_seed(0)
    channels = 128
    conv = nn.Conv3d(channels, channels, 3, padding=(0, 1, 1)).to("mps", torch.bfloat16)
    conv = conv.to(memory_format=torch.channels_last_3d)
    workspace = torch.randn(1, 2 * channels, 10, 18, 32, device="mps", dtype=torch.bfloat16)
    workspace = workspace.contiguous(memory_format=torch.channels_last_3d)
    _assert_matches_stock(conv, workspace[:, :channels, 2:8], atol=0.1)


@needs_mps
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
        actual = mps_conv3d_as_conv2d._forward(conv, x)
    assert torch.equal(actual, expected)


@needs_mps
def test_input_shorter_than_temporal_kernel_uses_original_forward() -> None:
    conv = nn.Conv3d(16, 16, 3, padding=(0, 1, 1)).to("mps")
    x = torch.randn(1, 16, 2, 8, 8, device="mps")
    assert not mps_conv3d_as_conv2d._supported(conv, x)


@needs_mps
def test_last_partial_chunk_matches_stock_conv3d(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    channels, height, width = 32, 8, 8
    frame_bytes = channels * height * width * 4
    # 3 frames per chunk over 8 output frames: chunks of 3, 3, 2.
    monkeypatch.setattr(mps_conv3d_as_conv2d, "_CHUNK_BYTES", 3 * frame_bytes)
    conv = nn.Conv3d(channels, channels, 3, padding=(0, 1, 1)).to("mps")
    x = torch.randn(1, channels, 10, height, width, device="mps")
    _assert_matches_stock(conv, x, atol=1e-4)


@needs_mps
@pytest.mark.parametrize(("batch", "bias"), [(2, True), (1, False)], ids=["batch_2", "no_bias"])
def test_batch_above_one_and_missing_bias_match_stock_conv3d(batch: int, bias: bool) -> None:
    # Frames are batched into N for conv2d: a wrong batch/frame reshape only shows when batch > 1.
    torch.manual_seed(0)
    conv = nn.Conv3d(16, 24, 3, padding=(0, 1, 1), bias=bias).to("mps")
    x = torch.randn(batch, 16, 6, 10, 10, device="mps")
    _assert_matches_stock(conv, x, atol=1e-4)


def test_toggle_off_runs_original_forward_and_on_runs_patched_path(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = torch.full((1,), 7.0)
    monkeypatch.setattr(mps_conv3d_as_conv2d, "_installed", True)
    monkeypatch.setattr(mps_conv3d_as_conv2d, "_enabled", False)
    monkeypatch.setattr(mps_conv3d_as_conv2d, "_supported", lambda conv, x: True)
    monkeypatch.setattr(mps_conv3d_as_conv2d, "_conv3d_as_conv2d", lambda conv, x: sentinel)
    conv = nn.Conv3d(2, 2, 3, padding=(0, 1, 1))
    x = torch.randn(1, 2, 4, 5, 5)

    mps_conv3d_as_conv2d.set_enabled(False)
    assert torch.equal(mps_conv3d_as_conv2d._forward(conv, x), mps_conv3d_as_conv2d._original_forward(conv, x))

    mps_conv3d_as_conv2d.set_enabled(True)
    assert mps_conv3d_as_conv2d._forward(conv, x) is sentinel


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_install_leaves_conv3d_forward_untouched_off_darwin(platform: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(mps_conv3d_as_conv2d, "_installed", False)
    monkeypatch.setattr(mps_conv3d_as_conv2d, "_enabled", False)
    before = nn.Conv3d.forward
    mps_conv3d_as_conv2d.install()
    assert nn.Conv3d.forward is before
    assert mps_conv3d_as_conv2d._installed is False
