"""Run MPS ``nn.Conv3d`` as a sum of ``conv2d`` calls, one per temporal tap.

The conv video VAE decoder (``ltx-2.5-video-vae-conv``, the Fast decode path) is
built from 3x3x3 ``CausalConv3d`` layers. On Apple Silicon, torch's MPS
``conv3d`` runs at ~3 TFLOPS while ``conv2d`` runs at ~16 TFLOPS on the same
M3 Max. A stride-1 Conv3d with no temporal padding is exactly the sum over its
temporal taps of a Conv2d on time-shifted frames batched into N, so the decoder
can take the conv2d path without changing its math. Measured on an M3 Max
48 GB, 540p/5s t2v, headless backend, purged page cache: VAE decode 1:30 ->
0:21, whole generation 6:20 -> 5:12. Decoded frames match stock at 48.7 dB
PSNR / 0.991 SSIM.

MPS ``conv2d`` returns wrong values, with no error, when its weight or input is
a strided view (torch 2.11): ``weight[:, :, i]``, or a channel slice of the
decoder's ``channels_last_3d`` workspace. Both are copied to a dense layout
before each call. Frames are processed in ~512 MiB chunks to bound these
copies: for a 540p/5s decode, peak MPS driver memory goes from 9.7 to 10.7 GiB.

Any other Conv3d (non-MPS, stride, dilation, groups, temporal padding,
non-zero padding mode) runs the original ``forward``.

Remove once torch MPS ``conv3d`` reaches ``conv2d`` throughput.

Usage:
    import services.patches.mps_conv3d_as_conv2d as mps_conv3d_as_conv2d
    mps_conv3d_as_conv2d.install()
"""

from __future__ import annotations

import logging
import sys

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

# Input bytes per conv2d batch. Bounds the dense copies and the partial sums.
_CHUNK_BYTES = 512 * 2**20

_installed = False
_original_forward = nn.Conv3d.forward


def _supported(conv: nn.Conv3d, x: torch.Tensor) -> bool:
    return (
        x.device.type == "mps"
        and x.dim() == 5
        and conv.stride == (1, 1, 1)
        and conv.dilation == (1, 1, 1)
        and conv.groups == 1
        and conv.padding_mode == "zeros"
        and not isinstance(conv.padding, str)
        and conv.padding[0] == 0
    )


def _conv3d_as_conv2d(conv: nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    weight = conv.weight
    taps = [weight[:, :, tap].contiguous() for tap in range(weight.shape[2])]
    batch, in_channels, in_frames, height, width = x.shape
    out_frames = in_frames - len(taps) + 1
    kernel_height, kernel_width = conv.kernel_size[1], conv.kernel_size[2]
    pad_height, pad_width = conv.padding[1], conv.padding[2]
    out_height = height + 2 * pad_height - kernel_height + 1
    out_width = width + 2 * pad_width - kernel_width + 1
    memory_format = (
        torch.channels_last_3d if x.is_contiguous(memory_format=torch.channels_last_3d) else torch.contiguous_format
    )
    out = torch.empty(
        (batch, conv.out_channels, out_frames, out_height, out_width),
        device=x.device,
        dtype=x.dtype,
        memory_format=memory_format,
    )
    frames_per_chunk = max(1, _CHUNK_BYTES // (batch * in_channels * height * width * x.element_size()))
    for start in range(0, out_frames, frames_per_chunk):
        end = min(start + frames_per_chunk, out_frames)
        chunk_frames = batch * (end - start)

        def tap_input(tap: int) -> torch.Tensor:
            # Frames start+tap .. end+tap, batched into N and copied dense: MPS conv2d misreads strided views.
            shifted = x[:, :, start + tap : end + tap].transpose(1, 2).reshape(chunk_frames, in_channels, height, width)
            return shifted.contiguous(memory_format=torch.channels_last)

        total = F.conv2d(tap_input(0), taps[0], conv.bias, padding=(pad_height, pad_width))
        for tap in range(1, len(taps)):
            total.add_(F.conv2d(tap_input(tap), taps[tap], padding=(pad_height, pad_width)))
        out[:, :, start:end] = total.reshape(batch, end - start, conv.out_channels, out_height, out_width).transpose(1, 2)
    return out


def _patched_forward(self: nn.Conv3d, input: torch.Tensor) -> torch.Tensor:  # noqa: A002  # torch's parameter name
    if not _supported(self, input):
        return _original_forward(self, input)
    return _conv3d_as_conv2d(self, input)


def install() -> None:
    """Patch ``nn.Conv3d.forward`` on Darwin. Idempotent. No-op elsewhere."""
    global _installed
    if _installed or sys.platform != "darwin":
        return
    nn.Conv3d.forward = _patched_forward  # type: ignore[method-assign]
    _installed = True
    logger.info("MPS Conv3d runs as per-tap conv2d (conv VAE decode)")
