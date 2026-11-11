"""Monkey-patch: upload tile blend weights to MPS with a blocking copy.

ltx-core 1.6.0 added ``_device_weights`` (``ltx_core/modality_tiling.py``). It builds a tile's blend
weights on the host and uploads them once per plan with ``non_blocking=True`` on every backend.
``ltx_core.devices.allow_async_transfer`` says only CUDA orders a non-blocking transfer against the
kernel that consumes it, and ``upload_from_host`` follows that rule. ``_device_weights`` does not.
On MPS ``VideoModalityTilingHelper.blend`` can then multiply the tile prediction by weights that
have not landed yet.

Every job that tiles the transformer is affected. On an M5 Pro (48 GB, macOS 26.5.1) Refine Details
finished as ``succeeded`` and wrote a video whose frames are all zero in Y, U and V (solid green),
at 540p, 720p and 1080p and with every tiling setting (LTX-Desktop#199).

Measured on that machine by running ``LTXIcLoraPipeline`` with hooks on ``LTXModel.forward`` and
``EulerDiffusionStep.step`` (refine-details, 9 frames, 576 x 1024, one tile, same seed):

- As shipped: transformer output of step 1 is in -6.9 .. 6.8, the blended ``denoised`` passed to the
  Euler step is in -3.0e28 .. 3.8e28, step 2 is non-finite, the output is solid green.
- Blocking copy: ``denoised`` is in -5.9 .. 4.4, every step is finite, the picture is correct.

This patch keeps upstream's upload wherever ``allow_async_transfer`` allows it (CUDA) and copies
with a blocking ``.to`` everywhere else, so it is a no-op off Mac.

Remove once ltx-core's ``_device_weights`` passes ``non_blocking=allow_async_transfer(like.device)``
and Refine Details gives a real picture on a Mac without this patch.

Usage:
    import services.patches.mps_tiling_weights_sync  # noqa: F401
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import torch
from ltx_core import modality_tiling as _tiling
from ltx_core.devices import allow_async_transfer

_WeightsCache = dict[tuple[torch.dtype, torch.device], torch.Tensor]

_orig_device_weights: Callable[[_WeightsCache, Callable[[], torch.Tensor], torch.Tensor], torch.Tensor] = (
    _tiling._device_weights  # pyright: ignore[reportPrivateUsage]
)


def _device_weights_ordered(
    cache: _WeightsCache,
    weights: Callable[[], torch.Tensor],
    like: torch.Tensor,
) -> torch.Tensor:
    """Like upstream ``_device_weights``, but the upload blocks where the backend does not order it."""
    if allow_async_transfer(like.device):
        return _orig_device_weights(cache, weights, like)
    key = (like.dtype, like.device)
    if key not in cache:
        cache[key] = weights().to(dtype=like.dtype).to(device=like.device)
    return cache[key]


if list(inspect.signature(_orig_device_weights).parameters) != ["cache", "weights", "like"]:
    raise RuntimeError("modality_tiling._device_weights changed its signature — patch needs updating.")
_tiling._device_weights = _device_weights_ordered  # pyright: ignore[reportPrivateUsage]
