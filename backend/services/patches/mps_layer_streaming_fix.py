"""Monkey-patch: skip pin_memory() in _LayerStore.move_to_gpu on MPS.

pin_memory() is a CUDA concept for host-pinned memory that enables async
DMA H2D transfers.  MPS (Apple Metal Performance Shaders) does not support
it: calling tensor.pin_memory() on a CPU tensor when the target device is
MPS raises:

    RuntimeError: Attempted to set the storage of a tensor on device "cpu"
    to a storage on different device "mps:0".  This is no longer allowed;
    the devices must match.

This patch replaces _LayerStore.move_to_gpu with an implementation that
skips pin_memory() when the target device is MPS and does a direct
synchronous .to(device) copy instead.

Remove this patch once ltx-core adds MPS awareness to _LayerStore.

Usage:
    import services.patches.mps_layer_streaming_fix  # noqa: F401
"""

from __future__ import annotations

import itertools

import torch
from torch import nn

from ltx_core.layer_streaming import _LayerStore  # type: ignore[reportPrivateImportUsage]


_original_move_to_gpu = _LayerStore.move_to_gpu


def _patched_move_to_gpu(
    self: _LayerStore,
    idx: int,
    layer: nn.Module,
    *,
    non_blocking: bool = False,
) -> None:
    """Move layer idx to GPU, skipping pin_memory() on MPS targets."""
    self._check_idx(idx)
    if idx in self._on_gpu:
        return
    source = self._source_data[idx]

    if self.target_device.type == "mps":
        # MPS does not support pinned host memory — copy directly.
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            param.data = source[name].to(self.target_device, non_blocking=False)
        # No pinned_in_flight tracking needed for MPS (transfers are synchronous).
        self._on_gpu.add(idx)
    else:
        _original_move_to_gpu(self, idx, layer, non_blocking=non_blocking)


# Apply patch.
assert hasattr(_LayerStore, "move_to_gpu") and callable(getattr(_LayerStore, "move_to_gpu")), (
    "_LayerStore.move_to_gpu not found — was it renamed? The mps_layer_streaming_fix patch needs updating."
)
_LayerStore.move_to_gpu = _patched_move_to_gpu  # type: ignore[assignment]
