"""Monkey-patch: replace record_stream with manual GPU reference tracking.

record_stream corrupts the CUDA caching allocator on certain GPU/driver
combinations (observed on RTX 5090 / CUDA 12.8 / PyTorch 2.10), causing
access violations during subsequent safetensors model loads.

This patch replaces record_stream calls in LayerStreamingWrapper with
explicit Python reference holding + per-layer CUDA events, achieving the
same protection (prevent allocator reuse while compute reads a tensor)
without touching allocator metadata.

Remove this patch once the upstream ltx-core package includes the fix.

Usage:
    import services.patches.record_stream_fix  # noqa: F401
"""

from __future__ import annotations

import functools
import itertools
from typing import Any

import torch
from torch import nn

from ltx_core.layer_streaming import LayerStreamingWrapper


def _patched_register_hooks(self: LayerStreamingWrapper) -> None:
    idx_map: dict[int, int] = {id(layer): idx for idx, layer in enumerate(self._layers)}
    num_layers = len(self._layers)

    compute_stream = torch.cuda.current_stream(self._target_device)

    gpu_refs: dict[int, list[torch.Tensor]] = {}
    ref_events: dict[int, torch.cuda.Event] = {}

    def _drain_completed_refs() -> None:
        for layer_idx in list(ref_events):
            if ref_events[layer_idx].query():
                del gpu_refs[layer_idx]
                del ref_events[layer_idx]

    def _pre_hook(
        module: nn.Module,
        _args: Any,
        *,
        idx: int,
    ) -> None:
        self._prefetcher.wait(idx)
        if not self._store.is_on_gpu(idx):
            self._store.move_to_gpu(idx, module)

        gpu_refs[idx] = [param.data for param in itertools.chain(module.parameters(), module.buffers())]

        _drain_completed_refs()

        for offset in range(1, self._prefetch_count + 1):
            self._prefetcher.prefetch((idx + offset) % num_layers)

    def _post_hook(
        module: nn.Module,
        _args: Any,
        _output: Any,
        *,
        idx: int,
    ) -> None:
        event = torch.cuda.Event()
        event.record(compute_stream)
        ref_events[idx] = event

        self._store.evict_to_cpu(idx, module)

    self._gpu_refs = gpu_refs
    self._ref_events = ref_events

    for layer in self._layers:
        idx = idx_map[id(layer)]
        h1 = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
        h2 = layer.register_forward_hook(functools.partial(_post_hook, idx=idx))
        self._hooks.extend([h1, h2])


_original_teardown = LayerStreamingWrapper.teardown


def _patched_teardown(self: LayerStreamingWrapper) -> None:
    # Clear held GPU references before the original teardown evicts layers.
    if hasattr(self, "_gpu_refs"):
        torch.cuda.synchronize(device=self._target_device)
        self._gpu_refs.clear()
        self._ref_events.clear()
    _original_teardown(self)


# Apply patches.
LayerStreamingWrapper._register_hooks = _patched_register_hooks  # type: ignore[assignment]
LayerStreamingWrapper.teardown = _patched_teardown  # type: ignore[assignment]
