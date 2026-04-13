"""Monkey-patch: replace torch.cuda.synchronize() calls with device-aware sync.

ltx_pipelines calls torch.cuda.synchronize() unconditionally in two places:
- gpu_model() in ltx_pipelines.utils.gpu_model
- _streaming_model() in ltx_pipelines.utils.blocks

On MPS (Apple Silicon) both raise:

    AssertionError: Torch not compiled with CUDA enabled

This patch replaces both context managers with implementations that dispatch to
the correct synchronization primitive based on the actual device in use:
  CUDA → torch.cuda.synchronize, MPS → torch.mps.synchronize, CPU → no-op.

For _streaming_model() on MPS, LayerStreamingWrapper cannot be used because it
relies on torch.cuda.Stream, torch.cuda.Event, and torch.cuda.current_stream
throughout (in _AsyncPrefetcher and _register_hooks).  Instead, we provide
_MpsLayerStreamingWrapper, a synchronous layer-streaming implementation that
moves each layer to MPS immediately before its forward pass and evicts it
immediately after, without any CUDA stream machinery.

cleanup_memory() is called indirectly via _cleanup_memory(), which looks up the
function through ltx_pipelines.utils.helpers at call time rather than capturing
it at import time.  This ensures we pick up any later patches applied to
cleanup_memory (e.g. by LTXTextEncoder._install_cleanup_memory_patch).

Remove this patch once ltx-pipelines adds MPS awareness to these functions.

Usage:
    import services.patches.mps_gpu_model_fix  # noqa: F401
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

import torch
import torch.nn as nn

import ltx_pipelines.utils.gpu_model as _gpu_model_module
import ltx_pipelines.utils.helpers as _helpers_module

_M = TypeVar("_M", bound=torch.nn.Module)


def _cleanup_memory() -> None:
    # Always look up through the module so we pick up any later patches
    # (e.g. the cleanup_memory patch installed by LTXTextEncoder).
    _helpers_module.cleanup_memory()


def _synchronize_device(device: torch.device) -> None:
    """Run a device synchronization appropriate for the given device."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    # CPU: no synchronization needed.


def _synchronize_model(model: torch.nn.Module) -> None:
    """Run a device synchronization for all devices a model's tensors live on."""
    devices: set[torch.device] = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        devices.add(tensor.device)
    for device in devices:
        _synchronize_device(device)


def _resolve_attr(module: nn.Module, dotted_path: str) -> nn.ModuleList:
    obj: Any = module
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, nn.ModuleList):
        raise TypeError(f"Expected nn.ModuleList at '{dotted_path}', got {type(obj).__name__}")
    return obj


class _MpsLayerStreamingWrapper(nn.Module):
    """Synchronous layer-streaming wrapper for MPS (Apple Silicon).

    LayerStreamingWrapper cannot be used on MPS because _AsyncPrefetcher and
    _register_hooks use torch.cuda.Stream/Event/current_stream throughout.

    This wrapper achieves the same memory-reduction goal — keep transformer
    layers on CPU and move them to MPS one at a time — using synchronous
    transfers and forward hooks.  There is no async prefetch, so throughput
    may be slightly lower than CUDA, but peak GPU memory is the same.

    Non-layer parameters and buffers are moved to MPS at setup time (matching
    the behaviour of LayerStreamingWrapper._setup).
    """

    def __init__(
        self,
        model: nn.Module,
        layers_attr: str,
        target_device: torch.device,
    ) -> None:
        super().__init__()
        self._model = model
        self._layers = _resolve_attr(model, layers_attr)
        self._target_device = target_device
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        # Record source (CPU) tensors for each layer so we can restore them
        # after eviction — same approach as _LayerStore.
        self._source_data: list[dict[str, torch.Tensor]] = []
        for layer in self._layers:
            source: dict[str, torch.Tensor] = {}
            for name, tensor in itertools.chain(layer.named_parameters(), layer.named_buffers()):
                source[name] = tensor.data
            self._source_data.append(source)

        self._setup()

    def _setup(self) -> None:
        # Move non-layer params/buffers to MPS so the rest of the model is
        # ready for computation without waiting for layer transfers.
        layer_tensor_ids: set[int] = set()
        for layer in self._layers:
            for t in itertools.chain(layer.parameters(), layer.buffers()):
                layer_tensor_ids.add(id(t))

        for p in self._model.parameters():
            if id(p) not in layer_tensor_ids:
                p.data = p.data.to(self._target_device)
        for b in self._model.buffers():
            if id(b) not in layer_tensor_ids:
                b.data = b.data.to(self._target_device)

        self._register_hooks()

    def _move_layer_to_device(self, idx: int, layer: nn.Module) -> None:
        source = self._source_data[idx]
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            param.data = source[name].to(self._target_device, non_blocking=False)

    def _evict_layer_to_cpu(self, idx: int, layer: nn.Module) -> None:
        source = self._source_data[idx]
        for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            param.data = source[name]

    def _register_hooks(self) -> None:
        idx_map: dict[int, int] = {id(layer): idx for idx, layer in enumerate(self._layers)}

        def _pre_hook(module: nn.Module, _args: Any, *, idx: int) -> None:
            self._move_layer_to_device(idx, module)

        def _post_hook(module: nn.Module, _args: Any, _output: Any, *, idx: int) -> None:
            self._evict_layer_to_cpu(idx, module)

        for layer in self._layers:
            idx = idx_map[id(layer)]
            h1 = layer.register_forward_pre_hook(functools.partial(_pre_hook, idx=idx))
            h2 = layer.register_forward_hook(functools.partial(_post_hook, idx=idx))
            self._hooks.extend([h1, h2])

    def teardown(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._source_data.clear()

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)


@contextmanager
def _patched_gpu_model(model: _M) -> Iterator[_M]:
    """Device-aware replacement for gpu_model().

    Identical to the original except it dispatches synchronization based on
    the model's actual device rather than assuming CUDA.
    """
    try:
        yield model
    finally:
        _synchronize_model(model)
        model.to("meta")
        _cleanup_memory()


import ltx_pipelines.utils.blocks as _blocks_module  # noqa: E402
from ltx_pipelines.utils.blocks import LayerStreamingWrapper  # noqa: E402


@contextmanager
def _patched_streaming_model(
    model: _M,
    layers_attr: str,
    target_device: torch.device,
    prefetch_count: int,
) -> Iterator[_M]:
    """Device-aware replacement for _streaming_model().

    On CUDA: delegates to LayerStreamingWrapper (async prefetch with CUDA streams).
    On MPS:  uses _MpsLayerStreamingWrapper (synchronous, no CUDA streams).
    """
    if target_device.type == "mps":
        mps_wrapped = _MpsLayerStreamingWrapper(
            model,
            layers_attr=layers_attr,
            target_device=target_device,
        )
        try:
            yield mps_wrapped  # type: ignore[misc]
        finally:
            mps_wrapped.teardown()
            mps_wrapped.to("meta")
            _cleanup_memory()
            torch.mps.synchronize()
            try:
                if hasattr(torch._C, "_host_emptyCache"):
                    torch._C._host_emptyCache()  # type: ignore[attr-defined]
            except Exception:
                pass
    else:
        wrapped = LayerStreamingWrapper(
            model,
            layers_attr=layers_attr,
            target_device=target_device,
            prefetch_count=prefetch_count,
        )
        try:
            yield wrapped  # type: ignore[misc]
        finally:
            wrapped.teardown()
            wrapped.to("meta")
            _cleanup_memory()
            _synchronize_device(target_device)
            try:
                if hasattr(torch._C, "_host_emptyCache"):
                    torch._C._host_emptyCache()  # type: ignore[attr-defined]
            except Exception:
                pass


# Apply patches.
# 1. Replace gpu_model in the defining module (catches any future dynamic lookups).
assert hasattr(_gpu_model_module, "gpu_model") and callable(getattr(_gpu_model_module, "gpu_model")), (
    "ltx_pipelines.utils.gpu_model.gpu_model not found — was it renamed? "
    "The mps_gpu_model_fix patch needs updating."
)
_gpu_model_module.gpu_model = _patched_gpu_model  # type: ignore[assignment]

# 2. Replace gpu_model in ltx_pipelines.utils.blocks, which does
#    `from ltx_pipelines.utils.gpu_model import gpu_model` at import time,
#    binding the old function directly into its own namespace.
assert hasattr(_blocks_module, "gpu_model") and callable(getattr(_blocks_module, "gpu_model")), (
    "ltx_pipelines.utils.blocks.gpu_model not found — was it renamed or removed? "
    "The mps_gpu_model_fix patch needs updating."
)
_blocks_module.gpu_model = _patched_gpu_model  # type: ignore[assignment]

# 3. Replace _streaming_model in ltx_pipelines.utils.blocks.
#    This function also calls torch.cuda.synchronize() unconditionally on
#    target_device in its finally block.
assert hasattr(_blocks_module, "_streaming_model") and callable(getattr(_blocks_module, "_streaming_model")), (
    "ltx_pipelines.utils.blocks._streaming_model not found — was it renamed or removed? "
    "The mps_gpu_model_fix patch needs updating."
)
_blocks_module._streaming_model = _patched_streaming_model  # type: ignore[assignment]
