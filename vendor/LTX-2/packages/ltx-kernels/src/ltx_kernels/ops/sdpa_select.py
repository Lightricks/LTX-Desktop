"""Unmasked SDPA that runs whichever of cuDNN and flash attention is faster for each shape.
cuDNN's fused attention is the fastest kernel for the transformer's long self-attention, but
for a short query against a long key sequence (audio queries over video keys) it falls back to
an sm80 kernel with no split over the keys, which leaves most SMs idle and runs several times
slower than flash attention's split-KV kernel.
The first call per input geometry times both backends and caches the winner; later calls, and
calls made while a CUDA graph is being captured, reuse it. A geometry first seen during capture
runs cuDNN, since timing inside a capture is impossible. A geometry neither backend takes runs
unpinned, so SDPA falls back to its other kernels as the rewritten call would. The choice is
cached per deterministic-algorithms mode, which decides whether cuDNN is usable at all.
Candidates are timed as replays of a CUDA graph, as the transformer runs them. Timing eager
launches measures host launch cost for short attentions, where cuDNN's is higher, and picks
flash for shapes where cuDNN's kernel is several times faster.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

_CANDIDATES: tuple[SDPBackend, ...] = (SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION)
_TIMING_ITERS = 10

_Key = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype, int, bool]
# ``None`` is a geometry neither candidate takes: it runs unpinned.
_choices: dict[_Key, SDPBackend | None] = {}


def _key(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> _Key:
    deterministic = torch.are_deterministic_algorithms_enabled()
    return (tuple(q.shape), tuple(k.shape), tuple(v.shape), q.dtype, q.device.index, deterministic)


def _run(backend: SDPBackend | None, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if backend is None:
        return F.scaled_dot_product_attention(q, k, v)
    with sdpa_kernel([backend]):
        return F.scaled_dot_product_attention(q, k, v)


def _time(backend: SDPBackend, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> float:
    try:
        # Eager first, so plan building and workspace allocation stay out of the timed graph.
        _run(backend, q, k, v)
    except RuntimeError:
        return float("inf")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(_TIMING_ITERS):
            _run(backend, q, k, v)
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _select(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> SDPBackend | None:
    key = _key(q, k, v)
    if key in _choices:
        return _choices[key]
    if torch.cuda.is_current_stream_capturing():
        return SDPBackend.CUDNN_ATTENTION
    times = {backend: _time(backend, q, k, v) for backend in _CANDIDATES}
    fastest = min(times, key=times.__getitem__)
    choice = fastest if times[fastest] != float("inf") else None
    _choices[key] = choice
    return choice


@torch.library.custom_op("ltx_kernels::sdpa_select", mutates_args=(), device_types="cuda")
def sdpa_select(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Unmasked, non-causal ``scaled_dot_product_attention`` over ``(B, H, S, D)`` inputs."""
    out = _run(_select(q, k, v), q, k, v)
    if not out.transpose(1, 2).is_contiguous():
        out = out.transpose(1, 2).contiguous().transpose(1, 2)
    return out


def _sdpa_select_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
    # A (B, S, H, D)-contiguous buffer viewed as (B, H, S, D): what both backends return for the
    # head-interleaved inputs attention builds, so the head merge that follows is a free view.
    b, h, s, _ = q.shape
    return q.new_empty((b, s, h, v.shape[-1])).transpose(1, 2)


sdpa_select.register_fake(_sdpa_select_fake)
