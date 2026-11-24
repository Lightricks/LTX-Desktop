# ruff: noqa: N803, ANN001, ANN202
"""Upcast many fp8 tensors to bf16 in one launch, into one flat buffer.
``Fp8CastLinear`` upcasts its weight and bias on every forward. Compiled, that is one
elementwise kernel per tensor -- about fifty per transformer block, a third of them biases of a
few thousand elements whose cost is launch latency rather than traffic. This op converts a whole
list in a single launch: every program converts one ``_BLOCK``-element chunk of one tensor,
located through a device-side table of source addresses.
The table is keyed by the sources' addresses and built on the first eager call, so it must exist
before a CUDA graph is captured over the op (the transformer's capture runs eager warmups first).
A list first seen during capture is converted tensor by tensor instead, since building the table
needs a host-to-device copy.
The upcast is exact: every float8_e4m3fn value is representable in bf16.
"""

import torch
import triton
from triton import language as tl

_BLOCK = 4096


@triton.jit
def _grouped_upcast_kernel(
    Src_ptrs,
    Dst,
    Dst_offsets,
    Numels,
    Chunk_starts,
    NUM_SEGS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    # Padding entries of ``Chunk_starts`` hold INT64_MAX, so they never count.
    starts = tl.load(Chunk_starts + tl.arange(0, NUM_SEGS))
    seg = tl.sum((starts <= pid).to(tl.int32)) - 1
    # Every source address and destination offset is 16-element aligned (checked on the host);
    # without the hints the loads through a table-read pointer are not vectorized.
    src = tl.multiple_of(tl.load(Src_ptrs + seg).to(tl.pointer_type(tl.float8e4nv)), 16)
    numel = tl.multiple_of(tl.load(Numels + seg), 16)
    dst_offset = tl.multiple_of(tl.load(Dst_offsets + seg), 16)
    chunk = (pid - tl.load(Chunk_starts + seg)).to(tl.int64)
    idx = chunk * BLOCK + tl.arange(0, BLOCK)
    mask = idx < numel
    x = tl.load(src + idx, mask=mask, eviction_policy="evict_first")
    tl.store(Dst + dst_offset + idx, x.to(tl.bfloat16), mask=mask)


class _Table:
    """Device-side segment table for one list of sources."""

    def __init__(self, sources: list[torch.Tensor]) -> None:
        numels = torch.tensor([t.numel() for t in sources], dtype=torch.int64)
        chunks = (numels + _BLOCK - 1) // _BLOCK
        self.num_chunks = int(chunks.sum())
        self.num_segs = triton.next_power_of_2(len(sources))
        pad = self.num_segs - len(sources)

        def padded(values: torch.Tensor, fill: int = 0) -> torch.Tensor:
            return torch.cat([values, values.new_full((pad,), fill)]).to(sources[0].device)

        self.src_ptrs = padded(torch.tensor([t.data_ptr() for t in sources], dtype=torch.int64))
        self.dst_offsets = padded(numels.cumsum(0) - numels)
        self.numels = padded(numels)
        self.chunk_starts = padded(chunks.cumsum(0) - chunks, fill=2**63 - 1)


# Keyed by the sources' addresses and sizes -- freed memory is handed to new tensors of other sizes --
# and holding nothing but their own few bytes.
_tables: dict[tuple[tuple[int, int], ...], _Table] = {}


@torch.library.custom_op("ltx_kernels::grouped_upcast", mutates_args=(), device_types="cuda")
def grouped_upcast(sources: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate the flattened ``float8_e4m3fn`` ``sources``, upcast to bf16, as one 1-D tensor."""
    total = sum(t.numel() for t in sources)
    out = torch.empty(total, dtype=torch.bfloat16, device=sources[0].device)
    # The table reads each source as flat contiguous storage, so an unaligned list must not be served
    # by the table of an aligned one at the same addresses (a transposed view, say).
    table = None
    if all(t.is_contiguous() and t.data_ptr() % 16 == 0 and t.numel() % 16 == 0 for t in sources):
        key = tuple((t.data_ptr(), t.numel()) for t in sources)
        table = _tables.get(key)
        if table is None and not torch.cuda.is_current_stream_capturing():
            table = _tables[key] = _Table(sources)
    if table is None:
        offset = 0
        for t in sources:
            out[offset : offset + t.numel()].copy_(t.reshape(-1))
            offset += t.numel()
        return out
    _grouped_upcast_kernel[(table.num_chunks,)](
        table.src_ptrs,
        out,
        table.dst_offsets,
        table.numels,
        table.chunk_starts,
        NUM_SEGS=table.num_segs,
        BLOCK=_BLOCK,
        num_warps=4,
    )
    return out


@grouped_upcast.register_fake
def _grouped_upcast_fake(sources: list[torch.Tensor]) -> torch.Tensor:
    return sources[0].new_empty(sum(t.numel() for t in sources), dtype=torch.bfloat16)
