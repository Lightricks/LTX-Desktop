"""Absolute 3D RoPE over a whole ``(B, T, H, W, NH, HD)`` volume in one CuTe DSL launch.
The rotation is :func:`ltx_kernels.vae.fna_attn_core.apply_rope`, the same code the fused
DiffVAE block runs on its Q/K rows; here thread ``i`` of a block owns head row ``i``, rows
are staged through SMEM so global traffic stays coalesced, and the cos/sin come from
global tables. No tensor-core instructions are
involved, so it builds for any CUDA architecture the DSL targets.
The caller builds the tables, so they can hold exactly the angles a reference
implementation uses -- integer or fractional positions alike.
"""

# NOTE: no `from __future__ import annotations` -- cute.jit needs real annotations.

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator

from ltx_kernels.vae.aot import KERNEL_CACHE, fake_dynamic_matrix
from ltx_kernels.vae.fna_attn_core import apply_rope
from ltx_kernels.vae.fna_smem import _row_vec_ptr
from ltx_kernels.vae.fna_types import ACC, HD, IO

_THREADS = 128
# Global traffic moves in 16B slices, consecutive lanes on consecutive slices, so a warp's
# loads and stores cover whole 128B lines. Rows are staged through SMEM, padded by one
# slice so the per-thread row reads of an 8-lane phase land on distinct banks.
_VEC = 8
_LANES_PER_ROW = HD // _VEC
_ROW_STRIDE = HD + _VEC


@cute.kernel
def _rope_kernel(
    x: cute.Tensor,  # (rows, HD) bf16, rows ordered (B, T, H, W, NH)
    y: cute.Tensor,  # (rows, HD) bf16
    cos_t: cute.Tensor,  # (T, d_t / 2) f32
    sin_t: cute.Tensor,
    cos_h: cute.Tensor,  # (H, d_h / 2) f32
    sin_h: cute.Tensor,
    cos_w: cute.Tensor,  # (W, d_w / 2) f32
    sin_w: cute.Tensor,
    n_rows: cutlass.Int32,
    T: cutlass.Int32,
    H: cutlass.Int32,
    W: cutlass.Int32,
    NH: cutlass.Int32,
    d_t: cutlass.Constexpr,
    d_h: cutlass.Constexpr,
    d_w: cutlass.Constexpr,
) -> None:
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    row0 = bidx * _THREADS
    smem = SmemAllocator()
    tile = smem.allocate_tensor(IO, cute.make_layout((_THREADS * _ROW_STRIDE,)), byte_alignment=16)
    frag = cute.make_rmem_tensor(cute.make_layout(_VEC), IO)

    # Coalesced fill: consecutive lanes take consecutive 16B slices of consecutive rows.
    for k in range(_LANES_PER_ROW):
        r = (k * _THREADS + tidx) // _LANES_PER_ROW
        c = (k * _THREADS + tidx) % _LANES_PER_ROW * _VEC
        if row0 + r < n_rows:
            cute.autovec_copy(cute.make_tensor(_row_vec_ptr(x, row0 + r, c, HD), cute.make_layout(_VEC)), frag)
            cute.autovec_copy(frag, _smem_slice(tile, r, c))
    cute.arch.sync_threads()

    # Thread ``i`` rotates row ``i`` in registers.
    row = row0 + tidx
    if row < n_rows:
        token = row // NH
        w = token % W
        h = (token // W) % H
        t = (token // (W * H)) % T
        reg = cute.make_rmem_tensor(cute.make_layout(HD), ACC)
        for k in range(_LANES_PER_ROW):
            cute.autovec_copy(_smem_slice(tile, tidx, k * _VEC), frag)
            for e in range(_VEC):
                reg[k * _VEC + e] = ACC(frag[e])
        apply_rope(reg, d_t, d_h, d_w, (cos_t, sin_t, cos_h, sin_h, cos_w, sin_w), t, h, w)
        for k in range(_LANES_PER_ROW):
            for e in range(_VEC):
                frag[e] = IO(reg[k * _VEC + e])
            cute.autovec_copy(frag, _smem_slice(tile, tidx, k * _VEC))
    cute.arch.sync_threads()

    for k in range(_LANES_PER_ROW):
        r = (k * _THREADS + tidx) // _LANES_PER_ROW
        c = (k * _THREADS + tidx) % _LANES_PER_ROW * _VEC
        if row0 + r < n_rows:
            cute.autovec_copy(_smem_slice(tile, r, c), frag)
            cute.autovec_copy(frag, cute.make_tensor(_row_vec_ptr(y, row0 + r, c, HD), cute.make_layout(_VEC)))


def _smem_slice(tile, row, col):
    """The 16B slice at ``(row, col)`` of the padded SMEM row tile."""
    return cute.make_tensor((tile.iterator + (row * _ROW_STRIDE + col)).align(16), cute.make_layout(_VEC))


@cute.jit
def _launch(
    x: cute.Tensor,
    y: cute.Tensor,
    cos_t: cute.Tensor,
    sin_t: cute.Tensor,
    cos_h: cute.Tensor,
    sin_h: cute.Tensor,
    cos_w: cute.Tensor,
    sin_w: cute.Tensor,
    n_rows: cutlass.Int32,
    T: cutlass.Int32,
    H: cutlass.Int32,
    W: cutlass.Int32,
    NH: cutlass.Int32,
    d_t: cutlass.Constexpr,
    d_h: cutlass.Constexpr,
    d_w: cutlass.Constexpr,
) -> None:
    _rope_kernel(x, y, cos_t, sin_t, cos_h, sin_h, cos_w, sin_w, n_rows, T, H, W, NH, d_t, d_h, d_w).launch(
        grid=[(n_rows + _THREADS - 1) // _THREADS, 1, 1], block=[_THREADS, 1, 1]
    )


def _compile_kernel(*, rope_dims: "tuple[int, int, int]", options: str | None = None):
    """Compile one ``(d_t, d_h, d_w)`` split from storage-free descriptors."""
    rows = fake_dynamic_matrix(cutlass.BFloat16, assumed_align=16)
    tables = [fake_dynamic_matrix(cutlass.Float32, assumed_align=4) for _ in range(6)]
    return cute.compile(
        _launch,
        rows,
        fake_dynamic_matrix(cutlass.BFloat16, assumed_align=16),
        *tables,
        1,
        1,
        1,
        1,
        1,
        *rope_dims,
        **({"options": options} if options else {}),
    )


def abs_rope_supported(x: torch.Tensor, rope_dims: "tuple[int, int, int]") -> bool:
    """Whether :func:`run_abs_rope` takes ``x``: a contiguous CUDA bf16 ``(B, T, H, W, NH, 64)`` volume."""
    d_t, d_h, d_w = rope_dims
    return (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() == 6
        and x.shape[-1] == HD
        and x.is_contiguous()
        and x.numel() // HD < 2**31
        and d_t + d_h + d_w == HD
        and d_t % 2 == 0
        and d_h % 2 == 0
    )


def run_abs_rope(
    x: torch.Tensor,
    rope_dims: "tuple[int, int, int]",
    tables: "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]",
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate every head row of ``x`` by its (t, h, w) angles, into ``out`` (a new tensor by default).
    ``tables`` is ``(cos_t, sin_t, cos_h, sin_h, cos_w, sin_w)``, fp32 ``(span, dim / 2)``
    tables over ``T``, ``H`` and ``W`` of ``x``. Check :func:`abs_rope_supported` first.
    ``out`` may be ``x`` itself: each CTA reads all of its rows into shared memory before it
    writes any of them back. Otherwise it must not overlap ``x``. It must be on ``x``'s device.
    """
    _, t, h, w, nh, _ = x.shape
    rows = x.view(-1, HD)
    if out is None:
        y = torch.empty_like(rows)
    else:
        if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device or not out.is_contiguous():
            raise ValueError(
                f"out must be a contiguous {tuple(x.shape)} {x.dtype} tensor on {x.device}, "
                f"got {tuple(out.shape)} {out.dtype} on {out.device}"
            )
        y = out.view(-1, HD)
    with torch.cuda.device(x.device):
        compiled = KERNEL_CACHE.get(_compile_kernel, x.device, rope_dims=tuple(rope_dims))
        compiled(
            from_dlpack(rows, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(y, assumed_align=16).mark_layout_dynamic(),
            *(from_dlpack(table.contiguous(), assumed_align=4).mark_layout_dynamic() for table in tables),
            rows.shape[0],
            t,
            h,
            w,
            nh,
        )
    return y.view(x.shape)
