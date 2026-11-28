"""The whole DiffVAE ``DiffusionNABlock`` in one CuTe DSL launch.
The stage-4 hop, context projection, residual, AdaLN, 3D neighborhood attention, SwiGLU
and the closing residual, with **no durable full-volume Q/K/V**: the workspace is the
output, a K/V cache sized by the slab the call picks, and a two-int barrier. The compiled
NATTEN path materialises Q/K/V for the whole volume instead, which is what forces
spatial tiling at 4K and above -- and tiling recomputes most of the volume.
Structure, in the order the kernel runs it:
* **A deferred stage-4 context.** The input is stage-4 itself, at half the output
  resolution per axis, and the pixel-shuffle upsample that would have produced a
  full-resolution context is folded into ``context_proj`` at load time
  (:func:`pack_fused_ctx_weights`) -- one Linear per subpixel. So the context volume is
  never materialised, at any resolution, by anyone.
* **3D query tiles.** One CTA owns a ``(TT, TH, TW)`` box of exactly ``M`` query
  tokens, so every UMMA runs at full ``M``. The NA halo panel is that box grown by
  the kernel radius.
* **Thread ``i`` owns accumulator row ``i``; warpgroup ``g`` owns a slice of its
  columns.** A tcgen05 accumulator hands thread ``i`` the whole of row ``i``, which
  makes RMSNorm, RoPE, bias and the softmax register-local; splitting the *columns*
  keeps that while giving the CTA ``4 * NUM_WARPGROUPS`` warps to hide latency behind.
* **A grid-shared K/V slab.** Panel tokens are shared by neighbouring query tiles, so
  :func:`_fill_slab_caches` projects a whole slab's halo box into a fixed K/V cache
  once, a device-wide barrier follows, and :func:`_run_tile` gathers from it.
  Per-query redundancy drops from ~27x (at an 11^3 window) to ~1.4x.
* **One body per stream pair.** The keyframe stream's rows go through the same fill and the
  same query tile as the video stream's, with :func:`ltx_kernels.vae.fna_smem.select_tensor`
  picking the buffer at runtime. Tracing either of them twice cost ~70 s of backend compile
  on a kernel whose plain build is already 25 s.
* **One body per row pass.** Heads and C-wide chunks are walked by device loops, and each
  warpgroup reaches its own column slice through an SMEM operand view shifted by whole
  K-stages (:func:`ltx_kernels.vae.fna_smem._alias`) rather than through a constexpr arm
  per warpgroup. Only what has to stay static -- slice coordinates within a stage, the
  KV pipeline's buffer parity -- is still unrolled.
* **The attention half is not here.** The window mask, the pipelined KV loop and the
  softmax are :func:`ltx_kernels.vae.fna_attn_core.attention_tile`, shared verbatim
  with the standalone NA kernel.
The box vocabulary the comments use -- volume, slab core, slab box, query tile, halo
panel, NA window -- is defined once in :mod:`ltx_kernels.vae.fna_geometry`, which also
holds the coordinate arithmetic both kernels share.
``T``/``H``/``W`` stay dynamic, and so does the slab extent. Channel count, hidden
width, head count, kernel size, tile shape and the slab *bounds* are compile-time.
Datacenter Blackwell only -- see :mod:`ltx_kernels.vae.availability` for exactly which
parts and why.
"""

# NOTE: no `from __future__ import annotations` -- cute.struct needs real types.

import dataclasses
import functools
from typing import Any, NamedTuple

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.utils import SmemAllocator

from ltx_kernels.devices import upload_from_host
from ltx_kernels.vae.aot import (
    KERNEL_CACHE,
    fake_dynamic_matrix,
    fake_dynamic_vector,
    fake_dynamic_weight,
)
from ltx_kernels.vae.availability import UNSUPPORTED_MESSAGE, gpu_supports_dsl_kernels
from ltx_kernels.vae.fna_attn_core import (
    _as_rows,
    _dyn,
    _dyn_act,
    _dyn_w,
    apply_rope,
    apply_rope_float_t,
    attention_tile,
    attn_epilogue,
    attn_zero_accumulators,
    cta_count,
    default_tile_thw,
    fill_rope_axis,
    grid_barrier,
    head_sum2,
    row_sum,
)
from ltx_kernels.vae.fna_gemm import (
    GemmJob,
    UmmaSharedStorage,
    _mma_op,
    _reg_shape,
    _tmem_load,
    _tmem_shift,
    acquire_tmem,
    gemm_tma,
    release_tmem,
)
from ltx_kernels.vae.fna_geometry import (
    at_least,
    at_most,
    clamp_window_start,
    flatten_3d,
    halo_panel_origin,
    query_row_indices,
    query_slot_in_tile,
    unflatten_3d,
    window_bounds_in_panel,
    window_origin_in_panel,
)
from ltx_kernels.vae.fna_smem import (
    _alias,
    _fill_a_chunk,
    _fill_a_gather,
    _read_a_chunk,
    _scatter_a,
    _store_head_row,
    cooperative_copy_slot,
    select_tensor,
)
from ltx_kernels.vae.fna_types import (
    ACC,
    ACC_COLS,
    EPS,
    HD,
    HEAD_COLS,
    HEADS_PER_TILE,
    IO,
    LOG2E,
    NUM_WARPGROUPS,
    THREADS,
    TILE_K,
    TILE_N,
    TILER_N64,
    TILER_N128,
    WARPGROUPS_PER_HEAD,
    M,
)
from ltx_kernels.vae.keyframe_slots import (
    KEYFRAME_CONTEXT_SLOTS,
    nearest_keyframe_slots,
    nearest_video_frames,
    tile_plane_union,
)
from ltx_kernels.vae.workspace import workspace_volume

# Channel counts the kernel is compiled and validated for. The shipped DiffVAE only
# ever asks for 256 (stage 5); 128 is kept because it is the cheap shape to validate
# a pipeline change against -- it compiles in a third of the time.
SUPPORTED_CHANNELS = (128, 256)

# Slab bounds. The slab's *extent* in query tiles is chosen per call by
# :func:`_choose_slab` from the volume, the tile and the grid, so T/H/W stay dynamic;
# only these bounds are compile-time.
# Raising them is worth a lot, because what they cap is the cache fill's redundancy:
# the slab box is the core grown by the NA radius, so a slab one tile wider amortises
# the same halo over more output. Geometric mean over the DiffVAE's stage-5 shapes, in
# tokens projected per token of output: 1.83 at (42,58,58), 1.33 at (80,80,80), 1.23
# at (96,96,96), 1.12 at (128,128,128). A symmetric bound is deliberate -- one fitted
# to a single production shape scored worse on average, because it starves whichever
# axis the next shape happens to be long in.
# What the bounds cost directly is SMEM for the rope tables, ``4 * 2 * sum_i B_i *
# rope_dim_i / 2`` bytes -- 32 KiB here. They do *not* size the cache, whose per-head stride
# is a runtime argument set to the box the call actually chose, and at every shape the
# shipped decoder asks for ``_choose_slab`` lands inside these bounds: what binds is its
# cost model, not the cap.
MAXP_TOKENS = 2097152
SLAB_BOX_MAX = (128, 128, 128)

#: Rows an output buffer must carry past the volume, to absorb the stores of
#: out-of-range lanes in partial tiles. Public so a caller passing ``out`` to
#: :func:`run_block_fna_dsl` can size the buffer without importing ``M``.
OUT_SLACK_ROWS = M


def _check_out(out: torch.Tensor, rows: int, channels: int, x_: torch.Tensor, name: str = "out") -> torch.Tensor:
    """Validate a caller-supplied output buffer for :func:`run_block_fna_dsl`.
    ``x_`` is the stream's own input rows -- the video volume for ``out``, the anchor
    planes for ``out_keyframes`` -- since that is what the buffer must not alias.
    """
    if tuple(out.shape) != (rows, channels):
        raise ValueError(
            f"{name} must be {(rows, channels)} -- the volume plus {OUT_SLACK_ROWS} lane-slack "
            f"rows -- got {tuple(out.shape)}"
        )
    if out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError(f"{name} must be contiguous bfloat16, got {out.dtype}, contiguous={out.is_contiguous()}")
    if out.device != x_.device:
        raise ValueError(f"{name} is on {out.device}, its input on {x_.device}")
    if out.data_ptr() == x_.data_ptr():
        raise ValueError(f"{name} must not alias its input: the kernel reads neighborhoods while writing")
    return out


#: Bytes in one N=128 B-operand stage, which is what ``gemm_tma``'s TMA transaction
#: count has to match.
_B_STAGE_BYTES = TILE_N * TILE_K * 2
# Weight tiles the MLP keeps in flight; four N=128 x K=64 bf16 stages fill a C=256 s_q exactly.
MLP_B_STAGES = 4


# --------------------------------------------------------------------------
# Data passed between the host, the kernel and its helpers
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _SlabPlan:
    """How one call cuts its query-tile grid into slabs.
    Attributes:
        core: the slab core, ``(t, h, w)`` query tiles.
        box: the halo box those tiles imply, ``(t, h, w)`` tokens. Bounds the cache
            and the rope tables; the kernel shrinks it per slab at the volume edge.
        grid: how many slabs the volume is cut into, ``(t, h, w)``.
    """

    core: "tuple[int, int, int]"
    box: "tuple[int, int, int]"
    grid: "tuple[int, int, int]"

    @property
    def count(self) -> int:
        """Total slabs, which is also how many cache-fill/barrier rounds the kernel runs."""
        return self.grid[0] * self.grid[1] * self.grid[2]

    @property
    def cache_row_stride(self) -> int:
        """Rows between consecutive heads in the K/V cache: one whole slab box."""
        return self.box[0] * self.box[1] * self.box[2]


@dataclasses.dataclass(frozen=True)
class _BlockShape:
    """The block's compile-time shape. Every field is a Python value at trace time."""

    channels: int
    ctx_channels: int  # stage-4's width, a multiple of ``channels``
    stride: "tuple[int, int, int]"  # the stage-4 pixel-shuffle stride, ``(1,1,1)`` if none
    hidden: int  # SwiGLU hidden width
    n_heads: int  # n_heads * HD == channels
    kernel: "tuple[int, int, int]"  # NA window extents
    tile: "tuple[int, int, int]"  # query tile; its product is M
    panel: "tuple[int, int, int]"  # halo panel: the tile grown by the radius
    rope_dims: "tuple[int, int, int]"  # RoPE dims per axis; they sum to HD
    n_kv_tiles: int  # KV tiles spanning one panel

    @property
    def radius(self) -> "tuple[int, int, int]":
        """The NA window's half-extents."""
        return ((self.kernel[0] - 1) // 2, (self.kernel[1] - 1) // 2, (self.kernel[2] - 1) // 2)

    @property
    def n_chunks(self) -> int:
        """N-tiles of a C-wide output, which is also K-chunks of a C-wide input."""
        return self.channels // TILE_N

    @property
    def n_k_tiles(self) -> int:
        """K-tiles spanning one C-wide reduction."""
        return self.channels // TILE_K

    @property
    def n_subpixels(self) -> int:
        """``P``: pixel-shuffle subpixels per stage-4 cell, one fused weight slice each."""
        return self.stride[0] * self.stride[1] * self.stride[2]

    @property
    def ctx_groups(self) -> int:
        """C-wide operands a stage-4 row spans: ``s_a``, and ``s_q`` when it is 2C wide."""
        return self.ctx_channels // self.channels

    @property
    def param(self) -> "dict[str, int]":
        """Offsets into the packed per-channel constants."""
        return _channel_param_offsets(self.channels, HD, self.n_subpixels)


@dataclasses.dataclass(frozen=True)
class _ThreadIds:
    """This thread's place in the CTA and in the grid."""

    tidx: Any  # thread index within the CTA
    lane: Any  # the accumulator row it owns, ``tidx % M``
    wg: Any  # its warpgroup: which slice of that row's columns it holds
    warp: Any  # warp-uniform warp index, for the elect-one paths
    cta: Any  # block index -- this CTA's slot in the persistent grid


@dataclasses.dataclass(frozen=True)
class _Workspace:
    """Everything the device helpers need that is fixed for the whole launch.
    The compile-time :class:`_BlockShape` deliberately does *not* live here: the DSL
    stages a dataclass argument's numeric fields into runtime values, so anything that
    has to reach ``range_constexpr`` or a statically aligned SMEM slice travels
    separately as a ``cutlass.Constexpr`` parameter.
    """

    ids: _ThreadIds

    # ---- global tensors ----
    x: cute.Tensor
    stage4: cute.Tensor  # (T_lo*H_lo*W_lo, Ctx) -- the context before its upsample
    y: cute.Tensor  # the output, and the residual accumulator between passes
    kv_cache: cute.Tensor  # the slab's K/V, K's heads then V's
    norm_cache: cute.Tensor  # the slab's per-token RMSNorm factor for ``post``
    barrier: cute.Tensor

    # ---- the keyframe stream, mirroring the video tensors above ----
    x_kf: cute.Tensor
    stage4_kf: cute.Tensor
    y_kf: cute.Tensor
    kv_cache_kf: cute.Tensor
    norm_cache_kf: cute.Tensor
    kf_times: cute.Tensor
    kf_slots: cute.Tensor
    kf_planes: cute.Tensor
    kf_counts: cute.Tensor
    kf_video: cute.Tensor
    kf_video_counts: cute.Tensor
    kf_slab: cute.Tensor

    # ---- SMEM operands, all aliased onto one pool; see the map in _fna_kernel ----
    s_a: cute.Tensor  # the activation / A operand
    s_b: cute.Tensor  # the streamed weight tile / B operand
    s_q: cute.Tensor  # queries; dead after attention
    s_b4: cute.Tensor  # the MLP's weight queue, aliasing s_q
    s_k: cute.Tensor
    s_v: cute.Tensor
    s_p_own: cute.Tensor  # this warpgroup's slice of P (s_p), at stage 0
    s_h_own: cute.Tensor  # this warpgroup's slice of the MLP hidden chunk (s_h), at stage 0
    # Raw storage and operand layouts, for views shifted by a runtime number of K-stages.
    pool: cute.Tensor  # under s_a, s_p, s_k, s_b, s_v and s_h
    q_pool: cute.Tensor  # under s_q and s_b4
    a_layout: Any
    q_layout: Any
    storage: Any  # barrier storage for the pipelines gemm_tma stands up

    # ---- MMA operand fragments over those buffers ----
    fr_a: Any
    fr_q: Any
    fr_h: Any
    fr_k: Any
    fr_p: Any
    fr_v: Any

    # ---- weights, each as its ``(tma_atom, tensor)`` pair ----
    w_ctx: tuple
    w_q: tuple
    w_kv: tuple  # packed K|V, interleaved per head by _pack_kv
    w_o: tuple
    w_gate: tuple
    w_up: tuple
    w_down: tuple

    # ---- TMEM accumulators; see the TMEM map in _fna_kernel ----
    acc_main: list
    acc_qk: list
    acc_scratch: Any
    acc_o: list
    acc_down: list

    # ---- register scratch ----
    areg: Any
    breg: Any
    oreg: Any
    kvfrag: Any

    # ---- SMEM scratch ----
    rowix: cute.Tensor  # in-bounds source row per lane; feeds every gather
    rowix_lo: cute.Tensor  # the stage-4 cell row per lane, for the fused-context gather
    rowix_w: cute.Tensor  # destination row per lane; scratch when out of range
    kv_rowtab: cute.Tensor  # (2, M) KV-tile source rows, for the cooperative K/V copies
    kv_copy_bars: cute.Tensor  # K-full x2, P/V-full: the KV loop's operand-ready counters
    copy_slot: tuple  # this thread's slot in the cooperative copies
    red_c: cute.Tensor  # row reduction across all warpgroups
    red_head: cute.Tensor  # row reduction across the warpgroups sharing a head
    m_sm: cute.Tensor  # fixed softmax offset per (row, head)
    l_sm: cute.Tensor  # softmax denominator per (row, warpgroup, head)
    qkbar: cute.Tensor  # Q@K completion barriers, one per accumulator
    pvbar: cute.Tensor  # P@V completion barrier
    rope: tuple  # (cos_t, sin_t, cos_h, sin_h, cos_w, sin_w) over the slab box
    rope_inv: tuple  # per-axis inverse RoPE frequencies, the tables' source
    params: cute.Tensor  # the packed per-channel constants, staged in SMEM
    query_slot: tuple  # this lane's (dt, dh, dw) within whatever query tile it runs

    # ---- runtime scalars, fixed for the launch ----
    volume: tuple  # (T, H, W)
    ctx_volume: tuple  # (T_lo, H_lo, W_lo), the stage-4 grid the volume shuffles out of
    drop_leading: Any  # 1 when the caller dropped the shuffle's duplicate leading frame
    n_ctas: Any
    slab_count: Any
    slab_grid_hw: Any  # strides for unflattening a slab index
    slab_grid_w: Any
    slab_tiles: tuple  # query tiles per slab, per axis
    max_box: tuple  # the largest halo box this call chose
    cache_row_stride: Any  # rows between heads in kv_cache; one whole max box
    n_kf: Any
    kf_cache_row_stride: Any

    @property
    def n_volume_tokens(self):
        """Rows of the output that are real; anything at or past this is scratch."""
        return self.volume[0] * self.volume[1] * self.volume[2]


@dataclasses.dataclass(frozen=True)
class _Slab:
    """The slab in flight: the output it owns, and the box its K/V cache covers."""

    core_origin: tuple  # first token of the core, per axis
    core_end: tuple  # one past its last, clamped into the volume
    t_index: Any  # this slab's position along the temporal slab axis
    box_origin: tuple  # first token of the halo box
    box_shape: tuple  # the box's extents
    box_hw: Any  # box_shape[1] * box_shape[2], hoisted
    n_box_tokens: Any  # tokens in the box -- what the cache fill walks


# --------------------------------------------------------------------------
# The kernel
# --------------------------------------------------------------------------


@cute.kernel
def _fna_kernel(
    # ---- tensors ----
    x: cute.Tensor,  # (T*H*W, C) bf16, the block's input
    stage4: cute.Tensor,  # (T_lo*H_lo*W_lo, Ctx) bf16, the context before its upsample
    y: cute.Tensor,  # (T*H*W + M, C) bf16, output; also the residual accumulator
    kv_cache: cute.Tensor,  # (2*NH*cache_row_stride, HD) bf16, K's heads then V's
    norm_cache: cute.Tensor,  # (cache_row_stride + M,) f32, post's RMSNorm factor
    barrier: cute.Tensor,  # (2,) int32 device-wide barrier: arrivals, then sense
    # ---- the keyframe stream; inert placeholders when KF is False ----
    x_kf: cute.Tensor,  # (n_kf*H*W, C) bf16
    stage4_kf: cute.Tensor,  # (n_kf*H_lo*W_lo, Ctx) bf16
    y_kf: cute.Tensor,  # (n_kf*H*W + M, C) bf16
    kv_cache_kf: cute.Tensor,  # (2*NH*kf_row_stride, HD) bf16
    norm_cache_kf: cute.Tensor,  # (kf_row_stride + M,) f32
    kf_times: cute.Tensor,  # (n_kf,) f32 stage-local chunk-center positions
    kf_slots: cute.Tensor,  # (T*SLOTS,) i32
    kf_planes: cute.Tensor,  # (n_t_tiles*P_MAX,) i32
    kf_counts: cute.Tensor,  # (n_t_tiles,) i32
    kf_video: cute.Tensor,  # (n_kf*SLOTS,) i32
    kf_video_counts: cute.Tensor,  # (n_kf,) i32
    kf_slab: cute.Tensor,  # (n_kf,) i32 -- which slab owns this plane's query tiles
    # ---- weights, each as its TMA atom and the tensor it reads ----
    a_ctx,  # fused upsample x context projection, (P*C, Ctx)
    m_ctx,
    a_q,  # query projection
    m_q,
    a_kv,  # packed K|V projection, interleaved per head by _pack_kv
    m_kv,
    a_o,  # attention out-projection
    m_o,
    a_gate,  # SwiGLU gate
    m_gate,
    a_up,  # SwiGLU up
    m_up,
    a_down,  # SwiGLU down
    m_down,
    # ---- constant tables ----
    channel_params: cute.Tensor,  # f32, laid out by _CHANNEL_PARAM_RUNS
    inv_t: cute.Tensor,  # f32 inverse RoPE frequencies, per axis
    inv_h: cute.Tensor,
    inv_w: cute.Tensor,
    # ---- runtime volume ----
    T,
    H,
    W,
    T_lo,  # the stage-4 grid this volume pixel-shuffles out of
    H_lo,
    W_lo,
    drop_leading,  # 1 when the caller dropped the shuffle's duplicate leading frame
    # ---- runtime schedule ----
    n_ctas,  # persistent grid width; every CTA must be resident for the barrier
    slab_count,  # slabs the volume is cut into
    slab_grid_hw,  # strides for unflattening a slab index: grid_h * grid_w ...
    slab_grid_w,  # ... and grid_w
    slab_tiles_t,  # query tiles per slab, per axis
    slab_tiles_h,
    slab_tiles_w,
    # ---- runtime slab geometry ----
    max_box_t,  # the largest halo box this call chose; caps each slab's own box
    max_box_h,
    max_box_w,
    cache_row_stride,  # rows between heads in kv_cache == max_box_t*h*w
    n_kf,  # keyframe planes carried alongside the video stream
    kf_cache_row_stride,  # rows between heads in kv_cache_kf == n_kf*max_box_h*max_box_w
    # ---- compile-time block shape ----
    C: cutlass.Constexpr,  # channels
    Ctx: cutlass.Constexpr,  # stage-4 channels; a multiple of C
    SP_T: cutlass.Constexpr,  # the stage-4 pixel-shuffle stride
    SP_H: cutlass.Constexpr,
    SP_W: cutlass.Constexpr,
    Hidd: cutlass.Constexpr,  # SwiGLU hidden width
    NH: cutlass.Constexpr,  # heads; NH * HD == C
    ROPE_T: cutlass.Constexpr,  # RoPE dims per axis; they sum to HD
    ROPE_H: cutlass.Constexpr,
    ROPE_W: cutlass.Constexpr,
    KT: cutlass.Constexpr,  # NA window extents
    KH: cutlass.Constexpr,
    KW: cutlass.Constexpr,
    # ---- compile-time tile geometry ----
    TT: cutlass.Constexpr,  # query tile; TT*TH*TW == M
    TH: cutlass.Constexpr,
    TW: cutlass.Constexpr,
    PT: cutlass.Constexpr,  # halo panel; PT == TT + KT - 1, and so on
    PH: cutlass.Constexpr,
    PW: cutlass.Constexpr,
    NKV: cutlass.Constexpr,  # KV tiles spanning one panel
    BT: cutlass.Constexpr,  # SLAB_BOX_MAX; sizes the rope tables only
    BH: cutlass.Constexpr,
    BW: cutlass.Constexpr,
    KF: cutlass.Constexpr,  # keyframe stream compiled in at all
    SLOTS: cutlass.Constexpr,
    P_MAX: cutlass.Constexpr,  # cap on a query tile's keyframe-plane union
    NKV_KF: cutlass.Constexpr,  # KV tiles over one keyframe panel, ceil(PH*PW / M)
    # ---- compile-time layouts ----
    mma128: cute.TiledMma,
    mma64: cute.TiledMma,
    a_layout: cute.ComposedLayout,
    b_layout: cute.ComposedLayout,
    p_layout: cute.ComposedLayout,
    q_layout: cute.ComposedLayout,
    k_layout: cute.ComposedLayout,
    v_layout: cute.ComposedLayout,
    b4_layout: cute.ComposedLayout,
    h_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    cta_index, _, _ = cute.arch.block_idx()
    # ``lane`` is the accumulator row this thread owns; ``wg`` picks which slice of
    # that row's columns. Every per-row SMEM map is indexed by lane, and every
    # epilogue chunk index is ``wg``.
    ids = _ThreadIds(
        tidx=tidx,
        lane=tidx % M,
        wg=tidx // M,
        warp=cute.arch.make_warp_uniform(cute.arch.warp_idx()),
        cta=cta_index,
    )
    shape = _BlockShape(
        channels=C,
        ctx_channels=Ctx,
        stride=(SP_T, SP_H, SP_W),
        hidden=Hidd,
        n_heads=NH,
        kernel=(KT, KH, KW),
        tile=(TT, TH, TW),
        panel=(PT, PH, PW),
        rope_dims=(ROPE_T, ROPE_H, ROPE_W),
        n_kv_tiles=NKV,
    )
    param = shape.param

    smem = SmemAllocator()
    storage = smem.allocate(UmmaSharedStorage)
    # One 96 KiB operand pool, because the two halves of the block never need their
    # operands at the same time. Outside the KV loop it is the activation (``s_a``)
    # plus one weight tile (``s_b``); inside it, the attention operands -- ``s_p`` and
    # ``s_k``/``s_v`` double-buffered, which is what lets a step's cache read and both
    # of its MMAs overlap the *previous* step's softmax. ``s_a`` is dead from the Q
    # projection, which consumes it, until the out-projection refills it; ``s_b`` is
    # dead between those same two GEMMs. So the pipeline costs no SMEM at all.
    pool_elems = 6 * M * TILE_K  # s_a | (s_p, s_k) ; s_b | s_v
    pool = smem.allocate_tensor(IO, cute.make_layout((pool_elems,)), byte_alignment=128)
    s_a = _alias(pool, a_layout)
    s_p = _alias(pool, p_layout)
    s_k = _alias(pool, k_layout, 2 * M * TILE_K)
    s_b = _alias(pool, b_layout, 4 * M * TILE_K)
    s_v = _alias(pool, v_layout, 4 * M * TILE_K)
    # s_q holds the queries until attention ends; the MLP then reuses it as its weight queue.
    # Sized for the larger view: at C=128 the queries take half of what the queue needs.
    q_pool_elems = max(cute.cosize(q_layout), cute.cosize(b4_layout))
    q_pool = smem.allocate_tensor(IO, cute.make_layout((q_pool_elems,)), byte_alignment=128)
    s_q = _alias(q_pool, q_layout)
    s_b4 = _alias(q_pool, b4_layout)
    s_h = _alias(pool, h_layout, 4 * M * TILE_K)
    # Warpgroup ``g``'s ``ACC_COLS`` chunk of a 128-wide row is K-stage ``g`` of it, so these
    # views -- shifted ``wg`` stages -- give every warpgroup its own chunk at static stage 0,
    # and a row pass over them is one code copy rather than one per warpgroup.
    wg_stage = ids.wg * (M * TILE_K)
    s_p_own = _alias(pool, p_layout, wg_stage)
    s_h_own = _alias(pool, h_layout, 4 * M * TILE_K + wg_stage)
    # Completion barriers: one per Q@K accumulator plus one the P@V chain signals.
    # ``attention_tile`` re-initialises them per query tile so the wait phases stay
    # compile-time constants.
    qkbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((2,)), byte_alignment=16)
    pvbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((1,)), byte_alignment=16)
    rowix = smem.allocate_tensor(cutlass.Int32, cute.make_layout((M,)), byte_alignment=16)
    rowix_lo = smem.allocate_tensor(cutlass.Int32, cute.make_layout((M,)), byte_alignment=16)
    rowix_w = smem.allocate_tensor(cutlass.Int32, cute.make_layout((M,)), byte_alignment=16)
    kv_rowtab = smem.allocate_tensor(cutlass.Int32, cute.make_layout((2, M)), byte_alignment=16)
    kv_copy_bars = smem.allocate_tensor(cutlass.Int64, cute.make_layout((3,)), byte_alignment=16)
    m_sm = smem.allocate_tensor(ACC, cute.make_layout((M, NH)), byte_alignment=16)
    l_sm = smem.allocate_tensor(ACC, cute.make_layout((M, NUM_WARPGROUPS, NH)), byte_alignment=16)
    red_c = smem.allocate_tensor(ACC, cute.make_layout((M, NUM_WARPGROUPS)), byte_alignment=16)
    red_head = smem.allocate_tensor(ACC, cute.make_layout((M, NUM_WARPGROUPS, 2)), byte_alignment=16)
    # The rope tables are sized by the compile-time slab bound, and filled per slab
    # over the extent that slab's box actually has.
    rope = tuple(
        smem.allocate_tensor(ACC, cute.make_layout((extent, dim // 2)), byte_alignment=16)
        for extent, dim in ((BT, ROPE_T), (BT, ROPE_T), (BH, ROPE_H), (BH, ROPE_H), (BW, ROPE_W), (BW, ROPE_W))
    )
    params = smem.allocate_tensor(ACC, cute.make_layout((param["total"],)), byte_alignment=16)
    i = tidx
    while i < param["total"]:
        params[i] = channel_params[i]
        i += THREADS
    cute.arch.sync_threads()

    tmem, tmem_ptr = acquire_tmem(storage)

    lay_main = mma128.make_fragment_C(mma128.partition_shape_C(TILER_N128[:2])).layout
    lay_n64 = mma64.make_fragment_C(mma64.partition_shape_C(TILER_N64[:2])).layout

    # TMEM map (512 columns). ``acc_o`` holds the flash-attention accumulator for every
    # head and is live for the whole KV loop; everything else is scratch that only
    # exists between two syncs, so it all shares the region above it.
    #   acc_o        [0, NH*HD)              per-head attention accumulator
    #   acc_main     [NH*HD, +n_chunks*TILE_N)  one per N-tile of a C-wide output
    #   acc_qk       aliases acc_main[0..1]  two Q@K accumulators, so step s's softmax
    #                                        reads one while step s+1's MMA fills the
    #                                        other (acc_main is dead in the KV loop)
    #   acc_scratch  aliases acc_main[1]     the K|V, Q and SwiGLU-up projections, which
    #                                        lets SiLU read gate and up from TMEM
    #   acc_down     aliases acc_o           (acc_o is dead once the MLP starts)
    scr = NH * HD
    n_chunks = shape.n_chunks
    acc_o = [cute.make_tensor(tmem_ptr + h * HD, lay_n64) for h in range(NH)]
    acc_main = [cute.make_tensor(tmem_ptr + scr + n * TILE_N, lay_main) for n in range(n_chunks)]
    acc_qk = [cute.make_tensor(tmem_ptr + scr + b * TILE_N, lay_main) for b in range(2)]
    acc_scratch = cute.make_tensor(tmem_ptr + scr + TILE_N, lay_main)
    acc_down = [cute.make_tensor(tmem_ptr + n * TILE_N, lay_main) for n in range(n_chunks)]
    rshape = _reg_shape(mma128, acc_main[0], ids.lane, TILE_N)
    rshape_o = _reg_shape(mma64, acc_o[0], ids.lane, HD, HEAD_COLS)

    ws = _Workspace(
        ids=ids,
        x=x,
        stage4=stage4,
        y=y,
        kv_cache=kv_cache,
        norm_cache=norm_cache,
        barrier=barrier,
        x_kf=x_kf,
        stage4_kf=stage4_kf,
        y_kf=y_kf,
        kv_cache_kf=kv_cache_kf,
        norm_cache_kf=norm_cache_kf,
        kf_times=kf_times,
        kf_slots=kf_slots,
        kf_planes=kf_planes,
        kf_counts=kf_counts,
        kf_video=kf_video,
        kf_video_counts=kf_video_counts,
        kf_slab=kf_slab,
        s_a=s_a,
        s_b=s_b,
        s_q=s_q,
        s_b4=s_b4,
        s_k=s_k,
        s_v=s_v,
        s_p_own=s_p_own,
        s_h_own=s_h_own,
        pool=pool,
        q_pool=q_pool,
        a_layout=a_layout,
        q_layout=q_layout,
        storage=storage,
        fr_a=mma128.make_fragment_A(s_a),
        fr_q=mma128.make_fragment_A(s_q),
        fr_h=mma128.make_fragment_A(s_h),
        fr_k=mma128.make_fragment_B(s_k),
        fr_p=mma64.make_fragment_A(s_p),
        fr_v=mma64.make_fragment_B(s_v),
        w_ctx=(a_ctx, m_ctx),
        w_q=(a_q, m_q),
        w_kv=(a_kv, m_kv),
        w_o=(a_o, m_o),
        w_gate=(a_gate, m_gate),
        w_up=(a_up, m_up),
        w_down=(a_down, m_down),
        acc_main=acc_main,
        acc_qk=acc_qk,
        acc_scratch=acc_scratch,
        acc_o=acc_o,
        acc_down=acc_down,
        areg=cute.make_rmem_tensor(rshape, ACC),
        breg=cute.make_rmem_tensor(rshape, ACC),
        oreg=cute.make_rmem_tensor(rshape_o, ACC),
        kvfrag=cute.make_rmem_tensor(cute.make_layout(HD), IO),
        rowix=rowix,
        rowix_lo=rowix_lo,
        rowix_w=rowix_w,
        kv_rowtab=kv_rowtab,
        kv_copy_bars=kv_copy_bars,
        copy_slot=cooperative_copy_slot(tidx),
        red_c=red_c,
        red_head=red_head,
        m_sm=m_sm,
        l_sm=l_sm,
        qkbar=qkbar,
        pvbar=pvbar,
        rope=rope,
        rope_inv=(inv_t, inv_h, inv_w),
        params=params,
        query_slot=query_slot_in_tile(ids.lane, TH, TW),
        volume=(T, H, W),
        ctx_volume=(T_lo, H_lo, W_lo),
        drop_leading=drop_leading,
        n_ctas=n_ctas,
        slab_count=slab_count,
        slab_grid_hw=slab_grid_hw,
        slab_grid_w=slab_grid_w,
        slab_tiles=(slab_tiles_t, slab_tiles_h, slab_tiles_w),
        max_box=(max_box_t, max_box_h, max_box_w),
        cache_row_stride=cache_row_stride,
        n_kf=n_kf,
        kf_cache_row_stride=kf_cache_row_stride,
    )

    slab_index = 0
    while slab_index < slab_count:
        slab = _slab_geometry(ws, shape, slab_index)
        _build_rope_tables(ws, shape, slab)
        _fill_slab_caches(ws, shape, slab, KF)
        # Every CTA must see the whole cache before any of them reads it.
        grid_barrier(barrier, tidx, n_ctas, 2 * slab_index)

        # One loop over both streams' tiles, video first. The keyframe stream's queries ride
        # this slab's barrier because both key sets they read -- their own plane, and the <=2
        # nearest video frames -- are single-plane panels in caches it has already filled.
        # A plane's tokens are split across slabs exactly like video tokens: the slab whose
        # *h/w core* holds the token, among those whose *temporal* core owns the plane. That
        # is what keeps every token written once and every read inside the box -- the h/w
        # panel is the core grown by the radius, and the plane's <=2 nearest video frames are
        # a step away from its own position, so the temporal box (radius >= 1) covers them.
        n_tiles, tile_plane, tile_row = _core_tile_grid(ws, shape, slab)
        tile_id = ids.cta
        if cutlass.const_expr(KF):
            # The two call sites below are one binary each: ``KF`` is compile-time, so a
            # keyframe build traces only the guarded call and a plain build only the bare
            # one. Written as a single call under a runtime-or-constant guard instead, the
            # plain build would carry a branch it can never take.
            # A keyframe tile carries ``TT`` planes (group ``g`` = planes ``g*TT .. g*TT+TT-1``), so the
            # projections and MLP run once per group of distinct queries. The group is live in this slab
            # if any of its planes is.
            n_groups = (ws.n_kf + shape.tile[0] - 1) // shape.tile[0]
            while tile_id < n_tiles + n_groups * tile_plane:
                live = cutlass.Int32(1)
                if tile_id >= n_tiles:
                    group = unflatten_3d(tile_id - n_tiles, tile_plane, tile_row)[0]
                    live = cutlass.Int32(0)
                    for j in cutlass.range_constexpr(shape.tile[0]):
                        plane_j = group * shape.tile[0] + j
                        if plane_j < ws.n_kf:  # noqa: SIM102
                            if ws.kf_slab[plane_j] == slab.t_index:
                                live = cutlass.Int32(1)
                if live != 0:
                    _run_tile(ws, shape, slab, tile_id, n_tiles, tile_plane, tile_row, KF, SLOTS, P_MAX, NKV_KF)
                tile_id += n_ctas
        else:
            while tile_id < n_tiles:
                _run_tile(ws, shape, slab, tile_id, n_tiles, tile_plane, tile_row, KF, SLOTS, P_MAX, NKV_KF)
                tile_id += n_ctas
        # ... and no CTA may start the next slab's cache fill until they all have.
        grid_barrier(barrier, tidx, n_ctas, 2 * slab_index + 1)
        slab_index += 1

    release_tmem(tmem, tmem_ptr)


# --------------------------------------------------------------------------
# Slab geometry
# --------------------------------------------------------------------------


def _slab_geometry(ws: _Workspace, shape: _BlockShape, slab_index) -> _Slab:
    """Locate one slab: the core it owns, and the halo box its K/V cache covers.
    Cores partition the volume, so nothing outside this one is written while it runs.
    The box is that core grown by the NA radius and clamped into the volume; every
    halo panel of every query tile in the core lies inside it, which is what lets one
    cache fill serve them all.
    """
    volume = ws.volume
    core_shape = (
        ws.slab_tiles[0] * shape.tile[0],
        ws.slab_tiles[1] * shape.tile[1],
        ws.slab_tiles[2] * shape.tile[2],
    )
    slab_t, slab_h, slab_w = unflatten_3d(slab_index, ws.slab_grid_hw, ws.slab_grid_w)
    core_origin = (slab_t * core_shape[0], slab_h * core_shape[1], slab_w * core_shape[2])
    core_end = tuple(at_most(core_origin[i] + core_shape[i], volume[i]) for i in range(3))
    box_shape = tuple(
        _slab_box_extent(core_origin[i], ws.slab_tiles[i], shape.tile[i], shape.kernel[i], volume[i], ws.max_box[i])
        for i in range(3)
    )
    box_origin = tuple(clamp_window_start(core_origin[i] - shape.radius[i], box_shape[i], volume[i]) for i in range(3))
    box_hw = box_shape[1] * box_shape[2]
    return _Slab(
        core_origin=core_origin,
        core_end=core_end,
        t_index=slab_t,
        box_origin=box_origin,
        box_shape=box_shape,
        box_hw=box_hw,
        n_box_tokens=box_shape[0] * box_hw,
    )


def _slab_box_extent(core_origin, core_tiles, tile, kernel, volume, max_box):
    """One axis of a slab's halo-box extent.
    The box grows from this slab's *own* core, not from the largest slab's, so a slab
    cut short by the volume edge projects fewer tokens -- and since the cache fill
    re-runs the box once per slab, that difference is real work. It never drops below
    one query tile's halo panel, which is the box :func:`panel_key_row` indexes into.
    """
    core_extent = at_most(core_tiles * tile, volume - core_origin)
    core_extent = at_least(core_extent, cutlass.Int32(tile))
    return at_most(core_extent + (kernel - 1), max_box)


def _core_tile_grid(ws: _Workspace, shape: _BlockShape, slab: _Slab):
    """How many query tiles this slab's core actually holds.
    Returns ``(count, plane_stride, row_stride)`` -- the total, and the two strides
    :func:`unflatten_3d` needs to turn a tile index back into a position in the core.
    A slab at the far edge of the volume is cut short, and the tiles it does not have
    must not be scheduled: a CTA taking one would write outside the volume.
    """
    n = tuple(
        at_most(
            (ws.volume[i] - slab.core_origin[i] + shape.tile[i] - 1) // shape.tile[i],
            ws.slab_tiles[i],
        )
        for i in range(3)
    )
    return n[0] * (n[1] * n[2]), n[1] * n[2], n[2]


def _build_rope_tables(ws: _Workspace, shape: _BlockShape, slab: _Slab) -> None:
    """Tabulate cos/sin over the slab box, once per slab.
    RoPE is keyed to the box rather than to each query tile's panel: the cache fill
    rotates K here, and the query projection reads the same tables with a box-local
    index, so the two cannot disagree about a token's position.
    """
    cos_t, sin_t, cos_h, sin_h, cos_w, sin_w = ws.rope
    inv_t, inv_h, inv_w = ws.rope_inv
    half = tuple(d // 2 for d in shape.rope_dims)
    cute.arch.sync_threads()
    fill_rope_axis(cos_t, sin_t, ws.ids.tidx, slab.box_origin[0], slab.box_shape[0], half[0], inv_t)
    fill_rope_axis(cos_h, sin_h, ws.ids.tidx, slab.box_origin[1], slab.box_shape[1], half[1], inv_h)
    fill_rope_axis(cos_w, sin_w, ws.ids.tidx, slab.box_origin[2], slab.box_shape[2], half[2], inv_w)
    cute.arch.sync_threads()


# --------------------------------------------------------------------------
# Filling the slab's K/V cache
# --------------------------------------------------------------------------


@cute.jit
def _fill_slab_caches(ws, shape: cutlass.Constexpr, slab, KF: cutlass.Constexpr = False):
    """Project the whole slab box into the K/V cache, ``M`` tokens per CTA per step.
    Every token of the box gets the block's front half run over it exactly once:
    ``post = x + fused_ctx(stage4)``, its RMSNorm factor, AdaLN, then the K and V
    projections. K and V land in the cache; ``post`` and its factor land in the output
    and in ``norm_cache``, because every query tile in this slab's core needs the same
    two values and rebuilding them per tile would cost a context gather, the context
    GEMM, an x gather and the residual add.
    With ``KF`` the keyframe planes' rows are walked by this same loop, after the video box's
    and appended to its step space: every weight, bias, norm and AdaLN constant is the video
    pass's -- which is exactly why the softmax bound stays a bound once these keys join a
    video query's row -- so a second copy of the body would only have cost compile time.
    :func:`_keyframe_fill_step` supplies the row indices and ``select_tensor`` the buffers.
    ``fused_ctx`` is the stage-4 upsample composed with ``context_proj``: one Linear per
    pixel-shuffle subpixel, packed as ``P`` stacked N-tiles of one ``(C, Ctx)`` weight.
    So a token's context is the row of ``stage4`` at its parent cell projected by *its*
    subpixel's slice -- which is why the box is walked by stage-4 cell and then by
    subpixel (:func:`_subpixel_step`) rather than row-major: one pass's ``M`` tokens then
    share a subpixel, and with it the one B operand an ``M``-row UMMA can take.
    The subpixel is minor across CTAs rather than inside one, so the ``P`` CTAs that share a
    cell run are resident together and share its stage-4 rows in L2. It has to be: a
    subpixel loop *inside* the CTA is only worth its extra reuse as a compile-time loop, and
    that unrolls the whole fill body ``P`` times, tripling its register frame.
    Publishing ``post`` into the output is safe here: cores partition the volume and
    slab ``s+1``'s fill runs after slab ``s``'s tile loop, so these rows belong to no
    finished slab.
    """
    lane, wg = ws.ids.lane, ws.ids.wg
    param, channels = shape.param, shape.channels
    _, vol_h, vol_w = ws.volume
    _, lo_h, lo_w = ws.ctx_volume
    stride = shape.stride
    cells = tuple((slab.box_shape[i] + stride[i] - 1) // stride[i] for i in range(3))
    n_cells = cells[0] * cells[1] * cells[2]
    n_steps = shape.n_subpixels * ((n_cells + M - 1) // M)
    n_steps_kf = 0
    if cutlass.const_expr(KF):
        n_steps_kf = _keyframe_fill_steps(ws, shape, slab)

    step = ws.ids.cta
    while step < n_steps + n_steps_kf:
        subpixel, box_offset, in_box = _subpixel_step(ws, shape, slab, step, cells, n_cells)
        pos = tuple(slab.box_origin[i] + box_offset[i] for i in range(3))
        row = flatten_3d(pos[0], pos[1], pos[2], vol_h, vol_w)
        # The cache and the norm cache are indexed by the box's own row-major order, not
        # by the order the fill walks it: that is the order the query tiles read back.
        token = flatten_3d(box_offset[0], box_offset[1], box_offset[2], slab.box_shape[1], slab.box_shape[2])
        # A lane past the end of the box parks its norm factor on a scratch entry: its
        # clamped coordinates would otherwise target a real token, whose factor a
        # legitimate lane is already writing.
        norm_row = token if in_box else slab.n_box_tokens + lane
        row_lo = flatten_3d(
            (pos[0] + ws.drop_leading) // stride[0], pos[1] // stride[1], pos[2] // stride[2], lo_h, lo_w
        )
        # ``post`` is published only for the rows this slab owns. A halo row belongs to
        # another slab's core, and a lane past the end of the box has clamped
        # coordinates that would otherwise look in-core; both go to the scratch rows
        # past the end of the output.
        owned = in_box and pos[0] >= slab.core_origin[0] and pos[0] < slab.core_end[0]
        owned = owned and pos[1] >= slab.core_origin[1] and pos[1] < slab.core_end[1]
        owned = owned and pos[2] >= slab.core_origin[2] and pos[2] < slab.core_end[2]
        write_row = row if owned else ws.n_volume_tokens + lane
        src_x, src_ctx, out, norms = ws.x, ws.stage4, ws.y, ws.norm_cache
        cache, cache_stride, kf_time = ws.kv_cache, ws.cache_row_stride, None
        rope_offset = box_offset

        if cutlass.const_expr(KF):
            # The keyframe planes' rows, walked by the same loop. Every quantity above is
            # recomputed for a plane and then selected, which is a handful of instructions per
            # ``M``-token step; the body below -- three gathers, the context GEMM, the residual,
            # the AdaLN and ``n_heads`` K/V projections -- is traced once instead of twice.
            kf_row = step >= n_steps
            kf = _keyframe_fill_step(ws, shape, slab, step - n_steps)
            row = kf.row if kf_row else row
            token = kf.token if kf_row else token
            norm_row = kf.norm_row if kf_row else norm_row
            row_lo = kf.row_lo if kf_row else row_lo
            write_row = kf.write_row if kf_row else write_row
            subpixel = kf.subpixel if kf_row else subpixel
            rope_offset = (
                box_offset[0],
                kf.box_offset[1] if kf_row else box_offset[1],
                kf.box_offset[2] if kf_row else box_offset[2],
            )
            kf_time = kf.time
            src_x = select_tensor(kf_row, ws.x_kf, ws.x)
            src_ctx = select_tensor(kf_row, ws.stage4_kf, ws.stage4)
            out = select_tensor(kf_row, ws.y_kf, ws.y)
            norms = select_tensor(kf_row, ws.norm_cache_kf, ws.norm_cache)
            cache = select_tensor(kf_row, ws.kv_cache_kf, ws.kv_cache)
            cache_stride = ws.kf_cache_row_stride if kf_row else ws.cache_row_stride

        ws.rowix[lane] = row
        ws.rowix_lo[lane] = row_lo
        ws.rowix_w[lane] = write_row

        cute.arch.sync_threads()
        # A ``Ctx``-wide stage-4 row does not fit one C-wide operand, so it lands in two:
        # ``s_a``, and ``s_q``, which is dead until the query projection. Both gathers are
        # issued before the barrier and the reduction stays one GEMM call spanning them --
        # the barriers and the GEMM's own pipeline stand-up cost far more at these shapes
        # than the extra K-tiles do, so this is the whole difference between the deferred
        # context and a full-resolution one.
        _fill_a_gather(ws.s_a, src_ctx, ws.rowix_lo, ws.copy_slot, channels)
        if cutlass.const_expr(shape.ctx_groups > 1):
            _fill_a_gather(ws.s_q, src_ctx, ws.rowix_lo, ws.copy_slot, channels, channels)
        cute.arch.sync_threads()
        _run_jobs(
            ws,
            tuple(
                _job(
                    ws,
                    shape,
                    ws.w_ctx,
                    ws.acc_main[n],
                    subpixel * shape.n_chunks + n,
                    n_k_tiles=shape.ctx_groups * shape.n_k_tiles,
                    frag2=ws.fr_q,
                    a_split=shape.n_k_tiles if shape.ctx_groups > 1 else 0,
                )
                for n in range(shape.n_chunks)
            ),
        )
        cute.arch.sync_threads()
        _fill_a_gather(ws.s_a, src_x, ws.rowix, ws.copy_slot, channels)
        cute.arch.sync_threads()

        post_sq = _add_residual(ws, shape, ws.acc_main, param["ctx_bias"], subpixel * channels)
        rms_inv = cute.rsqrt(row_sum(ws.red_c, lane, wg, post_sq) / ACC(channels) + ACC(EPS))
        # Publish the factor with ``post``; both warpgroups hold the reduced value, so
        # one writes it. One f32 a token, a 256th of what caching the normed row costs.
        if wg == 0:
            norms[norm_row] = rms_inv
        _scatter_a(ws.s_a, out, ws.rowix_w, ws.copy_slot, channels)
        # A cooperative copy spans the whole operand -- every thread touches rows and
        # channels outside its own warpgroup's slice -- so it has to finish before the
        # per-warpgroup norm below rewrites s_a. Without this the two overlap whenever
        # the warpgroups fall out of lockstep, and the scatter writes normed values
        # where it meant to write the residual.
        cute.arch.sync_threads()
        _apply_adaln(ws, shape, rms_inv, param["msa_weight"], param["msa_shift"])
        cute.arch.sync_threads()

        # Heads go in pairs, one pipeline and one wait per pair, into the two accumulators the
        # residual above has finished with: ``acc_main[0]`` and ``acc_scratch``, ``TILE_N``
        # columns apart. Both loops are device loops, so the per-head pass is traced once;
        # the head reaches only the weight N-tile, a TMEM address, bias offsets and cache
        # rows. ``n_heads`` is even (``attention_tile`` needs it too).
        for pair in cutlass.range(shape.n_heads // 2):
            _run_jobs(
                ws,
                (
                    _job(ws, shape, ws.w_kv, ws.acc_main[0], 2 * pair),
                    _job(ws, shape, ws.w_kv, ws.acc_scratch, 2 * pair + 1),
                ),
            )
            for j in cutlass.range(2):
                head = 2 * pair + j
                acc = _tmem_shift(ws.acc_main[0], j * TILE_N)
                if cutlass.const_expr(KF):
                    _project_head_kv(ws, shape, head, acc, token, rope_offset, cache, cache_stride, kf_time, kf_row)
                else:
                    _project_head_kv(ws, shape, head, acc, token, rope_offset)
        step += ws.n_ctas


class _KeyframeFillStep(NamedTuple):
    """Where one keyframe fill step reads and writes, mirroring the video walk's quantities."""

    row: Any  # into ``x_kf`` / the output: (plane, h, w) over the whole volume grid
    row_lo: Any  # into ``stage4_kf``: its parent stage-4 cell
    token: Any  # into the keyframe K/V cache, and the norm cache
    norm_row: Any  # ``token``, or a scratch entry for a lane past the end of the box
    write_row: Any  # ``row`` if this slab owns it, else a scratch row past the output
    subpixel: Any  # which fused-context N-tile this row's parent projects through
    box_offset: Any  # (0, h, w) inside the slab box -- the spatial rope index
    time: Any  # the plane's float chunk-center position, for the temporal rope


@cute.jit
def _keyframe_fill_steps(ws, shape: cutlass.Constexpr, slab):
    """How many fill steps the keyframe planes need in this slab.
    Only the slab box's ``h/w`` rectangle is filled: a query in the core reads a keyframe
    window around its own ``(h, w)``, and the box is the core grown by the NA radius, so that
    rectangle is already the whole reachable set. The temporal axis is the plane index, not a
    box offset -- a keyframe plane is one slice however wide the box is in ``t``, so the walk
    covers the spatial subpixels only.
    """
    stride = shape.stride
    cells_h = (slab.box_shape[1] + stride[1] - 1) // stride[1]
    cells_w = (slab.box_shape[2] + stride[2] - 1) // stride[2]
    n_cells = ws.n_kf * cells_h * cells_w
    return stride[1] * stride[2] * ((n_cells + M - 1) // M)


@cute.jit
def _keyframe_fill_step(ws, shape: cutlass.Constexpr, slab, step):
    """Locate keyframe fill step ``step``: the video walk's geometry, one plane deep.
    Two things differ from the video pass and both are forced by the plane being a single
    temporal slice. Its stage-4 parent is looked up without a temporal divide, and under a
    joint ``(2,2,2)`` upsample it takes the temporal subpixel a one-frame input would shuffle
    out to -- ``drop_leading % SP_T``, which is subpixel 1 for the shipped stride and 0 when
    the hop is spatial-only. Never both: a keyframe stays ``n_kf`` planes across every stage.
    Called for video steps too, whose results are then selected away, so ``step`` may be
    negative and ``plane`` out of range: both are clamped, and every table read below lands
    on a real entry (row 0 at worst) rather than off the end.
    """
    lane = ws.ids.lane
    _, vol_h, vol_w = ws.volume
    _, lo_h, lo_w = ws.ctx_volume
    stride = shape.stride
    zero = cutlass.Int32(0)
    cells_h = (slab.box_shape[1] + stride[1] - 1) // stride[1]
    cells_w = (slab.box_shape[2] + stride[2] - 1) // stride[2]
    n_cells = ws.n_kf * cells_h * cells_w
    sub_t = ws.drop_leading % stride[0]
    kf_hw = slab.box_shape[1] * slab.box_shape[2]

    step = at_least(step, zero)
    sub_hw = step % (stride[1] * stride[2])
    cell = (step // (stride[1] * stride[2])) * M + lane
    in_box = cell < n_cells
    plane, cell_h, cell_w = unflatten_3d(at_most(cell, at_least(n_cells - 1, zero)), cells_h * cells_w, cells_w)
    plane = at_most(plane, at_least(ws.n_kf - 1, zero))
    sub_h = sub_hw // stride[2]
    sub_w = sub_hw - sub_h * stride[2]
    # Smallest box offset whose absolute position carries this subpixel, then step by
    # the stride -- the video pass's rule with the temporal axis removed.
    first_h = (sub_h + stride[1] - slab.box_origin[1] % stride[1]) % stride[1]
    first_w = (sub_w + stride[2] - slab.box_origin[2] % stride[2]) % stride[2]
    off_h = first_h + stride[1] * at_most(cell_h, (slab.box_shape[1] - 1 - first_h) // stride[1])
    off_w = first_w + stride[2] * at_most(cell_w, (slab.box_shape[2] - 1 - first_w) // stride[2])
    in_box = in_box and first_h + stride[1] * cell_h < slab.box_shape[1]
    in_box = in_box and first_w + stride[2] * cell_w < slab.box_shape[2]
    pos_h = slab.box_origin[1] + off_h
    pos_w = slab.box_origin[2] + off_w

    row = flatten_3d(plane, pos_h, pos_w, vol_h, vol_w)
    token = flatten_3d(plane, off_h, off_w, slab.box_shape[1], slab.box_shape[2])
    # ``post`` is published only for the rows this slab owns -- same rule as the video
    # pass, and load-bearing for the same reason. Every slab projects every plane over
    # its own halo rectangle (that is what makes the keyframe keys available to its video
    # queries), so without the gate a later slab would overwrite the *finished* block
    # output an earlier slab left in ``y_kf`` with ``post`` again. A plane's rows belong
    # to the slab whose temporal core owns the plane and whose h/w core holds the token.
    owned = in_box and ws.kf_slab[plane] == slab.t_index
    owned = owned and pos_h >= slab.core_origin[1] and pos_h < slab.core_end[1]
    owned = owned and pos_w >= slab.core_origin[2] and pos_w < slab.core_end[2]
    return _KeyframeFillStep(
        row=row,
        row_lo=flatten_3d(plane, pos_h // stride[1], pos_w // stride[2], lo_h, lo_w),
        token=token,
        norm_row=token if in_box else ws.n_kf * kf_hw + lane,
        write_row=row if owned else ws.n_kf * vol_h * vol_w + lane,
        subpixel=sub_t * stride[1] * stride[2] + sub_hw,
        box_offset=(zero, off_h, off_w),
        time=ws.kf_times[plane],
    )


def _subpixel_step(ws, shape: cutlass.Constexpr, slab, step, cells, n_cells):
    """Where fill step ``step`` sits: its subpixel, this lane's box offset, and validity.
    The box is walked as ``P`` passes over the stage-4 cells it spans, subpixel-minor so
    that consecutive CTAs take the ``P`` subpixels of the *same* cells and share their
    stage-4 rows in L2. Within a pass, lane ``l`` takes one cell and the box offset is
    that cell's position for this subpixel::
        offset = first + stride * cell
    where ``first`` is the smallest box offset whose absolute position has this
    subpixel. ``cells`` is rounded up per axis, so a pass can overrun the box by up to
    one cell per axis; those lanes report ``in_box=False`` and are clamped back onto a
    real position of the same subpixel, which makes their duplicate K/V store harmless
    (it recomputes what the owning lane writes) exactly as a row-major overrun does.
    """
    lane, stride = ws.ids.lane, shape.stride
    sub = step % shape.n_subpixels
    cell = (step // shape.n_subpixels) * M + lane
    cell_offset = unflatten_3d(at_most(cell, n_cells - 1), cells[1] * cells[2], cells[2])
    # Per-axis subpixel, from the ``(c, p_t, p_h, p_w)`` channel layout the pack folded.
    sub_thw = (sub // (stride[1] * stride[2]), (sub // stride[2]) % stride[1], sub % stride[2])
    # The dropped duplicate leading frame shifts the whole temporal axis by one.
    shift = (ws.drop_leading, cutlass.Int32(0), cutlass.Int32(0))
    # ``+ 2 * stride`` keeps the dividend positive: integer ``%`` here is C's, not
    # Python's, so a negative one would return a negative offset.
    first = tuple(
        (sub_thw[i] + 2 * stride[i] - shift[i] - slab.box_origin[i] % stride[i]) % stride[i] for i in range(3)
    )
    # ``&`` rather than ``and``: this helper touches no runtime branch, so it traces
    # inline like the rest of the geometry instead of costing a device call frame.
    fits = (cell < n_cells) & (first[0] + stride[0] * cell_offset[0] < slab.box_shape[0])
    fits = fits & (first[1] + stride[1] * cell_offset[1] < slab.box_shape[1])
    fits = fits & (first[2] + stride[2] * cell_offset[2] < slab.box_shape[2])
    # ``first < stride <= box_shape`` (a tile holds a whole cell), so the clamp target is
    # always a real offset of this subpixel.
    box_offset = tuple(
        first[i] + stride[i] * at_most(cell_offset[i], (slab.box_shape[i] - 1 - first[i]) // stride[i])
        for i in range(3)
    )
    return sub, box_offset, fits


@cute.jit
def _project_head_kv(
    ws,
    shape: cutlass.Constexpr,
    head,
    acc,
    cache_row,
    box_offset,
    dst=None,
    row_stride=None,
    kf_time=None,
    kf_row=None,
):
    """Finish one head's K and V projection out of ``acc`` and store both to the cache.
    The packed K|V weight puts K in the accumulator's first ``HD`` columns and V in
    the second, which lines the warpgroup split up with the two paths: the first
    ``WARPGROUPS_PER_HEAD`` warpgroups take K (bias, RMSNorm, RoPE), the rest take V (bias only).
    ``dst`` / ``row_stride`` retarget the store, and ``kf_time`` swaps the temporal RoPE for a
    float chunk-center position: that is the whole difference between filling the video cache
    and filling the keyframe one. The projection, both biases, ``k_norm`` and the spatial RoPE
    are identical -- which is also what keeps the softmax bound a bound for keyframe keys.
    ``kf_row`` decides that swap at *runtime*, for the merged fill that walks both streams'
    rows in one loop. Given, both temporal ropes are traced and one is selected; left ``None``
    only the lattice one is, which is what a plain build gets.
    ``acc`` holds the head's projection, which the caller already ran. ``head`` may be a
    runtime value: it reaches only bias offsets and cache rows.
    """
    lane, wg = ws.ids.lane, ws.ids.wg
    areg, params, param = ws.areg, ws.params, shape.param
    n_heads, rope_dims = shape.n_heads, shape.rope_dims
    dst = ws.kv_cache if dst is None else dst
    row_stride = ws.cache_row_stride if row_stride is None else row_stride

    k_sq = ACC(0.0)
    for g in cutlass.range_constexpr(NUM_WARPGROUPS):
        if wg == g:
            c0 = (g % WARPGROUPS_PER_HEAD) * ACC_COLS
            _tmem_load(acc, lane, areg, g)
            if cutlass.const_expr(g < WARPGROUPS_PER_HEAD):
                s = ACC(0.0)
                for d in cutlass.range(ACC_COLS, unroll_full=True):
                    z = areg[d] + params[param["k_bias"] + head * HD + c0 + d]
                    areg[d] = z
                    s += z * z
                k_sq = s
            else:
                for d in cutlass.range(ACC_COLS, unroll_full=True):
                    areg[d] = areg[d] + params[param["v_bias"] + head * HD + c0 + d]
    k_rms, _unused = head_sum2(ws.red_head, lane, wg, k_sq, ACC(0.0))
    k_rms = cute.rsqrt(k_rms / ACC(HD) + ACC(EPS))
    for g in cutlass.range_constexpr(NUM_WARPGROUPS):
        if wg == g:
            c0 = (g % WARPGROUPS_PER_HEAD) * ACC_COLS
            if cutlass.const_expr(g < WARPGROUPS_PER_HEAD):
                for d in cutlass.range(ACC_COLS, unroll_full=True):
                    areg[d] = areg[d] * k_rms * params[param["k_norm"] + c0 + d]
                if cutlass.const_expr(kf_time is None):
                    apply_rope(areg, *rope_dims, ws.rope, *box_offset, c0, ACC_COLS)
                elif kf_row:
                    apply_rope_float_t(
                        areg, *rope_dims, ws.rope, kf_time, box_offset[1], box_offset[2], ws.rope_inv[0], c0, ACC_COLS
                    )
                else:
                    apply_rope(areg, *rope_dims, ws.rope, *box_offset, c0, ACC_COLS)
                _store_head_row(dst, head * row_stride + cache_row, areg, ACC_COLS, c0)
            else:
                _store_head_row(dst, (n_heads + head) * row_stride + cache_row, areg, ACC_COLS, c0)


# --------------------------------------------------------------------------
# One query tile
# --------------------------------------------------------------------------


class _TileGeometry(NamedTuple):
    """Where one query tile reads and writes, for either stream.
    Everything :func:`_run_tile` needs to locate itself, which is also the only part of it
    that differs between the two streams. Under ``KF`` the keyframe quantities are computed
    unconditionally -- pure arithmetic plus small table lookups, cheaper than branching
    around them -- so a video tile evaluates them too and selects the results away. That is
    why these are runtime selections rather than one branch's two exits.
    """

    kf_tile: Any  # this tile is a keyframe plane group's; None in a binary without that stream
    video_tile: Any  # its complement, which the deep temporal panel keys off
    plane: Any  # this lane's anchor plane (``kf_group * TT + slot_t``), clamped into range
    kf_group: Any  # the plane group a keyframe tile carries (planes ``kf_group*TT ..``)
    kf_live: Any  # per lane: its plane exists and this slab owns it (0/1 Int32)
    query_pos: tuple  # this lane's (t, h, w) in the volume
    read_row: Any  # the row ``post`` is gathered from
    write_row: Any  # where the result lands; a scratch row for an out-of-range lane
    norm_row: Any  # into the stream's norm cache, indexed over the slab box
    panel_origin: tuple  # the video halo panel's (t, h, w)
    panel_h0: Any  # the panel's spatial origin for *this* stream
    panel_w0: Any
    window: tuple  # the NA window's low bounds, panel-relative
    window_hi: Any  # its high bounds under clamp-and-mask; None under the inward shift
    rope_pos: tuple  # the box offset the rope tables are indexed by
    kf_time: Any  # a plane's float chunk-center time; None for a video tile
    t_idx: Any  # this tile's video temporal index, which the plane union is keyed by


@cute.jit
def _tile_geometry(
    ws,
    shape: cutlass.Constexpr,
    slab,
    tile_id,
    n_core_tiles,
    tile_plane,
    tile_row,
    KF: cutlass.Constexpr = False,
):
    """Locate query tile ``tile_id``: which stream owns it, and every index it reads with.
    ``tile_id`` indexes the two grids end to end: ``[0, n_core_tiles)`` are this slab's
    video tiles and the rest are ``(plane, h, w)`` keyframe tiles. Both unflatten with the
    same divisors -- a keyframe grid is ``(n_kf, tiles_h, tiles_w)`` against the video's
    ``(tiles_t, tiles_h, tiles_w)`` -- so only the temporal component's *meaning* differs:
    a tile origin for video, a plane index for keyframes.
    """
    lane = ws.ids.lane
    zero = cutlass.Int32(0)
    _, vol_h, vol_w = ws.volume

    index = tile_id
    kf_tile, video_tile, plane, kf_group, kf_live = None, None, None, None, None
    if cutlass.const_expr(KF):
        kf_tile = tile_id >= n_core_tiles
        video_tile = tile_id < n_core_tiles
        index = tile_id - n_core_tiles if kf_tile else tile_id
    local = unflatten_3d(index, tile_plane, tile_row)
    # Tiles partition the volume: edge tiles are partial and their out-of-range lanes
    # write to scratch, so no output row is touched by two CTAs. (The MMA still runs at
    # full M; the extra rows are garbage.)
    tile_origin = tuple(slab.core_origin[i] + local[i] * shape.tile[i] for i in range(3))
    panel_origin = halo_panel_origin(tile_origin, shape.radius, shape.panel, ws.volume)
    cute.arch.sync_threads()
    query_pos, read_row, write_row = query_row_indices(
        tuple(tile_origin[i] + ws.query_slot[i] for i in range(3)),
        ws.volume,
        lane,
        ws.n_volume_tokens,
    )
    # Two window rules. The joint keyframe operator is defined on clamp-and-mask, every
    # other caller on NATTEN's inward shift, and ``KF`` is exactly that distinction -- keyed
    # off the compile flag so the no-keyframe binary keeps a byte-identical trace.
    if cutlass.const_expr(KF):
        bounds = window_bounds_in_panel(query_pos, panel_origin, shape.radius, shape.kernel, ws.volume)
        window = tuple(lo for lo, _ in bounds)
        window_hi = tuple(hi for _, hi in bounds)
    else:
        window = window_origin_in_panel(query_pos, panel_origin, shape.radius, shape.kernel, ws.volume)
        window_hi = None
    # Where this query sits inside the slab box: the index for both the rope tables,
    # which were built over the box, and the cached RMSNorm factor.
    box_pos = tuple(query_pos[i] - slab.box_origin[i] for i in range(3))
    norm_row = flatten_3d(box_pos[0], box_pos[1], box_pos[2], slab.box_shape[1], slab.box_shape[2])
    panel_h0, panel_w0 = panel_origin[1], panel_origin[2]
    rope_pos, kf_time = box_pos, None
    t_idx = tile_origin[0] // shape.tile[0]

    if cutlass.const_expr(KF):
        # The keyframe tile's geometry: the video tile's with the temporal axis replaced by
        # the plane. Its ``h/w`` origin is already ``tile_origin[1:]``, so only the temporal
        # half, the row indices and the buffers are recomputed. Every read below is index-
        # clamped because a video tile evaluates this arithmetic too -- it is pure arithmetic
        # plus small table lookups, and selecting the results is cheaper than branching
        # around them.
        # ``TT`` planes per keyframe tile: slot ``slot_t`` of group ``local[0]`` is plane ``group * TT + slot_t``;
        # :func:`_run_tile` walks the planes one pass set at a time, each lane gated into its own plane's passes.
        kf_group = local[0]
        slot_t, slot_h, slot_w = ws.query_slot
        plane_raw = kf_group * shape.tile[0] + slot_t
        plane = at_most(at_least(plane_raw, zero), at_least(ws.n_kf - 1, zero))
        kf_live = cutlass.Int32(0)
        if plane_raw < ws.n_kf:  # noqa: SIM102
            if ws.kf_slab[plane] == slab.t_index:
                kf_live = cutlass.Int32(1)
        q_h = tile_origin[1] + slot_h
        q_w = tile_origin[2] + slot_w
        r_h = q_h if q_h < vol_h else vol_h - 1
        r_w = q_w if q_w < vol_w else vol_w - 1
        kf_read = flatten_3d(plane, r_h, r_w, vol_h, vol_w)
        kf_write = kf_read
        if kf_live == 0 or q_h >= vol_h or q_w >= vol_w:
            kf_write = ws.n_kf * vol_h * vol_w + lane
        # The *panel* still shifts inward -- it only has to contain every key the tile can
        # reach, and the inward-shifted box is the larger of the two -- but the per-query
        # window is clamp-and-mask, like every window on the joint path.
        kf_panel_h0 = clamp_window_start(tile_origin[1] - shape.radius[1], shape.panel[1], vol_h)
        kf_panel_w0 = clamp_window_start(tile_origin[2] - shape.radius[2], shape.panel[2], vol_w)
        kf_bounds = window_bounds_in_panel(
            (zero, r_h, r_w),
            (zero, kf_panel_h0, kf_panel_w0),
            (0, shape.radius[1], shape.radius[2]),
            (1, shape.kernel[1], shape.kernel[2]),
            (cutlass.Int32(1), vol_h, vol_w),
        )
        kf_box_h = r_h - slab.box_origin[1]
        kf_box_w = r_w - slab.box_origin[2]

        read_row = kf_read if kf_tile else read_row
        write_row = kf_write if kf_tile else write_row
        norm_row = flatten_3d(plane, kf_box_h, kf_box_w, slab.box_shape[1], slab.box_shape[2]) if kf_tile else norm_row
        panel_h0 = kf_panel_h0 if kf_tile else panel_h0
        panel_w0 = kf_panel_w0 if kf_tile else panel_w0
        window = (window[0], kf_bounds[1][0] if kf_tile else window[1], kf_bounds[2][0] if kf_tile else window[2])
        window_hi = (
            window_hi[0],
            kf_bounds[1][1] if kf_tile else window_hi[1],
            kf_bounds[2][1] if kf_tile else window_hi[2],
        )
        rope_pos = (
            box_pos[0],
            kf_box_h if kf_tile else box_pos[1],
            kf_box_w if kf_tile else box_pos[2],
        )
        # A keyframe query sits at a float chunk-center time, off the lattice the slab's
        # temporal rope table was built over.
        kf_time = ws.kf_times[plane]
        # The tail's plane union is indexed by video temporal tile; a keyframe tile has no
        # such index, and reads row 0 whose value it then selects away.
        t_idx = zero if kf_tile else t_idx

    return _TileGeometry(
        kf_tile=kf_tile,
        video_tile=video_tile,
        plane=plane,
        kf_group=kf_group,
        kf_live=kf_live,
        query_pos=query_pos,
        read_row=read_row,
        write_row=write_row,
        norm_row=norm_row,
        panel_origin=panel_origin,
        panel_h0=panel_h0,
        panel_w0=panel_w0,
        window=window,
        window_hi=window_hi,
        rope_pos=rope_pos,
        kf_time=kf_time,
        t_idx=t_idx,
    )


@cute.jit
def _run_tile(
    ws,
    shape: cutlass.Constexpr,
    slab,
    tile_id,
    n_core_tiles,
    tile_plane,
    tile_row,
    KF: cutlass.Constexpr = False,
    SLOTS: cutlass.Constexpr = 2,
    P_MAX: cutlass.Constexpr = 1,
    NKV_KF: cutlass.Constexpr = 1,
):
    """The whole back half of the block for one ``M``-query tile, of either stream.
    Reads ``post`` and its RMSNorm factor back out of the slab's caches, projects Q,
    runs windowed NA against the slab's K/V cache, then the out-projection, its
    residual and AdaLN, the SwiGLU MLP and the closing residual. Every intermediate
    lives in ``s_a`` or in TMEM; the only global traffic is the output row, read and
    rewritten as each residual needs it.
    One body serves video and keyframe tiles, and that is a compile-time decision, not a
    tidiness one. Everything after the geometry is identical between the two streams --
    same weights, same AdaLN constants, same MLP -- so a second copy of it bought nothing
    and cost 60 s of ptxas and MLIR on a kernel that is already the compile-time floor.
    What differs is which buffers the rows live in, and :func:`select_tensor` makes that a
    runtime address rather than a second instantiation.
    :func:`_tile_geometry` carries the whole of that difference except the buffers, which
    stay here because there is nothing to compute about them: given its ``kf_tile``, each
    pair is one ``select_tensor``.
    """
    ids, param = ws.ids, shape.param
    lane, channels = ids.lane, shape.channels
    zero = cutlass.Int32(0)

    # ---------------- which tile, and where it reads ----------------------
    geom = _tile_geometry(ws, shape, slab, tile_id, n_core_tiles, tile_plane, tile_row, KF)
    out, norms = ws.y, ws.norm_cache
    if cutlass.const_expr(KF):
        out = select_tensor(geom.kf_tile, ws.y_kf, ws.y)
        norms = select_tensor(geom.kf_tile, ws.norm_cache_kf, ws.norm_cache)

    # ---------------- residual, AdaLN, Q ---------------------------------
    ws.rowix[lane] = geom.read_row
    ws.rowix_w[lane] = geom.write_row
    # The cache fill already built ``post = x + ctx_proj(context)`` for every row this
    # slab owns and left it in the output, so read it back instead of rebuilding it per
    # tile -- and its RMSNorm factor with it, sparing this tile a pass over the whole
    # row and the CTA-wide reduction that pass feeds.
    cute.arch.sync_threads()
    _fill_a_gather(ws.s_a, out, ws.rowix, ws.copy_slot, channels)
    cute.arch.sync_threads()
    rms_inv = norms[geom.norm_row]
    _apply_adaln(ws, shape, rms_inv, param["msa_weight"], param["msa_shift"])
    cute.arch.sync_threads()

    _project_queries(ws, shape, geom.rope_pos, geom.kf_time, geom.kf_tile)
    attn_zero_accumulators(ws.acc_o, ws.l_sm, ws.oreg, lane, ids.wg, shape.n_heads)

    # ---------------- windowed neighborhood attention --------------------
    # The K/V source is this slab's fused cache, so both halves are ``kv_cache`` --
    # K's heads first, then V's -- the head stride is the halo box this call chose, and
    # one panel row is one cache row.
    # A video query runs this deep ``PT``-tall panel; a keyframe query has no local
    # temporal neighborhood at all, so it skips it and takes only the single-plane passes
    # below. That asymmetry is the reason the deep panel stays its own instantiation.
    deep = True
    if cutlass.const_expr(KF):
        deep = geom.video_tile
    if deep:
        attention_tile(
            (ws.fr_q, ws.fr_k, ws.fr_p, ws.fr_v),
            (ws.s_k, ws.s_v, ws.s_p_own),
            ws.acc_qk,
            ws.acc_o,
            (ws.qkbar, ws.pvbar),
            ws.m_sm,
            ws.l_sm,
            ws.areg,
            ws.kvfrag,
            (ws.kv_cache, ws.kv_cache),
            (cutlass.Int32(0), shape.n_heads * ws.cache_row_stride),
            ws.cache_row_stride,
            (*geom.panel_origin, *slab.box_origin, slab.box_shape[1], slab.box_shape[2]),
            geom.window,
            (ids.tidx, lane, ids.wg, ids.warp),
            shape.n_heads,
            1,
            *shape.panel,
            *shape.kernel,
            shape.n_kv_tiles,
            None,
            geom.window_hi,
            ws.kv_rowtab,
            ws.kv_copy_bars,
        )

    # Single-plane passes, into the same (m_sm, l_sm, acc_o). The fixed softmax offset means
    # continuing the accumulation *is* a joint softmax -- no rescale, no state merge -- and
    # the keyframe keys were normed by the same ``k_norm``, so the offset still bounds them.
    # Both streams' extra key sets are single planes on the query's own spatial window, so
    # one instantiation covers all three uses: a video query's selected keyframe planes, a
    # keyframe query's own plane, and its <=2 nearest video frames. Which cache a pass reads
    # is an address, and the pass count is runtime.
    if cutlass.const_expr(KF):
        # Each plane's key sets (its own plane, then its <=2 nearest video frames) are one pass set, walked
        # plane by plane with the other lanes gated out; a gated lane's exponentials are exact zeros, so its
        # softmax is untouched.
        n_outer = cutlass.Int32(1)
        if geom.kf_tile:
            n_outer = cutlass.Int32(shape.tile[0])
        outer = zero
        while outer < n_outer:
            plane_j = zero
            n_passes = ws.kf_counts[geom.t_idx]
            if geom.kf_tile:
                plane_raw_j = geom.kf_group * shape.tile[0] + outer
                plane_j = at_most(plane_raw_j, at_least(ws.n_kf - 1, zero))
                n_passes = 1 + ws.kf_video_counts[plane_j]
                # A plane past ``n_kf`` or owned by another slab is not ours: no passes, its lanes park on scratch.
                if plane_raw_j >= ws.n_kf:  # noqa: SIM114
                    n_passes = zero
                elif ws.kf_slab[plane_j] != slab.t_index:
                    n_passes = zero
            pass_i = zero
            while pass_i < n_passes:
                # Video tail: this tile's plane union, with each lane gated out of the planes its
                # own ``t`` did not pick -- the plane is a per-tile gather but the nearest-S set
                # is per ``t``, and a tile spans ``TT`` of them.
                source = ws.kf_planes[geom.t_idx * P_MAX + at_most(pass_i, P_MAX - 1)]
                sel = zero
                for s in cutlass.range_constexpr(SLOTS):
                    if ws.kf_slots[geom.query_pos[0] * SLOTS + s] == source:
                        sel = cutlass.Int32(1)
                video_kv = zero
                if geom.kf_tile:
                    # Keyframe tile: this pass set belongs to plane ``plane_j`` (the lanes of slot ``outer``).
                    sel = cutlass.Int32(1) if ws.query_slot[0] == outer else zero
                    if pass_i == 0:
                        source = plane_j
                    else:
                        source = ws.kf_video[plane_j * SLOTS + at_most(pass_i - 1, SLOTS - 1)]
                        video_kv = cutlass.Int32(1)
                cache = select_tensor(video_kv != 0, ws.kv_cache, ws.kv_cache_kf)
                row_stride = ws.cache_row_stride if video_kv != 0 else ws.kf_cache_row_stride
                # The keyframe cache is (n_kf, box_h, box_w): a plane index is absolute, so its
                # box origin is 0, while a video frame index is relative to the box.
                attention_tile(
                    (ws.fr_q, ws.fr_k, ws.fr_p, ws.fr_v),
                    (ws.s_k, ws.s_v, ws.s_p_own),
                    ws.acc_qk,
                    ws.acc_o,
                    (ws.qkbar, ws.pvbar),
                    ws.m_sm,
                    ws.l_sm,
                    ws.areg,
                    ws.kvfrag,
                    (cache, cache),
                    (zero, shape.n_heads * row_stride),
                    row_stride,
                    (
                        source,
                        geom.panel_h0,
                        geom.panel_w0,
                        slab.box_origin[0] if video_kv != 0 else zero,
                        slab.box_origin[1],
                        slab.box_origin[2],
                        slab.box_shape[1],
                        slab.box_shape[2],
                    ),
                    (zero, geom.window[1], geom.window[2]),
                    (ids.tidx, lane, ids.wg, ids.warp),
                    shape.n_heads,
                    1,
                    1,
                    shape.panel[1],
                    shape.panel[2],
                    1,
                    shape.kernel[1],
                    shape.kernel[2],
                    NKV_KF,
                    sel,
                    # One plane deep; the spatial pair is the query's own window, keeping both
                    # key sets aligned for the query sharing a softmax over them.
                    (cutlass.Int32(1), geom.window_hi[1], geom.window_hi[2]),
                    ws.kv_rowtab,
                    ws.kv_copy_bars,
                )
                pass_i += 1
            outer += 1

    # ---------------- out projection, residual, AdaLN --------------------
    cute.arch.sync_threads()
    attn_epilogue(ws.acc_o, ws.l_sm, ws.oreg, ws.pool, ws.a_layout, lane, ids.wg, shape.n_heads)
    cute.arch.sync_threads()
    _run_jobs(ws, tuple(_job(ws, shape, ws.w_o, ws.acc_main[n], n) for n in range(shape.n_chunks)))
    # s_a is free once the out-projection has consumed it, so ``post`` comes back
    # through it coalesced rather than one cache line per lane.
    cute.arch.sync_threads()
    _fill_a_gather(ws.s_a, out, ws.rowix, ws.copy_slot, channels)
    cute.arch.sync_threads()
    resid_sq = _add_residual(ws, shape, ws.acc_main, param["out_bias"])
    rms_inv = cute.rsqrt(row_sum(ws.red_c, lane, ids.wg, resid_sq) / ACC(channels) + ACC(EPS))
    cute.arch.sync_threads()
    _scatter_a(ws.s_a, out, ws.rowix_w, ws.copy_slot, channels)
    # A cooperative copy spans the whole operand -- every thread touches rows and
    # channels outside its own warpgroup's slice -- so it has to finish before the
    # per-warpgroup norm below rewrites s_a. Without this the two overlap whenever the
    # warpgroups fall out of lockstep, and the scatter writes normed values where it
    # meant to write the residual.
    cute.arch.sync_threads()
    _apply_adaln(ws, shape, rms_inv, param["mlp_weight"], param["mlp_shift"])
    cute.arch.sync_threads()

    # ---------------- SwiGLU MLP and the closing residual ----------------
    _swiglu_mlp(ws, shape)
    # s_a is free again after the last gate/up, so the final residual add reloads and
    # stores through it, coalesced.
    cute.arch.sync_threads()
    _fill_a_gather(ws.s_a, out, ws.rowix, ws.copy_slot, channels)
    cute.arch.sync_threads()
    _add_accumulator(ws, shape, ws.acc_down)
    cute.arch.sync_threads()
    _scatter_a(ws.s_a, out, ws.rowix_w, ws.copy_slot, channels)


@cute.jit
def _project_queries(ws, shape: cutlass.Constexpr, rope_pos, kf_time=None, kf_tile=None):
    """Project, normalise and rotate this tile's queries into ``s_q``.
    ``kf_time`` given, this tile may be a keyframe tile, and ``kf_tile`` says at runtime
    whether it is: a keyframe query sits at a float chunk-center time, off the lattice the
    slab's temporal rope table was built over, while h and w are still on it. Both temporal
    ropes are then traced, which is a few hundred instructions -- the alternative was tracing
    the whole projection twice.
    ``HEADS_PER_TILE`` heads at a time: an N=128 tile of ``w_q`` is already
    ``[head 2n | head 2n+1]``, so with one warpgroup per head the RMSNorm and RoPE stay
    inside a warpgroup -- no reduction, and half the GEMM calls of the per-head N=64
    projection it replaces.
    Also fills ``m_sm``, the fixed softmax offset the KV loop subtracts in place of an
    online row max.
    """
    lane, wg = ws.ids.lane, ws.ids.wg
    areg, params, param = ws.areg, ws.params, shape.param

    # Every head pair's GEMM goes through one pipeline and one wait, into the two
    # accumulators attention has not claimed yet: ``acc_main[0]`` and ``acc_scratch``,
    # ``TILE_N`` columns apart.
    n_pairs = shape.n_heads // HEADS_PER_TILE
    assert n_pairs <= 2, "the query projection has two free accumulators"
    _run_jobs(ws, tuple(_job(ws, shape, ws.w_q, _tmem_shift(ws.acc_main[0], hp * TILE_N), hp) for hp in range(n_pairs)))
    # A device loop over the pairs, and each warpgroup handles its own head of the pair: one
    # code copy of the passes below. The runtime head reaches only SMEM data indices, the
    # TMEM column and the query stage, which is a view shifted ``head`` stages.
    assert WARPGROUPS_PER_HEAD == 1, "each warpgroup owns one whole head of a pair"
    for hp in cutlass.range(n_pairs):
        head = hp * HEADS_PER_TILE + wg
        _tmem_load(_tmem_shift(ws.acc_main[0], hp * TILE_N), lane, areg, wg)
        # Pass one: bias, and the two row sums the normalisation needs. ``sum z^2``
        # gives the RMSNorm scale; ``sum (z * q_norm_w)^2`` gives ``||q||`` after it,
        # since the scale factors out. Both come off one pass and one barrier -- as two
        # dependent passes the allocator kept ``areg``'s floats live across the
        # reduction and spilled.
        q_sq = ACC(0.0)
        q_weighted_sq = ACC(0.0)
        for d in cutlass.range(ACC_COLS, unroll_full=True):
            z = areg[d] + params[param["q_bias"] + head * HD + d]
            areg[d] = z
            q_sq += z * z
            zw = z * params[param["q_norm"] + d]
            q_weighted_sq += zw * zw
        q_sum, q_norm_sq = head_sum2(ws.red_head, lane, wg, q_sq, q_weighted_sq)
        q_rms = cute.rsqrt(q_sum / ACC(HD) + ACC(EPS))
        # This row and head's fixed softmax offset, ``||q||`` times the ``||k||``
        # bound -- see :mod:`ltx_kernels.vae.softmax_bound`. ``||q||`` is the
        # reduced sum scaled after the fact, since the RMSNorm factor is uniform
        # over the head and so comes out of the square root. The ``q_norm`` run
        # already carries the attention scale and the base change, folded in by
        # :func:`_pack_channel_params`.
        ws.m_sm[lane, head] = cute.sqrt(q_norm_sq) * q_rms * params[param["k_bound"]] * LOG2E
        for d in cutlass.range(ACC_COLS, unroll_full=True):
            areg[d] = areg[d] * q_rms * params[param["q_norm"] + d]
        if cutlass.const_expr(kf_time is None):
            apply_rope(areg, *shape.rope_dims, ws.rope, *rope_pos, 0, ACC_COLS)
        elif kf_tile:
            apply_rope_float_t(
                areg,
                *shape.rope_dims,
                ws.rope,
                kf_time,
                rope_pos[1],
                rope_pos[2],
                ws.rope_inv[0],
                0,
                ACC_COLS,
            )
        else:
            apply_rope(areg, *shape.rope_dims, ws.rope, *rope_pos, 0, ACC_COLS)
        _fill_a_chunk(_alias(ws.q_pool, ws.q_layout, head * (M * TILE_K)), lane, areg, 0, ACC_COLS)


@cute.jit
def _swiglu_mlp(ws, shape: cutlass.Constexpr):
    """SwiGLU: N-tiled gate/up, then a K-accumulated down projection into ``acc_down``.
    Tiles along N (hidden channels), not fewer M-token rows: the CTA already owns a
    full ``M``-token tile from attention (lane ``i`` = row ``i``), and the same UMMA /
    ``gemm_tma`` shape serves every C-wide linear. N-tiling reuses that path and lets
    SiLU read gate/up from TMEM; token-chunking would fight lane ownership and force a
    second tiling scheme just for the MLP.
    The weights stream through ``s_b4``, ``MLP_B_STAGES`` tiles deep over ``s_q`` (dead once
    attention has read the queries); the hidden chunk is staged in ``s_h``, the two-stage weight
    slot the MLP no longer uses. Gate and up share one pipeline, as do the two down halves.
    """
    lane, wg = ws.ids.lane, ws.ids.wg
    areg, breg = ws.areg, ws.breg
    # A device loop, so the slice body is in the instruction stream once rather than
    # ``hidden / TILE_N`` times; the down projection's accumulate flag is a runtime predicate.
    for nt in cutlass.range(shape.hidden // TILE_N):
        # gate and up land in separate accumulators, so SiLU reads both straight from
        # TMEM instead of parking M gate values in registers.
        _run_jobs(
            ws,
            (_job(ws, shape, ws.w_gate, ws.acc_main[0], nt), _job(ws, shape, ws.w_up, ws.acc_scratch, nt)),
            ws.s_b4,
            MLP_B_STAGES,
        )
        _tmem_load(ws.acc_main[0], lane, areg, wg)
        _tmem_load(ws.acc_scratch, lane, breg, wg)
        for e in cutlass.range(ACC_COLS, unroll_full=True):
            gate = areg[e]
            # Two MUFU per element (ex2 and rcp.approx). The one-MUFU identity
            # ``silu(x) = x/2 (1 + tanh(x/2))`` is tempting and does not hold up:
            # ``tanh.approx.f32`` carries 2^-11 *absolute* error and the down
            # projection sums ``hidden`` of them, which cost 20 dB of PSNR
            # against torch.
            breg[e] = gate * cute.arch.rcp_approx(ACC(1.0) + cute.exp(-gate)) * breg[e]
        _fill_a_chunk(ws.s_h_own, lane, breg, 0, ACC_COLS)
        cute.arch.sync_threads()
        # Down projection: A is this hidden chunk, B is w_down's matching K-slice,
        # accumulated across chunks into one accumulator per N-tile. It reads ``s_h``,
        # so it takes ``fr_h`` rather than the ``fr_a`` every other projection uses.
        atom, tensor = ws.w_down
        _run_jobs(
            ws,
            tuple(
                GemmJob(atom, tensor, ws.fr_h, ws.acc_down[n], n, nt * 2, 2, 0, nt != 0) for n in range(shape.n_chunks)
            ),
            ws.s_b4,
            MLP_B_STAGES,
        )


# --------------------------------------------------------------------------
# Per-row channel passes
# Each walks the C-wide row in ``TILE_N`` chunks with a device loop, and each warpgroup
# handles its own ``ACC_COLS`` slice of a chunk, so the body is one code copy. A chunk's
# accumulator is ``acc[0]`` shifted ``n * TILE_N`` columns, which requires the
# accumulators to be evenly spaced -- ``acc_main`` and ``acc_down`` both are.
# --------------------------------------------------------------------------


def _own_chunk(ws, n):
    """``s_a`` shifted so stage 0 is this warpgroup's slice of chunk ``n``, K-stage ``n * NUM_WARPGROUPS + wg``."""
    return _alias(ws.pool, ws.a_layout, (n * NUM_WARPGROUPS + ws.ids.wg) * (M * TILE_K))


@cute.jit
def _add_residual(ws, shape: cutlass.Constexpr, acc, bias_off: cutlass.Constexpr, bias_row=0):
    """``s_a += acc + bias`` in place, returning the new row's sum of squares.
    ``acc`` is one TMEM accumulator per N-tile of the C-wide projection that just ran.
    The sum of squares comes off the same pass because the RMSNorm that follows needs
    it and a second pass would re-read the whole row.
    ``bias_row`` picks a C-wide run out of a multi-run bias, which is how the fused
    context's per-subpixel bias is reached; it is a runtime value and folds away at 0.
    """
    lane, wg = ws.ids.lane, ws.ids.wg
    areg, breg, params = ws.areg, ws.breg, ws.params
    total = ACC(0.0)
    for n in cutlass.range(shape.n_chunks):
        s_a = _own_chunk(ws, n)
        _tmem_load(_tmem_shift(acc[0], n * TILE_N), lane, breg, wg)
        _read_a_chunk(s_a, lane, areg, 0, ACC_COLS)
        for e in cutlass.range(ACC_COLS, unroll_full=True):
            v = areg[e] + breg[e] + params[bias_off + bias_row + n * TILE_N + wg * ACC_COLS + e]
            total += v * v
            areg[e] = v
        _fill_a_chunk(s_a, lane, areg, 0, ACC_COLS)
    return total


@cute.jit
def _apply_adaln(ws, shape: cutlass.Constexpr, rms_inv, weight_off: cutlass.Constexpr, shift_off: cutlass.Constexpr):
    """AdaLN in place on ``s_a``: ``value * rms_inv * weight + shift``, per channel.
    One multiply and one FFMA, because :func:`_pack_channel_params` already folded
    ``norm_w * (1 + scale)`` into the weight run.
    """
    lane, wg = ws.ids.lane, ws.ids.wg
    areg, params = ws.areg, ws.params
    for n in cutlass.range(shape.n_chunks):
        s_a = _own_chunk(ws, n)
        _read_a_chunk(s_a, lane, areg, 0, ACC_COLS)
        for e in cutlass.range(ACC_COLS, unroll_full=True):
            c = n * TILE_N + wg * ACC_COLS + e
            areg[e] = areg[e] * rms_inv * params[weight_off + c] + params[shift_off + c]
        _fill_a_chunk(s_a, lane, areg, 0, ACC_COLS)


@cute.jit
def _add_accumulator(ws, shape: cutlass.Constexpr, acc):
    """``s_a += acc`` in place -- the MLP's closing residual, no bias and no norm."""
    lane, wg = ws.ids.lane, ws.ids.wg
    areg, breg = ws.areg, ws.breg
    for n in cutlass.range(shape.n_chunks):
        s_a = _own_chunk(ws, n)
        _tmem_load(_tmem_shift(acc[0], n * TILE_N), lane, areg, wg)
        _read_a_chunk(s_a, lane, breg, 0, ACC_COLS)
        for e in cutlass.range(ACC_COLS, unroll_full=True):
            areg[e] = breg[e] + areg[e]
        _fill_a_chunk(s_a, lane, areg, 0, ACC_COLS)


def _job(ws, shape, weight, acc, n_idx, accumulate: bool = False, n_k_tiles=None, frag2=None, a_split=0):
    """One ``M``-row GEMM of the activation in ``s_a`` against a streamed weight.
    Unpacks the ``(atom, tensor)`` pair and fills in the operands every projection in this
    kernel shares. ``frag2`` continues the reduction in a second operand past K-tile
    ``a_split``, which is how the fused context's ``Ctx``-wide row -- too wide for one
    operand -- stays a single GEMM.
    """
    atom, tensor = weight
    n_kt = shape.n_k_tiles if n_k_tiles is None else n_k_tiles
    return GemmJob(atom, tensor, ws.fr_a, acc, n_idx, 0, n_kt, 0, accumulate, frag2, a_split)


def _run_jobs(ws, jobs, s_b=None, n_stages=2):
    """Run independent GEMMs through one :func:`gemm_tma` call: one pipeline, one wait.
    ``s_b`` / ``n_stages`` select the weight queue; the default is the two-stage ``ws.s_b``.
    """
    gemm_tma("n128", ws.storage, ws.s_b if s_b is None else s_b, _B_STAGE_BYTES, jobs, n_stages)


# --------------------------------------------------------------------------
# Packed per-channel constants
# --------------------------------------------------------------------------


#: Runs of the packed constant vector, in layout order, each with its length unit:
#: ``"C"`` for one value per channel, ``"PC"`` for one per channel per pixel-shuffle
#: subpixel, ``"HD"`` for one per head channel, ``"1"`` for a scalar.
_CHANNEL_PARAM_RUNS: "tuple[tuple[str, str], ...]" = (
    ("msa_weight", "C"),  # norm1_w * (1 + scale_msa) -- the AdaLN before attention
    ("msa_shift", "C"),  # shift_msa
    ("mlp_weight", "C"),  # norm2_w * (1 + scale_mlp) -- the AdaLN before the MLP
    ("mlp_shift", "C"),  # shift_mlp
    ("ctx_bias", "PC"),  # fused upsample x context_proj, one C-run per subpixel
    ("out_bias", "C"),  # attention out-projection
    ("q_bias", "C"),
    ("k_bias", "C"),
    ("v_bias", "C"),
    ("q_norm", "HD"),  # q_norm_w, carrying the attention scale
    ("k_norm", "HD"),
    ("k_bound", "1"),  # sqrt(HD) * max|k_norm_w| -- the ||k|| bound
)


def _channel_param_offsets(c: int, hd: int, n_subpixels: int) -> "dict[str, int]":
    """Start offset of every run in the packed vector, plus its ``total`` length."""
    unit = {"C": c, "PC": c * n_subpixels, "HD": hd, "1": 1}
    offsets, pos = {}, 0
    for name, length in _CHANNEL_PARAM_RUNS:
        offsets[name] = pos
        pos += unit[length]
    offsets["total"] = pos
    return offsets


def _pack_channel_params(
    norm1_w: torch.Tensor,
    norm2_w: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    scale_msa: torch.Tensor,
    shift_msa: torch.Tensor,
    scale_mlp: torch.Tensor,
    shift_mlp: torch.Tensor,
    b_ctx: torch.Tensor,
    b_o: torch.Tensor,
    b_q: torch.Tensor,
    b_k: torch.Tensor,
    b_v: torch.Tensor,
    softmax_bound: torch.Tensor,
    c: int,
    hd: int,
    n_subpixels: int,
    attn_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """Build the packed constant vector in :data:`_CHANNEL_PARAM_RUNS` order.
    ``norm * (1 + scale)`` is folded here, so AdaLN is one multiply and one FFMA per
    channel against two loads instead of three loads and five ops.
    """

    def flat(t: torch.Tensor) -> torch.Tensor:
        return t.detach().contiguous().float().to(device=device).view(-1)

    runs = {
        "msa_weight": flat(norm1_w) * (1.0 + flat(scale_msa)),
        "msa_shift": flat(shift_msa),
        "mlp_weight": flat(norm2_w) * (1.0 + flat(scale_mlp)),
        "mlp_shift": flat(shift_mlp),
        "ctx_bias": flat(b_ctx),
        "out_bias": flat(b_o),
        "q_bias": flat(b_q),
        "k_bias": flat(b_k),
        "v_bias": flat(b_v),
        # The kernel builds Q itself, so the attention scale folds in here rather than
        # costing a multiply per element per forward. The base change to 2 does not: it
        # rides on the softmax's row-offset subtraction (``fna_attn_core.LOG2E``).
        "q_norm": flat(q_norm_w) * attn_scale,
        "k_norm": flat(k_norm_w),
        "k_bound": softmax_bound.detach().to(device=device, dtype=torch.float32).reshape(1),
    }
    packed = torch.cat([runs[name] for name, _ in _CHANNEL_PARAM_RUNS])
    want = _channel_param_offsets(c, hd, n_subpixels)["total"]
    assert packed.numel() == want, f"packed {packed.numel()} != {want}"
    return packed


# --------------------------------------------------------------------------
# Host entry point
# --------------------------------------------------------------------------


def pack_fused_ctx_weights(
    w_upsample: torch.Tensor,
    b_upsample: torch.Tensor,
    w_ctx: torch.Tensor,
    b_ctx: torch.Tensor,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Compose ``context_proj(pixel_shuffle(upsample(.)))`` into one Linear per subpixel.
    The pixel-shuffle only *permutes* the upsample's output channels, so the whole
    stage-4 hop is ``P`` independent ``(C, Ctx)`` Linears -- one per subpixel -- over the
    same low-resolution row. Fusing them at load time is what lets the kernel read
    stage-4 directly and never see a full-resolution context.
    ``LinearPixelShuffleUpsample`` / einops lay the Linear's out channels as
    ``(c, p1, p2, p3)`` (``c`` slowest among those four). Flattening the three
    subpixel axes to ``P = p1*p2*p3``, row ``c * P + s`` of the upsample is channel
    ``c`` of subpixel ``s``. Equivalently ``w_up.view(C, P, Ctx).transpose(0, 1)[s]``
    is the ``(C, Ctx)`` slice for that subpixel, and
    ``W[s] = W_ctx @ that slice``, ``b[s] = W_ctx @ b_up_slice + b_ctx``. Returned
    stacked along N in subpixel order: ``(P*C, Ctx)`` and ``(P*C,)``, matching the
    kernel's N-tile / ``ctx_bias`` walk -- in ``context_proj``'s dtype and on its
    device, since a streaming load reaches this with a pre-read upsample still on
    the host.
    Requires square ``context_proj`` (``C_out == C_in``): shipped DiffVAE configs keep
    ``stage5_channels == stage_channels[-1]``. A checkpoint with ``c5 != c_ctx`` cannot
    use this pack (the deferred eager path still can).
    """
    channels, ctx_channels = int(w_ctx.shape[0]), int(w_upsample.shape[1])
    expanded = int(w_upsample.shape[0])
    if int(w_ctx.shape[1]) != channels:
        raise ValueError(
            f"context_proj must be square (C, C) to fuse into the DSL stage-4 hop; got "
            f"{tuple(w_ctx.shape)}. BLACKWELL_DSL needs stage5_channels == "
            f"stage_channels[-1]; use CHUNKED / COMBINED for non-square configs"
        )
    if expanded % channels != 0:
        raise ValueError(f"upsample out_features {expanded} is not a multiple of C={channels}")
    subpixels = expanded // channels
    device = w_ctx.device
    w_up = w_upsample.detach().to(device=device, dtype=torch.float32)
    b_up = b_upsample.detach().to(device=device, dtype=torch.float32).view(-1)
    w_c = w_ctx.detach().float()
    b_c = b_ctx.detach().to(device=device, dtype=torch.float32).view(-1)
    # (P, C, Ctx) and (P, C): batched GEMM over view(C, P, Ctx) then transpose to P-major.
    w_fused = w_c @ w_up.view(channels, subpixels, ctx_channels).transpose(0, 1)
    b_fused = b_up.view(channels, subpixels).t() @ w_c.t() + b_c
    return (
        w_fused.reshape(subpixels * channels, ctx_channels).to(dtype=w_ctx.dtype),
        b_fused.reshape(subpixels * channels).to(dtype=b_ctx.dtype),
    )


@functools.lru_cache(maxsize=8)
def _inert_keyframe_operands(channels: int, device: torch.device) -> "tuple":
    """Never-read stand-ins for the keyframe operands, wrapped once per device.
    Cached rather than built per call: the kernel takes all twelve unconditionally but reads
    none of them with the keyframe stream compiled out, so they only have to be *typed*
    right. Both the allocation and the ``from_dlpack`` wrap are hoisted here -- the wrap
    especially, since twelve of them on the hot path measured as ~9% of a small block launch.
    """
    act = _dyn_act(torch.zeros((1, channels), device=device, dtype=torch.bfloat16))
    f32 = _dyn(torch.zeros(1, device=device, dtype=torch.float32))
    i32 = _dyn(torch.zeros(1, device=device, dtype=torch.int32))
    return (act, act, act, act, f32, f32, i32, i32, i32, i32, i32, i32)


class _KeyframeWorkspace(NamedTuple):
    """The keyframe half of one launch: its compile-key facts, its operands, its output."""

    #: The caller supplied a keyframe stream -- so this launch is the *joint* operator, whose
    #: window rule is clamp-and-mask rather than NATTEN's inward shift. Deliberately not
    #: conditioned on the plane count: the rule belongs to the operator, and keying both it
    #: and the tail off "has planes" quietly served two different window rules depending on
    #: whether a caller passed zero planes or planes nothing selected. With no planes the
    #: tail is a zero-trip loop over real, empty tables.
    enabled: bool
    n_kf: int
    cache_row_stride: int
    #: The twelve keyframe arguments, already DSL-wrapped and in signature order.
    operands: tuple
    #: ``operands``' output buffer, unwrapped, because the return value slices it. ``None``
    #: with the stream compiled out, where there is nothing to slice.
    y: "torch.Tensor | None"


def _absent_keyframe_workspace(channels: int, device: torch.device) -> _KeyframeWorkspace:
    """No keyframe stream at all: inert operands, and nothing the kernel will read."""
    return _KeyframeWorkspace(
        enabled=False,
        n_kf=0,
        cache_row_stride=1,
        operands=_inert_keyframe_operands(channels, device),
        y=None,
    )


def _keyframe_workspace(
    x_keyframes: torch.Tensor | None,
    stage4_keyframes: torch.Tensor | None,
    keyframe_times: torch.Tensor | None,
    t: int,
    h: int,
    w: int,
    channels: int,
    ctx_channels: int,
    num_heads: int,
    head_dim: int,
    tile: "tuple[int, int, int]",
    plan: _SlabPlan,
    device: torch.device,
    out: torch.Tensor | None = None,
    keyframe_times_host: torch.Tensor | None = None,
) -> _KeyframeWorkspace:
    """Allocate the keyframe caches and precompute the two lookups, or hand back placeholders.
    The keyframe cache is the video cache with the box's temporal extent replaced by ``n_kf``:
    a plane is one slice however deep the slab is, and the box's ``h/w`` rectangle is already
    everything a core query can reach spatially. That makes it ``n_kf / box_t`` of the video
    cache -- a few percent at any real shape.
    ``out`` is the anchor stream's output buffer, ``(n_kf*H*W + OUT_SLACK_ROWS, C)``, so a
    chain can recycle it exactly as it recycles the video stream's.
    """
    supplied = (x_keyframes, stage4_keyframes, keyframe_times)
    if all(v is None for v in supplied):
        if out is not None:
            raise ValueError("out_keyframes needs a keyframe stream to write")
        return _absent_keyframe_workspace(channels, device)
    if any(v is None for v in supplied):
        raise ValueError("x_keyframes, stage4_keyframes and keyframe_times must be given together")

    n_kf = int(x_keyframes.shape[1])
    if n_kf == 0:
        # Still the joint operator, so still the joint binary -- only with nothing for the
        # tail to iterate. Everything the kernel indexes by *query tile* is real and sized;
        # everything it indexes by plane is a placeholder behind a zero-trip loop.
        inert = _dyn_act(torch.zeros((1, channels), device=device, dtype=torch.bfloat16))
        inert_f32 = _dyn(torch.zeros(1, device=device, dtype=torch.float32))
        n_tiles_t = -(-t // tile[0])
        # Only the lane slack, since there is no plane to store -- but still a real buffer
        # the out-of-range stores can land in, and still ``out``-able, so a chain does not
        # have to special-case an empty selection.
        empty_y = (
            torch.empty((M, channels), device=device, dtype=torch.bfloat16)
            if out is None
            else _check_out(out, M, channels, x_keyframes, "out_keyframes")
        )
        return _KeyframeWorkspace(
            enabled=True,
            n_kf=0,
            cache_row_stride=1,
            operands=(
                inert,
                inert,
                _dyn_act(empty_y),
                _dyn_act(torch.empty((2 * num_heads, head_dim), device=device, dtype=torch.bfloat16)),
                _dyn(torch.empty(1 + M, device=device, dtype=torch.float32)),
                inert_f32,
                _dyn(torch.full((t * KEYFRAME_CONTEXT_SLOTS,), -1, dtype=torch.int32, device=device)),
                _dyn(torch.full((n_tiles_t,), -1, dtype=torch.int32, device=device)),
                _dyn(torch.zeros(n_tiles_t, dtype=torch.int32, device=device)),
                _dyn(torch.zeros(1, dtype=torch.int32, device=device)),
                _dyn(torch.zeros(1, dtype=torch.int32, device=device)),
                _dyn(torch.zeros(1, dtype=torch.int32, device=device)),
            ),
            y=empty_y,
        )
    if tuple(x_keyframes.shape) != (1, n_kf, h, w, channels):
        raise ValueError(f"x_keyframes must be {(1, n_kf, h, w, channels)}, got {tuple(x_keyframes.shape)}")
    if tuple(stage4_keyframes.shape)[:2] != (1, n_kf) or int(stage4_keyframes.shape[-1]) != ctx_channels:
        raise ValueError(f"stage4_keyframes must be (1, {n_kf}, H_lo, W_lo, {ctx_channels})")
    if int(keyframe_times.numel()) != n_kf:
        raise ValueError(f"keyframe_times must have {n_kf} entries, got {int(keyframe_times.numel())}")

    # The plane tables are built on the host: from the caller's host copy of the times when given,
    # else from one read of ``keyframe_times``.
    host_times = keyframe_times.detach().cpu() if keyframe_times_host is None else keyframe_times_host
    slots = nearest_keyframe_slots(host_times, t)
    # ``SLOTS * tile[0]``, not ``min(n_kf, ...)``: the stride into ``planes`` is a compile-time
    # constant of the tile and must not move with the anchor layout. Slack columns are -1.
    planes, _ = tile_plane_union(slots, tile_t=tile[0], max_planes=KEYFRAME_CONTEXT_SLOTS * tile[0])
    counts = (planes >= 0).sum(dim=-1).to(torch.int32)
    video = nearest_video_frames(host_times, t)
    # Which slab owns each plane's query tiles: the one whose core holds the plane's own
    # position. Its box is that core grown by the NA radius (>=1 on every stage) and the <=2
    # nearest video frames are within a step, so the box always covers them.
    # ``plan.core`` counts query *tiles*, so the core's temporal extent in tokens is that
    # times the tile depth. Dividing by the tile count instead sends most planes to a slab
    # index that never comes up, and their query tiles silently never run.
    core_tokens = plan.core[0] * tile[0]
    positions = host_times.detach().to(torch.float32).round().clamp_(0, t - 1).to(torch.int64)
    owner = (positions // core_tokens).clamp_(0, plan.grid[0] - 1).to(torch.int32)

    kf_rows = n_kf * h * w
    cache_row_stride = n_kf * plan.box[1] * plan.box[2]

    def rows(x: torch.Tensor, ch: int) -> torch.Tensor:
        return _as_rows(x[0].detach().to(dtype=torch.bfloat16)).reshape(-1, ch)

    def i32(x: torch.Tensor) -> torch.Tensor:
        return upload_from_host(x.reshape(-1).to(dtype=torch.int32).contiguous(), device)

    x_rows = rows(x_keyframes, channels)
    y = (
        workspace_volume((kf_rows + M, channels), device=device, dtype=torch.bfloat16)
        if out is None
        # Against ``x_rows`` rather than ``x_keyframes``: what the kernel reads is the row
        # view, and for a non-contiguous input that is a copy the caller cannot alias anyway.
        else _check_out(out, kf_rows + M, channels, x_rows, "out_keyframes")
    )
    return _KeyframeWorkspace(
        enabled=True,
        n_kf=n_kf,
        cache_row_stride=cache_row_stride,
        operands=(
            _dyn_act(x_rows),
            _dyn_act(rows(stage4_keyframes, ctx_channels)),
            _dyn_act(y),
            _dyn_act(torch.empty((2 * num_heads * cache_row_stride, head_dim), device=device, dtype=torch.bfloat16)),
            _dyn(torch.empty((cache_row_stride + M,), device=device, dtype=torch.float32)),
            _dyn(keyframe_times.detach().to(device=device, dtype=torch.float32).reshape(-1)),
            _dyn(i32(slots)),
            _dyn(i32(planes)),
            _dyn(i32(counts)),
            _dyn(i32(video)),
            _dyn(i32((video >= 0).sum(dim=-1))),
            _dyn(i32(owner)),
        ),
        y=y,
    )


def run_block_fna_dsl(
    x: torch.Tensor,
    stage4: torch.Tensor,
    *,
    norm1_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    softmax_bound: torch.Tensor,
    scale_msa: torch.Tensor,
    shift_msa: torch.Tensor,
    scale_mlp: torch.Tensor,
    shift_mlp: torch.Tensor,
    w_ctx: torch.Tensor,
    b_ctx: torch.Tensor,
    upsample_stride: "tuple[int, int, int]" = (1, 1, 1),
    drop_leading_frame: bool = False,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    b_q: torch.Tensor,
    b_k: torch.Tensor,
    b_v: torch.Tensor,
    b_o: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    inv_t: torch.Tensor,
    inv_h: torch.Tensor,
    inv_w: torch.Tensor,
    num_heads: int,
    attn_scale: float,
    rope_dim_split: "tuple[int, int, int]",
    kernel_size: "tuple[int, int, int]",
    tile_thw: "tuple[int, int, int] | None" = None,
    out: torch.Tensor | None = None,
    x_keyframes: torch.Tensor | None = None,
    stage4_keyframes: torch.Tensor | None = None,
    keyframe_times: torch.Tensor | None = None,
    keyframe_times_host: torch.Tensor | None = None,
    out_keyframes: torch.Tensor | None = None,
) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
    """Run one fused ``DiffusionNABlock``. ``x`` is ``(1, T, H, W, C)``.
    ``stage4`` is the context *before* its pixel-shuffle upsample,
    ``(1, T/p_t, H/p_h, W/p_w, Ctx)``, and ``w_ctx`` / ``b_ctx`` are that upsample fused
    with ``context_proj`` by :func:`pack_fused_ctx_weights` -- ``(P*C, Ctx)`` and
    ``(P*C,)``, one Linear per subpixel. The kernel projects each token straight off its
    stage-4 parent row, so the full-resolution context is never materialised. Pass
    ``upsample_stride=(1,1,1)`` with a plain ``(C, C)`` / ``(C,)`` ``context_proj`` to
    feed a context that is already at full resolution.
    ``drop_leading_frame`` must match the flag the caller would have passed to
    ``LinearPixelShuffleUpsample``: with ``p_t == 2`` it drops the shuffle's duplicate
    leading frame, so ``T == T_lo * 2 - 1`` and the temporal axis is offset by one.
    ``attn_scale`` is the caller's own attention scale; it is folded into the packed
    ``q_norm`` constants rather than applied to Q per element. ``softmax_bound`` is
    :func:`ltx_kernels.vae.softmax_row_bound` of ``k_norm_weight``, hoisted so it is
    derived once per weight rather than once per call.
    ``out`` supplies the output buffer instead of allocating one, so a chain of blocks
    can alternate between two buffers rather than allocating a whole volume per block.
    It must be ``(T*H*W + OUT_SLACK_ROWS, C)`` contiguous bf16: the kernel stores
    out-of-range lanes of partial tiles past the volume. It must not alias ``x`` --
    the kernel reads neighborhoods of ``x`` while writing, so in-place is not safe.
    ``out_keyframes`` is the same thing for the anchor stream, sized
    ``(n_kf*H*W + OUT_SLACK_ROWS, C)``; without it a joint launch allocates that stream's
    output itself and the chain's alternation stops one stream short.
    ``keyframe_times_host`` is a host copy of ``keyframe_times``. The anchor stream's plane
    tables are built on the host; with the copy that needs no read of the device.
    """
    if not block_fna_available(x.device.index or 0):
        raise RuntimeError(UNSUPPORTED_MESSAGE)
    if x.shape[0] != 1 or stage4.shape[0] != 1:
        raise ValueError("B=1 only")
    x_ = x[0].detach().to(dtype=torch.bfloat16)
    ctx_ = stage4[0].detach().to(dtype=torch.bfloat16)
    T, H, W, C = map(int, x_.shape)
    T_lo, H_lo, W_lo, Ctx = map(int, ctx_.shape)
    Hidd = int(w_gate.shape[0])
    tile = tile_thw or default_tile_thw(kernel_size)
    subpixels = upsample_stride[0] * upsample_stride[1] * upsample_stride[2]
    drop = 1 if (drop_leading_frame and upsample_stride[0] == 2) else 0
    if (T + drop, H, W) != tuple(lo * p for lo, p in zip((T_lo, H_lo, W_lo), upsample_stride, strict=True)):
        raise ValueError(
            f"stage4 {(T_lo, H_lo, W_lo)} at stride {upsample_stride} does not shuffle out to "
            f"x {(T, H, W)} (drop_leading_frame={drop_leading_frame})"
        )
    if tuple(w_ctx.shape) != (subpixels * C, Ctx) or int(b_ctx.numel()) != subpixels * C:
        raise ValueError(
            f"fused context weights must be {(subpixels * C, Ctx)} / {(subpixels * C,)} -- see "
            f"pack_fused_ctx_weights -- got {tuple(w_ctx.shape)} / {tuple(b_ctx.shape)}"
        )
    if not fna_supported(
        C=C,
        Ctx=Ctx,
        Hidd=Hidd,
        num_heads=num_heads,
        kernel_size=kernel_size,
        T=T,
        H=H,
        W=W,
        tile_thw=tile,
        upsample_stride=upsample_stride,
    ):
        raise ValueError(f"fused NA block unsupported: C={C} Ctx={Ctx} Hidd={Hidd} shape={(T, H, W)}")

    grid = tuple(-(-v // t) for v, t in zip((T, H, W), tile, strict=True))
    n_ctas = cta_count(grid[0] * grid[1] * grid[2], x_.device)
    plan = _choose_slab(grid, tile, kernel_size, (T, H, W), n_ctas)

    # M extra rows absorb the stores of out-of-range lanes in partial tiles -- which is
    # why a caller-supplied ``out`` has to carry the same slack.
    y_rows = T * H * W + OUT_SLACK_ROWS
    # Under an active ltx_kernels.vae.workspace the volume is leased per tile.
    y_buf = (
        workspace_volume((y_rows, C), device=x_.device, dtype=torch.bfloat16)
        if out is None
        else _check_out(out, y_rows, C, x_)
    )
    hd = C // num_heads
    # One cache buffer, K's heads then V's. The per-head stride is the slab box this
    # call actually chose, not ``MAXP_TOKENS``, so the workspace scales with the volume
    # rather than with the compile-time bound.
    kv_cache = torch.empty((2 * num_heads * plan.cache_row_stride, hd), device=x_.device, dtype=torch.bfloat16)
    # The cache fill's RMSNorm factor per box token; M extra entries absorb
    # out-of-range lanes.
    norm_cache = torch.empty((plan.cache_row_stride + M,), device=x_.device, dtype=torch.float32)
    barrier = torch.zeros(2, device=x_.device, dtype=torch.int32)

    kf = _keyframe_workspace(
        x_keyframes,
        stage4_keyframes,
        keyframe_times,
        T,
        H,
        W,
        C,
        Ctx,
        num_heads,
        hd,
        tile,
        plan,
        x_.device,
        out_keyframes,
        keyframe_times_host,
    )

    def f32(t: torch.Tensor):
        return _dyn(t.detach().contiguous().float().to(device=x_.device).view(-1))

    def bf(t: torch.Tensor):
        return _dyn_w(t.detach().contiguous().to(device=x_.device, dtype=torch.bfloat16))

    channel_params = _pack_channel_params(
        norm1_weight,
        norm2_weight,
        q_norm_weight,
        k_norm_weight,
        scale_msa,
        shift_msa,
        scale_mlp,
        shift_mlp,
        b_ctx,
        b_o,
        b_q,
        b_k,
        b_v,
        softmax_bound,
        C,
        hd,
        subpixels,
        attn_scale,
        x_.device,
    )

    with torch.cuda.device(x_.device):
        # ``keyframes`` keys the binary so the no-keyframe kernel keeps exactly the trace, and
        # the register allocation, it had before the keyframe stream existed.
        # Everything else here is structural too, and must stay that way. The per-tile plane
        # union bound used to be in this key; because it moves with the clip length it recompiled
        # the ladder on every new duration, which cost 12+ minutes of in-process MLIR before a
        # 545-frame decode could start. It is now ``SLOTS * TT``, a constant of ``tile_thw``.
        compiled = KERNEL_CACHE.get(
            _compile_kernel,
            x_.device,
            C=C,
            Ctx=Ctx,
            upsample_stride=tuple(upsample_stride),
            Hidd=Hidd,
            num_heads=num_heads,
            rope_dim_split=rope_dim_split,
            kernel_size=kernel_size,
            tile_thw=tile,
            slab_box_max=SLAB_BOX_MAX,
            keyframes=kf.enabled,
        )
        compiled(
            _dyn_act(_as_rows(x_)),
            _dyn_act(_as_rows(ctx_)),
            _dyn_act(y_buf),
            _dyn_act(kv_cache),
            _dyn(norm_cache),
            _dyn(barrier),
            *kf.operands,
            bf(w_ctx),
            bf(w_q),
            bf(_pack_kv(w_k, w_v, num_heads)),
            bf(w_o),
            bf(w_gate),
            bf(w_up),
            bf(w_down),
            _dyn(channel_params),
            f32(inv_t),
            f32(inv_h),
            f32(inv_w),
            *_runtime_scalars(
                volume=(T, H, W),
                ctx_volume=(T_lo, H_lo, W_lo),
                drop_leading=drop,
                n_ctas=n_ctas,
                plan=plan,
                n_kf=kf.n_kf,
                kf_cache_row_stride=kf.cache_row_stride,
            ),
        )
    video_out = y_buf[: T * H * W].view(T, H, W, C).unsqueeze(0).to(dtype=x.dtype)
    if not kf.enabled:
        return video_out
    # The arity follows the call, so an empty stream still returns a (possibly empty) pair.
    return video_out, kf.y[: kf.n_kf * H * W].view(kf.n_kf, H, W, C).unsqueeze(0).to(dtype=x.dtype)


def block_fna_available(device_index: int = 0) -> bool:
    """Whether this GPU can run the fused block at all (datacenter Blackwell)."""
    return gpu_supports_dsl_kernels(device_index)


def fna_supported(
    *,
    C: int,
    Ctx: int,
    Hidd: int,
    num_heads: int,
    kernel_size: "tuple[int, int, int]",
    T: int,
    H: int,
    W: int,
    tile_thw: "tuple[int, int, int] | None" = None,
    upsample_stride: "tuple[int, int, int]" = (1, 1, 1),
) -> bool:
    """Whether this block shape and volume are served by the fused kernel."""
    if C not in SUPPORTED_CHANNELS or C % TILE_N != 0 or Hidd % TILE_N != 0:
        return False
    # The fused-context reduction splits a stage-4 row across ``s_a`` and ``s_q``, one
    # C-wide group each, so stage-4 may be one or two C-wide groups and no more.
    if Ctx not in (C, 2 * C):
        return False
    if num_heads * HD != C:
        return False
    tt, th, tw = tile_thw or default_tile_thw(kernel_size)
    # A slab box is at least one query tile, and the fill walks it one stage-4 cell per
    # lane -- so a tile has to hold at least one whole cell on every axis.
    if any(t % p for t, p in zip((tt, th, tw), upsample_stride, strict=True)):
        return False
    kt, kh, kw = kernel_size
    return tt + kt - 1 <= T and th + kh - 1 <= H and tw + kw - 1 <= W


# --------------------------------------------------------------------------
# Launch and compile
# --------------------------------------------------------------------------


@cute.jit
def _launch(
    x,
    stage4,
    y,
    kv_cache,
    norm_cache,
    barrier,
    x_kf,
    stage4_kf,
    y_kf,
    kv_cache_kf,
    norm_cache_kf,
    kf_times,
    kf_slots,
    kf_planes,
    kf_counts,
    kf_video,
    kf_video_counts,
    kf_slab,
    w_ctx,
    w_q,
    w_kv,
    w_o,
    w_gate,
    w_up,
    w_down,
    channel_params,
    inv_t,
    inv_h,
    inv_w,
    T: cutlass.Int32,
    H: cutlass.Int32,
    W: cutlass.Int32,
    T_lo: cutlass.Int32,
    H_lo: cutlass.Int32,
    W_lo: cutlass.Int32,
    drop_leading: cutlass.Int32,
    n_ctas: cutlass.Int32,
    slab_count: cutlass.Int32,
    slab_grid_hw: cutlass.Int32,
    slab_grid_w: cutlass.Int32,
    slab_tiles_t: cutlass.Int32,
    slab_tiles_h: cutlass.Int32,
    slab_tiles_w: cutlass.Int32,
    max_box_t: cutlass.Int32,
    max_box_h: cutlass.Int32,
    max_box_w: cutlass.Int32,
    cache_row_stride: cutlass.Int32,
    n_kf: cutlass.Int32,
    kf_cache_row_stride: cutlass.Int32,
    C: cutlass.Constexpr,
    Ctx: cutlass.Constexpr,
    SP_T: cutlass.Constexpr,
    SP_H: cutlass.Constexpr,
    SP_W: cutlass.Constexpr,
    Hidd: cutlass.Constexpr,
    NH: cutlass.Constexpr,
    ROPE_T: cutlass.Constexpr,
    ROPE_H: cutlass.Constexpr,
    ROPE_W: cutlass.Constexpr,
    KT: cutlass.Constexpr,
    KH: cutlass.Constexpr,
    KW: cutlass.Constexpr,
    TT: cutlass.Constexpr,
    TH: cutlass.Constexpr,
    TW: cutlass.Constexpr,
    PT: cutlass.Constexpr,
    PH: cutlass.Constexpr,
    PW: cutlass.Constexpr,
    NKV: cutlass.Constexpr,
    BT: cutlass.Constexpr,
    BH: cutlass.Constexpr,
    BW: cutlass.Constexpr,
    KF: cutlass.Constexpr,
    SLOTS: cutlass.Constexpr,
    P_MAX: cutlass.Constexpr,
    NKV_KF: cutlass.Constexpr,
):
    mma128 = cute.make_tiled_mma(_mma_op("n128"))
    mma64 = cute.make_tiled_mma(_mma_op("pv"))
    # sA spans the full C-wide activation (2 K-stages per M channels); every weight
    # and probability operand is one 128-wide slice, so SMEM is flat in C.
    a_layout = sm100_utils.make_smem_layout_a(mma128, TILER_N128, IO, C // TILE_K)
    b_layout = sm100_utils.make_smem_layout_b(mma128, TILER_N128, IO, 2)
    # K and V are double-buffered inside the attention pipeline: 2 buffers of one
    # 128-key K stage and of two 64-key V stages. P stays single-buffered -- the step
    # waits for the previous P@V before overwriting it, and that wait is needed anyway
    # to free the V buffer.
    p_layout = sm100_utils.make_smem_layout_a(mma64, TILER_N64, IO, 2)
    q_layout = sm100_utils.make_smem_layout_a(mma128, TILER_N128, IO, NH)
    k_layout = sm100_utils.make_smem_layout_b(mma128, TILER_N128, IO, 2)
    v_layout = sm100_utils.make_smem_layout_b(mma64, TILER_N64, IO, 4)
    # The MLP's weights stream through a deeper queue over s_q, which is dead after attention,
    # and its hidden chunk is staged in the freed two-stage weight slot instead.
    b4_layout = sm100_utils.make_smem_layout_b(mma128, TILER_N128, IO, MLP_B_STAGES)
    h_layout = sm100_utils.make_smem_layout_a(mma128, TILER_N128, IO, TILE_N // TILE_K)

    # One TMA atom per weight; ``local_tile`` inside ``gemm_tma`` picks the N-tile and
    # K-tile, so one atom serves every call site for that weight.
    tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    b1 = cute.select(b_layout, mode=[0, 1, 2])

    def _atom128(w):
        return cute.nvgpu.make_tiled_tma_atom_B(tma_op, w, b1, TILER_N128, mma128)

    a_ctx, m_ctx = _atom128(w_ctx)
    a_o, m_o = _atom128(w_o)
    a_gate, m_gate = _atom128(w_gate)
    a_up, m_up = _atom128(w_up)
    a_down, m_down = _atom128(w_down)
    a_q, m_q = _atom128(w_q)
    a_kv, m_kv = _atom128(w_kv)

    _fna_kernel(
        x,
        stage4,
        y,
        kv_cache,
        norm_cache,
        barrier,
        x_kf,
        stage4_kf,
        y_kf,
        kv_cache_kf,
        norm_cache_kf,
        kf_times,
        kf_slots,
        kf_planes,
        kf_counts,
        kf_video,
        kf_video_counts,
        kf_slab,
        a_ctx,
        m_ctx,
        a_q,
        m_q,
        a_kv,
        m_kv,
        a_o,
        m_o,
        a_gate,
        m_gate,
        a_up,
        m_up,
        a_down,
        m_down,
        channel_params,
        inv_t,
        inv_h,
        inv_w,
        T,
        H,
        W,
        T_lo,
        H_lo,
        W_lo,
        drop_leading,
        n_ctas,
        slab_count,
        slab_grid_hw,
        slab_grid_w,
        slab_tiles_t,
        slab_tiles_h,
        slab_tiles_w,
        max_box_t,
        max_box_h,
        max_box_w,
        cache_row_stride,
        n_kf,
        kf_cache_row_stride,
        C,
        Ctx,
        SP_T,
        SP_H,
        SP_W,
        Hidd,
        NH,
        ROPE_T,
        ROPE_H,
        ROPE_W,
        KT,
        KH,
        KW,
        TT,
        TH,
        TW,
        PT,
        PH,
        PW,
        NKV,
        BT,
        BH,
        BW,
        KF,
        SLOTS,
        P_MAX,
        NKV_KF,
        mma128,
        mma64,
        a_layout,
        b_layout,
        p_layout,
        q_layout,
        k_layout,
        v_layout,
        b4_layout,
        h_layout,
    ).launch(grid=[n_ctas, 1, 1], block=[THREADS, 1, 1])


def _compile_kernel(
    *,
    C: int,
    Hidd: int,
    num_heads: int,
    rope_dim_split: "tuple[int, int, int]",
    kernel_size: "tuple[int, int, int]",
    tile_thw: "tuple[int, int, int]",
    Ctx: int | None = None,
    upsample_stride: "tuple[int, int, int]" = (1, 1, 1),
    slab_box_max: "tuple[int, int, int]" = SLAB_BOX_MAX,
    keyframes: bool = False,
    options: str | None = None,
):
    """Compile one structural key from storage-free ABI descriptors."""
    Ctx = C if Ctx is None else Ctx
    kt, kh, kw = kernel_size
    tt, th, tw = tile_thw
    panel = (tt + kt - 1, th + kh - 1, tw + kw - 1)
    # The per-KV-tile window mask packs each key's panel-local (dt, dh, dw) into 5-bit
    # fields of one int32, so the panel may not exceed 32 along any axis.
    if max(panel) > 32:
        raise ValueError(f"halo panel {panel} exceeds the packed-offset limit of 32")
    # Halo-box bounds are compile-time, so the rope tables are fixed-size; the box's
    # actual extents come from ``_choose_slab`` at runtime, which is what keeps T/H/W
    # dynamic. A one-tile slab is the floor, so the panel itself has to fit.
    if any(p > b for p, b in zip(panel, slab_box_max, strict=True)) or panel[0] * panel[1] * panel[2] > MAXP_TOKENS:
        raise ValueError(f"halo panel {panel} exceeds the slab box bound {slab_box_max}")

    def activation():
        return fake_dynamic_matrix(cutlass.BFloat16, assumed_align=16)

    def weight(columns: int):
        return fake_dynamic_weight(cutlass.BFloat16, columns=columns)

    def f32_vector():
        return fake_dynamic_vector(cutlass.Float32)

    def i32_vector():
        return fake_dynamic_vector(cutlass.Int32)

    # Shapes here only have to be *typed* right: every extent the kernel reads is a
    # runtime argument, so the stand-ins can be as small as their layouts allow.
    trace_plan = _SlabPlan(core=(1, 1, 1), box=(1, 1, 1), grid=(1, 1, 1))

    return cute.compile(
        _launch,
        activation(),  # x
        activation(),  # stage4
        activation(),  # y
        activation(),  # kv_cache -- its head stride is a runtime argument
        f32_vector(),  # norm_cache
        i32_vector(),  # barrier
        activation(),  # x_kf
        activation(),  # stage4_kf
        activation(),  # y_kf
        activation(),  # kv_cache_kf
        f32_vector(),  # norm_cache_kf
        f32_vector(),  # kf_times
        *(i32_vector() for _ in range(6)),
        weight(Ctx),  # w_ctx -- the fused upsample x context_proj
        weight(C),  # w_q
        weight(C),  # w_kv
        weight(C),  # w_o
        weight(C),  # w_gate
        weight(C),  # w_up
        weight(Hidd),  # w_down
        f32_vector(),  # channel_params
        f32_vector(),  # inv_t
        f32_vector(),  # inv_h
        f32_vector(),  # inv_w
        *_runtime_scalars(volume=(4, 4, 4), ctx_volume=(4, 4, 4), drop_leading=0, n_ctas=1, plan=trace_plan),
        *_static_shape(
            C=C,
            Ctx=Ctx,
            upsample_stride=upsample_stride,
            Hidd=Hidd,
            num_heads=num_heads,
            rope_dim_split=rope_dim_split,
            kernel_size=kernel_size,
            tile_thw=tile_thw,
            slab_box_max=slab_box_max,
            keyframes=keyframes,
        ),
        **({"options": options} if options else {}),
    )


def _runtime_scalars(
    *,
    volume: "tuple[int, int, int]",
    ctx_volume: "tuple[int, int, int]",
    drop_leading: int,
    n_ctas: int,
    plan: _SlabPlan,
    n_kf: int = 0,
    kf_cache_row_stride: int = 1,
) -> tuple:
    """The kernel's runtime scalar arguments, in signature order.
    Both :func:`_compile`'s trace call and the real call site build their argument
    list from here, so the order is written once.
    """
    return (
        *volume,
        *ctx_volume,
        drop_leading,
        n_ctas,
        plan.count,
        plan.grid[1] * plan.grid[2],
        plan.grid[2],
        *plan.core,
        *plan.box,
        plan.cache_row_stride,
        n_kf,
        kf_cache_row_stride,
    )


def _static_shape(
    *,
    C: int,
    Ctx: int,
    upsample_stride: "tuple[int, int, int]",
    Hidd: int,
    num_heads: int,
    rope_dim_split: "tuple[int, int, int]",
    kernel_size: "tuple[int, int, int]",
    tile_thw: "tuple[int, int, int]",
    slab_box_max: "tuple[int, int, int]",
    keyframes: bool = False,
) -> tuple:
    """The kernel's compile-time arguments, in signature order."""
    kt, kh, kw = kernel_size
    tt, th, tw = tile_thw
    panel = (tt + kt - 1, th + kh - 1, tw + kw - 1)
    n_kv = (panel[0] * panel[1] * panel[2] + TILE_N - 1) // TILE_N
    # Keyframe panels drop the temporal axis: the video panel's h/w rectangle, one plane deep.
    n_kv_kf = (panel[1] * panel[2] + TILE_N - 1) // TILE_N
    return (
        C,
        Ctx,
        *upsample_stride,
        Hidd,
        num_heads,
        *rope_dim_split,
        kt,
        kh,
        kw,
        tt,
        th,
        tw,
        *panel,
        n_kv,
        *slab_box_max,
        keyframes,
        KEYFRAME_CONTEXT_SLOTS,
        # The stride into ``kf_planes``: a tile's ``TT`` timesteps contribute at most
        # ``SLOTS`` planes each, so this bounds the union without consulting the layout.
        KEYFRAME_CONTEXT_SLOTS * tt,
        n_kv_kf,
    )


def _pack_kv(w_k: torch.Tensor, w_v: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Interleave the K and V projections head by head into one ``(2C, C)`` weight.
    N-tile ``h`` of the result is ``[k_head_h | v_head_h]``, so a single N=128 GEMM
    yields both of a head's operands from one pass over the activation -- half the
    weight-streaming GEMM calls of the two N=64 projections it replaces.
    """
    c = w_k.shape[1]
    kv = torch.stack([w_k.view(num_heads, -1, c), w_v.view(num_heads, -1, c)], dim=1)
    return kv.reshape(2 * w_k.shape[0], c)


# --------------------------------------------------------------------------
# Choosing the slab
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=256)
def _choose_slab(
    grid: "tuple[int, int, int]",
    tile: "tuple[int, int, int]",
    kernel: "tuple[int, int, int]",
    vol: "tuple[int, int, int]",
    n_ctas: int,
) -> _SlabPlan:
    """Pick the slab extent that minimises total time for this volume and grid.
    The tile loop costs ``sum_slabs ceil(tiles_in_slab / n_ctas)`` waves -- every CTA a
    slab cannot fill sits at the grid barrier for a whole tile -- and the cache fill
    re-projects the halo box once per slab. Both are exact counts, so this is a search
    rather than a heuristic; the only fitted number is their relative weight, and a
    cache-fill M-tile wave measures ~3% of a query-tile wave.
    """
    fill_weight = 0.03
    best, best_cost = None, float("inf")
    for st in range(1, grid[0] + 1):
        box_t = min(vol[0], st * tile[0] + kernel[0] - 1)
        if box_t > SLAB_BOX_MAX[0]:
            continue
        for sh in range(1, grid[1] + 1):
            box_h = min(vol[1], sh * tile[1] + kernel[1] - 1)
            if box_h > SLAB_BOX_MAX[1]:
                continue
            for sw in range(1, grid[2] + 1):
                box_w = min(vol[2], sw * tile[2] + kernel[2] - 1)
                if box_w > SLAB_BOX_MAX[2] or box_t * box_h * box_w > MAXP_TOKENS:
                    continue
                waves = 0
                for a, na in _axis_slabs(grid[0], st):
                    for b, nb in _axis_slabs(grid[1], sh):
                        for c, nc in _axis_slabs(grid[2], sw):
                            waves += na * nb * nc * -(-(a * b * c) // n_ctas)
                fill_waves = 0
                for xa in _axis_boxes(grid[0], st, tile[0], kernel[0], vol[0]):
                    for xb in _axis_boxes(grid[1], sh, tile[1], kernel[1], vol[1]):
                        for xc in _axis_boxes(grid[2], sw, tile[2], kernel[2], vol[2]):
                            fill_waves += -(-(xa * xb * xc) // (M * n_ctas))
                cost = waves + fill_weight * fill_waves
                if cost < best_cost:
                    best_cost, best = cost, ((st, sh, sw), (box_t, box_h, box_w))
    assert best is not None, f"no slab fits MAXP={MAXP_TOKENS} for grid={grid} tile={tile}"
    (st, sh, sw), box = best
    return _SlabPlan(
        core=(st, sh, sw),
        box=box,
        grid=(-(-grid[0] // st), -(-grid[1] // sh), -(-grid[2] // sw)),
    )


def _axis_slabs(grid: int, s: int) -> "tuple[tuple[int, int], ...]":
    """``(tiles_in_slab, how_many_slabs)`` when ``grid`` tiles are cut into runs of ``s``."""
    full, rem = divmod(grid, s)
    out = ((s, full),) if full else ()
    return out + (((rem, 1),) if rem else ())


def _axis_boxes(grid: int, s: int, tile: int, k: int, vol: int) -> "list[int]":
    """Halo extent of each slab along one axis.
    The kernel grows each slab's *own* core by the NA radius rather than using the
    largest slab's box, so a slab cut short by the volume edge projects less. The
    cache fill's cost is the sum of these, which is what the search below minimises.
    """
    out, pos = [], 0
    for i in range(-(-grid // s)):
        ext = min(min(s, grid - i * s) * tile, vol - pos)
        out.append(min(vol, max(ext, tile) + k - 1))
        pos += s * tile
    return out
