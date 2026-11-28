"""Standalone 3D neighborhood attention -- a drop-in for ``natten.na3d``.
Attention only: Q/K/V arrive already projected, RMS-normed and RoPE'd, so nothing
here scales with the channel count. It runs ``HG`` heads at a time and loops over
head groups, which keeps TMEM and SMEM independent of the head count while still
reading each (KV tile, head) row exactly once. Used for the DiffVAE's deterministic
stages, whose 32/16/8/8 heads the fused stage-5 kernel's TMEM map cannot hold.
The window mask, the two-deep pipeline and the softmax are not reimplemented here --
they are :func:`ltx_kernels.vae.fna_attn_core.attention_tile`, the same trace the
fused kernel runs.
Q must arrive carrying the attention scale, exactly as ``natten.na3d`` expects it.
The one extra thing this kernel needs is the ``k_norm`` weight, from which it builds a
fixed softmax offset in place of an online row max -- see
:mod:`ltx_kernels.vae.softmax_bound`, which is worth reading before calling this.
Compile keys are ``(kernel_size, tile_thw)`` per GPU architecture -- head count,
T, H and W are all runtime -- so the four deterministic stages share two binaries.
"""

# NOTE: no `from __future__ import annotations` -- cute.struct needs real types.

import functools
from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.utils import SmemAllocator

from ltx_kernels.devices import upload_from_host
from ltx_kernels.vae.aot import KERNEL_CACHE, fake_dynamic_matrix, fake_dynamic_vector
from ltx_kernels.vae.availability import UNSUPPORTED_MESSAGE, gpu_supports_dsl_kernels
from ltx_kernels.vae.fna_attn_core import (
    _as_rows,
    _dyn,
    _dyn_act,
    attention_tile,
    attn_epilogue,
    attn_zero_accumulators,
    cta_count,
    default_tile_thw,
)
from ltx_kernels.vae.fna_gemm import UmmaSharedStorage, _mma_op, _reg_shape, acquire_tmem, release_tmem
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
    _fill_a_gather,
    _read_a_chunk,
    _scatter_a,
    cooperative_copy_slot,
    select_tensor,
)
from ltx_kernels.vae.fna_types import (
    ACC,
    HD,
    HEAD_COLS,
    IO,
    LOG2E,
    NUM_WARPGROUPS,
    THREADS,
    TILE_K,
    TILE_N,
    TILER_N64,
    TILER_N128,
    M,
)
from ltx_kernels.vae.keyframe_slots import KEYFRAME_CONTEXT_SLOTS, nearest_video_frames, tile_plane_union
from ltx_kernels.vae.softmax_bound import softmax_row_bound

# Heads per group. ``HG * HD`` TMEM columns for the attention accumulators plus 256
# for the two Q@K accumulators must fit in 512, so 4 is the ceiling; it also has to
# be even (the pipeline's buffer parity is ``nh % 2``) and to divide ``NUM_WARPGROUPS`` evenly
# for the ``||q||`` pass below. Every DiffVAE stage's head count is a multiple of 4,
# so one value serves all of them and the head count stays a runtime argument.
HG = 4


@cute.jit
def _stage_queries(sQ, src, rowix, copy_slot, m_sm, areg, k_bound_t, lane, wg, c_off, HPG: cutlass.Constexpr):
    """Gather this head group's queries into ``sQ`` and fill their fixed softmax offsets.
    Shared by the video and keyframe query tiles: both read an already normed, scaled and
    RoPE'd query row, so the only difference is which tensor it comes from. Ends on the
    barrier inside ``attn_zero_accumulators``, which is what publishes ``m_sm``.
    """
    _fill_a_gather(sQ, src, rowix, copy_slot, HG * HD, c_off)
    cute.arch.sync_threads()
    # ``||q||`` per (row, head) -- the fixed softmax offset. Each lane owns its own row, so
    # a head's channels are register-local and this needs no reduction at all; the
    # warpgroups split the heads.
    for h_loc in cutlass.range_constexpr(HPG):
        for g in cutlass.range_constexpr(NUM_WARPGROUPS):
            if wg == g:
                nh = g * HPG + h_loc
                _read_a_chunk(sQ, lane, areg, nh * HD, HD)
                s = ACC(0.0)
                for d in cutlass.range(HD, unroll_full=True):
                    s += areg[d] * areg[d]
                # Read at the point of use rather than hoisted: this runs once per head
                # group, and a hoisted value would hold a register live across the KV loop.
                m_sm[lane, nh] = cute.sqrt(s) * k_bound_t[0] * LOG2E


@cute.kernel
def _na_kernel(
    q: cute.Tensor,  # (T*H*W, NH*HD) bf16, RMS-normed, scaled and RoPE'd
    k: cute.Tensor,  # (T*H*W*NH, HD) bf16, RMS-normed and RoPE'd
    v: cute.Tensor,  # (T*H*W*NH, HD) bf16
    y: cute.Tensor,  # (T*H*W + M, NH*HD) bf16
    k_bound_t: cute.Tensor,  # (1,) f32, see ltx_kernels.vae.softmax_bound
    k_kf: cute.Tensor,  # (n_kf*H*W*NH, HD) bf16, same norm as k, float-t RoPE
    v_kf: cute.Tensor,  # (n_kf*H*W*NH, HD) bf16
    kf_slots: cute.Tensor,  # (T*SLOTS,) i32 -- plane per (t, slot), -1 unused
    kf_planes: cute.Tensor,  # (n_t_tiles*P_MAX,) i32 -- per-tile union, -1 padded
    kf_counts: cute.Tensor,  # (n_t_tiles,) i32 -- how much of each union row is real
    q_kf: cute.Tensor,  # (n_kf*H*W, NH*HD) bf16, normed/scaled/float-t RoPE'd
    y_kf: cute.Tensor,  # (n_kf*H*W + M, NH*HD) bf16, the keyframe stream's output
    kf_video: cute.Tensor,  # (n_kf*SLOTS,) i32 -- nearest video frames per plane
    kf_video_counts: cute.Tensor,  # (n_kf,) i32
    T,
    H,
    W,
    NH,
    n_hg,
    n_tiles,
    n_ctas,
    grid_hw,
    grid_w_,
    n_kf,
    n_kfq_tiles,
    kt: cutlass.Constexpr,
    kh: cutlass.Constexpr,
    kw: cutlass.Constexpr,
    TT: cutlass.Constexpr,
    TH: cutlass.Constexpr,
    TW: cutlass.Constexpr,
    PT: cutlass.Constexpr,
    PH: cutlass.Constexpr,
    PW: cutlass.Constexpr,
    NKV: cutlass.Constexpr,
    KF: cutlass.Constexpr,  # keyframe tail compiled in at all
    SLOTS: cutlass.Constexpr,
    P_MAX: cutlass.Constexpr,  # cap on a tile's plane union
    NKV_KF: cutlass.Constexpr,  # KV tiles spanning one keyframe panel, ceil(PH*PW / M)
    KFQ: cutlass.Constexpr,  # keyframe *query* tiles compiled in
    mma128: cute.TiledMma,
    mma64: cute.TiledMma,
    p_layout: cute.ComposedLayout,
    q_layout: cute.ComposedLayout,
    k_layout: cute.ComposedLayout,
    v_layout: cute.ComposedLayout,
    o_layout: cute.ComposedLayout,
):
    tidx, _, _ = cute.arch.thread_idx()
    cta_index, _, _ = cute.arch.block_idx()
    lane = tidx % M
    wg = tidx // M
    warp_i = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    volume = (T, H, W)
    radius = ((kt - 1) // 2, (kh - 1) // 2, (kw - 1) // 2)
    panel_shape = (PT, PH, PW)
    kernel_shape = (kt, kh, kw)
    HPG = HG // NUM_WARPGROUPS  # heads whose ||q|| one warpgroup computes

    smem = SmemAllocator()
    storage = smem.allocate(UmmaSharedStorage)
    # One 96 KiB pool: ``sO`` (the epilogue's staging buffer) is dead for the whole
    # KV loop and ``sP``/``sK``/``sV`` are dead outside it, so they alias.
    pool_elems = 6 * M * TILE_K
    pool = smem.allocate_tensor(IO, cute.make_layout((pool_elems,)), byte_alignment=128)
    sO = _alias(pool, o_layout)
    sP = _alias(pool, p_layout)
    # This warpgroup's chunk of ``sP`` at stage 0, for ``attention_tile``'s P fill.
    sP_own = _alias(pool, p_layout, wg * (M * TILE_K))
    sK = _alias(pool, k_layout, 2 * M * TILE_K)
    sV = _alias(pool, v_layout, 4 * M * TILE_K)
    sQ = smem.allocate_tensor(IO, q_layout.outer, byte_alignment=128, swizzle=q_layout.inner)
    qkbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((2,)), byte_alignment=16)
    pvbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout((1,)), byte_alignment=16)
    # ``rowix`` is always an in-bounds row and feeds the gather; ``rowix_w`` is where
    # a lane's result goes, which for an out-of-range lane of a partial tile is a
    # scratch row past the end of the output.
    rowix = smem.allocate_tensor(cutlass.Int32, cute.make_layout((M,)), byte_alignment=16)
    rowix_w = smem.allocate_tensor(cutlass.Int32, cute.make_layout((M,)), byte_alignment=16)
    m_sm = smem.allocate_tensor(ACC, cute.make_layout((M, HG)), byte_alignment=16)
    l_sm = smem.allocate_tensor(ACC, cute.make_layout((M, NUM_WARPGROUPS, HG)), byte_alignment=16)
    cute.arch.sync_threads()

    tmem, tmem_ptr = acquire_tmem(storage)

    lay_main = mma128.make_fragment_C(mma128.partition_shape_C(TILER_N128[:2])).layout
    lay_n64 = mma64.make_fragment_C(mma64.partition_shape_C(TILER_N64[:2])).layout
    # Same TMEM map as the fused kernel, with ``HG`` standing in for its head count:
    # the per-head attention accumulators low, the two Q@K accumulators above.
    scr = HG * HD
    acc_o = [cute.make_tensor(tmem_ptr + h * HD, lay_n64) for h in range(HG)]
    acc_qk = [cute.make_tensor(tmem_ptr + scr + b * TILE_N, lay_main) for b in range(2)]
    rshape = _reg_shape(mma128, acc_qk[0], lane, TILE_N)
    rshape_o = _reg_shape(mma64, acc_o[0], lane, HD, HEAD_COLS)

    frQ = mma128.make_fragment_A(sQ)
    frK = mma128.make_fragment_B(sK)
    frP = mma64.make_fragment_A(sP)
    frV = mma64.make_fragment_B(sV)

    areg = cute.make_rmem_tensor(rshape, ACC)
    oreg = cute.make_rmem_tensor(rshape_o, ACC)
    kvfrag = cute.make_rmem_tensor(cute.make_layout(HD), IO)

    copy_slot = cooperative_copy_slot(tidx)
    slot_t, slot_h, slot_w = query_slot_in_tile(lane, TH, TW)
    zero_k = cutlass.Int32(0)

    # One loop over both streams' tiles, video first, so the single-plane pass below is one
    # instantiation rather than one per stream. A keyframe query tile is a plane on the video
    # tile's spatial shape, so both grids unflatten with the same divisors and only the
    # temporal component's meaning differs: a tile index for video, a plane for keyframes.
    kf_rows = n_kf * H * W
    tile_id = cta_index
    while tile_id < n_tiles + n_kfq_tiles:
        index = tile_id
        if cutlass.const_expr(KFQ):
            kfq_tile = tile_id >= n_tiles
            video_tile = tile_id < n_tiles
            index = tile_id - n_tiles if kfq_tile else tile_id
        tile_i, tile_j, tile_k = unflatten_3d(index, grid_hw, grid_w_)
        tile_origin = (tile_i * TT, tile_j * TH, tile_k * TW)

        panel_origin = halo_panel_origin(tile_origin, radius, panel_shape, volume)
        query_pos, read_row, write_row = query_row_indices(
            (tile_origin[0] + slot_t, tile_origin[1] + slot_h, tile_origin[2] + slot_w),
            volume,
            lane,
            T * H * W,
        )
        # Two window rules, picked by whether this binary carries the keyframe tail: the
        # joint operator is defined on clamp-and-mask, every other caller on NATTEN's inward
        # shift, and ``KF`` is exactly the distinction. Keying it off the compile flag rather
        # than a runtime branch keeps the no-keyframe binary's trace byte-identical.
        if cutlass.const_expr(KF):
            bounds = window_bounds_in_panel(query_pos, panel_origin, radius, kernel_shape, volume)
            window = tuple(lo for lo, _ in bounds)
            window_hi = tuple(hi for _, hi in bounds)
        else:
            window = window_origin_in_panel(query_pos, panel_origin, radius, kernel_shape, volume)
            window_hi = None
        q_src, y_dst = q, y
        panel_h0, panel_w0 = panel_origin[1], panel_origin[2]
        # The tail's plane union is indexed by video temporal tile; a keyframe tile has no
        # such index, and reads row 0 whose value it then selects away.
        t_idx = tile_i

        if cutlass.const_expr(KFQ):
            # The keyframe tile's geometry: one plane, on this tile's spatial box. The *panel*
            # still shifts inward -- it only has to contain every key the tile can reach, and
            # the inward-shifted box is the larger of the two -- but the per-query window is
            # clamp-and-mask, like every window on the joint path.
            plane = at_most(tile_i, at_least(n_kf - 1, zero_k))
            q_h = tile_origin[1] + slot_h
            q_w = tile_origin[2] + slot_w
            r_h = q_h if q_h < H else H - 1
            r_w = q_w if q_w < W else W - 1
            kf_read = flatten_3d(plane, r_h, r_w, H, W)
            # A tile is one plane, so the borrowed tile's other temporal slots hold duplicates
            # of ``slot_t == 0``; they compute and park on a scratch row.
            kf_write = kf_read
            if slot_t != 0 or q_h >= H or q_w >= W:
                kf_write = kf_rows + lane
            kf_ph0 = clamp_window_start(tile_origin[1] - radius[1], PH, H)
            kf_pw0 = clamp_window_start(tile_origin[2] - radius[2], PW, W)
            kf_bounds = window_bounds_in_panel(
                (zero_k, r_h, r_w),
                (zero_k, kf_ph0, kf_pw0),
                (0, radius[1], radius[2]),
                (1, kh, kw),
                (cutlass.Int32(1), H, W),
            )
            read_row = kf_read if kfq_tile else read_row
            write_row = kf_write if kfq_tile else write_row
            panel_h0 = kf_ph0 if kfq_tile else panel_h0
            panel_w0 = kf_pw0 if kfq_tile else panel_w0
            window = (window[0], kf_bounds[1][0] if kfq_tile else window[1], kf_bounds[2][0] if kfq_tile else window[2])
            window_hi = (
                window_hi[0],
                kf_bounds[1][1] if kfq_tile else window_hi[1],
                kf_bounds[2][1] if kfq_tile else window_hi[2],
            )
            t_idx = zero_k if kfq_tile else t_idx
            q_src = select_tensor(kfq_tile, q_kf, q)
            y_dst = select_tensor(kfq_tile, y_kf, y)

        # ``panel_key_row`` indexes the caller's K/V directly, so the box it walks is
        # the whole volume: origin zero, extents H and W.
        panel = (*panel_origin, zero_k, zero_k, zero_k, H, W)

        cute.arch.sync_threads()
        rowix[lane] = read_row
        rowix_w[lane] = write_row
        cute.arch.sync_threads()

        hg = 0
        while hg < n_hg:
            c_off = hg * (HG * HD)
            # This group's queries: one coalesced gather straight into the swizzled
            # operand. The channel window moves with the group; the operand does not.
            _stage_queries(sQ, q_src, rowix, copy_slot, m_sm, areg, k_bound_t, lane, wg, c_off, HPG)
            attn_zero_accumulators(acc_o, l_sm, oreg, lane, wg, HG)

            # K and V are the caller's own tensors: consecutive heads are one
            # ``HD``-row apart and one panel row is ``NH`` of them.
            # A video query runs this deep ``PT``-tall panel; a keyframe query has no local
            # temporal neighborhood at all and takes only the single-plane passes below.
            deep = True
            if cutlass.const_expr(KFQ):
                deep = video_tile
            if deep:
                attention_tile(
                    (frQ, frK, frP, frV),
                    (sK, sV, sP_own),
                    acc_qk,
                    acc_o,
                    (qkbar, pvbar),
                    m_sm,
                    l_sm,
                    areg,
                    kvfrag,
                    (k, v),
                    (hg * HG, hg * HG),
                    cutlass.Int32(1),
                    panel,
                    window,
                    (tidx, lane, wg, warp_i),
                    HG,
                    NH,
                    PT,
                    PH,
                    PW,
                    kt,
                    kh,
                    kw,
                    NKV,
                    None,
                    window_hi,
                )

            # Single-plane passes, into the *same* (m_sm, l_sm, acc_o). The softmax offset is
            # fixed rather than an online max, so continuing the accumulation is already a
            # joint softmax over local and keyframe keys -- no rescale, no state merge.
            # A video query's passes are the planes in its tile's union: a plane is a per-tile
            # gather but the nearest-S set is per t, so each lane gates itself with ``sel``,
            # and an unselected lane's mask words are zero so it skips its exponentials. A
            # keyframe query's are its own plane and then the <=2 nearest video latent frames
            # -- local motion context around the anchor, deliberately not the video kernel's
            # full ``kt`` reach -- all tile-uniform, so no gating. Both are ``(1, PH, PW)``
            # panels on the query's own spatial window over the caller's whole K/V, and
            # ``attention_tile`` needs nothing else: at PT=1 its panel walk and its dt mask
            # both collapse to the single plane.
            if cutlass.const_expr(KF):
                pass_i = cutlass.Int32(0)
                n_passes = kf_counts[t_idx]
                if cutlass.const_expr(KFQ):
                    n_passes = (1 + kf_video_counts[plane]) if kfq_tile else n_passes
                while pass_i < n_passes:
                    source = kf_planes[t_idx * P_MAX + at_most(pass_i, P_MAX - 1)]
                    sel = cutlass.Int32(0)
                    for s in cutlass.range_constexpr(SLOTS):
                        if kf_slots[query_pos[0] * SLOTS + s] == source:
                            sel = cutlass.Int32(1)
                    k_src, v_src = k_kf, v_kf
                    if cutlass.const_expr(KFQ):
                        # Pass 0 is the keyframe query's own plane, the rest its nearest video
                        # frames. Selected rather than branched: a tensor bound inside a
                        # runtime ``if`` would differ in structure between the arms, which the
                        # DSL rejects, and a select is one ``selp`` either way.
                        own = pass_i == 0
                        video_kv = cutlass.Int32(0) if own else cutlass.Int32(1)
                        frame = kf_video[plane * SLOTS + at_most(at_least(pass_i - 1, zero_k), SLOTS - 1)]
                        kfq_source = plane if own else frame
                        video_kv = video_kv if kfq_tile else cutlass.Int32(0)
                        source = kfq_source if kfq_tile else source
                        sel = cutlass.Int32(1) if kfq_tile else sel
                        k_src = select_tensor(video_kv != 0, k, k_kf)
                        v_src = select_tensor(video_kv != 0, v, v_kf)
                    attention_tile(
                        (frQ, frK, frP, frV),
                        (sK, sV, sP_own),
                        acc_qk,
                        acc_o,
                        (qkbar, pvbar),
                        m_sm,
                        l_sm,
                        areg,
                        kvfrag,
                        (k_src, v_src),
                        (hg * HG, hg * HG),
                        cutlass.Int32(1),
                        (source, panel_h0, panel_w0, zero_k, zero_k, zero_k, H, W),
                        (zero_k, window[1], window[2]),
                        (tidx, lane, wg, warp_i),
                        HG,
                        NH,
                        1,
                        PH,
                        PW,
                        1,
                        kh,
                        kw,
                        NKV_KF,
                        sel,
                        # One plane deep, so the temporal bound is the single slice; the
                        # spatial pair is the query's own window, which is what keeps the two
                        # key sets aligned for the query sharing a softmax over them.
                        (cutlass.Int32(1), window_hi[1], window_hi[2]),
                    )
                    pass_i += 1

            cute.arch.sync_threads()
            attn_epilogue(acc_o, l_sm, oreg, pool, o_layout, lane, wg, HG)
            cute.arch.sync_threads()
            _scatter_a(sO, y_dst, rowix_w, copy_slot, HG * HD, c_off)
            cute.arch.sync_threads()
            hg += 1

        tile_id += n_ctas

    release_tmem(tmem, tmem_ptr)


@cute.jit
def _launch(
    q,
    k,
    v,
    y,
    k_bound_t,
    k_kf,
    v_kf,
    kf_slots,
    kf_planes,
    kf_counts,
    q_kf,
    y_kf,
    kf_video,
    kf_video_counts,
    T: cutlass.Int32,
    H: cutlass.Int32,
    W: cutlass.Int32,
    NH: cutlass.Int32,
    n_hg: cutlass.Int32,
    n_tiles: cutlass.Int32,
    n_ctas: cutlass.Int32,
    grid_hw: cutlass.Int32,
    grid_w_: cutlass.Int32,
    n_kf: cutlass.Int32,
    n_kfq_tiles: cutlass.Int32,
    kt: cutlass.Constexpr,
    kh: cutlass.Constexpr,
    kw: cutlass.Constexpr,
    TT: cutlass.Constexpr,
    TH: cutlass.Constexpr,
    TW: cutlass.Constexpr,
    PT: cutlass.Constexpr,
    PH: cutlass.Constexpr,
    PW: cutlass.Constexpr,
    NKV: cutlass.Constexpr,
    KF: cutlass.Constexpr,
    SLOTS: cutlass.Constexpr,
    P_MAX: cutlass.Constexpr,
    NKV_KF: cutlass.Constexpr,
    KFQ: cutlass.Constexpr,
):
    mma128 = cute.make_tiled_mma(_mma_op("n128"))
    mma64 = cute.make_tiled_mma(_mma_op("pv"))
    p_layout = sm100_utils.make_smem_layout_a(mma64, TILER_N64, IO, 2)
    q_layout = sm100_utils.make_smem_layout_a(mma128, TILER_N128, IO, HG)
    k_layout = sm100_utils.make_smem_layout_b(mma128, TILER_N128, IO, 2)
    v_layout = sm100_utils.make_smem_layout_b(mma64, TILER_N64, IO, 4)
    # ``sO`` is an A-operand only so the epilogue can reach it with the same
    # ``_fill_a_chunk`` the fused kernel uses; nothing multiplies by it.
    o_layout = sm100_utils.make_smem_layout_a(mma128, TILER_N128, IO, HG * HD // 64)

    _na_kernel(
        q,
        k,
        v,
        y,
        k_bound_t,
        k_kf,
        v_kf,
        kf_slots,
        kf_planes,
        kf_counts,
        q_kf,
        y_kf,
        kf_video,
        kf_video_counts,
        T,
        H,
        W,
        NH,
        n_hg,
        n_tiles,
        n_ctas,
        grid_hw,
        grid_w_,
        n_kf,
        n_kfq_tiles,
        kt,
        kh,
        kw,
        TT,
        TH,
        TW,
        PT,
        PH,
        PW,
        NKV,
        KF,
        SLOTS,
        P_MAX,
        NKV_KF,
        KFQ,
        mma128,
        mma64,
        p_layout,
        q_layout,
        k_layout,
        v_layout,
        o_layout,
    ).launch(grid=[n_ctas, 1, 1], block=[THREADS, 1, 1])


def na_attn_available(device_index: int = 0) -> bool:
    """Whether this GPU can run the NA kernel at all (datacenter Blackwell)."""
    return gpu_supports_dsl_kernels(device_index)


def na_supported(
    *,
    num_heads: int,
    head_dim: int,
    kernel_size: "tuple[int, int, int]",
    T: int,
    H: int,
    W: int,
    tile_thw: "tuple[int, int, int] | None" = None,
) -> bool:
    """Whether this shape is served by the standalone NA kernel."""
    if head_dim != HD or num_heads % HG != 0:
        return False
    tt, th, tw = tile_thw or default_tile_thw(kernel_size)
    kt, kh, kw = kernel_size
    pt, ph, pw = tt + kt - 1, th + kh - 1, tw + kw - 1
    if max(pt, ph, pw) > 32:
        return False
    return pt <= T and ph <= H and pw <= W


def _compile_kernel(
    *,
    kernel_size: "tuple[int, int, int]",
    tile_thw: "tuple[int, int, int]",
    keyframes: bool = False,
    keyframe_queries: bool = False,
    options: str | None = None,
):
    """Compile one structural key from storage-free ABI descriptors."""
    kt, kh, kw = kernel_size
    tt, th, tw = tile_thw
    pt, ph, pw = tt + kt - 1, th + kh - 1, tw + kw - 1
    # The mask packs each key's panel-local offset into 5-bit fields.
    if max(pt, ph, pw) > 32:
        raise ValueError(f"halo panel {(pt, ph, pw)} exceeds the packed-offset limit of 32")
    nkv = (pt * ph * pw + TILE_N - 1) // TILE_N
    # A keyframe panel is the same spatial box with the temporal axis collapsed. Keyframe
    # *query* tiles borrow it too, which is what keeps them on one instantiation with the
    # video tail (see the keyframe query loop in ``_na_kernel``).
    nkv_kf = (ph * pw + TILE_N - 1) // TILE_N

    def activation():
        return fake_dynamic_matrix(cutlass.BFloat16, assumed_align=16)

    def i32_vector():
        return fake_dynamic_vector(cutlass.Int32)

    return cute.compile(
        _launch,
        activation(),
        activation(),
        activation(),
        activation(),
        fake_dynamic_vector(cutlass.Float32),
        activation(),
        activation(),
        i32_vector(),
        i32_vector(),
        i32_vector(),
        activation(),
        activation(),
        i32_vector(),
        i32_vector(),
        4,
        4,
        4,
        HG,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        kt,
        kh,
        kw,
        tt,
        th,
        tw,
        pt,
        ph,
        pw,
        nkv,
        keyframes,
        KEYFRAME_CONTEXT_SLOTS,
        KEYFRAME_CONTEXT_SLOTS * tt,
        nkv_kf,
        keyframe_queries,
        **({"options": options} if options else {}),
    )


@functools.lru_cache(maxsize=8)
def _inert_operands(head_dim: int, channels: int, device: torch.device) -> "tuple":
    """Never-read stand-ins for the keyframe operands, wrapped once per device.
    The kernel takes the keyframe tensors unconditionally but reads none of them when the
    tail is compiled out, so these only have to be *typed* right -- and since they never
    change, both the allocation and the ``from_dlpack`` wrap are hoisted here. That matters:
    a deterministic-stage launch is well under a millisecond, and seven per-call allocations
    and wraps measured as a fifth of it.
    """
    kv = _dyn_act(torch.zeros((1, head_dim), device=device, dtype=torch.bfloat16))
    i32 = _dyn(torch.zeros(1, device=device, dtype=torch.int32))
    q_kf = _dyn_act(torch.zeros((M, channels), device=device, dtype=torch.bfloat16))
    y_kf = _dyn_act(torch.zeros((M, channels), device=device, dtype=torch.bfloat16))
    return kv, i32, q_kf, y_kf


class _KeyframeOperands(NamedTuple):
    """The five keyframe tensors the kernel takes, plus the two compile-key facts."""

    #: The caller supplied keyframe tensors -- so this launch is the *joint* operator, whose
    #: window rule is clamp-and-mask rather than NATTEN's inward shift. Deliberately not
    #: conditioned on the plane count: the rule is a property of the operator, and an earlier
    #: version keyed both it and the tail off "has planes", which quietly served two different
    #: window rules depending on whether a caller passed zero planes or planes nothing
    #: selected. With no planes the tail is a zero-trip loop over real, empty tables.
    enabled: bool
    k: torch.Tensor
    v: torch.Tensor
    slots: torch.Tensor
    planes: torch.Tensor
    counts: torch.Tensor


def _keyframe_operands(
    k_keyframes: torch.Tensor | None,
    v_keyframes: torch.Tensor | None,
    keyframe_slots: torch.Tensor | None,
    t: int,
    h: int,
    w: int,
    num_heads: int,
    head_dim: int,
    tile_t: int,
    device: torch.device,
) -> _KeyframeOperands:
    """Validate and lay out the keyframe operands, or hand back inert placeholders.
    Disabled is not "n_kf = 0 planes": it compiles the tail out entirely, so the
    no-keyframe kernel keeps the trace it had before this existed. The placeholder
    tensors below are never read.
    """

    if k_keyframes is None or v_keyframes is None or keyframe_slots is None:
        if not (k_keyframes is None and v_keyframes is None and keyframe_slots is None):
            raise ValueError("k_keyframes, v_keyframes and keyframe_slots must be given together or not at all")
        inert_kv, inert_i32, _, _ = _inert_operands(head_dim, 1, device)
        return _KeyframeOperands(False, inert_kv, inert_kv, inert_i32, inert_i32, inert_i32)

    if k_keyframes.shape != v_keyframes.shape:
        raise ValueError(f"keyframe k/v shapes differ: {tuple(k_keyframes.shape)} {tuple(v_keyframes.shape)}")
    if k_keyframes.ndim != 6 or k_keyframes.shape[0] != 1:
        raise ValueError(f"keyframe k/v must be (1, n_kf, H, W, NH, HD), got {tuple(k_keyframes.shape)}")
    n_kf, kf_h, kf_w, kf_nh, kf_hd = (int(s) for s in k_keyframes.shape[1:])
    if n_kf == 0:
        # Still the joint operator, so still the joint binary -- only with nothing for the
        # tail to iterate. The tables are real and correctly sized (every one of them is
        # indexed by a query tile, not by a plane) so the loop is a zero-trip, not an
        # out-of-bounds read on a placeholder.
        n_tiles_t = -(-t // tile_t)
        inert_kv, _, _, _ = _inert_operands(head_dim, 1, device)
        return _KeyframeOperands(
            enabled=True,
            k=inert_kv,
            v=inert_kv,
            slots=_dyn(torch.full((t * KEYFRAME_CONTEXT_SLOTS,), -1, dtype=torch.int32, device=device)),
            planes=_dyn(torch.full((n_tiles_t,), -1, dtype=torch.int32, device=device)),
            counts=_dyn(torch.zeros(n_tiles_t, dtype=torch.int32, device=device)),
        )
    if (kf_h, kf_w, kf_nh, kf_hd) != (h, w, num_heads, head_dim):
        raise ValueError(
            f"keyframe k/v must share the video grid and head layout: got {(kf_h, kf_w, kf_nh, kf_hd)}, "
            f"expected {(h, w, num_heads, head_dim)}"
        )
    if tuple(keyframe_slots.shape) != (t, KEYFRAME_CONTEXT_SLOTS):
        raise ValueError(f"keyframe_slots must be {(t, KEYFRAME_CONTEXT_SLOTS)}, got {tuple(keyframe_slots.shape)}")
    slots_cpu = keyframe_slots.detach().to(device="cpu", dtype=torch.int32)
    if int(slots_cpu.max()) >= n_kf:
        raise ValueError(f"keyframe_slots names a plane past n_kf={n_kf}")
    # A repeated plane in one row would be counted twice by the local pass's own logic --
    # the tail visits distinct planes, so a duplicate silently loses a slot instead.
    for row in range(t):
        live = [int(p) for p in slots_cpu[row].tolist() if p >= 0]
        if len(set(live)) != len(live):
            raise ValueError(f"keyframe_slots row {row} repeats a plane: {live}")

    # ``SLOTS * tile_t``, not ``min(n_kf, ...)``: the stride into ``planes`` is a compile-time
    # constant of the tile, so it must not move with the anchor layout. Slack columns are -1
    # and cost one int32 each; a data-dependent stride costs a recompile.
    planes, _ = tile_plane_union(slots_cpu, tile_t=tile_t, max_planes=KEYFRAME_CONTEXT_SLOTS * tile_t)
    counts = (planes >= 0).sum(dim=-1).to(torch.int32)

    def rows(x: torch.Tensor) -> torch.Tensor:
        return x[0].reshape(n_kf, h, w, num_heads * head_dim).to(dtype=torch.bfloat16).contiguous().view(-1, head_dim)

    return _KeyframeOperands(
        enabled=True,
        k=_dyn_act(rows(k_keyframes)),
        v=_dyn_act(rows(v_keyframes)),
        slots=_dyn(upload_from_host(slots_cpu.reshape(-1).contiguous(), device)),
        planes=_dyn(upload_from_host(planes.reshape(-1).contiguous(), device)),
        counts=_dyn(upload_from_host(counts, device)),
    )


def run_na_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kernel_size: "tuple[int, int, int]",
    k_norm_weight: torch.Tensor,
    tile_thw: "tuple[int, int, int] | None" = None,
) -> torch.Tensor:
    """Windowed 3D neighborhood attention over ``(1, T, H, W, NH, HD)`` Q/K/V.
    ``q`` must already carry the attention scale, and ``k`` must be the output of an
    RMSNorm with ``k_norm_weight``. Returns ``(1, T, H, W, NH * HD)``.
    Derives the softmax bound from ``k_norm_weight`` on every call, which is a small
    reduction per call; a caller in a hot loop should hoist it and use
    :func:`run_na_attention_bound`.
    """
    k_bound = softmax_row_bound(k_norm_weight)
    return run_na_attention_bound(q, k, v, kernel_size=kernel_size, k_bound=k_bound, tile_thw=tile_thw)


def run_na_attention_bound(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kernel_size: "tuple[int, int, int]",
    k_bound: torch.Tensor,
    tile_thw: "tuple[int, int, int] | None" = None,
    k_keyframes: torch.Tensor | None = None,
    v_keyframes: torch.Tensor | None = None,
    keyframe_slots: torch.Tensor | None = None,
    q_keyframes: torch.Tensor | None = None,
    keyframe_times: torch.Tensor | None = None,
) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
    """:func:`run_na_attention` with the softmax bound already computed.
    ``k_bound`` is the 0-d tensor :func:`ltx_kernels.vae.softmax_row_bound` returns; it
    is read by the kernel as an operand, so it never leaves the device.
    Keyframe context is optional and all-or-nothing. ``k_keyframes`` / ``v_keyframes`` are
    ``(1, n_kf, H, W, NH, HD)`` -- single-frame planes on the query grid's spatial lattice,
    projected by the *same* QKV and normed by the same ``k_norm`` as ``k`` (the softmax
    bound is only a bound because of that), and already carrying their float chunk-center
    temporal RoPE. ``keyframe_slots`` is the ``(T, S)`` int32 map from
    :func:`ltx_kernels.vae.keyframe_slots.nearest_keyframe_slots`.
    Each video query then attends, in one softmax, to its local ``kt x kh x kw`` window and
    to the ``kh x kw`` spatial window of every plane its row of ``keyframe_slots`` names --
    with no bound on how far those planes are in time.
    Passing ``q_keyframes`` (and ``keyframe_times``) additionally runs the *keyframe* stream's
    own queries, and the return becomes ``(video_out, keyframe_out)``. A keyframe query shares
    one softmax over its own plane's ``kh x kw`` window and that window on each of the <=2
    nearest video latent frames -- local motion context around the anchor, deliberately not
    the video kernel's full ``kt`` reach.
    """
    if not na_attn_available(q.device.index or 0):
        raise RuntimeError(UNSUPPORTED_MESSAGE)
    if q.shape[0] != 1:
        raise ValueError("B=1 only")
    if not (q.shape == k.shape == v.shape):
        raise ValueError(f"q/k/v shapes differ: {q.shape} {k.shape} {v.shape}")
    _, T, H, W, NH, hd = (int(s) for s in q.shape)
    tt, th, tw = tile_thw or default_tile_thw(kernel_size)
    if not na_supported(num_heads=NH, head_dim=hd, kernel_size=kernel_size, T=T, H=H, W=W, tile_thw=(tt, th, tw)):
        raise ValueError(f"na kernel unsupported: NH={NH} hd={hd} shape={(T, H, W)} k={kernel_size}")

    q_ = _as_rows(q[0].reshape(T, H, W, NH * hd).to(dtype=torch.bfloat16))
    # K and V are addressed one head-row at a time, so they are ``(tokens*NH, HD)``.
    k_ = k[0].reshape(T, H, W, NH * hd).to(dtype=torch.bfloat16).contiguous().view(-1, hd)
    v_ = v[0].reshape(T, H, W, NH * hd).to(dtype=torch.bfloat16).contiguous().view(-1, hd)

    grid_t = (T + tt - 1) // tt
    grid_h = (H + th - 1) // th
    grid_w = (W + tw - 1) // tw
    n_tiles = grid_t * grid_h * grid_w
    n_ctas = cta_count(n_tiles, q_.device)
    # M extra rows absorb the stores of out-of-range lanes in partial tiles.
    y_buf = torch.empty((T * H * W + M, NH * hd), device=q_.device, dtype=torch.bfloat16)

    kf = _keyframe_operands(k_keyframes, v_keyframes, keyframe_slots, T, H, W, NH, hd, tt, q_.device)

    n_kf = int(k_keyframes.shape[1]) if k_keyframes is not None else 0
    # Keyframe query tiles: one plane each, on the video tile's spatial shape. Compiled in
    # only when there is a plane to run -- unlike ``kf.enabled``, which selects the *window
    # rule* and so must not depend on the plane count.
    # A keyframe query tile borrows the video tile's spatial shape, so the two grids share
    # their h/w divisors and the kernel unflattens both with ``grid_h * grid_w``.
    kfq = q_keyframes is not None and kf.enabled and n_kf > 0
    if kfq:
        if tuple(q_keyframes.shape) != (1, n_kf, H, W, NH, hd):
            raise ValueError(f"q_keyframes must be {(1, n_kf, H, W, NH, hd)}, got {tuple(q_keyframes.shape)}")
        q_kf_rows = _as_rows(q_keyframes[0].reshape(n_kf, H, W, NH * hd).to(dtype=torch.bfloat16))
        video_frames = nearest_video_frames(keyframe_times, T) if keyframe_times is not None else None
        if video_frames is None:
            raise ValueError("q_keyframes requires keyframe_times, to place each plane on the video grid")
        y_kf_buf = torch.empty((n_kf * H * W + M, NH * hd), device=q_.device, dtype=torch.bfloat16)
        y_kf_dyn = _dyn_act(y_kf_buf)
        q_kf_ = _dyn_act(q_kf_rows)
        kfv = _dyn(upload_from_host(video_frames.reshape(-1).contiguous(), q_.device))
        kfv_counts = _dyn(upload_from_host((video_frames >= 0).sum(dim=-1).to(dtype=torch.int32), q_.device))
        n_kfq_tiles = n_kf * grid_h * grid_w
    else:
        _, _, q_kf_, y_kf_dyn = _inert_operands(hd, NH * hd, q_.device)
        kfv = kfv_counts = kf.counts
        n_kfq_tiles = 0

    with torch.cuda.device(q_.device):
        # ``keyframes`` is part of the key so the no-keyframe binary keeps the exact trace it
        # had before the tail existed -- ``n_kf = 0`` must stay bit-identical, and a runtime
        # zero-trip loop would still perturb the codegen around it.
        # Everything in this key is *structural*. Nothing derived from the anchor layout may join
        # it: the per-tile plane union bound used to, and since it moves with the clip length it
        # recompiled the whole stage ladder on every new duration -- 12+ minutes of in-process
        # MLIR before a 545-frame decode could start. It is now ``SLOTS * TT``, the true upper
        # bound (each of a tile's ``TT`` timesteps contributes at most ``SLOTS`` planes), which is
        # a constant of ``tile_thw`` and so already covered here.
        compiled = KERNEL_CACHE.get(
            _compile_kernel,
            q_.device,
            kernel_size=kernel_size,
            tile_thw=(tt, th, tw),
            keyframes=kf.enabled,
            keyframe_queries=kfq,
        )
        compiled(
            _dyn_act(q_),
            _dyn_act(k_),
            _dyn_act(v_),
            _dyn_act(y_buf),
            _dyn(k_bound.detach().reshape(1).float().to(device=q_.device)),
            kf.k,
            kf.v,
            kf.slots,
            kf.planes,
            kf.counts,
            q_kf_,
            y_kf_dyn,
            kfv,
            kfv_counts,
            T,
            H,
            W,
            NH,
            NH // HG,
            n_tiles,
            n_ctas,
            grid_h * grid_w,
            grid_w,
            n_kf,
            n_kfq_tiles,
        )
    video_out = y_buf[: T * H * W].view(1, T, H, W, NH * hd).to(dtype=q.dtype)
    if q_keyframes is None:
        return video_out
    if not kfq:
        # Keyframe queries asked for over an empty stream: no output buffer was allocated,
        # so return the empty shape rather than an arity the call site cannot predict.
        return video_out, q.new_empty((1, 0, H, W, NH * hd))
    return video_out, y_kf_buf[: n_kf * H * W].view(1, n_kf, H, W, NH * hd).to(dtype=q.dtype)
