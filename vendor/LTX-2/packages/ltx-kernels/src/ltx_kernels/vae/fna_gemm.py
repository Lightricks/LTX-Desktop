"""TMEM plumbing and TMA/UMMA GEMM helpers for the DiffVAE CuTe DSL kernels.
Extracted from :mod:`ltx_kernels.vae.fna_attn_core`. Constants:
:mod:`ltx_kernels.vae.fna_types`.
"""

# NOTE: no `from __future__ import annotations` -- cute.struct needs real types.

from typing import Any, NamedTuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as cutlass_utils
from cutlass.cute.nvgpu import cpasync, tcgen05

from ltx_kernels.vae.fna_types import (
    ACC,
    ACC_COLS,
    INST_N64,
    INST_N128,
    IO,
    THREADS,
    TILER_N64,
    TILER_N128,
    M,
)


@cute.struct
class UmmaSharedStorage:
    """Barrier storage for the TMA/UMMA pipelines :func:`gemm_tma` stands up."""

    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 8]  # ab_stages * 2
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]  # acc_stage * 2
    tmem_holding_buf: cutlass.Int32


# MMA variants, by name:
#   "n128"  wide linears and Q@K       (N=128, B K-major)
#   "pv"    P@V                        (N=64,  B MN-major)
_MMA_KINDS = ("n128", "pv")


def _mma_op(kind: str):
    """The tcgen05 MMA op for one of ``_MMA_KINDS``."""
    inst = INST_N128 if kind == "n128" else INST_N64
    b_major = cute.nvgpu.OperandMajorMode.MN if kind == "pv" else cute.nvgpu.OperandMajorMode.K
    return tcgen05.MmaF16BF16Op(
        IO,
        ACC,
        inst,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        cute.nvgpu.OperandMajorMode.K,
        b_major,
    )


def _mma_tiler(kind: str):
    return TILER_N128 if kind == "n128" else TILER_N64


def acquire_tmem(storage):
    """Take all 512 TMEM columns; returns ``(allocator, base_pointer)``.
    Both kernels hold the entire allocation for their whole lifetime, which is what
    pins them to one CTA per SM and forces the multi-warpgroup split described at
    :data:`NUM_WARPGROUPS`.
    """
    tmem = cutlass_utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=2, num_threads=THREADS),
    )
    tmem.allocate(512)
    tmem.wait_for_alloc()
    return tmem, tmem.retrieve_ptr(ACC)


def release_tmem(tmem, tmem_ptr):
    """Hand the TMEM allocation back. The last thing either kernel does."""
    tmem.relinquish_alloc_permit()
    pipeline.sync(barrier_id=2)
    tmem.free(tmem_ptr)


def _tmem_copies(acc, lane, w=ACC_COLS):
    """``(load, load_src, store, store_dst)`` for one TMEM accumulator.
    ``w`` is the chunk width: a 128-wide accumulator is cut into ``NUM_WARPGROUPS`` chunks of
    ``ACC_COLS`` and a per-head one into ``NUM_WARPGROUPS`` of ``HEAD_COLS``, so warpgroup ``g`` takes chunk
    ``g`` of either. ``lane`` is ``tidx % M`` -- the row -- because the copy's thread
    layout is one warpgroup wide.
    Rebuilt at each use: this is pure layout algebra with no runtime cost, and a
    long-lived object here would be classified as ``while``-loop state, which the DSL
    cannot stage.
    """
    epi = ((cute.size(acc, mode=[0, 0]), w),)
    tacc = cute.zipped_divide(acc, epi)
    ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(w)), ACC)
    st_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(w)), ACC)
    ld = tcgen05.make_tmem_copy(ld_atom, tacc[None, 0])
    st = tcgen05.make_tmem_copy(st_atom, tacc[None, 0])
    return ld, ld.get_slice(lane).partition_S(tacc), st, st.get_slice(lane).partition_D(tacc)


def _tmem_load(acc, lane, regs, i=0, w=ACC_COLS):
    ld, src, _, _ = _tmem_copies(acc, lane, w)
    cute.copy(ld, src[None, None, i], regs)


def _tmem_store(acc, lane, regs, i=0, w=ACC_COLS):
    """Write registers back into TMEM.
    This is what lets the flash-attention output accumulator live in TMEM rather than
    costing ``NH * HD`` registers per thread.
    """
    _, _, st, dst = _tmem_copies(acc, lane, w)
    cute.copy(st, regs, dst[None, None, i])


def _tmem_shift(acc, cols):
    """``acc`` moved ``cols`` TMEM columns along, with its layout.
    How a runtime index picks one of an evenly spaced run of accumulators -- a head's
    ``acc_o``, an N-tile's ``acc_main`` -- without unrolling the loop that indexes it: a
    TMEM address is a register, so ``cols`` may be a runtime value.
    """
    return cute.make_tensor(acc.iterator + cols, acc.layout)


def _reg_shape(tiled_mma, acc, lane, n_cols, w=ACC_COLS):
    """Shape every register array must use: the TMEM copy's ``partition_D``.
    An identity tensor stands in for the destination -- it carries the layout without
    needing real storage. A plain ``make_rmem_tensor(64, f32)`` used as a
    ``cute.copy`` destination miscompiles instead of erroring.
    """
    epi = ((cute.size(acc, mode=[0, 0]), w),)
    tacc = cute.zipped_divide(acc, epi)
    gC = cute.zipped_divide(tiled_mma.get_slice(0).partition_C(cute.make_identity_tensor((M, n_cols))), epi)
    atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(w)), ACC)
    tcp = tcgen05.make_tmem_copy(atom, tacc[None, 0])
    return tcp.get_slice(lane).partition_D(gC)[None, None, 0].shape


@cute.jit
def _mma_issue(
    kind: cutlass.Constexpr,
    tCrA,
    tCrB,
    acc,
    n_stage: cutlass.Constexpr,
    a_st,
    b_st,
    accumulate: cutlass.Constexpr,
    mbar=None,
):
    """Issue one UMMA and return immediately -- the caller decides when to wait.
    The MMA signals ``mbar`` on completion (``tcgen05.commit`` covers *every* MMA the
    warp issued before it, so a chain needs only the last one to carry a barrier) and
    the consumer waits a whole loop step later. Waiting inline instead pins the
    attention loop: the CTA sits on a ~370-cycle MMA with nothing else in flight.
    Operand correctness comes from program order: the issuing warp is an ordinary
    warp, so a ``sync_threads`` between an SMEM fill and this call publishes the
    operand, and a barrier *after* a wait on a later MMA proves every earlier one has
    drained.
    """
    tiled_mma = cute.make_tiled_mma(_mma_op(kind))
    tiled_mma.set(tcgen05.Field.ACCUMULATE, accumulate)
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        for st in cutlass.range_constexpr(n_stage):
            for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                cute.gemm(
                    tiled_mma,
                    acc,
                    tCrA[None, None, kb, a_st + st],
                    tCrB[None, None, kb, b_st + st],
                    acc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        if cutlass.const_expr(mbar is not None):
            with cute.arch.elect_one():
                tcgen05.commit(mbar)


class GemmJob(NamedTuple):
    """One ``M=128`` GEMM of a :func:`gemm_tma` call: B from ``(tma_atom, m_b)`` against ``fr_a``.
    ``fr_a2`` continues the reduction in a second A operand from K-tile ``a_split`` on, for a
    reduction wider than one operand can hold. ``kt0`` / ``n_kt`` pick the K-tiles of B.
    ``n_idx``, ``kt0`` and ``accumulate`` may be device values (a device loop's index and the
    expressions built from it); the other counts are trace-time ints.
    """

    tma_atom: Any
    m_b: Any
    fr_a: Any
    acc: Any
    n_idx: Any
    kt0: Any
    n_kt: int
    a_st: int
    accumulate: Any
    fr_a2: Any = None
    a_split: int = 0


@cute.jit
def gemm_tma(
    kind: cutlass.Constexpr,
    storage,
    sB,
    stage_bytes: cutlass.Constexpr,
    jobs: cutlass.Constexpr,
    num_stages: cutlass.Constexpr = 2,
):
    """Independent ``M=128`` GEMMs with their B operands streamed in by TMA.
    A is already in SMEM (the kernel computes it); only B is fetched. The
    ``PipelineTmaUmma`` cycles ``sB``'s ``num_stages`` stages over every job's K-tiles as
    one sequence, keeping ``num_stages - 1`` copies -- the next job's first ones included --
    in flight during the current tile's MMA, and the CTA waits for the accumulators once.
    Standing up the two pipelines and that final wait cost more than the K-tiles
    themselves at these shapes, which is why independent GEMMs share a call.
    ``jobs`` is a trace-time tuple of :class:`GemmJob`.
    """
    tiled_mma = cute.make_tiled_mma(_mma_op(kind))
    tiler = _mma_tiler(kind)
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        for job in jobs:
            cpasync.prefetch_descriptor(job.tma_atom)

    ab_prod, ab_cons = pipeline.PipelineTmaUmma.create(
        num_stages=num_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=stage_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    ).make_participants()
    acc_prod, acc_cons = pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    parts = []
    for job in jobs:
        gB = cute.local_tile(job.m_b, tiler, (0, job.n_idx, None), proj=(None, 1, 1))
        tCgB = tiled_mma.get_slice(0).partition_B(gB)
        parts.append(
            cpasync.tma_partition(
                job.tma_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 3),
                cute.group_modes(tCgB, 0, 3),
            )
        )
    frB = tiled_mma.make_fragment_B(sB)
    steps = [(j, kt) for j, job in enumerate(jobs) for kt in range(job.n_kt)]

    tiled_mma.set(tcgen05.Field.ACCUMULATE, jobs[0].accumulate)
    if warp_idx == 0:
        acc_empty = acc_prod.acquire_and_advance()
        # ``num_stages - 1`` copies ahead: step i+look's copy is issued before the wait on
        # step i, so sB's stages actually overlap TMA with MMA.
        look = num_stages - 1
        for i in cutlass.range_constexpr(min(look, len(steps))):
            pj, pkt = steps[i]
            pre = ab_prod.acquire_and_advance()
            cute.copy(
                jobs[pj].tma_atom,
                parts[pj][1][(None, jobs[pj].kt0 + pkt)],
                parts[pj][0][(None, pre.index)],
                tma_bar_ptr=pre.barrier,
            )
        for i in cutlass.range_constexpr(len(steps)):
            if cutlass.const_expr(i + look < len(steps)):
                nj, nkt = steps[i + look]
                nxt = ab_prod.acquire_and_advance()
                cute.copy(
                    jobs[nj].tma_atom,
                    parts[nj][1][(None, jobs[nj].kt0 + nkt)],
                    parts[nj][0][(None, nxt.index)],
                    tma_bar_ptr=nxt.barrier,
                )
            j, kt = steps[i]
            job = jobs[j]
            if cutlass.const_expr(j > 0 and kt == 0):
                tiled_mma.set(tcgen05.Field.ACCUMULATE, job.accumulate)
            f = ab_cons.wait_and_advance()
            first_operand = job.a_split == 0 or kt < job.a_split
            src = job.fr_a if cutlass.const_expr(first_operand) else job.fr_a2
            a_kt = job.a_st + kt if cutlass.const_expr(first_operand) else kt - job.a_split
            for kb in cutlass.range_constexpr(cute.size(job.fr_a, mode=[2])):
                cute.gemm(tiled_mma, job.acc, src[None, None, kb, a_kt], frB[None, None, kb, f.index], job.acc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            f.release()
        acc_empty.commit()
    acc_cons.wait_and_advance().release()
