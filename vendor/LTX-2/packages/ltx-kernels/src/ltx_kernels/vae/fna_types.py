"""Shared compile-time constants for the DiffVAE CuTe DSL kernels.
Imported by :mod:`ltx_kernels.vae.fna_smem`, :mod:`ltx_kernels.vae.fna_gemm`, and
:mod:`ltx_kernels.vae.fna_attn_core` so those modules do not import each other for
constants. Re-exported from ``fna_attn_core`` for existing call sites.
"""

# NOTE: no `from __future__ import annotations` -- cute.struct needs real types.

import cutlass

IO = cutlass.BFloat16
ACC = cutlass.Float32
M = 128
INST_N128 = (128, 128, 16)
TILER_N128 = (128, 128, 64)
INST_N64 = (128, 64, 16)
TILER_N64 = (128, 64, 64)
# The wide MMA's N and K tile widths. ``TILE_N`` is numerically ``M``, but it is a
# different axis: ``M`` counts rows -- tokens, one per lane -- while ``N`` counts an
# accumulator's columns, which are channels for a projection, keys for Q@K and hidden
# units for gate/up. Anything sized by an accumulator's width takes ``TILE_N``,
# anything sized by the reduction depth takes ``TILE_K``, and ``M`` keeps only its
# row/token/lane meaning.
TILE_N = TILER_N128[1]
TILE_K = TILER_N128[2]
# The two axes do have to agree numerically: the KV loop hands lane ``i`` key ``i`` of
# an N-wide key tile, so a wider N would leave keys no lane owns.
assert TILE_N == M, "attention_tile indexes an N=TILE_N key tile by lane (0..M-1)"
EPS = 1e-6
HD = 64  # head_dim; fixed by the DiffVAE attention
# A head is one K-stage of an A operand, so a runtime head index reaches its channels by
# shifting the operand view whole stages (``fna_smem._alias``) with static slice coordinates.
assert HD == TILE_K, "per-head operand views assume one head per K-stage"

# Warpgroups per CTA -- 128 threads, four warps, the tcgen05 scheduling unit. Both
# kernels hold all 512 TMEM columns, so they are pinned to one CTA per SM and the only
# way to get more than one warp per scheduler is more warps inside the CTA. Thread
# ``i`` still owns accumulator row ``i``; warpgroup ``g`` owns a different slice of
# that row's columns.
NUM_WARPGROUPS = 2
THREADS = M * NUM_WARPGROUPS
# How many columns of an accumulator row one thread owns, for the two accumulator
# widths the kernels use: 128 for anything C-wide or a Q@K score row, 64 for a
# per-head attention output.
ACC_COLS = TILE_N // NUM_WARPGROUPS
HEAD_COLS = HD // NUM_WARPGROUPS
HEADS_PER_TILE = TILE_N // HD  # heads in one N=128 projection tile
# Warpgroups sharing a head. Two warpgroups gives one, so the Q/K RMSNorms and RoPE
# stay register-local; four gives two, and they then need a partial reduction over the
# pair (``head_sum2``) and a channel-ranged RoPE. ``head_sum2`` and the K/V pass are
# written generically, so the two-warpgroup build is the ``WARPGROUPS_PER_HEAD == 1``
# special case of the same code.
WARPGROUPS_PER_HEAD = NUM_WARPGROUPS // HEADS_PER_TILE
assert HEADS_PER_TILE * WARPGROUPS_PER_HEAD == NUM_WARPGROUPS, "NUM_WARPGROUPS must be a multiple of TILE_N // HD"
# The other row passes -- the query pass, the residual and AdaLN passes, the P and MLP
# hidden fills -- are one code copy for all warpgroups: each reaches its ``ACC_COLS``
# chunk through an operand view shifted by whole K-stages, so a chunk must be one stage.
assert ACC_COLS == TILE_K, "warpgroup-agnostic row passes need one K-stage per warpgroup chunk"
# One cooperative-copy step moves a whole ``_KRUN`` run per thread, so both ends of
# the copy vectorise: 32B from global (2x LDG.E.128) and one swizzled-slice store.
VEC = 16  # bf16 elements per thread per cooperative-copy step
VEC_ROWS = THREADS // (M // VEC)  # rows of the operand in flight at once

# The softmax exponentiates with ``ex2``, so scores have to reach it in base 2. That
# rides on the subtraction of the row offset -- ``fma(score, LOG2E, -offset)`` instead
# of ``score - offset``, one instruction either way -- so callers hand over ordinary
# base-e scores and nothing outside this file knows about the base. Folding it into Q
# host-side instead measures the same to within noise at every DiffVAE shape.
LOG2E = 1.4426950408889634
