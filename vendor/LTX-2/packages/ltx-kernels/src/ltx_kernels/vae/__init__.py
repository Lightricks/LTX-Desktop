"""CuTe DSL kernels for the DiffVAE decoder.
Two kernels sharing one attention half (``fna_attn_core.attention_tile``):
- :func:`~ltx_kernels.vae.block_fna_dsl.run_block_fna_dsl` -- the whole stage-5
  ``DiffusionNABlock`` in one launch, with no full-volume Q/K/V.
- :func:`~ltx_kernels.vae.na_attn_dsl.run_na_attention` -- standalone 3D neighborhood
  attention for the deterministic stages, a drop-in for ``natten.na3d``.
Both need datacenter Blackwell (sm_100 / sm_103) or Jetson Thor (sm_110): they
are built on ``tcgen05`` UMMA and Tensor Memory, which consumer Blackwell, Hopper
and Ada do not have. Each launcher enforces that itself; ``block_fna_available`` /
``na_attn_available`` let a caller ask first and pick another path.
"""

from ltx_kernels.vae.block_fna_dsl import (
    OUT_SLACK_ROWS,
    block_fna_available,
    fna_supported,
    pack_fused_ctx_weights,
    run_block_fna_dsl,
)
from ltx_kernels.vae.na_attn_dsl import (
    na_attn_available,
    na_supported,
    run_na_attention,
    run_na_attention_bound,
)
from ltx_kernels.vae.softmax_bound import softmax_row_bound

__all__ = [
    "OUT_SLACK_ROWS",
    "block_fna_available",
    "fna_supported",
    "na_attn_available",
    "na_supported",
    "pack_fused_ctx_weights",
    "run_block_fna_dsl",
    "run_na_attention",
    "run_na_attention_bound",
    "softmax_row_bound",
]
