"""Fused RMS-norm + split RoPE.
Wraps the CUDA kernel in :mod:`ops_cpp` (built from
``csrc/ops/rms_norm_split_rope*``) as ``torch.library.custom_op`` so it
appears as a single ``call_function`` node under ``torch.compile`` and is
safe to use as a pattern rewrite target.
Mirrors :func:`ltx_core.model.transformer.rope.apply_split_rotary_emb`: the
``cos_freqs`` / ``sin_freqs`` tables are accepted in ``(cos, sin)`` order and
``(B, N, T, D)`` layout (what the eager LTX RoPE precomputation produces).
"""

import torch


@torch.library.custom_op("ltx_kernels::rms_norm_split_rope", mutates_args=())
def rms_norm_split_rope(
    x: torch.Tensor,
    weights: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """BF16 fused RMS-norm + split RoPE.
    Args:
        x: ``(B, T, H)`` bf16 input.
        weights: ``(H,)`` bf16 RMSNorm scale.
        cos_freqs: ``(B, N, T, D)`` bf16 cos table; ``N * D == H / 2``.
        sin_freqs: ``(B, N, T, D)`` bf16 sin table; same shape as ``cos_freqs``.
        eps: RMSNorm epsilon.
    Returns:
        ``(B, T, H)`` bf16 output.
    Arg order ``(x, weight, cos, sin)`` mirrors the order Dynamo emits these
    placeholders when capturing the eager PreAttention composite
    (``rms_norm(x, ..., weight, ...)`` first, then RoPE on cos/sin) -- letting
    :class:`SubgraphRewritePass` zip pattern and replacement placeholders by
    position without cross-wiring them.
    """
    import ops_cpp  # noqa: PLC0415

    return ops_cpp.rms_norm_split_rope(x, sin_freqs, cos_freqs, weights, eps, False)


@rms_norm_split_rope.register_fake
def _rms_norm_split_rope_fake(
    x: torch.Tensor,
    weights: torch.Tensor,  # noqa: ARG001
    cos_freqs: torch.Tensor,  # noqa: ARG001
    sin_freqs: torch.Tensor,  # noqa: ARG001
    eps: float,  # noqa: ARG001
) -> torch.Tensor:
    return torch.empty_like(x)
