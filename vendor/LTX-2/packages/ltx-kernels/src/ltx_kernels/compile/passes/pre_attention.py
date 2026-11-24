"""Pass that rewrites the eager RMSNorm + split-RoPE composite to a single custom op."""

from typing import ClassVar

import torch
from einops import rearrange
from torch.fx.passes.utils.matcher_utils import InternalMatch

from ltx_kernels.compile.subgraph_rewrite_pass import SubgraphRewritePass, is_cuda_bf16, static_dim


class PreAttentionPass(SubgraphRewritePass):
    """Rewrites the bf16 PreAttention composite::
        normed = F.rms_norm(x, (H,), weight, eps)
        return apply_split_rotary_emb(normed, cos, sin)
    to a single ``ltx_kernels::rms_norm_split_rope(x, weight, cos, sin, eps)``
    call. Fires once per (q, q_norm) and (k, k_norm) call in
    ``Attention.forward``.
    Pattern mirrors the Dynamo IR ``apply_split_rotary_emb`` emits in
    production (the ``needs_reshape=True`` branch: ``(B, T, H)`` input +
    ``(B, N, T, D//2)`` cos/sin), in its non-in-place form.
    Uses ``ignore_literals=False`` so ``eps`` is compared rather than wildcarded:
    the replacement passes this instance's ``eps`` to the kernel, so a site normed
    with a different one is not this pass's. That also pins the ``hidden_dim`` and
    ``heads`` literals the pattern carries, so callers register one instance per
    triple (LTX-2.3: video 4096 x 32, audio 2048 x 32, both ``eps=1e-6``).
    """

    # The CUDA kernel is instantiated for these hidden sizes only.
    supported_hidden: ClassVar[tuple[int, ...]] = (2048, 4096, 8192)
    ignore_literals: ClassVar[bool] = False

    def __init__(self, hidden_dim: int, heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._heads = heads

        def pattern_fn(
            x: torch.Tensor,
            weight: torch.Tensor,
            cos_freqs: torch.Tensor,
            sin_freqs: torch.Tensor,
        ) -> torch.Tensor:
            normed = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)
            # Inlined apply_split_rotary_emb (needs_reshape=True branch only).
            h = cos_freqs.shape[1]
            input_tensor = normed.unflatten(-1, (h, -1)).transpose(1, 2)
            split_input = rearrange(input_tensor, "... (d r) -> ... d r", d=2)
            first_half_input = split_input[..., :1, :]
            second_half_input = split_input[..., 1:, :]
            output = split_input * cos_freqs.unsqueeze(-2)
            first_half_output = output[..., :1, :].addcmul(-sin_freqs.unsqueeze(-2), second_half_input)
            second_half_output = output[..., 1:, :].addcmul(sin_freqs.unsqueeze(-2), first_half_input)
            output = torch.cat([first_half_output, second_half_output], dim=-2)
            output = rearrange(output, "... d r -> ... (d r)")
            return output.transpose(1, 2).flatten(-2)

        def replacement_fn(
            x: torch.Tensor,
            weight: torch.Tensor,
            cos_freqs: torch.Tensor,
            sin_freqs: torch.Tensor,
        ) -> torch.Tensor:
            return torch.ops.ltx_kernels.rms_norm_split_rope.default(x, weight, cos_freqs, sin_freqs, eps)

        self.pattern_fn = pattern_fn
        self.replacement_fn = replacement_fn

    def _is_match_eligible(self, match: InternalMatch) -> bool:
        inputs = self._matched_inputs(match)
        if not super()._is_match_eligible(match) or not inputs:
            return False
        x, weight, cos_freqs, sin_freqs = inputs
        if not is_cuda_bf16(x, weight, cos_freqs, sin_freqs):
            return False
        if x.ndim != 3 or weight.ndim != 1 or cos_freqs.ndim != 4 or sin_freqs.ndim != 4:
            return False
        hidden = static_dim(x.shape[-1])
        heads = static_dim(cos_freqs.shape[1])
        if hidden not in self.supported_hidden or heads is None or hidden % (2 * heads):
            return False
        # One thread owns 16 rotation pairs of a single head, so head_dim/2 must be a
        # multiple of 16, and the frequency tables must be exactly half a head wide.
        # Layout is the op's problem, not this check's: it makes x contiguous, the freq
        # tables innermost-contiguous, and expands a batch-shared table.
        half = hidden // (2 * heads)
        return (
            (hidden // heads) % 32 == 0
            and static_dim(cos_freqs.shape[-1]) == half
            and static_dim(sin_freqs.shape[-1]) == half
        )

    def example_inputs(self) -> tuple:
        batch, seq = 2, 64
        hidden, heads = self._hidden_dim, self._heads
        half = hidden // (2 * heads)
        return (
            torch.empty(batch, seq, hidden, dtype=torch.bfloat16, device="cuda"),
            torch.empty(hidden, dtype=torch.bfloat16, device="cuda"),
            torch.empty(batch, heads, seq, half, dtype=torch.bfloat16, device="cuda"),
            torch.empty(batch, heads, seq, half, dtype=torch.bfloat16, device="cuda"),
        )

    def _register_ops(self) -> None:
        from ltx_kernels.ops import rms_norm_split_rope as _rms_norm_split_rope_ops  # noqa: F401, PLC0415
