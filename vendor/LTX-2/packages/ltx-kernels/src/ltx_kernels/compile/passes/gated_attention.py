"""Pass that rewrites the bf16 gated-attention composite to a single custom op."""

import torch
from torch.fx.passes.utils.matcher_utils import InternalMatch

from ltx_kernels.compile.subgraph_rewrite_pass import SubgraphRewritePass, is_cuda_bf16, static_dim


class GatedAttentionPass(SubgraphRewritePass):
    """Rewrites the bf16 gated-attention composite::
        out = attn_out.view(B, T, H, D)
        out = out * (2 * sigmoid(gate_logits)).unsqueeze(-1)
        return out.view(B, T, H * D)
    to a single ``ltx_kernels::gated_attention(attn_out, gate_logits)`` call.
    """

    @staticmethod
    def pattern_fn(attn_out: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
        b, t, hd = attn_out.shape
        h = gate_logits.shape[-1]
        d = hd // h
        out = attn_out.view(b, t, h, d)
        gates = 2.0 * torch.sigmoid(gate_logits)
        out = out * gates.unsqueeze(-1)
        return out.view(b, t, hd)

    @staticmethod
    def replacement_fn(attn_out: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
        return torch.ops.ltx_kernels.gated_attention.default(attn_out, gate_logits)

    def _is_match_eligible(self, match: InternalMatch) -> bool:
        inputs = self._matched_inputs(match)
        if not super()._is_match_eligible(match) or not inputs:
            return False
        attn_out, gate_logits = inputs
        if not is_cuda_bf16(attn_out, gate_logits) or attn_out.ndim != 3 or gate_logits.ndim != 3:
            return False
        hidden = static_dim(attn_out.shape[-1])
        heads = static_dim(gate_logits.shape[-1])
        if hidden is None or heads is None or hidden % heads:
            return False
        # The head count and head dim are ``tl.constexpr`` the kernel spans a whole row
        # with, via ``tl.arange``, so both must be powers of two.
        dim_head = hidden // heads
        return heads & (heads - 1) == 0 and dim_head & (dim_head - 1) == 0

    def example_inputs(self) -> tuple:
        # Production LTX-2.3 shapes (heads=32, dim_head=128 for video, B=2 for CFG).
        return (
            torch.empty(2, 64, 32 * 128, dtype=torch.bfloat16, device="cuda"),
            torch.empty(2, 64, 32, dtype=torch.bfloat16, device="cuda"),
        )

    def _register_ops(self) -> None:
        from ltx_kernels.ops import gated_attention as _gated_attention_ops  # noqa: F401, PLC0415
