"""Fuse a trailing ``blockwise_quantize`` into the producer's quantized twin.
Each pass matches a producer together with the ``blockwise_quantize`` that consumes
it, and replaces the pair with the producer's ``*_quantized`` twin, which returns
the same ``(fp8, scales)`` tuple.
Assumes :class:`BlockwiseQuantizeCSE` has run, so a fusible producer has a single
quantize consumer. A producer that also feeds a bf16 consumer (e.g. the
FP8-excluded ``to_gate_logits``) is left alone, because the matcher rejects a match
whose internal nodes have outside consumers -- no dequant is ever introduced.
"""

from typing import ClassVar

import torch
from torch.fx.passes.utils.matcher_utils import InternalMatch

from ltx_kernels.blockwise._common import _BLOCK_SIZE
from ltx_kernels.compile.subgraph_rewrite_pass import SubgraphRewritePass, is_cuda_bf16, static_dim


class AdaZeroTrailingQuantizePass(SubgraphRewritePass):
    """``blockwise_quantize(rms_norm(x) * (1 + scale) + shift)`` -> ``ada_zero_quantized(...)``.
    Carries the eager composite because :class:`AdaZeroPass` does not run under every
    backend, so there may be no ``ada_zero`` node to bind. ``hidden_dim`` and ``eps``
    are baked as pattern literals, as there; one instance per pair.
    """

    ignore_literals: ClassVar[bool] = False

    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim

        def pattern_fn(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            normed = torch.nn.functional.rms_norm(x, (x.shape[-1],), None, eps)
            fp8, scales = torch.ops.ltx_kernels.blockwise_quantize.default(normed * (1 + scale) + shift)
            return fp8, scales

        def replacement_fn(
            x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.ltx_kernels.ada_zero_quantized.default(x, scale, shift, eps)

        self.pattern_fn = pattern_fn
        self.replacement_fn = replacement_fn

    def _is_match_eligible(self, match: InternalMatch) -> bool:
        inputs = self._matched_inputs(match)
        if not super()._is_match_eligible(match) or not inputs:
            return False
        x, scale, shift = inputs
        if not is_cuda_bf16(x, scale, shift) or x.ndim != 3 or scale.ndim != 3 or shift.ndim != 3:
            return False
        # H is a ``tl.constexpr`` the kernel spans a whole row with, via
        # ``tl.arange(0, H)``, so it must be a power of two and must be the row length
        # of the affine tensors too. The twin quantizes internally, so a row must also
        # split into whole 128-element blocks.
        hidden = static_dim(x.shape[-1])
        return (
            hidden is not None
            and hidden & (hidden - 1) == 0
            and hidden % _BLOCK_SIZE == 0
            and static_dim(scale.shape[-1]) == hidden
            and static_dim(shift.shape[-1]) == hidden
        )

    def example_inputs(self) -> tuple:
        batch, seq = 2, 128
        hidden = self._hidden_dim
        return (
            torch.empty(batch, seq, hidden, dtype=torch.bfloat16, device="cuda"),
            torch.empty(batch, 1, hidden, dtype=torch.bfloat16, device="cuda"),
            torch.empty(batch, 1, hidden, dtype=torch.bfloat16, device="cuda"),
        )

    def _register_ops(self) -> None:
        from ltx_kernels.blockwise.ops import quantize as _quantize_ops  # noqa: F401, PLC0415
        from ltx_kernels.ops import ada_zero as _ada_zero_ops  # noqa: F401, PLC0415


class GatedAttentionTrailingQuantizePass(SubgraphRewritePass):
    """``blockwise_quantize(gated_attention(attn_out, gate_logits))`` -> ``gated_attention_quantized(...)``.
    Binds the fused op :class:`GatedAttentionPass` leaves behind, so it must run
    after it. Fires at ``gated_attention -> to_out``.
    """

    @staticmethod
    def pattern_fn(attn_out: torch.Tensor, gate_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gated = torch.ops.ltx_kernels.gated_attention.default(attn_out, gate_logits)
        fp8, scales = torch.ops.ltx_kernels.blockwise_quantize.default(gated)
        return fp8, scales

    @staticmethod
    def replacement_fn(attn_out: torch.Tensor, gate_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.ltx_kernels.gated_attention_quantized.default(attn_out, gate_logits)

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
        # with, via ``tl.arange``, so both must be powers of two. The twin quantizes
        # internally, so a row must also split into whole 128-element blocks.
        dim_head = hidden // heads
        return heads & (heads - 1) == 0 and dim_head & (dim_head - 1) == 0 and hidden % _BLOCK_SIZE == 0

    def example_inputs(self) -> tuple:
        # Production LTX-2.3 shapes (heads=32, dim_head=128 for video, B=2 for CFG).
        return (
            torch.empty(2, 64, 32 * 128, dtype=torch.bfloat16, device="cuda"),
            torch.empty(2, 64, 32, dtype=torch.bfloat16, device="cuda"),
        )

    def _register_ops(self) -> None:
        from ltx_kernels.blockwise.ops import quantize as _quantize_ops  # noqa: F401, PLC0415
        from ltx_kernels.ops import gated_attention as _gated_attention_ops  # noqa: F401, PLC0415
