"""Pass that routes unmasked SDPA calls to ``ltx_kernels::sdpa_select``."""

from typing import ClassVar

import torch
from torch.fx.passes.utils.matcher_utils import InternalMatch

from ltx_kernels.compile.subgraph_rewrite_pass import SubgraphRewritePass, is_cuda_bf16


class SDPASelectPass(SubgraphRewritePass):
    """Rewrite the unmasked ``scaled_dot_product_attention(q, k, v)`` to ``ltx_kernels::sdpa_select``.
    The pattern carries the keyword arguments ``Attention``'s SDPA backends pass, and
    ``ignore_literals=False`` pins them: only a call with no mask, no dropout and
    ``is_causal=False`` is the op's to serve. The surrounding ``sdpa_kernel`` priority context
    is left in place and has no effect on the op, which pins its own backend per shape.
    """

    ignore_literals: ClassVar[bool] = False

    @staticmethod
    def pattern_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)

    @staticmethod
    def replacement_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.ops.ltx_kernels.sdpa_select.default(q, k, v)

    def _is_match_eligible(self, match: InternalMatch) -> bool:
        inputs = self._matched_inputs(match)
        if not super()._is_match_eligible(match) or not inputs:
            return False
        return is_cuda_bf16(*inputs) and all(t.ndim == 4 for t in inputs)

    def example_inputs(self) -> tuple:
        return tuple(torch.empty(1, 4, 64, 64, dtype=torch.bfloat16, device="cuda") for _ in range(3))

    def _register_ops(self) -> None:
        from ltx_kernels.ops import sdpa_select as _sdpa_select_ops  # noqa: F401, PLC0415
