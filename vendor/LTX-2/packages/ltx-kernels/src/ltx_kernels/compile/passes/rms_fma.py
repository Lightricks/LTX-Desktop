"""Pass that fuses the post-self-attention residual update and RMSNorm."""

from typing import ClassVar

import torch
from torch.fx.passes.utils.matcher_utils import InternalMatch

from ltx_kernels.compile.subgraph_rewrite_pass import SubgraphRewritePass, is_cuda_bf16, static_dim


def _supports_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> bool:
    """Whether ``_rms_sum_mult_kernel`` handles these residual inputs."""
    if not is_cuda_bf16(x, y, gate) or x.ndim != 3 or y.ndim != 3 or gate.ndim not in (2, 3):
        return False
    # The kernel spans a whole row with ``tl.arange(0, H)``, so H must be a power
    # of two, and reaches x and y row-major from one base pointer each.
    hidden = static_dim(x.shape[-1])
    if hidden is None or hidden & (hidden - 1) or not x.is_contiguous() or not y.is_contiguous():
        return False
    # ``gate`` is indexed by batch (and by token unless broadcast); it is
    # broadcast over H only when its last dim is 1.
    return (
        static_dim(y.shape[-1]) == hidden
        and static_dim(gate.shape[0]) == static_dim(x.shape[0])
        and static_dim(gate.shape[-1]) in (hidden, 1)
    )


class RMSFmaAdaLNPass(SubgraphRewritePass):
    """Rewrite PostSA and cross-attention query modulation together.
    This is the largest PostSA pattern and must run before both
    :class:`RMSFmaPass` and :class:`AdaZeroPass`.
    """

    ignore_literals: ClassVar[bool] = False

    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim

        def pattern_fn(
            x: torch.Tensor,
            y: torch.Tensor,
            gate: torch.Tensor,
            scale: torch.Tensor,
            shift: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            residual = x + y * gate
            normed = torch.nn.functional.rms_norm(residual, (x.shape[-1],), None, eps)
            attn_input = normed * (1 + scale) + shift
            return residual, attn_input

        def replacement_fn(
            x: torch.Tensor,
            y: torch.Tensor,
            gate: torch.Tensor,
            scale: torch.Tensor,
            shift: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.ltx_kernels.rms_fma_adaln.default(x, y, gate, scale, shift, eps)

        self.pattern_fn = pattern_fn
        self.replacement_fn = replacement_fn

    def _is_match_eligible(self, match: InternalMatch) -> bool:
        inputs = self._matched_inputs(match)
        if not super()._is_match_eligible(match) or not inputs:
            return False
        x, y, gate, scale, shift = inputs
        if not _supports_residual(x, y, gate) or not is_cuda_bf16(scale, shift):
            return False
        # ``scale`` / ``shift`` are indexed by batch and, unless broadcast over
        # the seq dim, by token -- and the broadcast flag is read off ``scale``
        # alone, so ``shift`` has to agree with it.
        hidden = static_dim(x.shape[-1])
        batch = static_dim(x.shape[0])
        return (
            scale.ndim == shift.ndim
            and scale.ndim in (2, 3)
            and static_dim(scale.shape[0]) == batch
            and static_dim(shift.shape[0]) == batch
            and static_dim(scale.shape[1]) == static_dim(shift.shape[1])
            and static_dim(scale.shape[-1]) == hidden
            and static_dim(shift.shape[-1]) == hidden
        )

    def example_inputs(self) -> tuple:
        batch, seq = 2, 64
        return (
            torch.empty(batch, seq, self._hidden_dim, dtype=torch.bfloat16, device="cuda"),
            torch.empty(batch, seq, self._hidden_dim, dtype=torch.bfloat16, device="cuda"),
            torch.empty(batch, 1, self._hidden_dim, dtype=torch.bfloat16, device="cuda"),
            torch.empty(batch, 1, self._hidden_dim, dtype=torch.bfloat16, device="cuda"),
            torch.empty(batch, 1, self._hidden_dim, dtype=torch.bfloat16, device="cuda"),
        )

    def _register_ops(self) -> None:
        from ltx_kernels.ops import rms_fma as _rms_fma_ops  # noqa: F401, PLC0415


class RMSFmaPass(SubgraphRewritePass):
    """Rewrite the PostSA composite to ``ltx_kernels::rms_fma``.
    This pass must run before :class:`AdaZeroPass`. With cross-attention AdaLN,
    AdaZero's pattern is a subgraph of PostSA followed by query modulation.
    """

    ignore_literals: ClassVar[bool] = False

    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim

        def pattern_fn(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            residual = x + y * gate
            normed = torch.nn.functional.rms_norm(residual, (x.shape[-1],), None, eps)
            return residual, normed

        def replacement_fn(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.ltx_kernels.rms_fma.default(x, y, gate, eps)

        self.pattern_fn = pattern_fn
        self.replacement_fn = replacement_fn

    def _is_match_eligible(self, match: InternalMatch) -> bool:
        inputs = self._matched_inputs(match)
        if not super()._is_match_eligible(match) or not inputs:
            return False
        return _supports_residual(*inputs)

    def example_inputs(self) -> tuple:
        batch, seq = 2, 64
        return (
            torch.empty(batch, seq, self._hidden_dim, dtype=torch.bfloat16, device="cuda"),
            torch.empty(batch, seq, self._hidden_dim, dtype=torch.bfloat16, device="cuda"),
            torch.empty(batch, 1, self._hidden_dim, dtype=torch.bfloat16, device="cuda"),
        )

    def _register_ops(self) -> None:
        from ltx_kernels.ops import rms_fma as _rms_fma_ops  # noqa: F401, PLC0415
