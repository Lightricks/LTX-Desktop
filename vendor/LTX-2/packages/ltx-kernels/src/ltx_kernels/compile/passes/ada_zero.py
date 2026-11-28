"""Pass that rewrites the eager AdaZero composite to a single custom op."""

from typing import ClassVar

import torch
from torch.fx.passes.utils.matcher_utils import InternalMatch

from ltx_kernels.compile.subgraph_rewrite_pass import SubgraphRewritePass, is_cuda_bf16, static_dim


class AdaZeroPass(SubgraphRewritePass):
    """Rewrites the bf16 AdaZero composite::
        normed = F.rms_norm(x, (H,), None, eps)
        return normed * (1 + scale) + shift
    to a single ``ltx_kernels::ada_zero(x, scale, shift, eps)`` call. Matches
    the inlined ``rms_norm(...) * (1 + scale) + shift`` sites in
    :class:`BasicAVTransformerBlock.forward` (MSA, A-V cross-attention, MLP).
    PostSA residual + RMSNorm (+ query modulation when AdaLN is on) is a
    larger pattern claimed first by :class:`RMSFmaAdaLNPass` /
    :class:`RMSFmaPass`.
    Uses ``ignore_literals=False`` because the pattern contains a bare
    ``1 + scale`` whose literal ``1`` would be wildcarded under the default
    setting and confuse placeholder binding. Both ``hidden_dim`` and ``eps``
    are accordingly baked as literals in the captured pattern; the
    constructor takes them so callers register one instance per distinct
    ``(hidden_dim, eps)`` pair encountered in the user graph (LTX-2.3:
    video ``hidden_dim=4096``, audio ``hidden_dim=2048``, both
    ``eps=1e-6``).
    """

    ignore_literals: ClassVar[bool] = False

    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim

        # Bake ``eps`` into pattern/replacement via closures. Dynamo bakes
        # Python scalar function args as graph literals (not placeholders),
        # so per-instance pattern_gms with the right literal are the only
        # way to match a particular ``eps`` value.
        def pattern_fn(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
            normed = torch.nn.functional.rms_norm(x, (x.shape[-1],), None, eps)
            return normed * (1 + scale) + shift

        def replacement_fn(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
            return torch.ops.ltx_kernels.ada_zero.default(x, scale, shift, eps)

        # Shadow the ``pattern_fn`` / ``replacement_fn`` ``ClassVar``
        # declarations on the base class with these instance-bound closures.
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
        # of the affine tensors too.
        hidden = static_dim(x.shape[-1])
        return (
            hidden is not None
            and hidden & (hidden - 1) == 0
            and static_dim(scale.shape[-1]) == hidden
            and static_dim(shift.shape[-1]) == hidden
        )

    def example_inputs(self) -> tuple:
        # ``batch`` and ``seq`` are immaterial -- the pattern doesn't bake
        # them as literals. Only ``hidden_dim`` shows up in the captured
        # ``F.rms_norm(x, (hidden_dim,), ...)``.
        batch, seq = 2, 64
        hidden = self._hidden_dim
        return (
            torch.empty(batch, seq, hidden, dtype=torch.bfloat16, device="cuda"),
            torch.empty(batch, 1, hidden, dtype=torch.bfloat16, device="cuda"),
            torch.empty(batch, 1, hidden, dtype=torch.bfloat16, device="cuda"),
        )

    def _register_ops(self) -> None:
        from ltx_kernels.ops import ada_zero as _ada_zero_ops  # noqa: F401, PLC0415
