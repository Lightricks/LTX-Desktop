"""BF16 linear projection with a fused approximate GELU epilogue."""

import torch


@torch.library.custom_op("ltx_kernels::linear_gelu", mutates_args=(), device_types="cuda")
def linear_gelu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Apply a BF16 linear projection and tanh-approximate GELU."""
    batch_shape = x.shape[:-1]
    out = torch.ops.aten._addmm_activation.default(
        bias,
        x.flatten(0, -2),
        weight.t(),
        use_gelu=True,
    )
    return out.unflatten(0, batch_shape)


@linear_gelu.register_fake
def _linear_gelu_fake(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
    return x.new_empty((*x.shape[:-1], weight.shape[0]))
