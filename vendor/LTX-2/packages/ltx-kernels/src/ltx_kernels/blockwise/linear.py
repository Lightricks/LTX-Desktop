"""Blockwise FP8/FP6 quantized linear layers.
Both store quantized weights with per-128-block FP32 scales and quantize the
bf16 activation through ``ltx_kernels::blockwise_quantize`` before dispatching to
the ``blockwise::fp8_gemm`` op (:mod:`.ops.gemm`). ``forward`` always quantizes
its bf16 input; under the compile backend a pass fuses that quantize node into
the preceding op. FP6 stores 6-bit-packed weights and unpacks them to fp8 on the
fly.
"""

import torch
from torch import nn

from ltx_kernels.arch import get_device_arch
from ltx_kernels.blockwise.ops.fp6_pack import fp6_pack_tensor, fp6_unpack_tensor
from ltx_kernels.blockwise.ops.gemm import blockwise_fp8_gemm
from ltx_kernels.blockwise.ops.quantize import (
    blockwise_quantize,
    fp6_blockwise_quantize_weights_torch,
    fp8_blockwise_quantize_weights_torch,
)

_BLOCK_SIZE = 128


def _blockwise_fp8_matmul(x, fp8_weight, weight_scale, bias):
    """Quantize the bf16 activation and run the FP8 GEMM.
    Always quantizes ``x``. Under the compile backend, FuseTrailingQuantize
    rewrites the ``blockwise_quantize`` node into the preceding op's quantized
    twin wherever that producer feeds only FP8 consumers -- so there is no
    runtime tuple-input path to handle here.
    """
    fp8_x, scales = blockwise_quantize(x)
    b, n, _ = fp8_x.shape
    out_h = fp8_weight.shape[0]
    flat = (fp8_x.view(-1, fp8_x.shape[-1]), scales)
    use_fast_accum = (not fp8_x.is_cuda) or get_device_arch() != "blackwell"
    out = blockwise_fp8_gemm(flat, [fp8_weight, weight_scale], bias, use_fast_accum)
    return out.view(b, n, out_h)


class BlockwiseFP8Linear(nn.Module):
    """Drop-in ``nn.Linear`` replacement with FP8 weights + bf16 activations.
    ``forward(x)`` takes bf16 ``(B, T, H)`` and returns bf16
    ``(B, T, out_features)``; the activation is quantized internally.
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.ones(out_features // _BLOCK_SIZE, in_features // _BLOCK_SIZE, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        if bias:
            # FP32 to match the SM89/SM90 GEMM bias contract (the kernels take float*).
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        bias = self.bias.data if self.bias is not None else None
        return _blockwise_fp8_matmul(x, self.weight, self.weight_scale, bias)

    @classmethod
    def from_linear(cls, linear, transform_weights=True):
        """Construct an FP8 module from a regular ``nn.Linear``.
        ``transform_weights=False`` allocates empty quantized buffers (the caller
        loads ``.weight`` / ``.weight_scale`` via state-dict); ``True``
        blockwise-quantizes the source linear's weight.
        """
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
        )
        if transform_weights:
            w_fp8, w_scales = fp8_blockwise_quantize_weights_torch(linear.weight.data.cuda())
        else:
            w_fp8 = torch.empty(
                linear.out_features, linear.in_features, device=linear.weight.device, dtype=torch.float8_e4m3fn
            )
            w_scales = torch.ones(
                w_fp8.shape[0] // _BLOCK_SIZE,
                w_fp8.shape[1] // _BLOCK_SIZE,
                device=w_fp8.device,
                dtype=torch.float32,
            )
        layer.weight.data = w_fp8
        layer.weight_scale.data = w_scales
        if linear.bias is not None:
            layer.bias.data = linear.bias.data.to(torch.float32)
        return layer


class BlockwiseFP6Linear(nn.Module):
    """``nn.Linear`` replacement with 6-bit-packed weights, unpacked to fp8 per call.
    FP6 is weight-only: the activation path is identical to FP8.
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        assert in_features % 8 == 0, f"in_features must be divisible by 8 for fp6_pack, got {in_features}"
        assert in_features % 128 == 0, f"in_features must be divisible by 128, got {in_features}"
        assert out_features % 128 == 0, f"out_features must be divisible by 128, got {out_features}"
        self.in_features = in_features
        self.out_features = out_features
        # Packed weight: [out_features, (in_features // 4) * 3] uint8.
        self.weight = nn.Parameter(
            torch.empty(out_features, (in_features // 4) * 3, device=device, dtype=torch.uint8),
            requires_grad=False,
        )
        # Scales are based on the unpacked dimensions.
        self.weight_scale = nn.Parameter(
            torch.ones(out_features // _BLOCK_SIZE, in_features // _BLOCK_SIZE, device=device, dtype=torch.float32),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    @property
    def fp8weight(self):
        """Unpack the 6-bit weights to float8_e4m3fn on the fly.
        Uses the ``ltx_kernels::fp6_unpack`` custom op so the call survives
        torch.compile without a graph break.
        """
        unpacked = fp6_unpack_tensor(self.weight, self.in_features)
        return unpacked.view(torch.float8_e4m3fn)

    def forward(self, x):
        bias = self.bias.data if self.bias is not None else None
        return _blockwise_fp8_matmul(x, self.fp8weight, self.weight_scale, bias)

    @classmethod
    def from_linear(cls, linear, transform_weights=True):
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
        )
        if transform_weights:
            # Quantize to FP6 (FP8 with a restricted range), then pack [out, in] -> [out, in*3/4].
            w_fp6, w_scales = fp6_blockwise_quantize_weights_torch(linear.weight.data.cuda())
            layer.weight.data = fp6_pack_tensor(w_fp6.view(torch.uint8))
            layer.weight_scale.data = w_scales
        else:
            layer.weight.data = torch.empty(
                linear.out_features, (linear.in_features // 4) * 3, device=linear.weight.device, dtype=torch.uint8
            )
            layer.weight_scale.data = torch.ones(
                linear.out_features // _BLOCK_SIZE,
                linear.in_features // _BLOCK_SIZE,
                device=linear.weight.device,
                dtype=torch.float32,
            )
        if linear.bias is not None:
            layer.bias.data = linear.bias.data.to(torch.float32)
        return layer
