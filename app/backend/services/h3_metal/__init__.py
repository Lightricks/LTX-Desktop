"""M5-native INT8 linear layers for the managed ComfyUI MiniMax H3 transformer."""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

NODE_CLASS_MAPPINGS = {}
_LOG = logging.getLogger("spacespeak.h3_metal")
_TAG = "[SpaceSpeak H3 Metal]"
_SHADER = Path(__file__).with_name("int8_gemm.metal")


def hardware_reason() -> str | None:
    if os.environ.get("LTX_H3_M5_INT8") == "0":
        return "disabled by LTX_H3_M5_INT8=0"
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return "requires a native Apple silicon process"
    major = platform.mac_ver()[0].partition(".")[0]
    if not major.isdecimal() or int(major) < 26:
        return "requires macOS 26 or newer"
    try:
        chip = subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL, text=True, timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "could not identify the chip"
    if not re.fullmatch(r"Apple M5(?: Pro| Max| Ultra)?", chip):
        return f"chip {chip!r} is outside the supported M5 family"
    return None


class NativeInt8:
    def __init__(self):
        import torch
        from comfy_kitchen.backends.eager.quantization import quantize_and_rotate_rowwise
        from comfy_kitchen.tensor.int8_utils import _build_hadamard

        self._hadamard = _build_hadamard
        self._quantize = quantize_and_rotate_rowwise
        self._shader = torch.mps.compile_shader(_SHADER.read_text(encoding="ascii"))
        self._kernels = {
            torch.bfloat16: self._shader.h3_bf16,
            torch.float16: self._shader.h3_fp16,
            torch.float32: self._shader.h3_fp32,
        }
        self.native_calls = 0
        self.enabled = True
        self._last_prompt = None

    def _gemm(self, a, w, rows, cols, bias, out):
        # The first six arguments are buffers. PyTorch's Metal binding sends
        # Python integer scalar arguments as 64-bit unless explicitly cast.
        m, k = a.shape
        n = w.shape[0]
        self._kernels[out.dtype](
            a, w, out, rows, cols, bias if bias is not None else out,
            m, n, k, cols.numel(), int(bias is not None),
            threads=(n, (m + 31) // 32), group_size=(128, 1), arg_casts="int32",
        )

    def linear(self, x, w, scale, bias, group):
        import torch

        n, k = w.shape
        flat = x.reshape(-1, k)
        out = torch.empty((flat.shape[0], n), device=x.device, dtype=x.dtype)
        h = self._hadamard(group, x.device, x.dtype)
        scale = scale.to(device=x.device, dtype=torch.float32).reshape(-1).contiguous()
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype).contiguous()
        # Keep transient rotated and quantized activations bounded. Only the
        # final output spans the complete video sequence, as in a stock linear.
        for start in range(0, flat.shape[0], 1024):
            block = flat[start:start + 1024]
            a, row_scale = self._quantize(block, h, group)
            self._gemm(a.contiguous(), w, row_scale.contiguous(), scale, bias, out[start:start + 1024])
        return out.reshape(*x.shape[:-1], n)

    def self_test(self):
        """Check the actual Metal pipeline, including layouts, tails, scales and bias."""
        import torch

        gen = torch.Generator(device="cpu").manual_seed(831)
        # Distinct M/N/K dimensions and partial M detect transposes and edge
        # loads. Nonzero product sums fit in FP32 exactly in the integer test.
        a = torch.randint(-127, 128, (65, 768), generator=gen, dtype=torch.int8)
        w = torch.randint(-127, 128, (256, 768), generator=gen, dtype=torch.int8)
        product = (a.int() @ w.int().T).float()
        a_mps, w_mps = a.to("mps"), w.to("mps")
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            plain = dtype == torch.float32
            # Powers of two keep rescaling exact and avoid rejecting correct
            # compiled kernels solely for reassociating float32 products.
            rows = torch.ones((65, 1)) if plain else (2.0 ** (-6 - torch.arange(65) % 3)).view(-1, 1)
            cols = torch.ones(1) if plain else 2.0 ** (-3 - torch.arange(256) % 4)
            bias = None if plain else ((torch.arange(256) - 128) / 64).to(dtype)
            want = (product * (rows * cols)).to(dtype)
            if bias is not None:
                want = want + bias
            out = torch.empty(want.shape, dtype=dtype, device="mps")
            self._gemm(a_mps, w_mps, rows.to("mps"), cols.to("mps"),
                       bias.to("mps") if bias is not None else None, out)
            torch.testing.assert_close(out.cpu(), want, rtol=0, atol=0)

        # One-hot inputs have an exactly representable regular Hadamard
        # rotation (group size 256). Compare the complete activation path
        # against a floating-point dequantized CPU reference.
        x = torch.zeros(65, 768, dtype=torch.bfloat16)
        x[torch.arange(65), torch.arange(65) * 11] = 1
        h = self._hadamard(256, "cpu", torch.float32)
        unrotated = (w.float().reshape(-1, 256) @ h).reshape(w.shape)
        want = (x.float() @ unrotated.T) * 0.0625
        actual = self.linear(x.to("mps"), w_mps, torch.tensor([0.0625]), None, 256)
        torch.testing.assert_close(actual.cpu().float(), want, rtol=0.015, atol=0.03125)

    def record_dispatch(self, prompt):
        import torch

        if self.native_calls == 0 or self._last_prompt != prompt:
            # Synchronize once per prompt to distinguish an actual completed
            # dispatch from a kernel that merely loaded or was queued.
            torch.mps.synchronize()
            _LOG.info("%s Native INT8 active for H3 transformer (prompt %s).", _TAG, prompt or "standalone")
            self._last_prompt = prompt
        self.native_calls += 1


@functools.lru_cache(maxsize=1)
def get_runtime():
    reason = hardware_reason()
    if reason:
        _LOG.info("%s Standard MPS: %s.", _TAG, reason)
        return None
    try:
        import torch

        if not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS unavailable")
        with torch.inference_mode():
            runtime = NativeInt8()
            runtime.self_test()
    except Exception as exc:
        _LOG.warning("%s Standard MPS: native kernel self-test failed: %s", _TAG, exc)
        return None
    _LOG.info("%s Self-test passed. Waiting for an eligible H3 transformer layer.", _TAG)
    return runtime


def wrap_operations(operations, runtime):
    """Override only MiniMaxH3Model's mixed-precision Linear instances."""
    import torch
    from comfy import model_management, ops
    from comfy_execution.utils import get_executing_context

    if operations is None or not issubclass(operations.Linear, ops.MixedPrecisionOp):
        return operations
    if getattr(operations, "_full_precision_mm", False):
        return operations

    class H3Operations(operations):
        class Linear(operations.Linear):
            def forward_comfy_cast_weights(self, input, *args, **kwargs):
                w = self.weight
                if (not runtime.enabled or input.device.type != "mps"
                        or input.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                        or isinstance(input, ops.QuantizedTensor) or input.ndim < 2 or not input.numel()
                        or input.requires_grad or model_management.in_training
                        or getattr(self, "_full_precision_mm_config", False)
                        or getattr(self, "comfy_force_cast_weights", False)
                        or self.weight_function or self.bias_function
                        or not isinstance(w, ops.QuantizedTensor) or w._layout_cls != "TensorWiseINT8Layout"
                        or not w._params.convrot or w._params.transposed):
                    return super().forward_comfy_cast_weights(input, *args, **kwargs)

                n, k = w._qdata.shape
                group = w._params.convrot_groupsize
                if (w._qdata.dtype == torch.int8 and w._qdata.is_contiguous() and n % 128 == 0
                        and 0 < k <= 65536 and k == input.shape[-1] and k % 16 == 0
                        and group in (4, 16, 64, 256, 1024) and k % group == 0
                        and w._params.scale.numel() in (1, n)):
                    # Preserve the current (possibly eagerly LoRA-patched)
                    # quantized weight and Comfy's offload lifetime. Live
                    # weight/bias functions above use Comfy's full stock path.
                    with ops.CastBiasWeightContext(
                        self, dtype=w.dtype, device=input.device, bias_dtype=input.dtype,
                        offloadable=True, compute_dtype=input.dtype, want_requant=True,
                    ) as (weight, bias):
                        if isinstance(weight, ops.QuantizedTensor):
                            try:
                                output = runtime.linear(input, weight._qdata, weight._params.scale, bias, group)
                                context = get_executing_context()
                                runtime.record_dispatch(context.prompt_id if context else None)
                                return output
                            except (RuntimeError, ValueError, TypeError) as exc:
                                runtime.enabled = False
                                _LOG.warning("%s Standard MPS for remaining layers: %s", _TAG, exc)
                return super().forward_comfy_cast_weights(input, *args, **kwargs)

    return H3Operations


def _install():
    from comfy.ldm.minimax.model import MiniMaxH3Model

    original = MiniMaxH3Model.__init__
    if getattr(original, "_spacespeak_m5", False):
        return

    @functools.wraps(original)
    def initialize(model, *args, **kwargs):
        runtime = get_runtime()
        if runtime is not None and runtime.enabled and "operations" in kwargs:
            kwargs["operations"] = wrap_operations(kwargs["operations"], runtime)
        original(model, *args, **kwargs)

    initialize._spacespeak_m5 = True
    MiniMaxH3Model.__init__ = initialize


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if sys.argv[1:] != ["--self-test"]:
        sys.exit("Usage: python __init__.py --self-test")
    if reason := hardware_reason():
        print("SKIP:", reason)
        sys.exit(77)
    sys.exit(0 if get_runtime() is not None else 1)
elif sys.platform == "darwin":
    _install()
