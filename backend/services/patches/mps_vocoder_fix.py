"""Monkey-patch: fix VocoderWithBWE dtype mismatch on MPS.

VocoderWithBWE.forward() uses torch.autocast(dtype=torch.float32) to upcast
bf16 weights per-op during inference.  MPS does not support float32 autocast:

    UserWarning: In MPS autocast, but the target dtype is not supported.
    Disabling autocast. MPS Autocast only supports dtypes of torch.bfloat16,
    torch.float16 currently.

When autocast is disabled, the model weights remain in bfloat16 but the input
is cast to float32 (mel_spec.float()), causing:

    RuntimeError: Input type (float) and bias type (c10::BFloat16) should be
    the same

On MPS we temporarily convert the vocoder submodule to float32 before the
forward pass and restore it to bfloat16 afterward.  This is the same approach
the upstream code explicitly benchmarked and rejected for CUDA ("+324 MB peak
VRAM, 149 ms") — on MPS the memory penalty is acceptable because unified
memory is not subject to the same VRAM constraints.

Remove this patch once ltx-core adds MPS awareness to VocoderWithBWE.forward.

Usage:
    import services.patches.mps_vocoder_fix  # noqa: F401
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn

from ltx_core.model.audio_vae.vocoder import VocoderWithBWE  # type: ignore[reportMissingImports]

_original_forward = VocoderWithBWE.forward


def _patched_forward(self: VocoderWithBWE, mel_spec: torch.Tensor) -> torch.Tensor:
    if mel_spec.device.type != "mps":
        return _original_forward(self, mel_spec)

    # MPS does not support float32 autocast — convert weights temporarily.
    # Suppress the expected "MPS Autocast only supports dtypes of torch.bfloat16"
    # warning: _original_forward still contains the autocast block but our
    # self.float() conversion makes the weights match the float32 input, so
    # the autocast no-op is harmless.
    original_dtype = next(self.parameters()).dtype
    try:
        self.float()  # convert all weights to float32
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="In MPS autocast", category=UserWarning)
            return _original_forward(self, mel_spec)
    finally:
        self.to(dtype=original_dtype)  # restore original dtype (bfloat16)


assert hasattr(VocoderWithBWE, "forward") and callable(VocoderWithBWE.forward), (
    "VocoderWithBWE.forward not found — was it renamed? "
    "The mps_vocoder_fix patch needs updating."
)
VocoderWithBWE.forward = _patched_forward  # type: ignore[method-assign]
