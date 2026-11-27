"""Accelerator backend detection.

ROCm PyTorch reports itself through the same `torch.cuda` interface as NVIDIA
CUDA builds (`torch.cuda.is_available()`, `device.type == "cuda"`, etc.), so
code that checks `device.type == "cuda"` to mean "this is an NVIDIA GPU" is
wrong under ROCm. `torch.version.hip` is the actual discriminator: it is set
on ROCm builds and `None` on CUDA builds.

Originally contributed by boxwrench in
https://github.com/Lightricks/LTX-Desktop/pull/160
"""

from __future__ import annotations

from typing import Literal

import torch

AcceleratorBackend = Literal["rocm", "cuda", "mps", "cpu"]


def accelerator_backend() -> AcceleratorBackend:
    if torch.cuda.is_available():
        return "rocm" if getattr(torch.version, "hip", None) else "cuda"

    if mps_available():
        return "mps"

    return "cpu"


def verify_rocm_execution(*, require_gfx1151: bool = False) -> str:
    """Execute two BF16 operations used by the model on the first HIP device."""
    if not getattr(torch.version, "hip", None):
        raise RuntimeError(
            f"The AMD edition requires ROCm PyTorch; this runtime has torch {torch.__version__} "
            "without HIP. Repair the AMD runtime and restart LTX Desktop - AMD 395 MAX."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "ROCm PyTorch cannot see an AMD GPU. Check the AMD graphics driver and install "
            "the matching Windows ROCm runtime for your Radeon, then restart LTX Desktop - AMD 395 MAX."
        )
    device = torch.device("cuda", 0)  # PyTorch exposes HIP devices through the CUDA interface.
    name = torch.cuda.get_device_name(device)
    if require_gfx1151:
        arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(":")[0]
        if (arch and arch != "gfx1151") or (not arch and "8060s" not in name.lower()):
            raise RuntimeError(f"The selected HIP adapter is {name} ({arch or 'unknown architecture'}). "
                               "Rerun the Windows installer to select the gfx1151 device.")
    try:
        with torch.inference_mode():
            tensor = torch.ones((1, 2, 16, 64), device=device, dtype=torch.bfloat16)
            product = tensor @ tensor.transpose(-1, -2)
            attention = torch.nn.functional.scaled_dot_product_attention(tensor, tensor, tensor)
            torch.cuda.synchronize(device)
            if product.device != device or attention.device != device:
                raise RuntimeError("a required operation returned a tensor on a different device")
            if not bool((product == 64).all().item()) or not bool((attention == 1).all().item()):
                raise RuntimeError("the Radeon returned an incorrect BF16 result")
    except RuntimeError as exc:
        raise RuntimeError(
            f"ROCm detected {name} but the BF16 GPU execution check failed. "
            f"Verify the AMD driver and GPU-specific ROCm wheels. Details: {exc}"
        ) from exc
    return name


def mps_available() -> bool:
    """Whether the MPS backend exists and is usable. False everywhere off-Mac."""
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and bool(mps.is_available())


def accelerator_supports_fp8(device_index: int = 0) -> bool:
    """Whether a CUDA device can run the fp8 transformer path.

    True only on NVIDIA CUDA with compute capability >= 8.9. ROCm, MPS, CPU,
    pre-Ada NVIDIA, and any probe failure all fail closed to False: without
    fp8 the full bf16 transformer cannot stay resident at the full-loading
    floor, so callers must take the streaming / unquantized path. Device 0 by
    default, matching the total- and free-VRAM probes.
    """
    if accelerator_backend() != "cuda":
        return False
    from inference.device import device_supports_fp8

    return device_supports_fp8(torch.device("cuda", device_index))
