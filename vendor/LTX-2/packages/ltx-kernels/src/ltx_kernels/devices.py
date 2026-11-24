"""Host-to-device transfer helpers shared by the kernels.
``ltx_kernels`` does not depend on ``ltx_core``, so it keeps its own counterpart of
``ltx_core.devices.upload_from_host``.
"""

from __future__ import annotations

import torch


def upload_from_host(host: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Copy a host tensor to ``device`` without waiting for the device.
    A pageable host-to-device copy synchronizes the stream; a pinned one is ordered on it instead,
    so ``host`` may be modified as soon as this returns. A tensor that is already on a device, or a
    target that is not CUDA, is moved with ``.to``.
    """
    if host.device.type != "cpu" or device.type != "cuda":
        return host.to(device)
    return host.pin_memory().to(device, non_blocking=True)
