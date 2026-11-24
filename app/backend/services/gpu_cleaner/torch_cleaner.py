"""GPU cleanup helper service."""

from __future__ import annotations

import gc

import torch

from services.services_utils import empty_device_cache


class TorchCleaner:
    """Wraps GPU memory cleanup operations."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self._device = device

    def cleanup(self) -> None:
        # gc first: unreachable tensors must release their storage back to
        # the caching allocator before empty_cache can return it to the
        # driver. Reversed, freed blocks stay cached and mem_get_info counts
        # them as used — an evict-retry re-sample would see no freed VRAM.
        gc.collect()
        empty_device_cache(self._device)
