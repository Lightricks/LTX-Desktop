"""Monkey-patch: replace safetensors safe_open with Python mmap loader.

safetensors' safe_open uses torch.UntypedStorage.from_file internally, which
causes access violations (STATUS_ACCESS_VIOLATION / 0xC0000005) under memory
pressure on Windows. The crash occurs in UntypedStorage.__getitem__ when the
OS cannot service an mmap page fault.

This patch replaces the load method of SafetensorsStateDictLoader with a
custom implementation that uses Python's mmap module + torch.frombuffer,
which handles memory pressure gracefully (raises a Python exception instead
of crashing the process).

Each tensor's storage holds a reference to the mmap via _ltx_tensor_mmap_refs,
ensuring the mapping stays alive exactly as long as the tensor does.

Remove this patch once safetensors / PyTorch fix the underlying issue.

Usage:
    import services.patches.safetensors_loader_fix  # noqa: F401
"""

from __future__ import annotations

import json
import mmap
import os
import struct
import warnings
from typing import Any

import torch

from ltx_core.loader.sft_loader import SafetensorsStateDictLoader
from ltx_core.loader.primitives import StateDict
from ltx_core.loader.sd_ops import SDOps

_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def _load_safetensors_direct(shard_path: str) -> dict[str, torch.Tensor]:
    """Load tensors from a safetensors file using Python mmap + torch.frombuffer."""
    f = open(shard_path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    mv = memoryview(mm)

    header_size = struct.unpack("<Q", mv[:8])[0]
    header = json.loads(bytes(mv[8 : 8 + header_size]).decode("utf-8"))
    data_offset = 8 + header_size
    mv_data = mv[data_offset:]

    mmap_refs = (mm, mv, mv_data, f)

    tensors: dict[str, torch.Tensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        if start == end:
            tensors[name] = torch.empty(info["shape"], dtype=_DTYPES[info["dtype"]])
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given buffer is not writable")
                tensor = torch.frombuffer(
                    mv_data[start:end], dtype=_DTYPES[info["dtype"]]
                ).reshape(info["shape"])
                storage = tensor.untyped_storage()
                setattr(storage, "_ltx_tensor_mmap_refs", mmap_refs)
                tensors[name] = tensor

    return tensors


def _patched_load(
    self: SafetensorsStateDictLoader,
    path: str | list[str],
    sd_ops: SDOps,
    device: torch.device | None = None,
) -> StateDict:
    sd: dict[str, Any] = {}
    size = 0
    dtype: set[torch.dtype] = set()
    device = device or torch.device("cpu")
    model_paths = path if isinstance(path, list) else [path]
    for shard_path in model_paths:
        tensors = _load_safetensors_direct(shard_path)
        for name, value in tensors.items():
            expected_name = name if sd_ops is None else sd_ops.apply_to_key(name)
            if expected_name is None:
                continue
            # copy=False is not valid when moving from CPU mmap storage to a
            # non-CPU device (e.g. MPS): the devices must match for in-place
            # storage reuse.  Always copy when crossing device boundaries.
            # non_blocking=True is also unsafe on MPS with mmap-backed tensors:
            # MPS has no equivalent of CUDA's pinned-memory DMA, so the async
            # transfer can read freed/remapped mmap pages → segfault.
            needs_copy = value.device != device
            non_blocking = device.type == "cuda"
            value = value.to(device=device, non_blocking=non_blocking, copy=needs_copy)
            key_value_pairs = ((expected_name, value),)
            if sd_ops is not None:
                key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
            for key, value in key_value_pairs:
                size += value.nbytes
                dtype.add(value.dtype)
                sd[key] = value

    return StateDict(sd=sd, device=device, size=size, dtype=dtype)


# Apply patch — assert the original method exists so a rename doesn't silently regress.
assert hasattr(SafetensorsStateDictLoader, "load") and callable(getattr(SafetensorsStateDictLoader, "load")), \
    "SafetensorsStateDictLoader.load not found — was it renamed? The safetensors_loader_fix patch needs updating."
SafetensorsStateDictLoader.load = _patched_load  # type: ignore[assignment]
