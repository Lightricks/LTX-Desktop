"""Hide Triton from this process (used when Windows Smart App Control is on)."""

from __future__ import annotations

import sys


def block_triton_import() -> bool:
    """Make ``import triton`` raise ImportError for the rest of this process.

    SageAttention is not the only Triton user: the ltx-core video VAE (swiglu,
    row_rms_norm, stage5_tail, loader kernels) launches Triton kernels whenever Triton
    and CUDA are present. Each of those modules guards its import with
    ``except ImportError`` and then takes a plain PyTorch path, so hiding Triton keeps
    them off the run-time compiled ``.pyd`` files that Smart App Control blocks.

    Must run before anything imports Triton. Returns False when Triton was already
    imported, because it cannot be undone then.
    """
    modules: dict[str, object] = sys.modules  # type: ignore[assignment]
    if "triton" in modules:
        return modules["triton"] is None
    modules["triton"] = None
    return True
