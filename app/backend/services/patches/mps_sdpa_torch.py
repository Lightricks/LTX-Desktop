"""Route Diffusers/Z-Image ``F.scaled_dot_product_attention`` through mps-sdpa.

Video already uses mps-sdpa via ltx-core. Z-Image goes through Diffusers native
attention, which calls ``torch.nn.functional.scaled_dot_product_attention``.
Stock MPS SDPA materializes S×S score matrices and can jetsam / freeze a Mac
(Windows is fine: SageAttention / fused CUDA SDPA).

mps-sdpa's stock fallback calls ``F.sdpa`` again. A contextvar sends those
nested calls to the original implementation so we do not recurse.

Do not catch ``sdpa_opt`` failures and fall back to stock MPS SDPA — that is
the freeze path. Let the generation fail instead.

Remove once Diffusers native attention on MPS uses a fused / mps-sdpa backend.

Usage:
    import services.patches.mps_sdpa_torch as mps_sdpa_torch
    mps_sdpa_torch.install()
"""

from __future__ import annotations

import contextvars
import logging
import sys
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_installed = False
_in_mps_sdpa: contextvars.ContextVar[bool] = contextvars.ContextVar("in_mps_sdpa", default=False)
_original_sdpa = F.scaled_dot_product_attention
_sdpa_opt: Callable[..., Any] | None = None


def _call_original(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    return _original_sdpa(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        **kwargs,
    )


def _patched_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    if _in_mps_sdpa.get() or getattr(query, "device", None) is None or query.device.type != "mps":
        return _call_original(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, kwargs=kwargs
        )
    # GQA is CUDA-only in this wrapper; mps-sdpa.sdpa_opt does not take enable_gqa.
    if kwargs.get("enable_gqa"):
        return _call_original(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, kwargs=kwargs
        )

    opt = _sdpa_opt
    if opt is None:
        raise RuntimeError("mps-sdpa F.sdpa patch called before install()")

    token = _in_mps_sdpa.set(True)
    try:
        # Only pass the kwargs sdpa_opt documents. Extra F.sdpa kwargs would
        # TypeError; do not catch that and run stock MPS SDPA (the freeze path).
        return opt(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
    finally:
        _in_mps_sdpa.reset(token)


def install() -> None:
    """Patch ``F.scaled_dot_product_attention`` on Darwin. Idempotent. No-op elsewhere."""
    global _installed, _original_sdpa, _sdpa_opt
    if _installed or sys.platform != "darwin":
        return

    try:
        from mps_sdpa import sdpa_opt  # noqa: PLC0415  # type: ignore[reportMissingModuleSource]
    except ImportError:
        logger.warning("mps-sdpa not installed; Z-Image will use stock MPS SDPA")
        return

    _sdpa_opt = sdpa_opt
    _original_sdpa = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _patched_sdpa
    _installed = True
    logger.info("mps-sdpa enabled for torch SDPA (Z-Image / Diffusers native attention)")
