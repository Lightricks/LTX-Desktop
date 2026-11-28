"""Pillow's 8-bit Lanczos-x2 resample as torch ops, so it runs on the frames' own device.
Pillow resizes in two separable passes of fixed-point Lanczos-3 weights (22 fractional bits),
rounding and clamping to a byte after each pass. The same integer arithmetic in float64 is exact
(every product and partial sum stays below 2**53), so the bytes match Pillow's on any device.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from ltx_core.devices import upload_from_host

FIXED_POINT_BITS = 22
LANCZOS_SUPPORT = 3.0


def rgb_to_uint8_fhwc(frames: torch.Tensor) -> torch.Tensor:
    """``(F, H, W, C)`` RGB in ``[0, 1]`` to bytes, rounding the way the Pillow path does on the host."""
    return (frames.detach().float().clamp(0, 1) * 255.0).round().to(torch.uint8)


def _lanczos3(x: float) -> float:
    if not -LANCZOS_SUPPORT <= x < LANCZOS_SUPPORT:
        return 0.0

    def sinc(v: float) -> float:
        if v == 0.0:
            return 1.0
        v *= math.pi
        return math.sin(v) / v

    return sinc(x) * sinc(x / LANCZOS_SUPPORT)


@lru_cache(maxsize=8)
def _lanczos_weights(in_size: int, out_size: int) -> torch.Tensor:
    """Pillow's fixed-point Lanczos-3 taps as a ``(2, out_size, taps)`` float64 tensor.
    ``[0]`` is each tap's source index and ``[1]`` its weight, normalised per output row and
    rounded to ``FIXED_POINT_BITS``. A row with fewer taps than the widest pads with weight 0
    on index 0. Indices are small integers, exact in float64, so one tensor carries both.
    """
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = LANCZOS_SUPPORT * filterscale
    ss = 1.0 / filterscale
    rows: list[tuple[int, list[float]]] = []
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size) - xmin
        k = [_lanczos3((x + xmin - center + 0.5) * ss) for x in range(xmax)]
        total = sum(k)
        fixed = []
        for w in k:
            weight = w / total if total != 0.0 else w
            v = weight * (1 << FIXED_POINT_BITS)
            fixed.append(float(int(-0.5 + v) if weight < 0 else int(0.5 + v)))
        rows.append((xmin, fixed))
    taps = torch.zeros(2, out_size, max(len(fixed) for _, fixed in rows), dtype=torch.float64)
    for xx, (xmin, fixed) in enumerate(rows):
        taps[0, xx, : len(fixed)] = torch.arange(xmin, xmin + len(fixed), dtype=torch.float64)
        taps[1, xx, : len(fixed)] = torch.tensor(fixed, dtype=torch.float64)
    return taps


def _fixed_point_pass(v: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Resample the last axis of ``v`` with taps ``k``: fixed-point accumulate, rounding shift, byte clamp.
    One gather and multiply-add per tap; every product and partial sum is an integer below
    2**53, so the float64 sum is exact in any order.
    """
    index = k[0].long()
    acc = torch.full((*v.shape[:-1], k.shape[1]), float(1 << (FIXED_POINT_BITS - 1)), dtype=v.dtype, device=v.device)
    for tap in range(k.shape[2]):
        acc.addcmul_(v[..., index[:, tap]], k[1, :, tap])
    return torch.floor(acc / float(1 << FIXED_POINT_BITS)).clamp_(0.0, 255.0)


@lru_cache(maxsize=8)
def _device_lanczos_weights(in_size: int, out_size: int, device: torch.device) -> torch.Tensor:
    """:func:`_lanczos_weights` on ``device``, uploaded once without waiting for the device."""
    return upload_from_host(_lanczos_weights(in_size, out_size), device)


def lanczos_x2_u8_hwc(frame: torch.Tensor) -> torch.Tensor:
    """``(H, W, C)`` uint8 to ``(2H, 2W, C)`` uint8, byte-identical to ``Image.resize((2W, 2H), LANCZOS)``."""
    height, width, _ = frame.shape
    kx = _device_lanczos_weights(width, 2 * width, frame.device)
    ky = _device_lanczos_weights(height, 2 * height, frame.device)
    x = frame.to(torch.float64).permute(2, 0, 1)
    x = _fixed_point_pass(x, kx)
    x = _fixed_point_pass(x.transpose(1, 2), ky).transpose(1, 2)
    return x.permute(1, 2, 0).to(torch.uint8)
