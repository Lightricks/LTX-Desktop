"""Binomial bit match of decoded bits against a baked payload.
Does not decide ``detected`` — each marker applies its own call.
"""

from __future__ import annotations

import torch
from scipy import stats


def score_bits(decoded: torch.Tensor, payload: torch.Tensor) -> tuple[float, float]:
    """Binomial bit match of ``decoded`` against ``payload``.
    Args:
        decoded: ``{0, 1}`` bits (any shape). Flattened before the test.
        payload: ``{0, 1}`` bits (any shape). Flattened length must match ``decoded``.
    Returns:
        ``(bit_accuracy, p_value)``. ``p_value`` is a one-tailed binomial test
        (``alternative="greater"``) against a chance match of ``p=0.5``.
    Raises:
        ValueError: If the flattened lengths differ.
    """
    decoded = decoded.reshape(-1).to(dtype=torch.float32)
    expected = payload.reshape(-1).to(device=decoded.device, dtype=torch.float32)
    if decoded.numel() != expected.numel():
        raise ValueError(f"detector bits {decoded.numel()} != payload bits {expected.numel()}")
    matches = int((decoded == expected).sum().item())
    nbits = int(expected.numel())
    bit_accuracy = matches / nbits
    p_value = float(stats.binomtest(matches, nbits, 0.5, alternative="greater").pvalue)
    return bit_accuracy, p_value
