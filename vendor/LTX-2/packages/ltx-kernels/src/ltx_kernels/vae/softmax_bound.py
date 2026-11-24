"""The one thing the kernels' softmax needs from the caller's weights.
Both kernels replace ``exp(s - rowmax)`` with ``exp2(s - bound)``. The base change is
internal (see ``fna_attn_core.LOG2E``); the *bound* is not, because it comes from a
weight only the caller has.
Using a fixed per-row bound instead of an online row max deletes the max pass, its
cross-warpgroup reduction and a read-modify-write of the output accumulator from every
step of the KV loop. It is legal because every score is a dot product of two
RMS-normed, RoPE'd vectors: Cauchy-Schwarz gives ``q . k <= ||q|| ||k||`` and RMSNorm
caps ``||k|| <= sqrt(head_dim) * max|k_norm_weight|``, which RoPE preserves.
Subtracting ``||q||`` times that is still *exactly* softmax -- one offset per row
divides out of the numerator and the denominator alike.
It is not a tolerance. Feed a kernel a K that was not RMS-normed with that weight, and
``exp2`` of a positive argument overflows.
"""

from __future__ import annotations

import torch


def softmax_row_bound(k_norm_weight: torch.Tensor) -> torch.Tensor:
    """``sqrt(head_dim) * max|k_norm_weight|`` -- the ``||k||`` bound, as a 0-d tensor.
    Left on the weight's own device: the kernels read it as an operand, so nothing here
    needs its value and no device-to-host sync happens.
    """
    head_dim = int(k_norm_weight.shape[-1])
    return k_norm_weight.detach().float().abs().amax() * head_dim**0.5
