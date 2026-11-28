"""Metal-specific correction to the pinned ComfyUI v0.38 MiniMax H3 VAE."""

import hashlib
from pathlib import Path

# On MPS, constant temporal padding of a large 5-D tensor can silently corrupt
# the encoder latent. Explicit zero frames produce the same values on all devices.
# See pytorch/pytorch#194922 and Comfy-Org/ComfyUI#15902.
_VAE_SHA256 = "0168d55410794a2171c0507ca2dcb8a80e8cd288fa3e40b52fb460e4c92f3ffa"
_PAD = '                    x = F.pad(x, (0, 0, 0, 0, front, 0), mode="constant")'
_ZERO_FRAMES = """                    prefix = x.new_zeros((*x.shape[:2], front, *x.shape[3:]))
                    x = torch.cat((prefix, x), dim=2)"""


def prepare_h3_mps_core(core: Path) -> None:
    """Apply only to a fresh, verified source tree before the worker is activated."""
    vae = core / "comfy" / "ldm" / "minimax" / "vae.py"
    raw = vae.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _VAE_SHA256:
        raise ValueError("Unexpected MiniMax H3 VAE source for the Apple silicon worker.")
    text = raw.decode("utf-8")
    if text.count(_PAD) != 1:
        raise ValueError("Cannot locate the MiniMax H3 temporal padding operation.")
    vae.write_text(text.replace(_PAD, _ZERO_FRAMES), encoding="utf-8")
