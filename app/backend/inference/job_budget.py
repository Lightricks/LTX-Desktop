"""Token count used by Pod logs. Desktop stream-GiB curves stay in ``runtime_config``."""

from __future__ import annotations


def video_tokens(height: int, width: int, frames: int) -> float:
    """LTX-2 latent tokens: 8-frame temporal, 32 px spatial."""
    temporal = (max(1, frames) - 1) / 8.0 + 1.0
    return temporal * (height / 32.0) * (width / 32.0)
