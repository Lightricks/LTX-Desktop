"""Small byte-size helpers shared across server utilities."""

from __future__ import annotations


def gib(n_bytes: int) -> float:
    """Bytes to GiB."""
    return n_bytes / (1024**3)
