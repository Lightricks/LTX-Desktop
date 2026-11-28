"""Detect a machine that runs heavy local generations slowly."""

from __future__ import annotations

from runtime_config.runtime_config import RuntimeConfig


def detect_low_performance_machine(config: RuntimeConfig) -> bool:
    """True when heavy local generations are likely to take a long time.

    For now only Darwin counts. Extend this with RAM or VRAM checks when needed.
    """
    return config.darwin_unified_memory
