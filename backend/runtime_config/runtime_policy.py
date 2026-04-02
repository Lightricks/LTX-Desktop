"""Runtime policy decisions for forced API mode."""

from __future__ import annotations


def decide_force_api_generations(system: str, cuda_available: bool, vram_gb: int | None) -> bool:
    """Return whether API-only generation must be forced for this runtime."""
    if system == "Darwin":
        return True

    if system in ("Windows", "Linux"):
        if not cuda_available:
            return True
        if vram_gb is None:
            return True
        return vram_gb < 15

    # Fail closed for non-target platforms unless explicitly relaxed.
    return True
