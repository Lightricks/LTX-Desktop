"""The frame rate of an IC-LoRA control video that is built from an image.

The frame count, the loop itself and the budget all read the rate from here. This module
imports nothing heavy, so the preprocessing layer can use it.
"""

from __future__ import annotations

DEFAULT_IC_LORA_FPS = 24


def resolve_loop_fps(step_fps: object, loop_fps: int | None) -> int:
    """The rate of a looped image. A request ``loop_fps`` wins over the step's ``fps`` param.

    The frame count and the loop itself both call this, so they cannot disagree.
    """
    return loop_fps or coerce_catalog_fps(step_fps)


def coerce_catalog_fps(value: object) -> int:
    """Catalog fps validator: positive ints pass, everything else defaults.

    Strict ``type() is`` checks with the bool case first (bool subclasses
    int, so ``isinstance(True, int)`` would admit it). Floats truncate,
    numeric strings parse, and None / objects / junk / non-positive fall
    back to ``DEFAULT_IC_LORA_FPS``.
    """
    if type(value) is bool:
        return DEFAULT_IC_LORA_FPS
    if type(value) is int and value > 0:
        return value
    if type(value) is float and value > 0:
        return int(value)
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return DEFAULT_IC_LORA_FPS
        if parsed > 0:
            return parsed
    return DEFAULT_IC_LORA_FPS
