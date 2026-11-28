"""ltx-2.5-fast T2V/I2V canvases for a host that copies ``inference/``.

Desktop keeps its own envelope and does not import this table.

TODO: the Pod image in the next PR reads this table.
"""

from __future__ import annotations

SIZES_STAGE1_OR_TWO_PASS: frozenset[tuple[int, int]] = frozenset(
    {
        (576, 576),
        (576, 704),
        (576, 1024),
        (704, 704),
        (704, 896),
        (704, 1280),
        (768, 576),
        (896, 576),
        (960, 704),
        (1024, 576),
        (1088, 704),
        (1280, 704),
        (1344, 576),
        (1664, 704),
    }
)
SIZES_1080: frozenset[tuple[int, int]] = frozenset(
    {
        (1088, 1088),
        (1088, 1344),
        (1088, 1920),
        (1472, 1088),
        (1664, 1088),
        (1920, 1088),
        (2560, 1088),
    }
)
ALL_SIZES = SIZES_STAGE1_OR_TWO_PASS | SIZES_1080
ENVELOPE_FPS: tuple[int, ...] = (24, 25, 48, 50)
LONG_SECONDS: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10, 20)
SHORT_SECONDS: tuple[int, ...] = (2, 3, 4, 5, 10)


def frames_for(seconds: tuple[int, ...]) -> dict[int, frozenset[int]]:
    return {
        fps: frozenset(max(((duration * fps) // 8) * 8 + 1, 9) for duration in seconds)
        for fps in ENVELOPE_FPS
    }


LONG_FRAMES = frames_for(LONG_SECONDS)
SHORT_FRAMES = frames_for(SHORT_SECONDS)


def is_allowed_fast_geometry(
    *,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    skip_stage_2: bool,
) -> bool:
    if height % 64 != 0 or width % 64 != 0:
        return False
    if num_frames < 1 or (num_frames - 1) % 8 != 0:
        return False
    if fps not in ENVELOPE_FPS:
        return False
    if (width, height) not in ALL_SIZES:
        return False
    band = SHORT_FRAMES if (width, height) in SIZES_1080 else LONG_FRAMES
    if num_frames not in band[fps]:
        return False
    if skip_stage_2 and (width, height) in SIZES_1080:
        return False
    return True
