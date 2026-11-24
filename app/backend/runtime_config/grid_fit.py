"""Aspect-preserving fit of a source size into a box, on a pixel grid."""

from __future__ import annotations

# The grid can force a shape error: 1280x720 has no 64-grid size with the same aspect
# that is not smaller than 1280x704 (2.3% off). Accept a size up to 3% off.
_ASPECT_TOLERANCE = 0.03


def _round_up(size: int, grid: int) -> int:
    return -(-size // grid) * grid


def grows_into_box(width: int, height: int, box_width: int, box_height: int, grid: int) -> bool:
    """True when the source, rounded up to the grid, is smaller than the box on both axes.

    A 1080p source rounds up to 1088 and already fills the 1080p cell, so only a source
    that would need a real upscale to reach the box counts. A wide clip that fills the
    width of the cell does not grow, whatever its height.
    """
    return min(box_width / _round_up(width, grid), box_height / _round_up(height, grid)) > 1.0


def _reach(size: int, max_scale: float, grid: int) -> int:
    """The edge a source edge may grow to: ``max_scale`` times it, rounded up to the grid.

    The round-up lets a 540p source (2x is 1080) fill the 1088 edge of the 1080p cell, the
    canvas a 1080p source gets. A scale of 1 keeps the source edge as it is.
    """
    if max_scale <= 1.0:
        return size
    return _round_up(round(size * max_scale), grid)


def fit_in_box(
    width: int,
    height: int,
    box_width: int,
    box_height: int,
    grid: int,
    *,
    max_scale: float = 1.0,
) -> tuple[int, int]:
    """Largest grid size inside the box and inside the source, close to the source aspect.

    The result never has an edge above the box or above ``max_scale`` times the source
    (rounded up to the grid). The default scale is 1, so it never upscales. Each grid size
    inside that limit has an aspect error against the source.
    The largest area wins among sizes within 3% of the source aspect. If the grid leaves
    none within 3%, the smallest error wins. A limit below one grid cell gives one cell:
    the minimum canvas, the same floor that the other two-stage recipes have.
    """
    limit_width = min(_reach(width, max_scale, grid), box_width)
    limit_height = min(_reach(height, max_scale, grid), box_height)
    aspect = width / height
    sizes = [
        (candidate_w, candidate_h)
        for candidate_w in range(grid, max(grid, limit_width // grid * grid) + 1, grid)
        for candidate_h in range(grid, max(grid, limit_height // grid * grid) + 1, grid)
    ]

    def aspect_error(size: tuple[int, int]) -> float:
        return abs(size[0] / size[1] - aspect) / aspect

    close = [size for size in sizes if aspect_error(size) <= _ASPECT_TOLERANCE]
    if close:
        return max(close, key=lambda size: (size[0] * size[1], -aspect_error(size)))
    return min(sizes, key=lambda size: (aspect_error(size), -size[0] * size[1]))
